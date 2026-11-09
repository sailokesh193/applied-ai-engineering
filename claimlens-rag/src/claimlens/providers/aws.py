"""AWS adapters.

Parsing      Amazon Textract (async, LAYOUT + TABLES) on documents uploaded to S3
Embeddings   Amazon Bedrock Titan Text Embeddings V2
Vector store Amazon OpenSearch Serverless (vector search collection), hybrid via RRF
Reranker     Amazon Bedrock Rerank API (e.g. Cohere Rerank 3.5, Amazon Rerank)
LLM          Amazon Bedrock Converse API
Managed RAG  Amazon Bedrock Knowledge Bases (managed or customer-managed)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from claimlens.providers.local import LocalParser, doc_id_for, rrf_fuse
from claimlens.types import Candidate, Chunk, ParsedDocument, Section


def _boto3():
    import boto3

    return boto3


# --------------------------------------------------------------------------- parsing


_SECTION_START = {"LAYOUT_TITLE", "LAYOUT_SECTION_HEADER"}
_BODY = {"LAYOUT_TEXT", "LAYOUT_LIST", "LAYOUT_TABLE", "LAYOUT_KEY_VALUE", "LAYOUT_FIGURE"}


def textract_blocks_to_sections(blocks: list[dict[str, Any]]) -> tuple[str, list[Section]]:
    """Turn Textract LAYOUT blocks (reading order) into heading-delimited sections."""
    by_id = {b["Id"]: b for b in blocks}

    def text_of(block: dict[str, Any]) -> str:
        lines = []
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cid in rel["Ids"]:
                    child = by_id.get(cid, {})
                    if child.get("BlockType") == "LINE":
                        lines.append(child.get("Text", ""))
        return " ".join(lines).strip()

    title, sections = "", []
    current, buf, page = "Overview", [], 1
    section_page = 1
    for b in blocks:
        bt = b.get("BlockType", "")
        if not bt.startswith("LAYOUT_"):
            continue
        page = b.get("Page", page)
        text = text_of(b)
        if not text:
            continue
        if bt in _SECTION_START:
            if buf:
                sections.append(Section(current, "\n\n".join(buf), section_page))
            if bt == "LAYOUT_TITLE" and not title:
                title = text
            current, buf, section_page = text, [], page
        elif bt in _BODY:
            buf.append(text)
    if buf:
        sections.append(Section(current, "\n\n".join(buf), section_page))
    return title, sections


class TextractParser:
    """Uploads the file to S3 and runs asynchronous Textract layout analysis.

    Plain text and markdown skip Textract and use the local parser.
    """

    def __init__(self, bucket: str, prefix: str = "claimlens/raw/", region: str = "us-east-1", poll_seconds: float = 2.0):
        if not bucket:
            raise ValueError("AWS_DOCS_BUCKET is required for the Textract parser")
        boto3 = _boto3()
        self.bucket, self.prefix, self.poll = bucket, prefix, poll_seconds
        self.s3 = boto3.client("s3", region_name=region)
        self.textract = boto3.client("textract", region_name=region)
        self.fallback = LocalParser(use_docling=False)

    def parse(self, path: Path) -> ParsedDocument:
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            return self.fallback.parse(path)
        key = f"{self.prefix}{path.name}"
        self.s3.upload_file(str(path), self.bucket, key)
        job = self.textract.start_document_analysis(
            DocumentLocation={"S3Object": {"Bucket": self.bucket, "Name": key}},
            FeatureTypes=["LAYOUT", "TABLES"],
        )
        blocks = self._collect(job["JobId"])
        title, sections = textract_blocks_to_sections(blocks)
        return ParsedDocument(
            doc_id=doc_id_for(path),
            source=path.name,
            title=title or path.stem,
            sections=sections,
            metadata={"s3_uri": f"s3://{self.bucket}/{key}"},
        )

    def _collect(self, job_id: str, timeout_s: float = 900) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout_s
        while True:
            resp = self.textract.get_document_analysis(JobId=job_id)
            status = resp["JobStatus"]
            if status in {"SUCCEEDED", "PARTIAL_SUCCESS"}:  # partial: keep the pages that worked
                break
            if status == "FAILED":
                raise RuntimeError(f"Textract job {job_id} failed: {resp.get('StatusMessage')}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Textract job {job_id} still {status} after {timeout_s}s")
            time.sleep(self.poll)
        blocks = list(resp["Blocks"])
        token = resp.get("NextToken")
        while token:
            resp = self.textract.get_document_analysis(JobId=job_id, NextToken=token)
            blocks.extend(resp["Blocks"])
            token = resp.get("NextToken")
        return blocks


# --------------------------------------------------------------------------- embeddings


class BedrockEmbedder:
    """Titan Text Embeddings V2 (one text per call, parallelised)."""

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", dimensions: int = 1024, region: str = "us-east-1", workers: int = 8, client: Any = None):
        self.name = f"bedrock:{model_id}"
        self.model_id, self.dimension, self.workers = model_id, dimensions, workers
        self.client = client or _boto3().client("bedrock-runtime", region_name=region)

    def _one(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text[:50000], "dimensions": self.dimension, "normalize": True})
        resp = self.client.invoke_model(modelId=self.model_id, body=body, contentType="application/json", accept="application/json")
        return json.loads(resp["body"].read())["embedding"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(self._one, texts))

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


# --------------------------------------------------------------------------- vector store


class OpenSearchServerlessStore:
    """OpenSearch Serverless vector collection with client-side hybrid fusion (RRF).

    Notes for vector search collections: refresh is automatic (writes appear
    within seconds) and documents are indexed without custom ``_id``s, so the
    chunk id is stored as a keyword field and deletes go through a search.
    """

    def __init__(self, endpoint: str, index: str, region: str = "us-east-1", client: Any = None):
        if client is None:
            if not endpoint:
                raise ValueError("AOSS_ENDPOINT is required for the OpenSearch Serverless store")
            from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

            creds = _boto3().Session().get_credentials()
            host = endpoint.replace("https://", "").rstrip("/")
            client = OpenSearch(
                hosts=[{"host": host, "port": 443}],
                http_auth=AWSV4SignerAuth(creds, region, "aoss"),
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                pool_maxsize=20,
                timeout=60,
            )
        self.client, self.index = client, index

    def ensure_index(self, dimension: int) -> None:
        if self.client.indices.exists(index=self.index):
            return
        self.client.indices.create(
            index=self.index,
            body={
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "method": {"name": "hnsw", "engine": "faiss", "space_type": "l2"},
                        },
                        "text": {"type": "text"},
                        "section": {"type": "text"},
                        "chunk_id": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "metadata": {"type": "object"},
                    }
                },
            },
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        from opensearchpy import helpers

        actions = [
            {
                "_index": self.index,
                "_source": {
                    "vector": v,
                    "text": c.text,
                    "section": c.section,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "page": c.page,
                    "metadata": c.metadata,
                },
            }
            for c, v in zip(chunks, vectors)
        ]
        ok, _ = helpers.bulk(self.client, actions, raise_on_error=True)
        return ok

    def delete_document(self, doc_id: str) -> int:
        if not self.client.indices.exists(index=self.index):
            return 0
        hits = self.client.search(
            index=self.index, body={"size": 10000, "_source": False, "query": {"term": {"doc_id": doc_id}}}
        )["hits"]["hits"]
        for h in hits:
            self.client.delete(index=self.index, id=h["_id"])
        return len(hits)

    @staticmethod
    def _filter_clauses(filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        clauses = []
        for key, value in (filters or {}).items():
            field = key if key in {"doc_id", "source", "page"} else f"metadata.{key}"
            if isinstance(value, (list, tuple, set)):
                clauses.append({"terms": {field: list(value)}})
            else:
                clauses.append({"term": {field: value}})
        return clauses

    def search(self, query: str, vector: list[float], k: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        clauses = self._filter_clauses(filters)
        knn: dict[str, Any] = {"vector": vector, "k": k * 2}
        if clauses:
            knn["filter"] = {"bool": {"filter": clauses}}
        dense = self.client.search(index=self.index, body={"size": k * 2, "query": {"knn": {"vector": knn}}, "_source": {"excludes": ["vector"]}})
        lexical = self.client.search(
            index=self.index,
            body={
                "size": k * 2,
                "query": {"bool": {"must": [{"multi_match": {"query": query, "fields": ["text", "section^2"]}}], "filter": clauses}},
                "_source": {"excludes": ["vector"]},
            },
        )
        docs: dict[str, dict[str, Any]] = {}
        rankings = []
        for res in (lexical, dense):
            ids = []
            for h in res["hits"]["hits"]:
                src = h["_source"]
                docs[src["chunk_id"]] = src
                ids.append(src["chunk_id"])
            rankings.append(ids)
        fused = sorted(rrf_fuse(rankings).items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            Candidate(
                chunk=Chunk(
                    chunk_id=cid,
                    doc_id=docs[cid]["doc_id"],
                    source=docs[cid]["source"],
                    section=docs[cid].get("section", ""),
                    text=docs[cid]["text"],
                    page=docs[cid].get("page"),
                    metadata=docs[cid].get("metadata") or {},
                ),
                score=score,
                retriever="aoss-hybrid",
            )
            for cid, score in fused
        ]


# --------------------------------------------------------------------------- reranker


class BedrockReranker:
    """Bedrock Rerank API. ``relevanceScore`` is already 0..1."""

    def __init__(self, model_arn: str, region: str = "us-east-1", client: Any = None):
        if not model_arn:
            raise ValueError("BEDROCK_RERANK_MODEL_ARN is required for the Bedrock reranker")
        self.name = f"bedrock-rerank:{model_arn.rsplit('/', 1)[-1]}"
        self.model_arn = model_arn
        self.client = client or _boto3().client("bedrock-agent-runtime", region_name=region)

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        if not candidates:
            return []
        resp = self.client.rerank(
            queries=[{"type": "TEXT", "textQuery": {"text": query}}],
            sources=[
                {"type": "INLINE", "inlineDocumentSource": {"type": "TEXT", "textDocument": {"text": c.chunk.embedding_text()}}}
                for c in candidates
            ],
            rerankingConfiguration={
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": min(top_n, len(candidates)),
                    "modelConfiguration": {"modelArn": self.model_arn},
                },
            },
        )
        out = []
        for r in resp["results"]:
            c = candidates[r["index"]]
            c.rerank_score = float(r["relevanceScore"])
            out.append(c)
        return sorted(out, key=lambda c: c.rerank_score or 0.0, reverse=True)


# --------------------------------------------------------------------------- LLM


class BedrockLLM:
    """Bedrock Converse API (works with Claude, Llama, Nova, Mistral...)."""

    def __init__(self, model_id: str, region: str = "us-east-1", max_tokens: int = 800, client: Any = None):
        if not model_id:
            raise ValueError("BEDROCK_LLM_MODEL is required (a model ID or inference profile ID)")
        self.name = f"bedrock:{model_id}"
        self.model_id, self.max_tokens = model_id, max_tokens
        self.client = client or _boto3().client("bedrock-runtime", region_name=region)

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": 0.0},
        )
        parts = resp["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts)


# --------------------------------------------------------------------------- managed RAG


def bedrock_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not filters:
        return None
    clauses = [
        {"in": {"key": k, "value": list(v)}} if isinstance(v, (list, tuple, set)) else {"equals": {"key": k, "value": v}}
        for k, v in filters.items()
    ]
    return clauses[0] if len(clauses) == 1 else {"andAll": clauses}


class BedrockKnowledgeBaseRetriever:
    """Retrieve from a Bedrock Knowledge Base (managed or customer-managed)."""

    name = "bedrock-kb"

    def __init__(self, kb_id: str, managed: bool = False, region: str = "us-east-1", client: Any = None):
        if not kb_id:
            raise ValueError("BEDROCK_KB_ID is required for managed retrieval on AWS")
        self.kb_id, self.managed = kb_id, managed
        self.client = client or _boto3().client("bedrock-agent-runtime", region_name=region)

    def retrieve(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        cfg: dict[str, Any] = {"numberOfResults": k}
        f = bedrock_filter(filters)
        if f:
            cfg["filter"] = f
        key = "managedSearchConfiguration" if self.managed else "vectorSearchConfiguration"
        resp = self.client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={key: cfg},
        )
        out = []
        for i, r in enumerate(resp.get("retrievalResults", [])):
            loc = r.get("location", {})
            uri = (
                loc.get("s3Location", {}).get("uri")
                or loc.get("webLocation", {}).get("url")
                or loc.get("confluenceLocation", {}).get("url")
                or loc.get("sharePointLocation", {}).get("url")
                or loc.get("salesforceLocation", {}).get("url")
                or loc.get("type", "kb")
            )
            meta = r.get("metadata", {}) or {}
            source = uri.rsplit("/", 1)[-1] if isinstance(uri, str) else "kb"
            page = meta.get("x-amz-bedrock-kb-document-page-number")
            out.append(
                Candidate(
                    chunk=Chunk(
                        chunk_id=str(meta.get("x-amz-bedrock-kb-chunk-id", f"{self.kb_id}-{i}")),
                        doc_id=Path(source).stem.lower(),
                        source=source,
                        section=str(meta.get("section", "")),
                        text=r.get("content", {}).get("text", ""),
                        page=int(page) if isinstance(page, (int, float)) else None,
                        metadata=meta,
                    ),
                    score=float(r.get("score", 0.0)),
                    retriever=self.name,
                )
            )
        return out


class BedrockKnowledgeBaseIngestor:
    """Uploads files to the KB's S3 data source and starts an ingestion (sync) job."""

    def __init__(self, kb_id: str, data_source_id: str, bucket: str, prefix: str, region: str = "us-east-1"):
        if not (kb_id and data_source_id and bucket):
            raise ValueError("BEDROCK_KB_ID, BEDROCK_KB_DATA_SOURCE_ID and AWS_DOCS_BUCKET are required")
        boto3 = _boto3()
        self.kb_id, self.ds_id, self.bucket, self.prefix = kb_id, data_source_id, bucket, prefix
        self.s3 = boto3.client("s3", region_name=region)
        self.agent = boto3.client("bedrock-agent", region_name=region)

    def ingest(self, paths: list[Path]) -> dict[str, Any]:
        uploaded = []
        for p in paths:
            key = f"{self.prefix}{p.name}"
            self.s3.upload_file(str(p), self.bucket, key)
            meta = p.with_name(p.name + ".meta.json")
            if meta.exists():  # Bedrock KB metadata sidecar format
                body = {"metadataAttributes": json.loads(meta.read_text())}
                self.s3.put_object(Bucket=self.bucket, Key=f"{key}.metadata.json", Body=json.dumps(body).encode())
            uploaded.append(key)
        job = self.agent.start_ingestion_job(knowledgeBaseId=self.kb_id, dataSourceId=self.ds_id)
        return {"uploaded": uploaded, "ingestion_job_id": job["ingestionJob"]["ingestionJobId"]}
