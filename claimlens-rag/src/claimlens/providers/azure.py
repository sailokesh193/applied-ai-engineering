"""Azure adapters.

Parsing      Azure AI Document Intelligence, prebuilt-layout model, markdown output
Embeddings   Azure OpenAI in Microsoft Foundry (text-embedding-3-*)
Vector store Azure AI Search: hybrid (BM25 + vector, RRF) + semantic ranker
Reranker     Azure AI Search semantic ranker (scores reused), or any other reranker
LLM          Azure OpenAI chat completions
Managed RAG  Foundry IQ knowledge base (Azure AI Search agentic retrieval, 2026-04-01 API)

Authentication uses Microsoft Entra ID (DefaultAzureCredential) unless an API
key is provided through AZURE_SEARCH_API_KEY / AZURE_OPENAI_API_KEY.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx

from claimlens.providers.local import LocalParser, doc_id_for, split_markdown
from claimlens.types import Candidate, Chunk, ParsedDocument

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"
SEARCH_SCOPE = "https://search.azure.com/.default"


def _credential():
    from azure.identity import DefaultAzureCredential

    return DefaultAzureCredential()


def _search_credential():
    key = os.environ.get("AZURE_SEARCH_API_KEY")
    if key:
        from azure.core.credentials import AzureKeyCredential

        return AzureKeyCredential(key)
    return _credential()


def _openai_client(endpoint: str, api_version: str):
    from openai import AzureOpenAI

    key = os.environ.get("AZURE_OPENAI_API_KEY")
    if key:
        return AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, api_key=key)
    from azure.identity import get_bearer_token_provider

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=get_bearer_token_provider(_credential(), COGNITIVE_SCOPE),
    )


# --------------------------------------------------------------------------- parsing


class DocumentIntelligenceParser:
    """prebuilt-layout with markdown output: headings, tables and page breaks preserved."""

    def __init__(self, endpoint: str, client: Any = None):
        if client is None:
            if not endpoint:
                raise ValueError("AZURE_DOCINTEL_ENDPOINT is required for the Document Intelligence parser")
            from azure.ai.documentintelligence import DocumentIntelligenceClient

            key = os.environ.get("AZURE_DOCINTEL_API_KEY")
            if key:
                from azure.core.credentials import AzureKeyCredential

                cred: Any = AzureKeyCredential(key)
            else:
                cred = _credential()
            client = DocumentIntelligenceClient(endpoint=endpoint, credential=cred)
        self.client = client
        self.fallback = LocalParser(use_docling=False)

    def parse(self, path: Path) -> ParsedDocument:
        if path.suffix.lower() in {".md", ".markdown", ".txt"}:
            return self.fallback.parse(path)
        with path.open("rb") as f:
            poller = self.client.begin_analyze_document("prebuilt-layout", body=f, output_content_format="markdown")
        result = poller.result()
        title, sections = split_markdown(result.content or "")
        return ParsedDocument(doc_id=doc_id_for(path), source=path.name, title=title or path.stem, sections=sections)


# --------------------------------------------------------------------------- embeddings


class AzureOpenAIEmbedder:
    def __init__(self, endpoint: str, deployment: str, dimensions: int = 1536, api_version: str = "2024-10-21", client: Any = None):
        if client is None and not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI embeddings")
        self.name = f"azure-openai:{deployment}"
        self.deployment, self.dimension = deployment, dimensions
        self.client = client or _openai_client(endpoint, api_version)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            resp = self.client.embeddings.create(model=self.deployment, input=texts[i : i + 64], dimensions=self.dimension)
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# --------------------------------------------------------------------------- vector store


def search_key(chunk_id: str) -> str:
    """Azure AI Search keys allow letters, digits, '_', '-' and '='."""
    return hashlib.sha1(chunk_id.encode()).hexdigest()


def odata_filter(filters: dict[str, Any] | None) -> str | None:
    if not filters:
        return None
    parts = []
    for key, value in filters.items():
        values = list(value) if isinstance(value, (list, tuple, set)) else [value]
        if key in {"doc_id", "source"}:
            parts.append("(" + " or ".join(f"{key} eq '{str(v).replace(chr(39), chr(39) * 2)}'" for v in values) + ")")
        elif key == "page":
            parts.append("(" + " or ".join(f"page eq {int(v)}" for v in values) + ")")
        else:
            tags = " or ".join(f"t eq '{key}:{str(v).replace(chr(39), chr(39) * 2)}'" for v in values)
            parts.append(f"tags/any(t: {tags})")
    return " and ".join(parts)


class AzureAISearchStore:
    """Azure AI Search index with HNSW vectors, BM25 and an optional semantic ranker.

    One request runs keyword + vector search (merged with RRF by the service).
    With semantic ranking on, ``@search.reranker_score`` (0-4) is stored on the
    candidate as a 0..1 ``rerank_score``.
    """

    SEMANTIC_CONFIG = "claimlens-semantic"

    def __init__(self, endpoint: str, index: str, semantic: bool = True, search_client: Any = None, index_client: Any = None):
        if search_client is None:
            if not endpoint:
                raise ValueError("AZURE_SEARCH_ENDPOINT is required for the Azure AI Search store")
            from azure.search.documents import SearchClient
            from azure.search.documents.indexes import SearchIndexClient

            cred = _search_credential()
            search_client = SearchClient(endpoint=endpoint, index_name=index, credential=cred)
            index_client = SearchIndexClient(endpoint=endpoint, credential=cred)
        self.search_client, self.index_client = search_client, index_client
        self.index, self.semantic = index, semantic

    def ensure_index(self, dimension: int) -> None:
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            SimpleField,
            VectorSearch,
            VectorSearchProfile,
        )

        if self.index in set(self.index_client.list_index_names()):
            return
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
            SearchableField(name="section", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="tags", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True),
            SimpleField(name="metadata_json", type=SearchFieldDataType.String),
            SearchField(
                name="vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=dimension,
                vector_search_profile_name="hnsw-profile",
            ),
        ]
        index = SearchIndex(
            name=self.index,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
                profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name=self.SEMANTIC_CONFIG,
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="section"),
                            content_fields=[SemanticField(field_name="content")],
                        ),
                    )
                ]
            ),
        )
        self.index_client.create_or_update_index(index)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        docs = [
            {
                "id": search_key(c.chunk_id),
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "page": c.page,
                "section": c.section,
                "content": c.text,
                "tags": [f"{k}:{v}" for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))],
                "metadata_json": json.dumps(c.metadata),
                "vector": v,
            }
            for c, v in zip(chunks, vectors)
        ]
        done = 0
        for i in range(0, len(docs), 500):
            results = self.search_client.merge_or_upload_documents(documents=docs[i : i + 500])
            done += sum(1 for r in results if r.succeeded)
        return done

    def delete_document(self, doc_id: str) -> int:
        try:
            hits = list(self.search_client.search(search_text="*", filter=odata_filter({"doc_id": doc_id}), select=["id"], top=1000))
        except Exception:  # index does not exist yet
            return 0
        if hits:
            self.search_client.delete_documents(documents=[{"id": h["id"]} for h in hits])
        return len(hits)

    def search(self, query: str, vector: list[float], k: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        from azure.search.documents.models import VectorizedQuery

        kwargs: dict[str, Any] = {
            "search_text": query,
            "vector_queries": [VectorizedQuery(vector=vector, k_nearest_neighbors=max(50, k), fields="vector")],
            "filter": odata_filter(filters),
            "select": ["chunk_id", "doc_id", "source", "page", "section", "content", "metadata_json"],
            "top": k,
        }
        if self.semantic:
            kwargs.update(query_type="semantic", semantic_configuration_name=self.SEMANTIC_CONFIG)
        out = []
        for r in self.search_client.search(**kwargs):
            rr = r.get("@search.reranker_score")
            out.append(
                Candidate(
                    chunk=Chunk(
                        chunk_id=r["chunk_id"],
                        doc_id=r["doc_id"],
                        source=r["source"],
                        section=r.get("section") or "",
                        text=r["content"],
                        page=r.get("page"),
                        metadata=json.loads(r.get("metadata_json") or "{}"),
                    ),
                    score=float(r.get("@search.score", 0.0)),
                    rerank_score=None if rr is None else float(rr) / 4.0,
                    retriever="azure-search-hybrid" + ("+semantic" if self.semantic else ""),
                )
            )
        return out


# --------------------------------------------------------------------------- reranker


class AzureSemanticReranker:
    """Uses the semantic ranker scores Azure AI Search already returned.

    Pair it with ``AzureAISearchStore(semantic=True)`` so no second call is needed.
    """

    name = "azure-semantic"

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        scored = [c for c in candidates if c.rerank_score is not None]
        if not scored:
            return candidates[:top_n]
        return sorted(scored, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]


# --------------------------------------------------------------------------- LLM


class AzureOpenAILLM:
    def __init__(self, endpoint: str, deployment: str, api_version: str = "2024-10-21", max_tokens: int = 800, client: Any = None):
        if client is None and not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for the Azure OpenAI LLM")
        self.name = f"azure-openai:{deployment}"
        self.deployment, self.max_tokens = deployment, max_tokens
        self.client = client or _openai_client(endpoint, api_version)

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.deployment,
            temperature=0,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- managed RAG


class FoundryIQRetriever:
    """Foundry IQ knowledge base retrieve action (Azure AI Search, API 2026-04-01).

    The stable API takes ``intents`` and returns extractive grounding data with
    ``references`` (sourceData + rerankerScore). Answer synthesis and LLM query
    planning are preview-only, so generation stays in our own LLM step.
    """

    name = "foundry-iq"

    def __init__(
        self,
        endpoint: str,
        knowledge_base: str,
        knowledge_source: str = "",
        content_field: str = "content",
        api_version: str = "2026-04-01",
        http: httpx.Client | None = None,
    ):
        if not (endpoint and knowledge_base):
            raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_KB_NAME are required for Foundry IQ retrieval")
        self.url = f"{endpoint.rstrip('/')}/knowledgebases('{knowledge_base}')/retrieve"
        self.knowledge_source, self.content_field, self.api_version = knowledge_source, content_field, api_version
        self.http = http or httpx.Client(timeout=60)
        self._cred = None

    def _headers(self) -> dict[str, str]:
        key = os.environ.get("AZURE_SEARCH_API_KEY")
        if key:
            return {"api-key": key, "Content-Type": "application/json"}
        if self._cred is None:
            self._cred = _credential()
        token = self._cred.get_token(SEARCH_SCOPE).token
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _body(self, query: str, filters: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "intents": [{"type": "semantic", "search": query}],
            "includeActivity": True,
            "maxRuntimeInSeconds": 30,
        }
        if self.knowledge_source:
            params: dict[str, Any] = {
                "knowledgeSourceName": self.knowledge_source,
                "kind": "searchIndex",
                "includeReferences": True,
                "includeReferenceSourceData": True,
            }
            f = odata_filter(filters)
            if f:
                params["filterAddOn"] = f
            body["knowledgeSourceParams"] = [params]
        return body

    def retrieve(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        resp = self.http.post(self.url, params={"api-version": self.api_version}, headers=self._headers(), json=self._body(query, filters))
        resp.raise_for_status()
        data = resp.json()
        out: list[Candidate] = []
        for i, ref in enumerate(data.get("references", [])[:k]):
            src = ref.get("sourceData") or {}
            text = src.get(self.content_field) or src.get("content") or src.get("page_chunk") or ""
            if not text:
                continue
            rr = ref.get("rerankerScore")
            source = str(src.get("source") or src.get("title") or ref.get("docKey") or f"ref-{i}")
            out.append(
                Candidate(
                    chunk=Chunk(
                        chunk_id=str(src.get("chunk_id") or ref.get("docKey") or ref.get("id") or i),
                        doc_id=str(src.get("doc_id") or Path(source).stem.lower()),
                        source=source,
                        section=str(src.get("section") or src.get("title") or ""),
                        text=text,
                        page=src.get("page") if isinstance(src.get("page"), int) else None,
                        metadata={k2: v for k2, v in src.items() if k2 not in {self.content_field, "vector"}},
                    ),
                    score=float(rr or 0.0),
                    rerank_score=None if rr is None else float(rr) / 4.0,
                    retriever=self.name,
                )
            )
        if not out:  # no references requested: fall back to the extractive response text
            for msg in data.get("response", []):
                for part in msg.get("content", []):
                    if part.get("type") != "text":
                        continue
                    try:
                        items = json.loads(part["text"])
                    except (ValueError, TypeError):
                        items = [{"content": part["text"]}]
                    for j, item in enumerate(items if isinstance(items, list) else [items]):
                        text = item.get("content") if isinstance(item, dict) else str(item)
                        if text:
                            out.append(
                                Candidate(
                                    chunk=Chunk(chunk_id=f"kb-{j}", doc_id="kb", source=str(item.get("title", "knowledge-base")) if isinstance(item, dict) else "knowledge-base", section="", text=text),
                                    score=1.0 / (j + 1),
                                    retriever=self.name,
                                )
                            )
        return out[:k]


class FoundryIQIngestor:
    """Uploads files to the Blob container behind the knowledge source, then runs its indexer."""

    def __init__(self, account_url: str, container: str, search_endpoint: str = "", indexer: str = ""):
        if not account_url:
            raise ValueError("AZURE_STORAGE_ACCOUNT_URL is required for managed ingestion on Azure")
        from azure.storage.blob import BlobServiceClient

        self.container = BlobServiceClient(account_url=account_url, credential=_credential()).get_container_client(container)
        self.search_endpoint, self.indexer = search_endpoint, indexer

    def ingest(self, paths: list[Path]) -> dict[str, Any]:
        uploaded = []
        for p in paths:
            with p.open("rb") as f:
                self.container.upload_blob(name=p.name, data=f, overwrite=True)
            uploaded.append(p.name)
        ran = False
        if self.indexer and self.search_endpoint:
            from azure.search.documents.indexes import SearchIndexerClient

            SearchIndexerClient(endpoint=self.search_endpoint, credential=_search_credential()).run_indexer(self.indexer)
            ran = True
        return {"uploaded": uploaded, "indexer_triggered": ran}
