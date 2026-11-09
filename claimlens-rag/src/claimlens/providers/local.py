"""Local adapters: run the whole pipeline on a laptop with no cloud account.

Defaults are dependency-light and deterministic (hashing embedder, BM25 +
cosine hybrid store, lexical reranker, extractive LLM) so tests run offline.
Real local models (BGE embeddings, bge-reranker, Docling) switch on when the
``local-models`` extra is installed and selected in settings.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from claimlens.prompts import REFUSAL
from claimlens.types import Candidate, Chunk, ParsedDocument, Section

_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = set(
    "a an the of and or to in on for with is are was were be been by at as it this that from "
    "what which who whom when where why how does do did has have had not no yes can could "
    "should would will shall may might than then there their them they his her its our your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def doc_id_for(path: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "-", path.stem).strip("-").lower() or hashlib.sha1(
        str(path).encode()
    ).hexdigest()[:12]


# --------------------------------------------------------------------------- parsing


def split_markdown(text: str) -> tuple[str, list[Section]]:
    """Split markdown into sections on headings. Tracks Azure-style page-break markers."""
    title = ""
    sections: list[Section] = []
    current_title, buf, page = "Overview", [], 1
    section_page = 1

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(title=current_title, text=body, page=section_page))

    for line in text.splitlines():
        if "<!-- PageBreak -->" in line:
            page += 1
            continue
        if line.strip().startswith("<!--") and line.strip().endswith("-->"):
            continue  # other Document Intelligence comments (PageNumber, PageHeader...)
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            heading = m.group(2).strip()
            if len(m.group(1)) == 1 and not title:
                title = heading
            current_title, buf, section_page = heading, [], page
        else:
            buf.append(line)
    flush()
    return title, sections


class LocalParser:
    """Markdown/text natively; PDF/DOCX through Docling (or pypdf) when installed."""

    def __init__(self, use_docling: bool = True):
        self.use_docling = use_docling

    def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown", ".txt"}:
            text = path.read_text(encoding="utf-8")
        elif self.use_docling and _has("docling"):
            from docling.document_converter import DocumentConverter

            text = DocumentConverter().convert(str(path)).document.export_to_markdown()
        elif suffix == ".pdf" and _has("pypdf"):
            from pypdf import PdfReader

            pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
            text = "\n<!-- PageBreak -->\n".join(pages)
        else:
            raise ValueError(f"No local parser for {suffix}; install the 'local-models' extra (Docling)")
        title, sections = split_markdown(text)
        return ParsedDocument(
            doc_id=doc_id_for(path),
            source=path.name,
            title=title or path.stem,
            sections=sections,
            metadata=_sidecar_metadata(path),
        )


def _sidecar_metadata(path: Path) -> dict[str, Any]:
    """Optional `<file>.meta.json` next to a document, used for metadata filters."""
    meta = path.with_name(path.name + ".meta.json")
    return json.loads(meta.read_text()) if meta.exists() else {}


def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- embeddings


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams). No model download."""

    name = "hashing"

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            vec[h % self.dimension] += 1.0 if (h >> 64) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        return (vec / norm).tolist() if norm else vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbedder:
    """BGE (or any sentence-transformers model) running locally."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.model = SentenceTransformer(model)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.query_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model.lower() else ""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True)[0].tolist()


# --------------------------------------------------------------------------- vector store


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return scores


def matches_filters(chunk: Chunk, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        actual = getattr(chunk, key, None) if key in {"doc_id", "source", "section", "page"} else chunk.metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class LocalVectorStore:
    """Hybrid (BM25 + cosine, fused with RRF) store persisted to a JSON file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._bm25: BM25Okapi | None = None
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.chunks = [Chunk.from_dict(c) for c in data["chunks"]]
            self.vectors = np.array(data["vectors"], dtype=np.float32)

    def ensure_index(self, dimension: int) -> None:
        if self.vectors.size and self.vectors.shape[1] != dimension:
            raise ValueError(
                f"Index dimension {self.vectors.shape[1]} != embedder dimension {dimension}; "
                f"delete {self.path} to rebuild"
            )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"chunks": [c.to_dict() for c in self.chunks], "vectors": self.vectors.tolist()})
        )
        self._bm25 = None

    def delete_document(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if self.vectors.size else self.vectors
            self._save()
        return removed

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if not chunks:
            return 0
        existing = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        new_vecs = np.array(vectors, dtype=np.float32)
        if not self.vectors.size:
            self.vectors = np.zeros((0, new_vecs.shape[1]), dtype=np.float32)
        for chunk, vec in zip(chunks, new_vecs):
            if chunk.chunk_id in existing:
                i = existing[chunk.chunk_id]
                self.chunks[i] = chunk
                self.vectors[i] = vec
            else:
                existing[chunk.chunk_id] = len(self.chunks)
                self.chunks.append(chunk)
                self.vectors = np.vstack([self.vectors, vec])
        self._save()
        return len(chunks)

    def search(self, query: str, vector: list[float], k: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        if not self.chunks:
            return []
        allowed = [i for i, c in enumerate(self.chunks) if matches_filters(c, filters)]
        if not allowed:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi([tokenize(c.embedding_text()) or ["_"] for c in self.chunks])
        bm25_scores = self._bm25.get_scores(tokenize(query))
        dense_scores = self.vectors @ np.array(vector, dtype=np.float32)
        lexical = sorted(allowed, key=lambda i: bm25_scores[i], reverse=True)[: k * 2]
        lexical = [i for i in lexical if bm25_scores[i] > 0]
        dense = sorted(allowed, key=lambda i: dense_scores[i], reverse=True)[: k * 2]
        fused = rrf_fuse([[str(i) for i in lexical], [str(i) for i in dense]])
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Candidate(chunk=self.chunks[int(i)], score=s, retriever="local-hybrid") for i, s in ranked]


# --------------------------------------------------------------------------- rerankers


class LexicalReranker:
    """Cheap reranker: query-term coverage, weighted by IDF over the candidate set."""

    name = "lexical"

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        q = set(tokenize(query))
        if not q:
            return candidates[:top_n]
        docs = [set(tokenize(c.chunk.embedding_text())) for c in candidates]
        n = len(docs)
        idf = {t: math.log(1 + n / (1 + sum(t in d for d in docs))) for t in q}
        total = sum(idf.values()) or 1.0
        for c, d in zip(candidates, docs):
            c.rerank_score = sum(idf[t] for t in q if t in d) / total
        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]


class CrossEncoderReranker:
    """bge-reranker (or any cross-encoder) running locally; scores squashed to 0..1."""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.name = model
        self.model = CrossEncoder(model)

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]:
        if not candidates:
            return []
        logits = self.model.predict([(query, c.chunk.embedding_text()) for c in candidates])
        for c, s in zip(candidates, logits):
            c.rerank_score = float(1 / (1 + math.exp(-float(s))))
        return sorted(candidates, key=lambda c: c.rerank_score or 0.0, reverse=True)[:top_n]


# --------------------------------------------------------------------------- LLMs


class ExtractiveLLM:
    """Offline stand-in for an LLM: answers with the best-matching source sentences.

    It follows the same contract as a real model (cited sentences, or the
    refusal text), which makes the pipeline testable without API keys.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def generate(self, system: str, prompt: str) -> str:
        question = prompt.rsplit("Question:", 1)[-1].split("\n")[0]
        q = set(tokenize(question))
        sources = re.findall(r"\[(\d+)\] \(source:[^\n]*\)\n(.*?)(?=\n\n\[\d+\] \(source:|\n\nQuestion:)", prompt, re.S)
        scored: list[tuple[float, int, str]] = []
        for num, body in sources:
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                toks = set(tokenize(sent))
                if toks and q:
                    overlap = len(q & toks) / len(q)
                    if overlap > 0:
                        scored.append((overlap, int(num), sent.strip()))
        if not scored:
            return REFUSAL
        scored.sort(key=lambda x: x[0], reverse=True)
        picked, seen = [], set()
        for _, num, sent in scored:
            if sent in seen:
                continue
            seen.add(sent)
            picked.append(f"{sent} [{num}]")
            if len(picked) >= self.max_sentences:
                break
        return " ".join(picked)


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, vLLM, NVIDIA NIM, Ollama)."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
