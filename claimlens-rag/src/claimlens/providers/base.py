"""Interfaces every cloud adapter implements.

The LangGraph graphs only talk to these protocols, so swapping AWS for Azure
(or for the local stack) is a configuration change, not a code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from claimlens.types import Candidate, Chunk, ParsedDocument


@runtime_checkable
class Parser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    def ensure_index(self, dimension: int) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...

    def delete_document(self, doc_id: str) -> int: ...

    def search(
        self,
        query: str,
        vector: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[Candidate]: ...


@runtime_checkable
class Retriever(Protocol):
    """Anything that turns a question into candidates (own pipeline or managed RAG)."""

    name: str

    def retrieve(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[Candidate]: ...


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Candidate], top_n: int) -> list[Candidate]: ...


@runtime_checkable
class LLM(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str: ...


@runtime_checkable
class ManagedIngestor(Protocol):
    """Pushes raw files to a managed RAG service and triggers its ingestion."""

    def ingest(self, paths: list[Path]) -> dict[str, Any]: ...


class PipelineRetriever:
    """Own-pipeline retrieval: embed the query, then hybrid search the vector store."""

    name = "pipeline"

    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[Candidate]:
        vector = self.embedder.embed_query(query)
        results = self.store.search(query, vector, k, filters)
        for c in results:
            c.retriever = self.name
        return results
