"""Core data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Section:
    """A logical section of a parsed document (heading + body)."""

    title: str
    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    doc_id: str
    source: str
    title: str
    sections: list[Section]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    section: str
    text: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """Text used for embeddings: the section header gives each chunk its context."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass
class Candidate:
    """A retrieved chunk with its retrieval score and (optional) rerank score.

    ``rerank_score`` is always normalised to 0..1 so the relevance gate works
    the same way for every reranker.
    """

    chunk: Chunk
    score: float
    rerank_score: float | None = None
    retriever: str = ""

    @property
    def relevance(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


@dataclass
class Citation:
    number: int
    chunk_id: str
    doc_id: str
    source: str
    section: str
    page: int | None
    quote: str


@dataclass
class Answer:
    question: str
    answer: str
    grounded: bool
    citations: list[Citation]
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
