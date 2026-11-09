"""LangGraph graphs for ingestion and question answering.

Ingest:  parse -> chunk -> embed -> index
Query:   retrieve -> rerank -> (relevance gate) -> generate -> verify citations
                                    \\-> refuse
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from claimlens.chunking import chunk_document
from claimlens.prompts import REFUSAL, SYSTEM_PROMPT, build_prompt
from claimlens.providers.base import LLM, Embedder, Parser, Reranker, Retriever, VectorStore
from claimlens.types import Answer, Candidate, Chunk, Citation, ParsedDocument

# --------------------------------------------------------------------------- ingest


class IngestState(TypedDict, total=False):
    paths: list[str]
    docs: list[ParsedDocument]
    chunks: list[Chunk]
    vectors: list[list[float]]
    indexed: int
    errors: list[str]
    timings_ms: dict[str, float]


def build_ingest_graph(
    parser: Parser,
    embedder: Embedder,
    store: VectorStore,
    chunk_tokens: int = 350,
    chunk_overlap: int = 50,
    embed_batch: int = 32,
):
    def _timed(state: IngestState, step: str, start: float) -> dict[str, float]:
        timings = dict(state.get("timings_ms", {}))
        timings[step] = round((time.perf_counter() - start) * 1000, 1)
        return timings

    def parse(state: IngestState) -> IngestState:
        t = time.perf_counter()
        docs, errors = [], list(state.get("errors", []))
        for p in state["paths"]:
            try:
                docs.append(parser.parse(Path(p)))
            except Exception as exc:  # keep going; report per-file failures
                errors.append(f"{p}: {exc}")
        return {"docs": docs, "errors": errors, "timings_ms": _timed(state, "parse", t)}

    def chunk(state: IngestState) -> IngestState:
        t = time.perf_counter()
        chunks: list[Chunk] = []
        for doc in state.get("docs", []):
            chunks.extend(chunk_document(doc, chunk_tokens, chunk_overlap))
        return {"chunks": chunks, "timings_ms": _timed(state, "chunk", t)}

    def embed(state: IngestState) -> IngestState:
        t = time.perf_counter()
        chunks = state.get("chunks", [])
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), embed_batch):
            batch = chunks[i : i + embed_batch]
            vectors.extend(embedder.embed_documents([c.embedding_text() for c in batch]))
        return {"vectors": vectors, "timings_ms": _timed(state, "embed", t)}

    def index(state: IngestState) -> IngestState:
        t = time.perf_counter()
        store.ensure_index(embedder.dimension)
        for doc in state.get("docs", []):
            store.delete_document(doc.doc_id)  # re-ingest replaces old chunks
        n = store.upsert(state.get("chunks", []), state.get("vectors", []))
        return {"indexed": n, "timings_ms": _timed(state, "index", t)}

    g = StateGraph(IngestState)
    g.add_node("parse", parse)
    g.add_node("chunk", chunk)
    g.add_node("embed", embed)
    g.add_node("index", index)
    g.add_edge(START, "parse")
    g.add_edge("parse", "chunk")
    g.add_edge("chunk", "embed")
    g.add_edge("embed", "index")
    g.add_edge("index", END)
    return g.compile()


# --------------------------------------------------------------------------- query


class QueryState(TypedDict, total=False):
    question: str
    filters: dict[str, Any]
    candidates: list[Candidate]
    reranked: list[Candidate]
    raw_answer: str
    answer: Answer
    timings_ms: dict[str, float]


_CITE = re.compile(r"\[(\d+)\]")


def build_query_graph(
    retriever: Retriever,
    reranker: Reranker | None,
    llm: LLM,
    retrieve_k: int = 20,
    rerank_top_n: int = 5,
    min_relevance: float = 0.15,
):
    def _timed(state: QueryState, step: str, start: float) -> dict[str, float]:
        timings = dict(state.get("timings_ms", {}))
        timings[step] = round((time.perf_counter() - start) * 1000, 1)
        return timings

    def retrieve(state: QueryState) -> QueryState:
        t = time.perf_counter()
        cands = retriever.retrieve(state["question"], retrieve_k, state.get("filters") or None)
        return {"candidates": cands, "timings_ms": _timed(state, "retrieve", t)}

    def rerank(state: QueryState) -> QueryState:
        t = time.perf_counter()
        cands = state.get("candidates", [])
        if reranker is None or not cands:
            top = sorted(cands, key=lambda c: c.relevance, reverse=True)[:rerank_top_n]
        else:
            top = reranker.rerank(state["question"], cands, rerank_top_n)
        return {"reranked": top, "timings_ms": _timed(state, "rerank", t)}

    def gate(state: QueryState) -> str:
        top = state.get("reranked", [])
        if not top:
            return "refuse"
        # Only enforce the threshold when scores are normalised rerank scores.
        if top[0].rerank_score is not None and top[0].rerank_score < min_relevance:
            return "refuse"
        return "generate"

    def refuse(state: QueryState) -> QueryState:
        return {
            "answer": Answer(
                question=state["question"],
                answer=REFUSAL,
                grounded=False,
                citations=[],
                trace=_trace(state, "refused: no source above relevance threshold"),
            )
        }

    def generate(state: QueryState) -> QueryState:
        t = time.perf_counter()
        prompt = build_prompt(state["question"], state["reranked"])
        raw = llm.generate(SYSTEM_PROMPT, prompt)
        return {"raw_answer": raw, "timings_ms": _timed(state, "generate", t)}

    def verify(state: QueryState) -> QueryState:
        top = state["reranked"]
        raw = state.get("raw_answer", "").strip()
        used = sorted({int(n) for n in _CITE.findall(raw)})
        valid = [n for n in used if 1 <= n <= len(top)]
        invalid = [n for n in used if n not in valid]
        for n in invalid:  # drop citations to sources that were never provided
            raw = raw.replace(f"[{n}]", "")
        refused = REFUSAL.split(".")[0].lower() in raw.lower()
        citations = [
            Citation(
                number=n,
                chunk_id=top[n - 1].chunk.chunk_id,
                doc_id=top[n - 1].chunk.doc_id,
                source=top[n - 1].chunk.source,
                section=top[n - 1].chunk.section,
                page=top[n - 1].chunk.page,
                quote=top[n - 1].chunk.text[:300],
            )
            for n in valid
        ]
        grounded = bool(citations) and not refused
        note = "ok" if grounded else ("model refused" if refused else "no valid citations")
        if invalid:
            note += f"; removed invalid citations {invalid}"
        return {
            "answer": Answer(
                question=state["question"],
                answer=raw if raw else REFUSAL,
                grounded=grounded,
                citations=citations,
                trace=_trace(state, note),
            )
        }

    def _trace(state: QueryState, note: str) -> dict[str, Any]:
        return {
            "note": note,
            "retriever": getattr(retriever, "name", "?"),
            "reranker": getattr(reranker, "name", "none") if reranker else "none",
            "llm": getattr(llm, "name", "?"),
            "candidates": len(state.get("candidates", [])),
            "top": [
                {
                    "chunk_id": c.chunk.chunk_id,
                    "source": c.chunk.source,
                    "section": c.chunk.section,
                    "score": round(c.score, 4),
                    "rerank_score": None if c.rerank_score is None else round(c.rerank_score, 4),
                }
                for c in state.get("reranked", [])
            ],
            "timings_ms": state.get("timings_ms", {}),
        }

    g = StateGraph(QueryState)
    g.add_node("retrieve", retrieve)
    g.add_node("rerank", rerank)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("refuse", refuse)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_conditional_edges("rerank", gate, {"generate": "generate", "refuse": "refuse"})
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    g.add_edge("refuse", END)
    return g.compile()
