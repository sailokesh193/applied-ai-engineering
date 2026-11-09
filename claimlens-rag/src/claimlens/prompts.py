"""Prompts for grounded, cited generation."""

from __future__ import annotations

from claimlens.types import Candidate

REFUSAL = (
    "I don't have enough evidence in the provided documents to answer this. "
    "Please route it to a human reviewer."
)

SYSTEM_PROMPT = """You are ClaimLens, an assistant for clinical claims reviewers.
Answer ONLY from the numbered sources provided. Rules:
- Cite every factual sentence with the source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "{refusal}"
- Never invent codes, amounts, dates or clinical facts.
- Be concise: 2-6 sentences, plain language.""".format(refusal=REFUSAL)


def format_sources(candidates: list[Candidate]) -> str:
    blocks = []
    for i, c in enumerate(candidates, start=1):
        page = f", page {c.chunk.page}" if c.chunk.page else ""
        blocks.append(
            f"[{i}] (source: {c.chunk.source}, section: {c.chunk.section}{page})\n{c.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, candidates: list[Candidate]) -> str:
    return f"Sources:\n\n{format_sources(candidates)}\n\nQuestion: {question}\n\nAnswer with citations:"
