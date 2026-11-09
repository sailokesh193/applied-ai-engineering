"""Section-aware chunking.

Chunks never cross a section boundary, so a citation always points to one
heading in one document. Long sections are split on sentence boundaries with
a small overlap.
"""

from __future__ import annotations

import hashlib
import re

from claimlens.types import Chunk, ParsedDocument

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def approx_tokens(text: str) -> int:
    """Cheap token estimate (about 1.3 tokens per word) that avoids a tokenizer dependency."""
    return int(len(text.split()) * 1.3) + 1


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        if para.lstrip().startswith(("|", "-", "*")) or "\n|" in para:
            parts.append(para)  # keep tables and lists intact
        else:
            parts.extend(s.strip() for s in _SENTENCE.split(para) if s.strip())
    return parts


def chunk_id(doc_id: str, section: str, index: int) -> str:
    return hashlib.sha1(f"{doc_id}|{section}|{index}".encode()).hexdigest()[:20]


def chunk_document(doc: ParsedDocument, max_tokens: int = 350, overlap_tokens: int = 50) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        sentences = split_sentences(section.text)
        if not sentences:
            continue
        current: list[str] = []
        index = 0

        def flush() -> None:
            nonlocal index
            text = " ".join(current).strip()
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id(doc.doc_id, section.title, index),
                        doc_id=doc.doc_id,
                        source=doc.source,
                        section=section.title,
                        text=text,
                        page=section.page,
                        metadata=dict(doc.metadata),
                    )
                )
                index += 1

        for sentence in sentences:
            if current and approx_tokens(" ".join(current + [sentence])) > max_tokens:
                flush()
                # carry the tail of the previous chunk forward as overlap
                tail: list[str] = []
                for s in reversed(current):
                    if approx_tokens(" ".join([s] + tail)) > overlap_tokens:
                        break
                    tail.insert(0, s)
                current = tail
            current.append(sentence)
        flush()
    return chunks
