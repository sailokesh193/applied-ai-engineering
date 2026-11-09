"""FastAPI service: /health, /ingest (file upload), /query."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from claimlens import __version__
from claimlens.factory import SUPPORTED_EXTENSIONS, RagPipeline

app = FastAPI(title="ClaimLens RAG", version=__version__)
_pipeline: RagPipeline | None = None


def pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline.from_settings()
    return _pipeline


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    filters: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, Any]:
    p = pipeline()
    return {
        "status": "ok",
        "version": __version__,
        "config": p.settings.summary(),
        "components": {
            "retriever": getattr(p.retriever, "name", "?"),
            "reranker": getattr(p.reranker, "name", "none") if p.reranker else "none",
            "llm": getattr(p.llm, "name", "?"),
        },
    }


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    try:
        return pipeline().ask(req.question, req.filters).to_dict()
    except Exception as exc:  # surface provider errors as 502, not a stack trace
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    tmp = Path(tempfile.mkdtemp(prefix="claimlens-"))
    try:
        for f in files:
            name = Path(f.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {name}")
            with (tmp / name).open("wb") as out:
                shutil.copyfileobj(f.file, out)
        return pipeline().ingest([tmp])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
