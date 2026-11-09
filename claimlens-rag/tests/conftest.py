from __future__ import annotations

from pathlib import Path

import pytest

from claimlens.config import Settings
from claimlens.factory import RagPipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture()
def local_pipeline(tmp_path, monkeypatch) -> RagPipeline:
    for var in ["CLOUD", "RETRIEVAL_MODE", "LLM", "RERANKER", "OPENAI_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    s = Settings(cloud="local", local_index_path=str(tmp_path / "index.json"))
    p = RagPipeline.from_settings(s)
    p.ingest([SAMPLE])
    return p
