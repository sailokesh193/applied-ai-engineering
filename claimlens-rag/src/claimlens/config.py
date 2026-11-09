"""Settings read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0]  # allow inline comments
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- what to run -------------------------------------------------------
    cloud: str = "local"  # local | aws | azure
    retrieval_mode: str = "pipeline"  # pipeline (own) | managed (Bedrock KB / Foundry IQ)
    parser: str = "auto"
    embedder: str = "auto"
    vector_store: str = "auto"
    reranker: str = "auto"  # auto | none | lexical | cross-encoder | bedrock | azure-semantic
    llm: str = "auto"  # auto | extractive | openai | bedrock | azure

    # --- pipeline tuning ---------------------------------------------------
    chunk_tokens: int = 350
    chunk_overlap: int = 50
    retrieve_k: int = 20
    rerank_top_n: int = 5
    min_relevance: float = 0.15

    # --- local -------------------------------------------------------------
    local_index_path: str = "data/index/local_index.json"
    local_embed_model: str = "BAAI/bge-small-en-v1.5"
    local_rerank_model: str = "BAAI/bge-reranker-base"
    openai_model: str = "gpt-4.1-mini"

    # --- AWS ---------------------------------------------------------------
    aws_region: str = "us-east-1"
    aws_docs_bucket: str = ""
    aws_docs_prefix: str = "claimlens/raw/"
    aoss_endpoint: str = ""
    aoss_index: str = "claimlens-chunks"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    bedrock_embed_dimensions: int = 1024
    bedrock_rerank_model_arn: str = ""
    bedrock_llm_model: str = ""
    bedrock_kb_id: str = ""
    bedrock_kb_data_source_id: str = ""
    bedrock_kb_managed: bool = False

    # --- Azure -------------------------------------------------------------
    azure_docintel_endpoint: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_embed_deployment: str = "text-embedding-3-large"
    azure_openai_embed_dimensions: int = 1536
    azure_openai_chat_deployment: str = "gpt-4.1"
    azure_search_endpoint: str = ""
    azure_search_index: str = "claimlens-chunks"
    azure_search_semantic: bool = True
    azure_search_api_version: str = "2026-04-01"
    azure_kb_name: str = ""
    azure_kb_source_name: str = ""
    azure_kb_content_field: str = "content"
    azure_storage_account_url: str = ""
    azure_storage_container: str = "claimlens-raw"
    azure_search_indexer: str = ""

    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        s = cls()
        for name, f in cls.__dataclass_fields__.items():
            if name == "extra":
                continue
            raw = os.environ.get(name.upper())
            if raw is None:
                continue
            if f.type in ("int", int):
                setattr(s, name, int(raw))
            elif f.type in ("float", float):
                setattr(s, name, float(raw))
            elif f.type in ("bool", bool):
                setattr(s, name, _env_bool(name.upper(), getattr(s, name)))
            else:
                setattr(s, name, raw.strip())
        s.cloud = s.cloud.lower()
        s.retrieval_mode = s.retrieval_mode.lower()
        return s

    def summary(self) -> dict[str, object]:
        return {
            "cloud": self.cloud,
            "retrieval_mode": self.retrieval_mode,
            "parser": self.parser,
            "embedder": self.embedder,
            "vector_store": self.vector_store,
            "reranker": self.reranker,
            "llm": self.llm,
            "retrieve_k": self.retrieve_k,
            "rerank_top_n": self.rerank_top_n,
            "min_relevance": self.min_relevance,
        }
