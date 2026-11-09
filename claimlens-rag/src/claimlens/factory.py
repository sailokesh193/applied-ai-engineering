"""Build the pipeline from settings: one switch for the cloud, one for the retrieval mode.

    CLOUD=local|aws|azure        RETRIEVAL_MODE=pipeline|managed

Every component can also be overridden individually (PARSER, EMBEDDER,
VECTOR_STORE, RERANKER, LLM), e.g. Azure search with a Bedrock reranker.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claimlens.config import Settings
from claimlens.graphs import build_ingest_graph, build_query_graph
from claimlens.providers.base import (
    LLM,
    Embedder,
    ManagedIngestor,
    Parser,
    PipelineRetriever,
    Reranker,
    Retriever,
    VectorStore,
)
from claimlens.types import Answer

DEFAULTS = {
    "local": {"parser": "local", "embedder": "hashing", "vector_store": "local", "reranker": "lexical", "llm": "extractive"},
    "aws": {"parser": "textract", "embedder": "bedrock", "vector_store": "opensearch", "reranker": "bedrock", "llm": "bedrock"},
    "azure": {"parser": "docintel", "embedder": "azure-openai", "vector_store": "azure-search", "reranker": "azure-semantic", "llm": "azure"},
}
SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ".docx", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _pick(s: Settings, component: str) -> str:
    value = getattr(s, component)
    if value and value != "auto":
        return value
    if component == "llm" and s.cloud == "local" and os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if component == "reranker" and s.retrieval_mode == "managed":
        return "none"  # managed services already rerank
    return DEFAULTS[s.cloud][component]


def make_parser(s: Settings) -> Parser:
    kind = _pick(s, "parser")
    if kind == "local":
        from claimlens.providers.local import LocalParser

        return LocalParser()
    if kind == "textract":
        from claimlens.providers.aws import TextractParser

        return TextractParser(s.aws_docs_bucket, s.aws_docs_prefix, s.aws_region)
    if kind == "docintel":
        from claimlens.providers.azure import DocumentIntelligenceParser

        return DocumentIntelligenceParser(s.azure_docintel_endpoint)
    raise ValueError(f"Unknown parser '{kind}'")


def make_embedder(s: Settings) -> Embedder:
    kind = _pick(s, "embedder")
    if kind == "hashing":
        from claimlens.providers.local import HashingEmbedder

        return HashingEmbedder()
    if kind in {"bge", "sentence-transformers"}:
        from claimlens.providers.local import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(s.local_embed_model)
    if kind == "bedrock":
        from claimlens.providers.aws import BedrockEmbedder

        return BedrockEmbedder(s.bedrock_embed_model, s.bedrock_embed_dimensions, s.aws_region)
    if kind == "azure-openai":
        from claimlens.providers.azure import AzureOpenAIEmbedder

        return AzureOpenAIEmbedder(
            s.azure_openai_endpoint, s.azure_openai_embed_deployment, s.azure_openai_embed_dimensions, s.azure_openai_api_version
        )
    raise ValueError(f"Unknown embedder '{kind}'")


def make_store(s: Settings) -> VectorStore:
    kind = _pick(s, "vector_store")
    if kind == "local":
        from claimlens.providers.local import LocalVectorStore

        return LocalVectorStore(s.local_index_path)
    if kind == "opensearch":
        from claimlens.providers.aws import OpenSearchServerlessStore

        return OpenSearchServerlessStore(s.aoss_endpoint, s.aoss_index, s.aws_region)
    if kind == "azure-search":
        from claimlens.providers.azure import AzureAISearchStore

        return AzureAISearchStore(s.azure_search_endpoint, s.azure_search_index, s.azure_search_semantic)
    raise ValueError(f"Unknown vector store '{kind}'")


def make_reranker(s: Settings) -> Reranker | None:
    kind = _pick(s, "reranker")
    if kind == "none":
        return None
    if kind == "lexical":
        from claimlens.providers.local import LexicalReranker

        return LexicalReranker()
    if kind == "cross-encoder":
        from claimlens.providers.local import CrossEncoderReranker

        return CrossEncoderReranker(s.local_rerank_model)
    if kind == "bedrock":
        from claimlens.providers.aws import BedrockReranker

        return BedrockReranker(s.bedrock_rerank_model_arn, s.aws_region)
    if kind == "azure-semantic":
        from claimlens.providers.azure import AzureSemanticReranker

        return AzureSemanticReranker()
    raise ValueError(f"Unknown reranker '{kind}'")


def make_llm(s: Settings) -> LLM:
    kind = _pick(s, "llm")
    if kind == "extractive":
        from claimlens.providers.local import ExtractiveLLM

        return ExtractiveLLM()
    if kind == "openai":
        from claimlens.providers.local import OpenAICompatibleLLM

        return OpenAICompatibleLLM(s.openai_model, os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_KEY"))
    if kind == "bedrock":
        from claimlens.providers.aws import BedrockLLM

        return BedrockLLM(s.bedrock_llm_model, s.aws_region)
    if kind == "azure":
        from claimlens.providers.azure import AzureOpenAILLM

        return AzureOpenAILLM(s.azure_openai_endpoint, s.azure_openai_chat_deployment, s.azure_openai_api_version)
    raise ValueError(f"Unknown LLM '{kind}'")


def make_managed_retriever(s: Settings) -> Retriever:
    if s.cloud == "aws":
        from claimlens.providers.aws import BedrockKnowledgeBaseRetriever

        return BedrockKnowledgeBaseRetriever(s.bedrock_kb_id, s.bedrock_kb_managed, s.aws_region)
    if s.cloud == "azure":
        from claimlens.providers.azure import FoundryIQRetriever

        return FoundryIQRetriever(
            s.azure_search_endpoint, s.azure_kb_name, s.azure_kb_source_name, s.azure_kb_content_field, s.azure_search_api_version
        )
    raise ValueError("RETRIEVAL_MODE=managed needs CLOUD=aws (Bedrock KB) or CLOUD=azure (Foundry IQ)")


def make_managed_ingestor(s: Settings) -> ManagedIngestor:
    if s.cloud == "aws":
        from claimlens.providers.aws import BedrockKnowledgeBaseIngestor

        return BedrockKnowledgeBaseIngestor(s.bedrock_kb_id, s.bedrock_kb_data_source_id, s.aws_docs_bucket, s.aws_docs_prefix, s.aws_region)
    if s.cloud == "azure":
        from claimlens.providers.azure import FoundryIQIngestor

        return FoundryIQIngestor(s.azure_storage_account_url, s.azure_storage_container, s.azure_search_endpoint, s.azure_search_indexer)
    raise ValueError("Managed ingestion needs CLOUD=aws or CLOUD=azure")


def collect_files(paths: list[str | Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    return files


@dataclass
class RagPipeline:
    """Facade used by the API, CLI and evals."""

    settings: Settings
    retriever: Retriever
    reranker: Reranker | None
    llm: LLM
    _ingest_graph: Any = None
    _managed_ingestor: ManagedIngestor | None = None
    _query_graph: Any = None

    @classmethod
    def from_settings(cls, s: Settings | None = None) -> "RagPipeline":
        s = s or Settings.from_env()
        if s.cloud not in DEFAULTS:
            raise ValueError(f"CLOUD must be one of {sorted(DEFAULTS)}, got '{s.cloud}'")
        if s.retrieval_mode == "managed":
            retriever = make_managed_retriever(s)
            ingest_graph, ingestor = None, None
        elif s.retrieval_mode == "pipeline":
            embedder, store = make_embedder(s), make_store(s)
            retriever = PipelineRetriever(embedder, store)
            ingest_graph = build_ingest_graph(make_parser(s), embedder, store, s.chunk_tokens, s.chunk_overlap)
            ingestor = None
        else:
            raise ValueError("RETRIEVAL_MODE must be 'pipeline' or 'managed'")
        reranker, llm = make_reranker(s), make_llm(s)
        p = cls(s, retriever, reranker, llm, ingest_graph, ingestor)
        p._query_graph = build_query_graph(retriever, reranker, llm, s.retrieve_k, s.rerank_top_n, s.min_relevance)
        return p

    def ingest(self, paths: list[str | Path]) -> dict[str, Any]:
        files = collect_files(paths)
        if self.settings.retrieval_mode == "managed":
            if self._managed_ingestor is None:
                self._managed_ingestor = make_managed_ingestor(self.settings)
            return {"mode": "managed", "files": len(files), **self._managed_ingestor.ingest(files)}
        state = self._ingest_graph.invoke({"paths": [str(f) for f in files]})
        return {
            "mode": "pipeline",
            "files": len(files),
            "documents": len(state.get("docs", [])),
            "chunks": len(state.get("chunks", [])),
            "indexed": state.get("indexed", 0),
            "errors": state.get("errors", []),
            "timings_ms": state.get("timings_ms", {}),
        }

    def ask(self, question: str, filters: dict[str, Any] | None = None) -> Answer:
        state = self._query_graph.invoke({"question": question, "filters": filters or {}})
        return state["answer"]
