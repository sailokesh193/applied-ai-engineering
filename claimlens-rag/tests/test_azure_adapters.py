"""Azure adapters tested offline with fake clients and an httpx mock transport."""

import json
from types import SimpleNamespace

import httpx

from claimlens.providers.azure import (
    AzureAISearchStore,
    AzureOpenAIEmbedder,
    AzureOpenAILLM,
    AzureSemanticReranker,
    FoundryIQRetriever,
    odata_filter,
)


def test_odata_filter_builds_tag_and_field_clauses():
    f = odata_filter({"doc_id": "a'b", "claim_id": ["CLM-1", "CLM-2"]})
    assert "doc_id eq 'a''b'" in f
    assert "tags/any(t: t eq 'claim_id:CLM-1' or t eq 'claim_id:CLM-2')" in f


def test_foundry_iq_request_and_reference_mapping(monkeypatch):
    monkeypatch.setenv("AZURE_SEARCH_API_KEY", "k")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["key"] = request.headers.get("api-key")
        return httpx.Response(200, json={
            "response": [{"content": [{"type": "text", "text": "[]"}]}],
            "references": [{
                "type": "searchIndex", "id": "0", "docKey": "k1", "rerankerScore": 3.2,
                "sourceData": {"chunk_id": "c1", "doc_id": "policy-knee-arthroscopy", "source": "policy-knee-arthroscopy.md",
                               "section": "Post-Payment Review Outcome Codes", "content": "Use outcome code PEND when records are missing."},
            }],
        })

    r = FoundryIQRetriever("https://s.search.windows.net", "claimlens-kb", "claimlens-ks", http=httpx.Client(transport=httpx.MockTransport(handler)))
    out = r.retrieve("missing records code?", 5, {"doc_type": "policy"})
    assert "/knowledgebases('claimlens-kb')/retrieve?api-version=2026-04-01" in seen["url"]
    assert seen["body"]["intents"] == [{"type": "semantic", "search": "missing records code?"}]
    assert seen["body"]["knowledgeSourceParams"][0]["filterAddOn"] == "tags/any(t: t eq 'doc_type:policy')"
    assert seen["key"] == "k"
    assert out[0].chunk.chunk_id == "c1" and abs(out[0].rerank_score - 0.8) < 1e-9


class _FakeSearchClient:
    def __init__(self):
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        return [
            {"chunk_id": "c1", "doc_id": "d", "source": "d.md", "page": 1, "section": "S", "content": "x",
             "metadata_json": "{}", "@search.score": 0.03, "@search.reranker_score": 3.0},
            {"chunk_id": "c2", "doc_id": "d", "source": "d.md", "page": 1, "section": "S", "content": "y",
             "metadata_json": "{}", "@search.score": 0.02, "@search.reranker_score": 1.0},
        ]


def test_ai_search_hybrid_semantic_query_and_scores():
    fake = _FakeSearchClient()
    store = AzureAISearchStore(endpoint="", index="i", semantic=True, search_client=fake, index_client=object())
    out = store.search("knee", [0.1, 0.2], k=5, filters={"claim_id": "CLM-1001"})
    assert fake.kwargs["query_type"] == "semantic"
    assert fake.kwargs["vector_queries"][0].fields == "vector"
    assert out[0].rerank_score == 0.75
    top = AzureSemanticReranker().rerank("knee", out, 1)
    assert [c.chunk.chunk_id for c in top] == ["c1"]


def test_azure_openai_embedder_and_llm_use_deployments():
    calls = {}

    def emb_create(model, input, dimensions):
        calls["emb"] = (model, len(input), dimensions)
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(input))])

    def chat_create(**kw):
        calls["chat"] = kw["model"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok [1]"))])

    fake = SimpleNamespace(embeddings=SimpleNamespace(create=emb_create), chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)))
    assert AzureOpenAIEmbedder("", "emb-dep", 1536, client=fake).embed_documents(["a", "b"]) == [[0.0], [1.0]]
    assert calls["emb"] == ("emb-dep", 2, 1536)
    assert AzureOpenAILLM("", "chat-dep", client=fake).generate("s", "p") == "ok [1]"
    assert calls["chat"] == "chat-dep"
