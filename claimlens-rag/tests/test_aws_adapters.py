"""AWS adapters tested offline with botocore Stubber and fake clients."""

import io
import json

import boto3
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

from claimlens.providers.aws import (
    BedrockEmbedder,
    BedrockKnowledgeBaseRetriever,
    BedrockLLM,
    BedrockReranker,
    OpenSearchServerlessStore,
    bedrock_filter,
    textract_blocks_to_sections,
)
from claimlens.types import Candidate, Chunk

REGION = "us-east-1"


def _cand(i, text):
    return Candidate(Chunk(f"c{i}", "d", "d.md", "S", text), score=0.1)


def test_converse_request_and_response():
    client = boto3.client("bedrock-runtime", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y")
    with Stubber(client) as stub:
        stub.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "Billed CPT 29881 [1]."}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                "metrics": {"latencyMs": 100},
            },
            {
                "modelId": "my-model",
                "system": [{"text": "sys"}],
                "messages": [{"role": "user", "content": [{"text": "prompt"}]}],
                "inferenceConfig": {"maxTokens": 800, "temperature": 0.0},
            },
        )
        assert BedrockLLM("my-model", client=client).generate("sys", "prompt") == "Billed CPT 29881 [1]."


def test_rerank_maps_indices_and_scores():
    client = boto3.client("bedrock-agent-runtime", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y")
    arn = "arn:aws:bedrock:us-west-2::foundation-model/cohere.rerank-v3-5:0"
    with Stubber(client) as stub:
        stub.add_response(
            "rerank",
            {"results": [{"index": 1, "relevanceScore": 0.92}, {"index": 0, "relevanceScore": 0.11}]},
            {"queries": ANY, "sources": ANY, "rerankingConfiguration": {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {"numberOfResults": 2, "modelConfiguration": {"modelArn": arn}}}},
        )
        out = BedrockReranker(arn, client=client).rerank("q", [_cand(0, "a"), _cand(1, "b")], top_n=5)
    assert [c.chunk.chunk_id for c in out] == ["c1", "c0"]
    assert out[0].rerank_score == 0.92


def test_titan_embedding_body():
    client = boto3.client("bedrock-runtime", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y")
    payload = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
    with Stubber(client) as stub:
        stub.add_response(
            "invoke_model",
            {"body": StreamingBody(io.BytesIO(payload), len(payload)), "contentType": "application/json"},
            {"modelId": "amazon.titan-embed-text-v2:0", "body": json.dumps({"inputText": "hello", "dimensions": 3, "normalize": True}),
             "contentType": "application/json", "accept": "application/json"},
        )
        assert BedrockEmbedder(dimensions=3, client=client).embed_query("hello") == [0.1, 0.2, 0.3]


def test_kb_retrieve_managed_config_and_mapping():
    client = boto3.client("bedrock-agent-runtime", region_name=REGION, aws_access_key_id="x", aws_secret_access_key="y")
    with Stubber(client) as stub:
        stub.add_response(
            "retrieve",
            {"retrievalResults": [{
                "content": {"text": "Use outcome code PEND when records are missing."},
                "location": {"type": "S3", "s3Location": {"uri": "s3://b/claimlens/raw/policy-knee-arthroscopy.md"}},
                "score": 0.81,
                "metadata": {"x-amz-bedrock-kb-chunk-id": "abc"},
            }]},
            {"knowledgeBaseId": "KB12345678", "retrievalQuery": {"text": "q"},
             "retrievalConfiguration": {"managedSearchConfiguration": {"numberOfResults": 5, "filter": {"equals": {"key": "doc_type", "value": "policy"}}}}},
        )
        out = BedrockKnowledgeBaseRetriever("KB12345678", managed=True, client=client).retrieve("q", 5, {"doc_type": "policy"})
    assert out[0].chunk.source == "policy-knee-arthroscopy.md"
    assert out[0].chunk.doc_id == "policy-knee-arthroscopy"
    assert out[0].score == 0.81


def test_bedrock_filter_shapes():
    assert bedrock_filter({"a": 1}) == {"equals": {"key": "a", "value": 1}}
    assert bedrock_filter({"a": 1, "b": ["x", "y"]}) == {"andAll": [{"equals": {"key": "a", "value": 1}}, {"in": {"key": "b", "value": ["x", "y"]}}]}


def test_textract_layout_blocks_to_sections():
    blocks = [
        {"Id": "l1", "BlockType": "LINE", "Text": "Claim CLM-1001"},
        {"Id": "l2", "BlockType": "LINE", "Text": "Operative Note"},
        {"Id": "l3", "BlockType": "LINE", "Text": "Partial medial meniscectomy performed."},
        {"Id": "t", "BlockType": "LAYOUT_TITLE", "Page": 1, "Relationships": [{"Type": "CHILD", "Ids": ["l1"]}]},
        {"Id": "h", "BlockType": "LAYOUT_SECTION_HEADER", "Page": 2, "Relationships": [{"Type": "CHILD", "Ids": ["l2"]}]},
        {"Id": "x", "BlockType": "LAYOUT_TEXT", "Page": 2, "Relationships": [{"Type": "CHILD", "Ids": ["l3"]}]},
        {"Id": "f", "BlockType": "LAYOUT_FOOTER", "Page": 2, "Relationships": [{"Type": "CHILD", "Ids": ["l1"]}]},
    ]
    title, sections = textract_blocks_to_sections(blocks)
    assert title == "Claim CLM-1001"
    assert [(s.title, s.page) for s in sections] == [("Operative Note", 2)]
    assert "meniscectomy" in sections[0].text


class _FakeOpenSearch:
    def __init__(self):
        self.calls = []

    def search(self, index, body):
        self.calls.append(body)
        hits = [{"_source": {"chunk_id": cid, "doc_id": "d", "source": "d.md", "section": "S", "text": cid, "page": 1}} for cid in (["a", "b"] if "knn" in body["query"] else ["b", "c"])]
        return {"hits": {"hits": hits}}


def test_opensearch_hybrid_fuses_lexical_and_dense():
    fake = _FakeOpenSearch()
    out = OpenSearchServerlessStore(endpoint="", index="i", client=fake).search("q", [0.0, 1.0], k=3, filters={"claim_id": "CLM-1"})
    assert out[0].chunk.chunk_id == "b"  # appears in both rankings
    knn_body = next(b for b in fake.calls if "knn" in b["query"])
    assert knn_body["query"]["knn"]["vector"]["filter"]["bool"]["filter"] == [{"term": {"metadata.claim_id": "CLM-1"}}]
