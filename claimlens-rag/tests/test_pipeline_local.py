from pathlib import Path

from claimlens.evals import run_eval
from claimlens.factory import RagPipeline
from claimlens.config import Settings

from .conftest import ROOT, SAMPLE


def test_answer_is_cited_from_the_right_document(local_pipeline):
    ans = local_pipeline.ask("What CPT code was billed on claim CLM-1001?")
    assert ans.grounded
    assert "29881" in ans.answer
    assert any(c.doc_id == "clm-1001-medical-record" for c in ans.citations)
    assert set(ans.trace["timings_ms"]) >= {"retrieve", "rerank", "generate"}


def test_out_of_scope_question_is_refused(local_pipeline):
    ans = local_pipeline.ask("Who won the football world cup in 2022?")
    assert not ans.grounded
    assert ans.citations == []
    assert "human reviewer" in ans.answer


def test_metadata_filter_limits_sources(local_pipeline):
    ans = local_pipeline.ask("What was the billed amount?", {"claim_id": "CLM-1002"})
    assert ans.grounded
    assert {c.doc_id for c in ans.citations} == {"clm-1002-medical-record"}
    assert "1,420" in ans.answer


def test_reingest_replaces_chunks_instead_of_duplicating(local_pipeline):
    before = len(local_pipeline.retriever.store.chunks)
    local_pipeline.ingest([SAMPLE])
    assert len(local_pipeline.retriever.store.chunks) == before


def test_index_persists_across_processes(tmp_path):
    s = Settings(cloud="local", local_index_path=str(tmp_path / "idx.json"))
    RagPipeline.from_settings(s).ingest([SAMPLE / "policy-knee-arthroscopy.md"])
    fresh = RagPipeline.from_settings(s)
    assert fresh.ask("Which outcome code is used when records are missing?").grounded


def test_golden_set_meets_ci_gate(local_pipeline):
    m = run_eval(local_pipeline, ROOT / "evals" / "golden.jsonl").metrics()
    assert m["retrieval_hit_rate"] >= 0.8
    assert m["refusal_accuracy"] == 1.0
    assert m["grounded_rate"] >= 0.8
