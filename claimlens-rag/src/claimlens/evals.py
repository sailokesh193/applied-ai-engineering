"""Evaluation harness: retrieval hit rate, citation precision, answer coverage, refusals.

Golden file format (JSON lines):
  {"question": "...", "answerable": true, "expected_doc": "clm-1001-medical-record",
   "expected_terms": ["29881", "medial meniscus"]}

Run with thresholds to use it as a CI gate:
  claimlens eval evals/golden.jsonl --min-hit-rate 0.8 --min-refusal-accuracy 1.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claimlens.factory import RagPipeline


@dataclass
class EvalReport:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        ans = [r for r in self.rows if r["answerable"]]
        unans = [r for r in self.rows if not r["answerable"]]

        def mean(xs: list[float]) -> float:
            return round(sum(xs) / len(xs), 3) if xs else 0.0

        return {
            "questions": len(self.rows),
            "retrieval_hit_rate": mean([r["hit"] for r in ans]),
            "citation_precision": mean([r["citation_precision"] for r in ans if r["citations"]]),
            "answer_term_coverage": mean([r["term_coverage"] for r in ans]),
            "grounded_rate": mean([r["grounded"] for r in ans]),
            "refusal_accuracy": mean([1.0 if not r["grounded"] else 0.0 for r in unans]) if unans else 1.0,
            "p50_latency_ms": sorted(r["latency_ms"] for r in self.rows)[len(self.rows) // 2] if self.rows else 0.0,
        }


def run_eval(pipeline: RagPipeline, golden: Path) -> EvalReport:
    report = EvalReport()
    for line in golden.read_text().splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        ans = pipeline.ask(case["question"], case.get("filters"))
        top_docs = [t["source"].rsplit(".", 1)[0].lower() for t in ans.trace.get("top", [])]
        expected = (case.get("expected_doc") or "").lower()
        cited_docs = [c.doc_id.lower() for c in ans.citations]
        terms = [t.lower() for t in case.get("expected_terms", [])]
        found = sum(1 for t in terms if t in ans.answer.lower())
        report.rows.append(
            {
                "question": case["question"],
                "answerable": case.get("answerable", True),
                "grounded": float(ans.grounded),
                "hit": float(bool(expected) and any(expected in d for d in top_docs)),
                "citations": len(ans.citations),
                "citation_precision": (sum(1 for d in cited_docs if expected and expected in d) / len(cited_docs)) if cited_docs else 0.0,
                "term_coverage": (found / len(terms)) if terms else 1.0,
                "latency_ms": sum(ans.trace.get("timings_ms", {}).values()),
                "answer": ans.answer,
            }
        )
    return report
