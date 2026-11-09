"""Command line: ingest, ask, eval, serve, config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from claimlens.config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="claimlens", description="ClaimLens RAG pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="parse, chunk, embed and index files or folders")
    p_ing.add_argument("paths", nargs="+")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--filter", action="append", default=[], help="metadata filter key=value (repeatable)")
    p_ask.add_argument("--json", action="store_true", help="print the full answer with trace as JSON")

    p_eval = sub.add_parser("eval", help="run the golden-set evaluation")
    p_eval.add_argument("golden", type=Path)
    p_eval.add_argument("--min-hit-rate", type=float, default=0.0)
    p_eval.add_argument("--min-refusal-accuracy", type=float, default=0.0)
    p_eval.add_argument("--min-grounded-rate", type=float, default=0.0)
    p_eval.add_argument("--out", type=Path, help="write per-question results as JSON")

    p_srv = sub.add_parser("serve", help="run the API")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8000)

    sub.add_parser("config", help="show the resolved configuration")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("claimlens.api:app", host=args.host, port=args.port)
        return 0

    if args.cmd == "config":
        print(json.dumps(Settings.from_env().summary(), indent=2))
        return 0

    from claimlens.factory import RagPipeline

    pipeline = RagPipeline.from_settings()

    if args.cmd == "ingest":
        print(json.dumps(pipeline.ingest(args.paths), indent=2))
        return 0

    if args.cmd == "ask":
        filters = dict(f.split("=", 1) for f in args.filter)
        ans = pipeline.ask(args.question, filters or None)
        if args.json:
            print(json.dumps(ans.to_dict(), indent=2))
        else:
            print(ans.answer)
            print()
            for c in ans.citations:
                page = f", p.{c.page}" if c.page else ""
                print(f"  [{c.number}] {c.source} - {c.section}{page}")
            if not ans.grounded:
                print(f"  (not grounded: {ans.trace.get('note')})")
        return 0

    if args.cmd == "eval":
        from claimlens.evals import run_eval

        report = run_eval(pipeline, args.golden)
        metrics = report.metrics()
        print(json.dumps(metrics, indent=2))
        if args.out:
            args.out.write_text(json.dumps(report.rows, indent=2))
        failed = (
            metrics["retrieval_hit_rate"] < args.min_hit_rate
            or metrics["refusal_accuracy"] < args.min_refusal_accuracy
            or metrics["grounded_rate"] < args.min_grounded_rate
        )
        if failed:
            print("EVAL GATE FAILED", file=sys.stderr)
            return 1
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
