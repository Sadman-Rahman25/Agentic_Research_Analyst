"""src/pipeline.py

End-to-end pipeline driver. Wraps the LangGraph from src/graph.py with
a CLI for single-query and batch-eval modes.

USAGE:
    # Run on a single query
    python -m src.pipeline --query "your question here"

    # Run on all 10 eval queries (saves reports to examples/)
    python -m src.pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.config import PROJECT_ROOT, TRACE_DIR
from src.graph import build_graph
from src.schemas import Finding, ResearchPlan, VerificationResult


# Compile the graph once at module load — building it per-query is wasteful.
_APP = build_graph()


async def run_pipeline(user_query: str) -> dict:
    """Run the full graph on one query. Returns the final state dict.

    Final state has keys: plan, results_by_sq, findings, report_md,
    verification, timings, errors.
    """
    result = await _APP.ainvoke({"user_query": user_query})
    return result


def _print_summary(state: dict) -> None:
    """One-block-per-stage summary for terminal readability."""
    timings = state.get("timings", {})
    findings = state.get("findings", []) or []
    verification = state.get("verification")
    errors = state.get("errors", []) or []

    print(f"\n{'=' * 78}")
    print(f"QUERY: {state.get('user_query', '?')}")
    print(f"{'-' * 78}")
    for stage in ("plan", "search", "synthesize", "write", "verify"):
        if stage in timings:
            print(f"  {stage:11s}{timings[stage]:6.2f}s")
    if findings:
        print(f"  findings:   {len(findings)}")
    if isinstance(verification, VerificationResult):
        print(
            f"  grounding:  {verification.overall_grounding_score:.0%} "
            f"({verification.verified}/{verification.total_claims} verified, "
            f"{len(verification.flagged)} flagged)"
        )
    total = sum(timings.values()) if timings else 0
    print(f"  TOTAL:      {total:6.2f}s")
    if errors:
        print(f"  ERRORS:")
        for e in errors:
            print(f"    - {e}")
    print(f"{'=' * 78}\n")


async def _run_single(query: str) -> None:
    state = await run_pipeline(query)
    _print_summary(state)
    print(state.get("report_md", "(no report produced)"))


async def _run_eval() -> None:
    """Run pipeline on all 10 eval queries, save reports + log to JSONL."""
    queries_path = PROJECT_ROOT / "tests" / "test_queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples_dir = PROJECT_ROOT / "examples"
    examples_dir.mkdir(exist_ok=True)

    eval_log: list[dict] = []
    grand_start = time.perf_counter()

    for i, q in enumerate(data["queries"], 1):
        print(f"\n[{i}/{len(data['queries'])}] {q['id']}: {q['query']}")
        try:
            state = await run_pipeline(q["query"])
            _print_summary(state)

            report = state.get("report_md", "")
            out_path = examples_dir / f"{q['id']}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"  Report saved: {out_path}")

            verification = state.get("verification")
            eval_log.append(
                {
                    "query_id": q["id"],
                    "user_query": q["query"],
                    "n_findings": len(state.get("findings", []) or []),
                    "timings": state.get("timings", {}),
                    "grounding_score": (
                        verification.overall_grounding_score
                        if isinstance(verification, VerificationResult)
                        else None
                    ),
                    "n_flagged": (
                        len(verification.flagged)
                        if isinstance(verification, VerificationResult)
                        else None
                    ),
                    "errors": state.get("errors", []) or [],
                    "report_path": str(out_path.relative_to(PROJECT_ROOT)),
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"  PIPELINE FAILED for {q['id']}: {e}")
            eval_log.append({"query_id": q["id"], "error": str(e)})

    grand_total = time.perf_counter() - grand_start
    log_path = TRACE_DIR / "pipeline_eval.jsonl"
    with open(log_path, "w", encoding="utf-8") as f:
        for rec in eval_log:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    succeeded = [r for r in eval_log if "timings" in r]
    print(f"\nPipeline eval complete. {grand_total:.1f}s total.")
    print(f"Success: {len(succeeded)}/{len(eval_log)}")
    if succeeded:
        valid_timings = [r for r in succeeded if r["timings"]]
        if valid_timings:
            avg = sum(sum(r["timings"].values()) for r in valid_timings) / len(valid_timings)
            print(f"Average end-to-end per query: {avg:.2f}s")
        scored = [r for r in succeeded if r.get("grounding_score") is not None]
        if scored:
            avg_score = sum(r["grounding_score"] for r in scored) / len(scored)
            print(f"Average grounding score: {avg_score:.0%}")
    print(f"Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run on a single query instead of the full eval.",
    )
    args = parser.parse_args()

    if args.query:
        asyncio.run(_run_single(args.query))
    else:
        asyncio.run(_run_eval())


if __name__ == "__main__":
    main()