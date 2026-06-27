"""src/pipeline.py

End-to-end pipeline: query → plan → search → synthesize → write.

This is the integration point for Day 3. Run as:
    python -m src.pipeline

The output is a markdown report per query, saved to examples/ and printed
to stdout. The Verifier (Day 4) plugs into the END of this pipeline.

USAGE:
    # Run on all 10 eval queries (cached search results — cheap)
    python -m src.pipeline

    # Run on one query (good for prompt iteration)
    python -m src.pipeline --query "your question here"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from src.agents.planner import plan as planner_plan
from src.agents.synthesizer import synthesize_all
from src.agents.writer import write_report
from src.config import PROJECT_ROOT, TRACE_DIR
from src.orchestrator import gather_results
from src.schemas import Finding, ResearchPlan


async def run_pipeline(user_query: str) -> tuple[ResearchPlan, list[Finding], str, dict]:
    """Run the full pipeline. Returns (plan, findings, markdown_report, timings)."""
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    plan_obj = planner_plan(user_query)
    timings["plan"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = await gather_results(plan_obj)
    timings["search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    findings = await synthesize_all(plan_obj.sub_questions, results)
    timings["synthesize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    report = write_report(plan_obj, findings)
    timings["write"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    return plan_obj, findings, report, timings


def _print_summary(query: str, findings: list[Finding], timings: dict) -> None:
    """One-liner per stage for terminal readability."""
    print(f"\n{'=' * 78}")
    print(f"QUERY: {query}")
    print(f"{'-' * 78}")
    print(f"  Plan:       {timings['plan']:5.2f}s")
    print(f"  Search:     {timings['search']:5.2f}s")
    print(f"  Synthesize: {timings['synthesize']:5.2f}s   ({len(findings)} findings)")
    print(f"  Write:      {timings['write']:5.2f}s")
    print(f"  TOTAL:      {timings['total']:5.2f}s")
    print(f"{'=' * 78}\n")


async def _run_single(query: str) -> None:
    plan_obj, findings, report, timings = await run_pipeline(query)
    _print_summary(query, findings, timings)
    print(report)


async def _run_eval() -> None:
    """Run pipeline on all 10 eval queries, save reports to examples/."""
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
            plan_obj, findings, report, timings = await run_pipeline(q["query"])
            _print_summary(q["query"], findings, timings)

            out_path = examples_dir / f"{q['id']}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved: {out_path}")

            eval_log.append(
                {
                    "query_id": q["id"],
                    "user_query": q["query"],
                    "n_findings": len(findings),
                    "timings": timings,
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
        avg = sum(r["timings"]["total"] for r in succeeded) / len(succeeded)
        print(f"Average end-to-end per query: {avg:.2f}s")
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