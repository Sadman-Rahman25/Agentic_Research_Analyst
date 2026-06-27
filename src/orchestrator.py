"""src/orchestrator.py

The orchestrator. Takes a ResearchPlan and dispatches each sub-question to
the appropriate researcher IN PARALLEL via asyncio.gather().

Parallelism is critical for UX: serial would take ~30s per query (10 sub-q ×
~3s each). Parallel completes in roughly the slowest single researcher call,
typically 3-5s.

Each researcher fails gracefully (returns []), so one bad researcher call
does not poison the rest of the pipeline.

Run end-to-end manual eval:
    python -m src.orchestrator
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Awaitable, Callable

from src.agents.planner import plan as planner_plan
from src.config import PROJECT_ROOT, TRACE_DIR
from src.researchers.arxiv import arxiv_search
from src.researchers.github import github_search
from src.researchers.web import web_search
from src.schemas import ResearchPlan, SearchResult, SubQuestion


# Researcher registry — maps source tag → async researcher function
_RESEARCHERS: dict[str, Callable[..., Awaitable[list[SearchResult]]]] = {
    "web": web_search,
    "arxiv": arxiv_search,
    "github": github_search,
}


async def _dispatch_one(sq: SubQuestion, top_k: int) -> tuple[str, list[SearchResult]]:
    """Send a single sub-question to the right researcher. Never raises."""
    researcher = _RESEARCHERS.get(sq.source)
    if researcher is None:
        print(f"  [orch] WARN: unknown source {sq.source!r} for {sq.id}")
        return (sq.id, [])
    try:
        results = await researcher(sq.id, sq.question, top_k=top_k)
    except Exception as e:  # noqa: BLE001 — defensive top-level guard
        print(f"  [orch] WARN: dispatch failed for {sq.id} ({sq.source}): {e}")
        results = []
    return (sq.id, results)


async def gather_results(
    plan_obj: ResearchPlan,
    top_k: int = 5,
) -> dict[str, list[SearchResult]]:
    """Fan out plan.sub_questions to researchers, gather results in parallel.

    Returns a dict mapping sub_question_id → list of SearchResult.
    """
    tasks = [_dispatch_one(sq, top_k) for sq in plan_obj.sub_questions]
    pairs = await asyncio.gather(*tasks)
    return dict(pairs)


# ---------------------------------------------------------------------------
# Manual end-to-end eval — `python -m src.orchestrator`
# ---------------------------------------------------------------------------


def _print_query_summary(
    plan_obj: ResearchPlan,
    results: dict[str, list[SearchResult]],
    plan_time: float,
    gather_time: float,
) -> None:
    """Pretty-print a single query's results for manual inspection."""
    total = sum(len(v) for v in results.values())
    print(f"  Plan:   {len(plan_obj.sub_questions)} sub-questions in {plan_time:5.2f}s")
    print(f"  Search: {total} results across all sources in {gather_time:5.2f}s")
    for sq in plan_obj.sub_questions:
        n = len(results.get(sq.id, []))
        marker = "OK " if n > 0 else "!! "
        snippet = sq.question[:55] + ("..." if len(sq.question) > 55 else "")
        print(f"    {marker}[{sq.source:6s}] {sq.id}: {n} hits  |  {snippet}")


async def _run_eval() -> None:
    """Plan + gather for all 10 eval queries. Saves full results to JSONL."""
    queries_path = PROJECT_ROOT / "tests" / "test_queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    print(f"End-to-end eval: planner + researchers on {len(queries)} queries.\n")
    print("=" * 78)

    eval_records: list[dict] = []
    grand_start = time.perf_counter()

    for i, q in enumerate(queries, 1):
        if i > 1:
            # Pace ourselves to stay under Groq 8B's 6K TPM burst limit.
            await asyncio.sleep(1.5)

        print(f"\n[{i}/{len(queries)}] {q['id']} :: {q['query']}")

        # Planner (sync, Groq call) — wrap so one failure doesn't kill the eval
        t0 = time.perf_counter()
        try:
            plan_obj = planner_plan(q["query"])
        except Exception as e:
            print(f"  [orch] PLANNER FAILED for {q['id']}: {e}")
            eval_records.append(
                {"query_id": q["id"], "user_query": q["query"], "error": str(e)}
            )
            print("-" * 78)
            continue
        plan_time = time.perf_counter() - t0

        # Researchers (async, parallel)
        t0 = time.perf_counter()
        results = await gather_results(plan_obj)
        gather_time = time.perf_counter() - t0

        _print_query_summary(plan_obj, results, plan_time, gather_time)

        eval_records.append(
            {
                "query_id": q["id"],
                "user_query": q["query"],
                "plan": plan_obj.model_dump(),
                "results": {
                    sq_id: [r.model_dump() for r in rs]
                    for sq_id, rs in results.items()
                },
                "timings": {
                    "plan_seconds": round(plan_time, 3),
                    "gather_seconds": round(gather_time, 3),
                    "total_seconds": round(plan_time + gather_time, 3),
                },
            }
        )
        print("-" * 78)

    grand_total = time.perf_counter() - grand_start
    out_path = TRACE_DIR / "orchestrator_eval.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nEval complete. {grand_total:.1f}s total.")
    print(f"Results saved to {out_path}")

    # Quick summary stats — only over successful records
    successful = [r for r in eval_records if "timings" in r]
    if successful:
        avg_total = sum(r["timings"]["total_seconds"] for r in successful) / len(successful)
        total_results = sum(
            sum(len(rs) for rs in r["results"].values()) for r in successful
        )
        print(f"Average end-to-end per query: {avg_total:.2f}s  (target: <15s)")
        print(f"Total search results gathered: {total_results}")
    failures = len(eval_records) - len(successful)
    if failures:
        print(f"Failed queries: {failures}/{len(eval_records)}")


if __name__ == "__main__":
    asyncio.run(_run_eval())