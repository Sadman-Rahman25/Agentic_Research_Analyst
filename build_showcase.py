"""build_showcase.py

Generates showcase/reports.json from the Day 5 evaluation artifacts.

Reads (in order of preference):
  1. trace_logs/pipeline_eval.jsonl  - structured per-query data (findings, timings, verification, plan)
  2. examples/q1.md ... q10.md       - the report markdown bodies
  3. tests/test_queries.json         - canonical query text for each qN

Writes:
  showcase/reports.json              - single JSON file consumed by showcase/index.html

Usage:
  python build_showcase.py

If some files are missing, defaults are used and warnings are printed. The
script is designed to always produce a valid reports.json even with partial data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).parent
TRACE_LOG = PROJECT_ROOT / "trace_logs" / "pipeline_eval.jsonl"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
TESTS_FILE = PROJECT_ROOT / "tests" / "test_queries.json"
OUTPUT = PROJECT_ROOT / "showcase" / "reports.json"

# Canonical outcome classification per query, based on Day 5 eval.
# Used to color-code pills and drive fallback rendering.
QUERY_OUTCOMES = {
    "q1": "verified",   # 76% clean
    "q2": "verified",   # 73% clean
    "q3": "verified",   # 74% clean
    "q4": "verified",   # 100% clean
    "q5": "verified",   # 100% clean
    "q6": "verified",   # 75% with 4 verifier_errors
    "q7": "failed",     # Planner exhausted both models
    "q8": "degraded",   # 21% - Writer fell back to 8B
    "q9": "stub",       # Writer returned stub
    "q10": "partial",   # 65% with 7 verifier_errors
}

# Fallback query text if tests/test_queries.json isn't found or is incomplete.
FALLBACK_QUERIES = {
    "q1": "What are the latest techniques for sparse retrieval in RAG systems in 2025-2026?",
    "q2": "How do multi-agent LLM systems handle disagreement between agents?",
    "q3": "Compare LangGraph vs CrewAI vs AutoGen for agent orchestration.",
    "q4": "How does hybrid BM25 + dense retrieval work in production RAG systems?",
    "q5": "What is the current state of the AI infrastructure market?",
    "q6": "How have transformer architectures evolved from the original 2017 paper?",
    "q7": "What are the latest developments in Bangla language models?",
    "q8": "How does Gemini 2.5 Flash compare to Llama-3.3-70B?",
    "q9": "What are the best practices for deploying LangGraph applications to production?",
    "q10": "How do you evaluate multi-step reasoning in LLM systems?",
}


# ============================================================
# Helpers
# ============================================================

def load_trace_log() -> dict[str, dict]:
    """Read the JSONL eval log. Return dict keyed by query_id."""
    if not TRACE_LOG.exists():
        print(f"[warn] {TRACE_LOG} not found - using fallback data only")
        return {}
    records: dict[str, dict] = {}
    with TRACE_LOG.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] {TRACE_LOG.name} line {line_num}: {e}")
                continue
            qid = obj.get("query_id") or obj.get("id")
            if qid:
                # If multiple runs of the same query, prefer the latest
                records[qid] = obj
    print(f"[info] loaded {len(records)} records from {TRACE_LOG.name}")
    return records


def load_query_texts() -> dict[str, str]:
    """Read the canonical benchmark query texts. Return dict keyed by query_id.
    Tolerates several plausible shapes for test_queries.json.
    """
    if not TESTS_FILE.exists():
        print(f"[warn] {TESTS_FILE} not found - using fallback query texts")
        return dict(FALLBACK_QUERIES)
    try:
        raw = json.loads(TESTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[warn] {TESTS_FILE.name}: {e} - using fallback query texts")
        return dict(FALLBACK_QUERIES)

    def extract_text(item) -> str:
        """Pull a query string out of a variety of nested shapes."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("query", "question", "text", "prompt", "user_query"):
                v = item.get(key)
                if isinstance(v, str) and v:
                    return v
            return ""
        if isinstance(item, list) and item:
            return extract_text(item[0])
        return ""

    out: dict[str, str] = {}

    # Case: dict at top level. May be {qN: ...} directly, or a wrapper like {"queries": [...]}
    if isinstance(raw, dict):
        for wrapper_key in ("queries", "test_queries", "items", "data"):
            if wrapper_key in raw and isinstance(raw[wrapper_key], list):
                raw = raw[wrapper_key]
                break
        else:
            # Flat {qid: <value>} - value can be string, dict, or list-with-string
            for qid, val in raw.items():
                text = extract_text(val)
                if text:
                    out[str(qid)] = text
            return out or dict(FALLBACK_QUERIES)

    # Case: list at top level (either originally or after unwrap above)
    if isinstance(raw, list):
        for i, item in enumerate(raw, start=1):
            if isinstance(item, dict):
                qid = item.get("id") or item.get("qid") or item.get("query_id") or f"q{i}"
            else:
                qid = f"q{i}"
            text = extract_text(item)
            if text:
                out[str(qid)] = text
        return out or dict(FALLBACK_QUERIES)

    print(f"[warn] {TESTS_FILE.name} has unexpected top-level type: {type(raw).__name__}")
    return dict(FALLBACK_QUERIES)


def load_report_md(qid: str) -> str:
    """Read examples/qN.md. Return empty string if missing."""
    path = EXAMPLES_DIR / f"{qid}.md"
    if not path.exists():
        print(f"[warn] {path} not found - {qid} will show empty report")
        return ""
    return path.read_text(encoding="utf-8")


def extract_citation_num(text: str) -> int | None:
    """Extract the first [N] citation number from a claim text. Returns None if not found."""
    match = re.search(r"\[(\d+)\]", text or "")
    return int(match.group(1)) if match else None


# ============================================================
# Normalize one query record
# ============================================================

def load_sub_questions(qid: str) -> list[dict]:
    """Try to read sub-questions for a query from planner_eval.jsonl.
    Returns [] if the file or query is not found."""
    planner_log = PROJECT_ROOT / "trace_logs" / "planner_eval.jsonl"
    if not planner_log.exists():
        return []
    try:
        with planner_log.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("query_id") == qid or obj.get("id") == qid:
                    # Try common paths to sub-questions
                    plan = obj.get("plan") or obj
                    sqs = plan.get("sub_questions") or plan.get("subquestions") or []
                    normalized = []
                    for sq in sqs:
                        if isinstance(sq, dict):
                            normalized.append({
                                "source": sq.get("source", "web"),
                                "priority": sq.get("priority", 1),
                                "question": sq.get("question") or sq.get("text", ""),
                            })
                    return normalized
    except Exception as e:
        print(f"[warn] failed to read planner_eval.jsonl for {qid}: {e}")
    return []


def build_query_record(qid: str, trace: dict, query_text: str) -> dict:
    """Merge trace-log summary metadata with report markdown and canonical query text.
    Produce the flat structure consumed by index.html.

    Note: pipeline_eval.jsonl in this project stores summary metadata only
    (n_findings/n_flagged counts, grounding_score fraction, timings, errors,
    report_path). Detailed per-finding and per-flagged-claim data is not
    persisted, so those arrays stay empty in the output. The UI handles this
    gracefully.
    """
    outcome = QUERY_OUTCOMES.get(qid, "verified")
    report_md = load_report_md(qid)

    # Fields present in Sadman's pipeline_eval.jsonl schema:
    n_findings = trace.get("n_findings", 0)
    n_flagged = trace.get("n_flagged", 0)
    grounding_frac = trace.get("grounding_score")  # 0.0 to 1.0 float
    timings = trace.get("timings") or {}
    errors = trace.get("errors") or []

    # Grounding: convert 0-1 fraction to 0-100 int for display.
    # For stub and failed outcomes, force score to None regardless of what the log says.
    grounding_score = None
    if outcome not in ("failed", "stub") and grounding_frac is not None:
        try:
            grounding_score = round(float(grounding_frac) * 100)
        except (TypeError, ValueError):
            grounding_score = None

    total_claims = int(n_findings) if n_findings else 0
    flagged_count = int(n_flagged) if n_flagged else 0
    verified_count = max(0, total_claims - flagged_count)

    # Wall clock: sum the per-stage timings.
    wall_clock = float(sum(timings.values())) if timings else None

    # Try planner_eval.jsonl for sub-questions - gracefully degrades to [].
    sub_questions = load_sub_questions(qid)

    return {
        "id": qid,
        "query_text": query_text,
        "outcome": outcome,
        "grounding_score": grounding_score,
        "verified": verified_count,
        "flagged_count": flagged_count,
        "total_claims": total_claims,
        "wall_clock_seconds": wall_clock,
        "sub_questions": sub_questions,
        "timings": {k: float(v) for k, v in timings.items()},
        "findings": [],   # not persisted in current benchmark JSONL - see report_md for narrative
        "report_md": report_md,
        "flagged": [],    # not persisted in current benchmark JSONL - counts still shown
        "errors": errors,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(f"[info] project root: {PROJECT_ROOT}")

    trace_records = load_trace_log()
    query_texts = load_query_texts()

    queries = []
    for qid in ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]:
        trace = trace_records.get(qid, {})
        query_text = query_texts.get(qid) or FALLBACK_QUERIES.get(qid, f"Query {qid}")
        queries.append(build_query_record(qid, trace, query_text))

    # Compute summary
    valid_scores = [q["grounding_score"] for q in queries if q["grounding_score"] is not None]
    avg_grounding_honest = round(sum(valid_scores) / len(valid_scores)) if valid_scores else 0
    total_runtime = sum(q["wall_clock_seconds"] or 0 for q in queries)

    summary = {
        "avg_grounding_honest": avg_grounding_honest,
        "total_queries": len(queries),
        "clean_runs": sum(1 for q in queries if q["outcome"] == "verified"),
        "stub_runs": sum(1 for q in queries if q["outcome"] == "stub"),
        "failed_runs": sum(1 for q in queries if q["outcome"] == "failed"),
        "total_runtime_seconds": round(total_runtime, 1),
        "avg_query_seconds": round(total_runtime / len(queries), 1) if queries else 0,
    }

    out = {"queries": queries, "summary": summary}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print(f"[ok] wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    print(f"[ok] {len(queries)} queries, avg grounding {avg_grounding_honest}%")
    print(f"[ok] summary: {summary['clean_runs']} clean, {summary['stub_runs']} stub, {summary['failed_runs']} failed")


if __name__ == "__main__":
    main()