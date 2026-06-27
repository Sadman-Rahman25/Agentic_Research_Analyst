"""src/graph.py

LangGraph StateGraph that wires all five agents into a single pipeline.

The graph is currently LINEAR (no branching, no loops, no conditional edges):

    START → planner → researchers → synthesizer → writer → verifier → END

Why linear? The brief's design is sequential — each stage depends on the
previous. The agentic-research-analyst value is multi-tool reasoning and
verification, not adaptive routing. Conditional edges (e.g. "if grounding
< 0.7, loop back and re-research") are a Day 6+ stretch goal.

Why LangGraph at all if it's linear? Three reasons:
1. CV/recruiter signal — "LangGraph" is the searched-for keyword.
2. LangSmith integration — every node's input/output becomes a traceable
   step automatically.
3. Resumability — the StateGraph supports checkpointing so a failed run
   can be restarted from the last successful node. Useful when 70B is
   rate-limited mid-pipeline.

USE:
    from src.graph import build_graph
    app = build_graph()
    result = await app.ainvoke({"user_query": "..."})
    print(result["report_md"])
    print(result["verification"])
"""

from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph

from src.agents.planner import plan as planner_plan
from src.agents.synthesizer import synthesize_all
from src.agents.verifier import format_verification_summary, verify_report
from src.agents.writer import write_report
from src.orchestrator import gather_results
from src.state import AgentState


# ---------------------------------------------------------------------------
# Node functions — each takes state, returns a partial state update
# ---------------------------------------------------------------------------


def _record_timing(state: AgentState, stage: str, seconds: float) -> dict:
    """Helper: merge a new timing into the running timings dict."""
    timings = dict(state.get("timings", {}))
    timings[stage] = seconds
    return {"timings": timings}


def _record_error(state: AgentState, stage: str, msg: str) -> dict:
    """Helper: append a stage error to the running errors list."""
    errors = list(state.get("errors", []))
    errors.append(f"[{stage}] {msg}")
    return {"errors": errors}


def planner_node(state: AgentState) -> dict:
    """Run the Planner. Sync call wrapped — Planner is internally sync."""
    t0 = time.perf_counter()
    try:
        plan_obj = planner_plan(state["user_query"])
        update = {"plan": plan_obj}
    except Exception as e:  # noqa: BLE001
        update = {"plan": None}
        update.update(_record_error(state, "planner", str(e)))
    update.update(_record_timing(state, "plan", time.perf_counter() - t0))
    return update


async def researchers_node(state: AgentState) -> dict:
    """Fan out search across web/arxiv/github researchers."""
    plan_obj = state.get("plan")
    if plan_obj is None:
        return {"results_by_sq": {}, **_record_error(state, "researchers", "no plan")}

    t0 = time.perf_counter()
    try:
        results = await gather_results(plan_obj)
        update = {"results_by_sq": results}
    except Exception as e:  # noqa: BLE001
        update = {"results_by_sq": {}}
        update.update(_record_error(state, "researchers", str(e)))
    update.update(_record_timing(state, "search", time.perf_counter() - t0))
    return update


async def synthesizer_node(state: AgentState) -> dict:
    """Extract Findings from SearchResults."""
    plan_obj = state.get("plan")
    results = state.get("results_by_sq", {})
    if plan_obj is None:
        return {"findings": [], **_record_error(state, "synthesizer", "no plan")}

    t0 = time.perf_counter()
    try:
        findings = await synthesize_all(plan_obj.sub_questions, results)
        update = {"findings": findings}
    except Exception as e:  # noqa: BLE001
        update = {"findings": []}
        update.update(_record_error(state, "synthesizer", str(e)))
    update.update(_record_timing(state, "synthesize", time.perf_counter() - t0))
    return update


def writer_node(state: AgentState) -> dict:
    """Compose markdown report from Findings."""
    plan_obj = state.get("plan")
    findings = state.get("findings", [])
    if plan_obj is None:
        return {"report_md": "", **_record_error(state, "writer", "no plan")}

    t0 = time.perf_counter()
    try:
        report = write_report(plan_obj, findings)
        update = {"report_md": report}
    except Exception as e:  # noqa: BLE001
        update = {"report_md": ""}
        update.update(_record_error(state, "writer", str(e)))
    update.update(_record_timing(state, "write", time.perf_counter() - t0))
    return update


async def verifier_node(state: AgentState) -> dict:
    """Check every cited claim in the report against its finding."""
    report_md = state.get("report_md", "")
    findings = state.get("findings", [])
    if not report_md or not findings:
        return {"verification": None, **_record_error(state, "verifier", "no report or findings")}

    t0 = time.perf_counter()
    try:
        verification = await verify_report(report_md, findings)
        # Append the verification footer to the report itself for human-readable output
        report_with_footer = report_md + format_verification_summary(verification)
        update = {"verification": verification, "report_md": report_with_footer}
    except Exception as e:  # noqa: BLE001
        update = {"verification": None}
        update.update(_record_error(state, "verifier", str(e)))
    update.update(_record_timing(state, "verify", time.perf_counter() - t0))
    return update


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph():
    """Construct and compile the AgentState graph.

    Returns a compiled LangGraph app. Call `await app.ainvoke({"user_query": "..."})`
    to run a single query end-to-end.
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researchers", researchers_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("writer", writer_node)
    graph.add_node("verifier", verifier_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researchers")
    graph.add_edge("researchers", "synthesizer")
    graph.add_edge("synthesizer", "writer")
    graph.add_edge("writer", "verifier")
    graph.add_edge("verifier", END)

    return graph.compile()