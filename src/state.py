"""src/state.py

AgentState — the shared state object that flows through the LangGraph.

LangGraph nodes receive state, return a dict of state updates. The graph
merges those updates into a new state for the next node. By declaring all
fields up front in a TypedDict, we get static typing and a clear contract
for what each agent can read and write.

CONVENTION: every field is optional in the dict sense (TypedDict, total=False)
because each node only writes the subset of fields it produces. The full
end-to-end state at the END of the graph has all fields populated.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from src.schemas import Finding, ResearchPlan, SearchResult, VerificationResult


class AgentState(TypedDict, total=False):
    # --- Input ---
    user_query: str

    # --- After Planner node ---
    plan: Optional[ResearchPlan]

    # --- After Researchers node ---
    results_by_sq: dict[str, list[SearchResult]]

    # --- After Synthesizer node ---
    findings: list[Finding]

    # --- After Writer node ---
    report_md: str

    # --- After Verifier node ---
    verification: Optional[VerificationResult]

    # --- Bookkeeping ---
    timings: dict[str, float]
    errors: list[str]