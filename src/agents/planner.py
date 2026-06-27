"""src/agents/planner.py

The Planner: decomposes a user research question into a ResearchPlan with
sub-questions, each tagged with the most appropriate source (web/arxiv/github)
and a priority (1-3). It also defines the report sections the Writer will use.

This agent is the entry point of the pipeline. Its output is the contract
that every downstream agent relies on — if the plan is bad, everything
downstream is fighting uphill.

DESIGN NOTES:
- One responsibility: decompose. Do not search, do not synthesize.
- Structured output via Gemini's native JSON mode + Pydantic validation.
- Deterministic-ish: low temperature (0.2). Plans don't benefit from creativity.
- Defensive: overwrite `user_query` in the output to guarantee it matches
  the input exactly (Gemini occasionally paraphrases it).

RESILIENCE:
- Primary: 70B (better structured-output reliability).
- Fallback: 8B (used only when 70B is rate-limited / TPD-exhausted).
- 8B plans are noticeably weaker but degraded > crashed.
"""

from __future__ import annotations

import json
from typing import Final

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GROQ_FAST_MODEL, PROJECT_ROOT, TRACE_DIR
from src.llm import get_llm
from src.schemas import ResearchPlan


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PLANNER_SYSTEM_PROMPT: Final[str] = """\
You are a research planner. Decompose the user's research question into a \
structured plan that downstream agents will execute via web/arXiv/GitHub \
searches and write up as a report.

CORE PRINCIPLES:
1. NON-OVERLAP. Each sub-question covers a distinct angle. No two sub-questions
   should be answerable from the same source passage.
2. COVERAGE. Together, the sub-questions cover what's needed to answer the
   user query well.
3. CONCRETENESS. Each sub-question is specific enough that a search engine
   can find good results. Avoid vague questions like "what is the state of X?"
   — prefer "what are the most-cited methods for X in 2024-2025?"
4. PARSIMONY. Use at most {max_sub_questions} sub-questions. Match the count
   to the query — broad queries get more (6-7), narrow queries get fewer (4-5).
   Don't default to a fixed count.
5. ON-TOPIC. Every sub-question must directly serve the user's original
   question. If the user asks how to EVALUATE X, do not include sub-questions
   about how to APPLY X. Stay inside the user's stated scope.

SOURCE TAGGING (pick one per sub-question):
- "web"    — current state, vendors, products, news, blog tutorials, comparisons,
             commercial AI models, framework documentation
- "arxiv"  — academic papers, theoretical methods, benchmarks of open research
             models, formal definitions
- "github" — implementations, libraries, code patterns, real-world usage,
             framework internals, issue trends

CRITICAL SOURCE TAGGING RULES:
- Commercial AI models (Gemini, GPT, Claude, Llama, Mistral, etc.) → "web".
  Their architectures and benchmarks live in product docs, blog posts, and
  engineering reports — NOT academic papers.
- Emerging frameworks/libraries (LangGraph, CrewAI, AutoGen, LangChain, etc.)
  → "web" for docs/blogs, "github" for code and issues. NEVER "arxiv" —
  these are engineering projects, not research subjects.
- When in doubt between arxiv and web for a recent topic, pick web.

PRIORITY:
- 1 (critical):     Report fails without this answer.
- 2 (important):    Substantive context. Default for most sub-questions.
- 3 (nice-to-have): Enriching background.

REPORT SECTIONS:
Always include: ["TL;DR", "Background", "Key Findings", "Open Questions / Gaps", "References"]
Add or rename when the query warrants — e.g., "Comparison Matrix" for vs-style
queries, "Implementation Notes" for engineering queries, "Timeline" for
historical queries.

GOOD EXAMPLE (research topic):
Query: "What are the latest techniques for sparse retrieval in 2025?"
Sub-questions:
- (sq_1, p=1, web)    "What is sparse retrieval and how does it differ from dense retrieval?"
- (sq_2, p=1, arxiv)  "What sparse retrieval methods have been published on arXiv in 2024-2025?"
- (sq_3, p=2, arxiv)  "How do learned sparse retrievers (e.g., SPLADE) compare to BM25 on standard benchmarks?"
- (sq_4, p=2, github) "Which open-source implementations of sparse retrieval are most widely used?"
- (sq_5, p=3, web)    "What are known limitations of sparse retrieval in production deployments?"

GOOD EXAMPLE (commercial model comparison — note ZERO arxiv tags):
Query: "Compare GPT-4 and Claude 3.5 Sonnet for coding tasks"
- (sq_1, p=1, web)    "What are the published benchmark results for GPT-4 vs Claude 3.5 Sonnet on coding tasks?"
- (sq_2, p=2, github) "What public repositories compare these models on real coding workflows?"
- (sq_3, p=2, web)    "What do developer reviews and engineering blogs say about each model's coding strengths?"
- (sq_4, p=3, web)    "What are the pricing and rate limit differences for these models?"

BAD PATTERNS TO AVOID:
- "Tell me about sparse retrieval" (vague — search engines return junk)
- "What is BM25?" + "What is TF-IDF?" (too narrow; low-value sub-questions)
- "What are the latest papers AND code on X?" (compound — split into two)
- All priority=1 (you haven't actually prioritized)
- Tagging "Gemini vs Llama architectural differences" as arxiv (commercial models → web)
- Tagging "LangGraph deployment failures" as arxiv (frameworks → web/github)
- A sub-question that drifts off-topic (user asks "how to evaluate X" → don't include "applications of X")
- All sub-questions same length, same source, same priority (you're following a template, not thinking)

Return ONLY the structured ResearchPlan. No prose, no commentary.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plan(user_query: str, max_sub_questions: int = 7) -> ResearchPlan:
    """Generate a ResearchPlan for the given user query.

    Tries 70B first. On rate-limit failure, falls back to 8B with a warning.

    Args:
        user_query: The research question to decompose.
        max_sub_questions: Upper bound on sub-questions. The model can return
            fewer if the query is narrow.

    Returns:
        A validated ResearchPlan with sub-questions and expected report sections.

    Raises:
        ValueError: If user_query is empty or whitespace.
        RuntimeError: If both 70B and 8B fallback fail.
    """
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    system = PLANNER_SYSTEM_PROMPT.format(max_sub_questions=max_sub_questions)
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_query),
    ]

    def _invoke(use_fast: bool) -> ResearchPlan:
        llm = get_llm(
            provider="groq",
            structured=True,
            temperature=0.2,
            model_override=GROQ_FAST_MODEL if use_fast else None,
        )
        structured_llm = llm.with_structured_output(ResearchPlan)
        return structured_llm.invoke(messages)

    # Primary attempt: 70B
    try:
        result = _invoke(use_fast=False)
    except Exception as primary_error:  # noqa: BLE001
        primary_msg = str(primary_error)
        if "rate_limit" in primary_msg.lower() or "429" in primary_msg:
            print(f"  [planner] WARN: 70B rate-limited, falling back to 8B: {primary_error}")
            try:
                result = _invoke(use_fast=True)
            except Exception as fallback_error:  # noqa: BLE001
                raise RuntimeError(
                    f"Planner failed on both 70B and 8B: {fallback_error}"
                ) from fallback_error
        else:
            raise RuntimeError(f"Planner LLM call failed: {primary_error}") from primary_error

    if not isinstance(result, ResearchPlan):
        raise RuntimeError(
            f"Planner expected ResearchPlan, got {type(result).__name__}"
        )

    # Defensive: guarantee user_query fidelity in case the model paraphrased.
    return result.model_copy(update={"user_query": user_query})


# ---------------------------------------------------------------------------
# Manual-eval runner — `python -m src.agents.planner`
# ---------------------------------------------------------------------------


def _pretty_print_plan(plan_obj: ResearchPlan) -> None:
    """Render a plan as readable text for manual inspection."""
    print(f"\nUSER QUERY: {plan_obj.user_query}")
    print(f"SECTIONS:   {plan_obj.expected_report_sections}")
    print(f"SUB-QUESTIONS ({len(plan_obj.sub_questions)}):")
    for sq in plan_obj.sub_questions:
        prio = {1: "[P1-CRIT]", 2: "[P2-IMP ]", 3: "[P3-NICE]"}[sq.priority]
        print(f"  {prio} [{sq.id} | {sq.source:6s}] {sq.question}")
        print(f"           rationale: {sq.rationale}")


def _run_eval() -> None:
    """Load the 10 eval queries, run the planner, pretty-print + log results."""
    queries_path = PROJECT_ROOT / "tests" / "test_queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    print(f"Running Planner on {len(queries)} eval queries...\n")
    print("=" * 78)

    results: list[dict] = []
    for i, q in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] {q['id']} :: category={q['category']}")
        try:
            p = plan(q["query"])
            _pretty_print_plan(p)
            results.append({"id": q["id"], "plan": p.model_dump()})
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            results.append({"id": q["id"], "error": str(e)})
        print("-" * 78)

    out_path = TRACE_DIR / "planner_eval.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nEval complete. Results saved to {out_path}")


if __name__ == "__main__":
    _run_eval()