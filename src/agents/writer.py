"""src/agents/writer.py

The Writer: composes a structured markdown report from Findings.

DESIGN PRINCIPLES:
- One call per query (low volume). Use the stronger model — quality matters
  here. Default: GROQ_MODEL (Llama-3.3-70B).
- COMPOSITION ONLY. The Writer reads Findings and arranges them. It must not
  introduce new factual claims that aren't in the Findings list.
- EVERY CLAIM IS CITED. Inline [1], [2] style citations that map to a numbered
  References section at the bottom of the report.
- STRUCTURED OUTPUT: report is a free-form markdown string (not Pydantic).
  Markdown is the natural format for the deliverable; the Verifier reads it
  with regex, not Pydantic.

INPUT SHAPE:
- The full ResearchPlan (gives the section structure via expected_report_sections)
- The full flat list of Findings (one URL → one numbered citation reference)

OUTPUT:
- A markdown string ready to render. Sections, inline citations, references.

RESILIENCE:
- Primary model: 70B. On 429 (rate limit / TPD exhausted) we fall back to 8B
  rather than crash. 8B-written reports are noticeably weaker but a degraded
  report beats no report — and this is the kind of fallback every production
  agent pipeline needs.
- If BOTH models fail, we emit a stub report containing the raw findings so
  no upstream work is lost.
"""

from __future__ import annotations

from typing import Final

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GROQ_FAST_MODEL
from src.llm import get_llm
from src.schemas import Finding, ResearchPlan


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

WRITER_SYSTEM_PROMPT: Final[str] = """\
You are a research writer. You will be given:
  1. The user's original research question.
  2. A list of expected report sections.
  3. A numbered list of Findings, each with a claim, a source URL, and a
     supporting quote from that source.

Your job is to compose a markdown report that answers the user's question
using ONLY the findings provided.

HARD RULES (NO EXCEPTIONS):
1. EVERY FACTUAL CLAIM IN THE REPORT MUST BE TRACEABLE TO A FINDING.
   You may not introduce facts, numbers, names, dates, percentages, or
   conclusions that don't appear in the findings list. Period.
2. EVERY FACTUAL CLAIM MUST BE FOLLOWED BY A CITATION [N], where N is the
   number of the finding that supports it. Multiple citations on one
   sentence are fine: "X is the case [1][3]."
3. USE THE EXPECTED SECTIONS as the markdown headers. Render them as
   `## Section Name`. You may add a brief intro under each.
4. TL;DR SECTION FIRST. 2-3 sentences. Direct answer to the user's question.
   Yes, every claim in the TL;DR also needs citations.
5. REFERENCES SECTION LAST. Numbered list of all cited finding sources.
   Format: `[N] <Title or domain> — <URL>`. Each cited finding's URL
   appears exactly once in References, even if cited multiple times in the
   body. If the body never cites a finding, do NOT include it in References.
6. NO INVENTED INFORMATION. If the findings don't cover something the user
   asked about, write "Open question" or note the gap in the Open Questions
   section. Do not paper over gaps with general knowledge.
7. WRITE NATURALLY. Despite the rules, the report should read like a
   human-written research summary. Cite inline, not in footnotes. Don't
   number every sentence — group related claims into paragraphs.

NUMBERING: The findings you receive are pre-numbered (Finding 1, Finding 2,
...). The citation [N] you write refers to that finding number. Map findings
to references such that each unique URL becomes one numbered reference.

OUTPUT: Pure markdown. No code fences. No prose around the report.\
"""


# ---------------------------------------------------------------------------
# Findings rendering
# ---------------------------------------------------------------------------


def _format_findings(findings: list[Finding]) -> str:
    """Render findings as a numbered list for the prompt."""
    if not findings:
        return "(no findings)"
    lines: list[str] = []
    for i, f in enumerate(findings, 1):
        lines.append(
            f"Finding {i} [sub-question: {f.sub_question_id}, confidence: {f.confidence}]\n"
            f"  CLAIM:  {f.claim}\n"
            f"  QUOTE:  \"{f.source_quote}\"\n"
            f"  SOURCE: {f.source_url}"
        )
    return "\n\n".join(lines)


def _stub_report(plan: ResearchPlan, findings: list[Finding], reason: str) -> str:
    """Emergency fallback when both 70B and 8B fail. Dump findings as markdown
    so upstream work isn't lost and the Verifier still has something to chew on."""
    lines = [
        f"# {plan.user_query}",
        "",
        "## TL;DR",
        "",
        f"_Writer agent failed: {reason}. Findings dumped below for manual review._",
        "",
        "## Raw Findings (writer unavailable)",
        "",
    ]
    for i, finding in enumerate(findings, 1):
        lines.append(f"**Finding {i}** ({finding.confidence}, sq={finding.sub_question_id})")
        lines.append(f"- Claim: {finding.claim}")
        lines.append(f"- Quote: \"{finding.source_quote}\"")
        lines.append(f"- Source: {finding.source_url}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_report(
    plan: ResearchPlan,
    findings: list[Finding],
    *,
    model_override: str | None = None,
) -> str:
    """Compose a markdown report from a plan + flat list of findings.

    Tries the 70B model first. On rate-limit failure, falls back to 8B.
    On total failure, emits a stub report containing the raw findings.

    Returns a markdown string.
    """
    if not findings:
        return (
            f"# {plan.user_query}\n\n"
            "## TL;DR\n\n"
            "No findings were extracted for this query. "
            "The Synthesizer found no extractable claims in the gathered sources.\n"
        )

    user_msg = (
        f"USER QUERY: {plan.user_query}\n\n"
        f"EXPECTED SECTIONS: {plan.expected_report_sections}\n\n"
        f"FINDINGS:\n\n{_format_findings(findings)}\n\n"
        f"Compose the markdown report now."
    )
    messages = [
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    def _invoke(use_fast: bool):
        llm = get_llm(
            provider="groq",
            structured=False,
            temperature=0.3,
            model_override=GROQ_FAST_MODEL if use_fast else model_override,
        )
        return llm.invoke(messages)

    # Primary attempt: 70B (or whatever model_override specifies)
    try:
        response = _invoke(use_fast=False)
    except Exception as primary_error:  # noqa: BLE001
        primary_msg = str(primary_error)
        if "rate_limit" in primary_msg.lower() or "429" in primary_msg:
            print(f"  [writer] WARN: 70B rate-limited, falling back to 8B: {primary_error}")
            try:
                response = _invoke(use_fast=True)
            except Exception as fallback_error:  # noqa: BLE001
                print(f"  [writer] WARN: 8B fallback also failed: {fallback_error}")
                return _stub_report(plan, findings, str(fallback_error))
        else:
            print(f"  [writer] WARN: writer call failed (non-rate-limit): {primary_error}")
            return _stub_report(plan, findings, str(primary_error))

    # response.content is a string for chat models; coerce defensively
    content = response.content if isinstance(response.content, str) else str(response.content)
    return content.strip()