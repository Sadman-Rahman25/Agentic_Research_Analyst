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

WRITER_SYSTEM_PROMPT: Final[str] = """
You are a research writer producing a thesis-style report grounded in evidence.

You will be given:
1. The user's research QUESTION.
2. A numbered list of FINDINGS. Each finding has:
   - Its number (used for citation as [N])
   - A one-sentence claim
   - A verbatim source quote
   - A source URL

Your job: compose a markdown report answering the question, structured like a
section of an academic paper, using the findings as evidence with inline [N]
citations.

STRUCTURE (use exactly these section headers, in this order):

## Summary
2-3 paragraphs, roughly 120-150 words. Plain-English answer to the question,
written for a reader unfamiliar with the topic. Cite the 2-4 strongest
findings inline. Define any acronyms on first use. This is what a reader gets
if they stop after this section.

## Introduction
2-3 paragraphs, roughly 180-220 words. Establish context: why the question
matters, what background the reader needs, what problem the topic addresses.
Introduce concepts before they appear in later sections. Cite 3-5 findings
that establish this background.

## Methodology
1-2 paragraphs, roughly 130-170 words. How the sources approach this problem:
what techniques, frameworks, or measurement approaches recur across the
findings. This is the "how do we know what we know" section. Cite findings
that describe methods, benchmarks, or implementations.

## Key Findings
2-3 paragraphs, roughly 220-280 words. The substantive answers. Group related
findings together thematically, not by finding number. Cite [N] for every
non-trivial claim. Explain each claim in enough depth that the reader could
act on it - do not just paraphrase the finding's one-sentence claim.

## Discussion
1-2 paragraphs, roughly 130-170 words. Synthesis. What do the sources agree
on? Where do they diverge? What tensions or unresolved questions emerge when
reading them together? This section is your voice integrating the evidence,
not another list of findings.

## Implementation Notes
1 paragraph, roughly 80-120 words. Practical takeaways for someone applying
this. When to use these techniques, common pitfalls, recommended starting
points. Cite where directly relevant.

## References
Numbered list matching every [N] citation you used, in order of first
appearance, with the source URL:
[1] <URL>
[2] <URL>
... etc.

TOTAL LENGTH: 900-1000 words. This is deliberately longer than a typical LLM
summary - you must explain concepts thoroughly, not just list them.

WRITING STYLE:
- Paragraphs of 4-7 sentences that build on each other coherently.
- Explanatory, not punchy. Assume a smart reader unfamiliar with the topic.
- Define acronyms on first use. Example: "SPLADE (Sparse Lexical AnD Expansion
  Model), a learned sparse retriever [1]" - not just "SPLADE [1]".
- Prefer concrete verbs over abstract ones. Say "measures" not "is concerned
  with", "reports" not "gives an indication of".
- Active voice. Present tense for what sources currently say, past tense for
  historical context.
- No section-opening filler like "This section discusses..." - start with the
  substance.

CITATION RULES:
- Every non-trivial factual claim must carry an inline [N] citation.
- Only cite finding numbers that exist in the input. If you write [5], Finding
  [5] must be in the input.
- Multiple citations for one claim: [1][3] (not [1,3] or [1, 3]).
- Do NOT cite the same finding more than 3 times across the whole report. If a
  finding supports one point, cite it once there.
- If findings contradict, present both sides and cite both.
- The References section MUST list every finding number you cited, in order
  of first appearance, with the source URL from the input.

DO NOT:
- Invent findings not in the input.
- Cite finding numbers that do not exist in the input.
- Use bullet points inside any section other than References.
- Include a table of contents.
- Include images, diagrams, or code blocks.
- Add meta-commentary about your task ("As requested", "In this report I
  will", "As an AI...").

Output ONLY the markdown report starting with `## Summary`. No preamble.
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
        "## Summary",
        "",
        f"_Writer agent failed: {reason}. Findings dumped below for manual review._",
        "",
        "## Key Findings (writer unavailable)",
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
            "## Summary\n\n"
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