"""src/agents/verifier.py

The Verifier: reads the Writer's markdown report claim by claim, checks each
factual statement against its cited Finding, and emits a structured verdict.

This is the agent that gives the project its credibility. Without it, we have
"an LLM that summarizes web search" — there are 1000 of those. With it, we
have a system that produces measurably grounded reports.

DESIGN PRINCIPLES:
- One call per query (low-volume). Use 70B (default GROQ_MODEL) for quality.
- INPUT: the markdown report + the numbered Findings list.
- OUTPUT: a VerificationResult with total / verified / flagged counts.
- The verifier is intentionally skeptical but NOT pedantic. It only flags
  claims that are genuinely unsupported, contradicted, or only partially
  supported by their cited finding. Stylistic disagreements are not flags.

HOW IT WORKS:
1. We pre-parse the markdown report into (claim_text, [cited_finding_numbers])
   tuples using regex. The LLM only sees claims it needs to verify, paired
   with the specific findings to check against. This dramatically reduces
   the verifier's hallucination surface.
2. For each claim, we call the LLM with (claim, cited_findings) and ask for
   a verdict + explanation.
3. We aggregate into a VerificationResult and compute the grounding score.

CONCURRENCY: claims-per-report is typically 8-15. We use a semaphore (max 2
in-flight) and small inter-call sleep to stay under Groq TPM caps. Reuse
the patterns from the Synthesizer.

RESILIENCE: 70B -> 8B fallback on rate limit, identical to Planner/Writer.
"""

from __future__ import annotations

import asyncio
import re
from typing import Final

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import GROQ_FAST_MODEL
from src.llm import get_llm
from src.schemas import Finding, FlaggedClaim, VerificationResult


# ---------------------------------------------------------------------------
# Concurrency control
# ---------------------------------------------------------------------------

# NOTE: no module-level asyncio.Semaphore.
# The Semaphore is created inside verify_report() so it binds to whatever
# event loop is running that call. A module-level Semaphore binds to the FIRST
# loop that touches it, then breaks with "bound to a different event loop"
# errors when a fresh loop (e.g. every Streamlit rerun) tries to use it.
_INTER_CALL_DELAY = 1.0
_MAX_CONCURRENT_LLM_CALLS = 2


# ---------------------------------------------------------------------------
# Per-claim verdict schema (internal, not exported)
# ---------------------------------------------------------------------------


class _ClaimVerdict(BaseModel):
    """Verifier's per-claim verdict. Structured output we ask the LLM for."""

    verdict: str = Field(
        description="One of: 'verified', 'unsupported', 'contradicted', 'partial_support'."
    )
    explanation: str = Field(
        description="One-sentence reason for the verdict. Must reference the quote."
    )


# ---------------------------------------------------------------------------
# Claim extraction from markdown report
# ---------------------------------------------------------------------------

# A claim is a sentence (or sentence fragment) followed by one or more [N]
# citation markers. We capture the sentence + the list of finding numbers.
# Example matches:
#   "SPLADE achieves SOTA performance [1]."
#   "Hybrid retrieval combines both approaches [2][3]."
#   "Multiple papers discuss this [1][5][7]."
_CLAIM_RE = re.compile(
    r"([A-Z][^.!?\n]*?)\s*((?:\[\d+\])+)",
    re.MULTILINE,
)
_CITATION_NUM_RE = re.compile(r"\[(\d+)\]")


def _extract_claims(report_md: str) -> list[tuple[str, list[int]]]:
    """Pull (claim_text, [finding_numbers]) tuples out of the markdown report.

    Only claims with at least one citation are returned. Uncited prose
    (section headers, intros, the References list) is ignored - the Writer's
    rule is "every factual claim must be cited", so uncited text is by
    definition not a factual claim the Verifier needs to check.
    """
    out: list[tuple[str, list[int]]] = []
    # Strip the References section - those lines contain [N] but aren't claims.
    body = report_md.split("## References")[0]
    body = body.split("# References")[0]

    for match in _CLAIM_RE.finditer(body):
        claim_text = match.group(1).strip()
        citations_str = match.group(2)
        nums = [int(n) for n in _CITATION_NUM_RE.findall(citations_str)]
        if claim_text and nums:
            out.append((claim_text, nums))
    return out


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

VERIFIER_SYSTEM_PROMPT: Final[str] = """\
You are a research verifier. You will be given:
  1. A factual CLAIM extracted from a research report.
  2. The FINDINGS that the report cited as support for that claim. Each
     finding has its own one-sentence summary and a verbatim source quote.

Your job is to determine whether the cited findings actually support the
claim. Return a verdict.

VERDICTS (pick exactly one):

- "verified": The finding(s) clearly and directly support the claim. A reader
  checking the source quote would agree the claim follows from it.

- "partial_support": The finding(s) support PART of the claim but not all of
  it. Example: claim says "X achieves SOTA on three benchmarks" but the
  finding only mentions one benchmark. Also use this when the finding implies
  the claim but doesn't state it.

- "unsupported": The finding(s) do not address the claim at all, or the
  source quote is irrelevant to what the claim says.

- "contradicted": The finding(s) directly contradict the claim. Rare - use
  only when the source actually says the opposite.

CRITICAL RULES:
1. Judge based on the SOURCE QUOTE in the finding, not the finding's own
   claim summary. The quote is the ground truth; the finding's claim is
   the previous agent's interpretation, which could itself be wrong.
2. Be strict but not pedantic. Minor paraphrase is fine. Stylistic difference
   is fine. Missing nuance is the difference between "verified" and
   "partial_support".
3. If the claim is unverifiable from the provided findings (e.g., it cites
   findings that don't actually relate to it), mark "unsupported".
4. Your EXPLANATION must reference what the source quote actually says.
   Don't just restate the verdict - explain WHY in one sentence.

OUTPUT: A JSON object with `verdict` and `explanation` fields. No prose.\
"""


# ---------------------------------------------------------------------------
# Single-claim verification (sync core)
# ---------------------------------------------------------------------------


def _verify_one_claim_sync(
    claim_text: str,
    cited_findings: list[Finding],
    model_override: str | None,
) -> _ClaimVerdict:
    """Verify one claim against its cited findings. Returns a verdict."""
    if not cited_findings:
        return _ClaimVerdict(
            verdict="unsupported",
            explanation="No valid citations found for this claim.",
        )

    findings_block = "\n\n".join(
        f"FINDING [{i+1}]:\n"
        f"  Claim: {f.claim}\n"
        f"  Quote: \"{f.source_quote}\"\n"
        f"  URL:   {f.source_url}"
        for i, f in enumerate(cited_findings)
    )

    user_msg = (
        f"CLAIM TO VERIFY:\n{claim_text}\n\n"
        f"CITED FINDINGS:\n{findings_block}\n\n"
        f"Return your verdict."
    )

    messages = [
        SystemMessage(content=VERIFIER_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    def _invoke(use_fast: bool) -> _ClaimVerdict:
        llm = get_llm(
            provider="groq",
            structured=True,
            temperature=0.1,  # verification wants determinism
            model_override=GROQ_FAST_MODEL if use_fast else model_override,
        )
        structured_llm = llm.with_structured_output(_ClaimVerdict)
        return structured_llm.invoke(messages)

    try:
        return _invoke(use_fast=False)
    except Exception as primary_error:  # noqa: BLE001
        primary_msg = str(primary_error)
        if "rate_limit" in primary_msg.lower() or "429" in primary_msg:
            print(f"  [verifier] WARN: 70B rate-limited, falling back to 8B: {primary_error}")
            try:
                return _invoke(use_fast=True)
            except Exception as fallback_error:  # noqa: BLE001
                print(f"  [verifier] WARN: 8B fallback also failed: {fallback_error}")
                return _ClaimVerdict(
                    verdict="verifier_error",
                    explanation=f"Verifier failed: {type(fallback_error).__name__}",
                )
        print(f"  [verifier] WARN: verifier call failed: {primary_error}")
        return _ClaimVerdict(
            verdict="verifier_error",
            explanation=f"Verifier failed: {type(primary_error).__name__}",
        )


# ---------------------------------------------------------------------------
# Async wrapper
# ---------------------------------------------------------------------------


async def _verify_one_claim_async(
    claim_text: str,
    cited_findings: list[Finding],
    model_override: str | None,
    sem: asyncio.Semaphore,
) -> tuple[str, list[Finding], _ClaimVerdict]:
    """Verify one claim under semaphore + pacing."""
    async with sem:
        verdict = await asyncio.to_thread(
            _verify_one_claim_sync, claim_text, cited_findings, model_override
        )
        await asyncio.sleep(_INTER_CALL_DELAY)
        return (claim_text, cited_findings, verdict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def verify_report(
    report_md: str,
    findings: list[Finding],
    *,
    model_override: str | None = None,
) -> VerificationResult:
    """Verify every claim in the report against its cited findings.

    Args:
        report_md: The markdown report produced by the Writer.
        findings: The flat list of Findings produced by the Synthesizer.
            Indexed 1-based to match the Writer's [N] citation scheme.
        model_override: Optional model name override.

    Returns:
        A VerificationResult with grounding score and flagged claims.
    """
    claims = _extract_claims(report_md)
    if not claims:
        return VerificationResult(
            total_claims=0,
            verified=0,
            flagged=[],
            overall_grounding_score=1.0,  # vacuously perfect
        )

    # Create the semaphore inside the coroutine so it binds to the running loop.
    # See the note near the top of this file for why this must not be module-scope.
    sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)

    # Build the claim -> findings mapping. Out-of-range citations get []
    # which produces an "unsupported" verdict downstream.
    tasks = []
    for claim_text, cite_nums in claims:
        cited = [findings[n - 1] for n in cite_nums if 1 <= n <= len(findings)]
        tasks.append(_verify_one_claim_async(claim_text, cited, model_override, sem))

    triples = await asyncio.gather(*tasks)

    # Aggregate verdicts
    verified_count = 0
    verifier_errors = 0
    flagged: list[FlaggedClaim] = []
    for claim_text, cited_findings, verdict in triples:
        v = verdict.verdict.lower().strip()
        if v == "verified":
            verified_count += 1
        elif v == "verifier_error":
            verifier_errors += 1
        else:
            issue_type = v if v in ("unsupported", "contradicted", "partial_support") else "unsupported"
            cited_url = cited_findings[0].source_url if cited_findings else "(no citation)"
            flagged.append(
                FlaggedClaim(
                    claim_text=claim_text,
                    cited_source=cited_url,
                    issue=issue_type,  # type: ignore[arg-type]
                    explanation=verdict.explanation,
                )
            )

    # Grounding score is verified / (claims we could actually check)
    checkable_total = len(claims) - verifier_errors
    total = len(claims)
    score = verified_count / checkable_total if checkable_total > 0 else 1.0
    if verifier_errors:
        print(f"  [verifier] {verifier_errors} claim(s) could not be verified due to rate limits.")

    total = len(claims)
    score = verified_count / total if total > 0 else 1.0
    return VerificationResult(
        total_claims=total,
        verified=verified_count,
        flagged=flagged,
        overall_grounding_score=round(score, 3),
    )


def format_verification_summary(result: VerificationResult) -> str:
    """Render a VerificationResult as a markdown footer for the report."""
    if result.total_claims == 0:
        return "\n\n## Verification\n\n_No citation-anchored claims to verify._\n"

    lines = [
        "",
        "## Verification",
        "",
        f"**Grounding score:** {result.overall_grounding_score:.0%} "
        f"({result.verified}/{result.total_claims} claims verified)",
        "",
    ]
    if not result.flagged:
        lines.append("_All claims verified against their cited findings._")
    else:
        lines.append(f"**{len(result.flagged)} flagged claim(s):**")
        lines.append("")
        for i, fc in enumerate(result.flagged, 1):
            lines.append(f"{i}. **[{fc.issue}]** {fc.claim_text}")
            lines.append(f"   - Cited: {fc.cited_source}")
            lines.append(f"   - Explanation: {fc.explanation}")
            lines.append("")
    return "\n".join(lines)