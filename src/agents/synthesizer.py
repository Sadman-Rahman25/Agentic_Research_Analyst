"""src/agents/synthesizer.py

The Synthesizer: free-form text output on Groq 8B, parsed with regex.
"""

from __future__ import annotations

import asyncio
import re
from typing import Final

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import GROQ_FAST_MODEL
from src.llm import get_llm
from src.schemas import Finding, SearchResult, SubQuestion


# NOTE: no module-level asyncio.Semaphore.
# The Semaphore is created inside synthesize_all() so it binds to whatever
# event loop is running that call. A module-level Semaphore binds to the FIRST
# loop that touches it, then breaks with "bound to a different event loop"
# errors when a fresh loop (e.g. every Streamlit rerun) tries to use it.
_INTER_CALL_DELAY = 0.5
_MAX_CONCURRENT_LLM_CALLS = 2


SYNTHESIZER_SYSTEM_PROMPT: Final[str] = """\
You are a research synthesizer. You will be given:
  1. A sub-question.
  2. A list of source passages with their URLs.

Your job is STRICTLY EXTRACTIVE: for each source, extract atomic factual
claims that directly answer the sub-question, cited exactly to that source.

OUTPUT FORMAT (EXACT — DO NOT DEVIATE):

For each finding, emit a block in this exact format:

###FINDING###
CLAIM: <one factual sentence stating what the source says>
QUOTE: <exact passage from the source, le 30 words, in double quotes>
URL: <the source URL exactly as given>
CONFIDENCE: <high|medium|low>
###END###

If a source has nothing relevant to the sub-question, emit NO block for it.
If NO sources are relevant, return the literal text: NO_FINDINGS

EXAMPLE OUTPUT:

###FINDING###
CLAIM: SPLADE achieves state-of-the-art zero-shot performance on TREC benchmarks.
QUOTE: "SPLADE has achieved state-of-the-art zero-shot performance and competitive results on TREC collections."
URL: http://arxiv.org/abs/2207.03834v1
CONFIDENCE: high
###END###

RULES (NO EXCEPTIONS):
1. ONE CLAIM PER FINDING. If a source supports two distinct claims, emit two findings.
2. ONE SOURCE PER FINDING. Never write a finding whose claim depends on two URLs.
3. QUOTE THE SOURCE VERBATIM. Copy text exactly. Do not invent quotes.
4. CLAIM IS YOUR ONE-SENTENCE SUMMARY of what the quote says.
5. ON-TOPIC ONLY. Skip irrelevant sources entirely.
6. CONFIDENCE: high = explicit support; medium = implied; low = tangential.
7. DO NOT SYNTHESIZE ACROSS SOURCES. No "multiple sources agree" findings.
8. DO NOT INVENT. If the source does not say it, do not write it.

Output ONLY the FINDING blocks (or NO_FINDINGS). No preamble, no commentary.\
"""


def _format_sources(results: list[SearchResult]) -> str:
    if not results:
        return "(no sources)"
    blocks: list[str] = []
    for r in results:
        body = r.full_text or r.snippet or ""
        body = body.strip().replace("\n", " ")
        body = body[:1500] if r.full_text else body[:500]
        blocks.append(
            f"--- SOURCE [{r.rank}] ---\n"
            f"TITLE: {r.title}\n"
            f"URL:   {r.url}\n"
            f"BODY:  {body}\n"
        )
    return "\n".join(blocks)


_FINDING_BLOCK_RE = re.compile(r"###FINDING###(.*?)###END###", re.DOTALL | re.IGNORECASE)
_CLAIM_RE = re.compile(r"^\s*CLAIM:\s*(.+?)$", re.MULTILINE)
_QUOTE_RE = re.compile(r'^\s*QUOTE:\s*"?(.+?)"?\s*$', re.MULTILINE)
_URL_RE = re.compile(r"^\s*URL:\s*(\S+)\s*$", re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^\s*CONFIDENCE:\s*(high|medium|low)\s*$", re.MULTILINE | re.IGNORECASE)


def _parse_findings(raw_text: str, sub_question_id: str) -> list[Finding]:
    if "NO_FINDINGS" in raw_text.upper() and "###FINDING###" not in raw_text:
        return []

    out: list[Finding] = []
    for block_match in _FINDING_BLOCK_RE.finditer(raw_text):
        block = block_match.group(1)
        claim_m = _CLAIM_RE.search(block)
        quote_m = _QUOTE_RE.search(block)
        url_m = _URL_RE.search(block)
        conf_m = _CONFIDENCE_RE.search(block)

        if not (claim_m and quote_m and url_m and conf_m):
            continue

        confidence_val = conf_m.group(1).lower()
        if confidence_val not in ("high", "medium", "low"):
            confidence_val = "medium"

        quote_text = quote_m.group(1).strip().strip('"').strip()
        if len(quote_text) > 290:
            quote_text = quote_text[:287] + "..."

        try:
            finding = Finding(
                sub_question_id=sub_question_id,
                claim=claim_m.group(1).strip(),
                source_url=url_m.group(1).strip(),
                source_quote=quote_text,
                confidence=confidence_val,  # type: ignore[arg-type]
            )
            out.append(finding)
        except Exception:
            continue

    return out


def _synthesize_sync(
    sub_question: SubQuestion,
    results: list[SearchResult],
    model_override: str | None,
) -> list[Finding]:
    if not results:
        return []

    llm = get_llm(
        provider="groq",
        structured=False,
        temperature=0.1,
        model_override=model_override or GROQ_FAST_MODEL,
    )

    user_msg = (
        f"SUB-QUESTION: {sub_question.question}\n\n"
        f"SOURCES:\n{_format_sources(results)}\n\n"
        f"Extract findings now in the required format."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])
    except Exception as e:
        print(f"  [synth] WARN: synthesizer call failed for {sub_question.id}: {e}")
        return []

    raw_text = response.content if isinstance(response.content, str) else str(response.content)
    findings = _parse_findings(raw_text, sub_question.id)

    if not findings and "NO_FINDINGS" not in raw_text.upper():
        sample = raw_text[:200].replace("\n", " ")
        print(f"  [synth] WARN: zero parsed findings for {sub_question.id}. Raw start: {sample!r}")

    return findings


def synthesize_one(
    sub_question: SubQuestion,
    results: list[SearchResult],
    *,
    model_override: str | None = None,
) -> list[Finding]:
    return _synthesize_sync(sub_question, results, model_override)


async def _synthesize_one_async(
    sub_question: SubQuestion,
    results: list[SearchResult],
    model_override: str | None,
    sem: asyncio.Semaphore,
) -> list[Finding]:
    async with sem:
        findings = await asyncio.to_thread(
            _synthesize_sync, sub_question, results, model_override
        )
        await asyncio.sleep(_INTER_CALL_DELAY)
        return findings


async def synthesize_all(
    sub_questions: list[SubQuestion],
    results_by_sq: dict[str, list[SearchResult]],
    *,
    model_override: str | None = None,
) -> list[Finding]:
    # Create the semaphore inside the coroutine so it binds to the running loop.
    # See the note near the top of this file for why this must not be module-scope.
    sem = asyncio.Semaphore(_MAX_CONCURRENT_LLM_CALLS)
    tasks = [
        _synthesize_one_async(sq, results_by_sq.get(sq.id, []), model_override, sem)
        for sq in sub_questions
    ]
    findings_per_sq = await asyncio.gather(*tasks)
    return [f for sublist in findings_per_sq for f in sublist]