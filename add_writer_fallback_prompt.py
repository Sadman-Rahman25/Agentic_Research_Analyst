"""add_writer_fallback_prompt.py

Step 1 of 2: adds WRITER_FALLBACK_PROMPT constant to src/agents/writer.py.

Why: WRITER_SYSTEM_PROMPT is tuned for 70B (900 words, many rules). When 70B
rate-limits and 8B takes over, 8B can't follow the full spec — it produces
short misshaped output. This adds a simpler prompt for the 8B path.

Step 2 will wire this constant into _write_report_fallback (or wherever the
8B call happens). That step needs to see the current code shape first.

Run from project root:
    python add_writer_fallback_prompt.py
"""

from __future__ import annotations

import re
from pathlib import Path


FALLBACK_PROMPT_CONTENT = """\
You are a research writer. You will be given a research QUESTION and a numbered
list of FINDINGS (each has a claim, verbatim quote, and source URL).

Write a markdown report with these EXACT section headers in this order:

## Summary
2-3 sentences answering the question. Cite [N] for main claims.

## Introduction
2-3 sentences of background context. Cite 2-3 findings.

## Methodology
1-2 sentences on how the sources approach this topic. Cite where relevant.

## Key Findings
3-4 sentences summarizing what the sources say. Cite heavily.

## Discussion
1-2 sentences on tensions or agreements between sources.

## Implementation Notes
1-2 sentences of practical takeaway.

## References
[1] <URL>
[2] <URL>
... (list every finding number you cited, in order, with source URL)

RULES:
- Every non-trivial claim needs an inline [N] citation.
- Only cite finding numbers that exist in the input.
- Multiple citations for one claim: [1][3] (not [1,3]).
- References section MUST list every [N] you cited, in order.
- Total length: 400-500 words.
- Do NOT invent findings.
- Do NOT add preamble like "As requested" or "In this report".

Output ONLY the markdown report starting with `## Summary`.
"""


def main() -> None:
    path = Path("src/agents/writer.py")
    if not path.exists():
        print(f"ERROR: {path} not found. Run from project root.")
        return

    content = path.read_text(encoding="utf-8")

    if "WRITER_FALLBACK_PROMPT" in content:
        print("NOTE: WRITER_FALLBACK_PROMPT already exists in file. Skipping.")
        print("If you want to update it, revert first with git and re-run.")
        return

    # Anchor: find the closing triple-quote of WRITER_SYSTEM_PROMPT
    pattern = r'(WRITER_SYSTEM_PROMPT[^=]*=\s*""".*?""")'

    new_constant = (
        '\n\n\n'
        'WRITER_FALLBACK_PROMPT: Final[str] = """\n'
        + FALLBACK_PROMPT_CONTENT
        + '"""'
    )

    new_content, n = re.subn(
        pattern,
        lambda m: m.group(1) + new_constant,
        content,
        count=1,
        flags=re.DOTALL,
    )

    if n == 0:
        print("ERROR: could not find WRITER_SYSTEM_PROMPT block to anchor to.")
        print("Expected pattern near top of writer.py:")
        print('  WRITER_SYSTEM_PROMPT: Final[str] = """... """')
        return

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)

    print(f"OK: added WRITER_FALLBACK_PROMPT to {path}")
    print(f"    File now: {path.stat().st_size:,} bytes")
    print(f"    Prompt: ~{len(FALLBACK_PROMPT_CONTENT.split())} words")
    print("")
    print("Next: paste your write_report / _write_report_fallback function")
    print("      to Claude so step 2 (wire the constant into the 8B path)")
    print("      can be done precisely.")


if __name__ == "__main__":
    main()