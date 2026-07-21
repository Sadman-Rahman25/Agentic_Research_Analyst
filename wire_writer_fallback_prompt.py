"""wire_writer_fallback_prompt.py

Step 2 of 2: wires WRITER_FALLBACK_PROMPT into the 8B fallback path.

Currently `messages` is built once (outside `_invoke`) using WRITER_SYSTEM_PROMPT
and reused for both 70B and 8B calls. This script moves the messages list
inside `_invoke` so it picks the appropriate prompt based on the model tier.

Run from project root:
    python wire_writer_fallback_prompt.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


def main() -> None:
    path = Path("src/agents/writer.py")
    if not path.exists():
        print(f"ERROR: {path} not found. Run from project root.")
        return

    content = path.read_text(encoding="utf-8")

    if "WRITER_FALLBACK_PROMPT" not in content:
        print("ERROR: WRITER_FALLBACK_PROMPT not found in file.")
        print("Run add_writer_fallback_prompt.py first.")
        return

    if "system_prompt = WRITER_FALLBACK_PROMPT" in content:
        print("NOTE: already wired. Nothing to do.")
        return

    # Match: the current outer messages list + blank line + def _invoke + first line of body
    pattern = re.compile(
        r'^(?P<indent>[ \t]+)messages\s*=\s*\[[\r\n]+'
        r'\s*SystemMessage\(content=WRITER_SYSTEM_PROMPT\),[\r\n]+'
        r'\s*HumanMessage\(content=user_msg\),[\r\n]+'
        r'\s*\][\r\n]+'
        r'\s*[\r\n]+'
        r'\s*def _invoke\(use_fast:\s*bool\)\s*:[\r\n]+'
        r'(?P<body_indent>[ \t]+)llm\s*=\s*get_llm\(',
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        outer = match.group("indent")
        body = match.group("body_indent")
        deeper = body + "    "
        return (
            f"{outer}def _invoke(use_fast: bool):\n"
            f"{body}# 70B: full thesis-style prompt (~900 words, many rules).\n"
            f"{body}# 8B: simpler prompt (~400 words). 8B cannot reliably follow the full 70B spec.\n"
            f"{body}system_prompt = WRITER_FALLBACK_PROMPT if use_fast else WRITER_SYSTEM_PROMPT\n"
            f"{body}messages = [\n"
            f"{deeper}SystemMessage(content=system_prompt),\n"
            f"{deeper}HumanMessage(content=user_msg),\n"
            f"{body}]\n"
            f"{body}llm = get_llm("
        )

    new_content, n = pattern.subn(repl, content, count=1)

    if n == 0:
        print("ERROR: could not find the expected messages/_invoke pattern.")
        print("The code shape may differ from what this script expects.")
        print("Please paste lines 258-275 of writer.py to Claude and we'll adjust.")
        return

    # Syntax check BEFORE overwriting
    try:
        ast.parse(new_content)
    except SyntaxError as e:
        print(f"ERROR: patched file would have a syntax error at line {e.lineno}: {e.msg}")
        print("Not writing changes.")
        return

    # Also verify semantic reasonableness: both prompt names should still appear
    # (WRITER_SYSTEM_PROMPT once in the ternary, WRITER_FALLBACK_PROMPT once too)
    sys_count = new_content.count("WRITER_SYSTEM_PROMPT")
    fb_count = new_content.count("WRITER_FALLBACK_PROMPT")
    if sys_count < 2 or fb_count < 2:
        print(f"WARN: expected each prompt name to appear >=2 times "
              f"(1 declaration + 1 use). Got SYSTEM={sys_count}, FALLBACK={fb_count}.")
        print("Aborting to be safe.")
        return

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(new_content)

    print(f"OK: wired 8B fallback in {path}")
    print(f"    File now: {path.stat().st_size:,} bytes")
    print(f"    WRITER_SYSTEM_PROMPT references:   {sys_count} (expected 2)")
    print(f"    WRITER_FALLBACK_PROMPT references: {fb_count} (expected 2)")


if __name__ == "__main__":
    main()