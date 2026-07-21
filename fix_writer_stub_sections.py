"""fix_writer_stub_sections.py

The Writer's emergency stub reports (used when both 70B and 8B rate-limit)
still reference old section names like '## TL;DR' and '## Raw Findings'.
This script updates them to match the new thesis-style schema so the stub
path stays consistent with what the LLM path produces.

Run from the project root:
    python fix_writer_stub_sections.py
"""

from __future__ import annotations

from pathlib import Path


# Each tuple: (old_string, new_string, expected_count).
# str.replace is used, not regex — safer for exact text.
REPLACEMENTS = [
    # Stub 1: emergency fallback when both models fail
    (
        '        "## TL;DR",\n        "",\n        f"_Writer agent failed: {reason}. Findings dumped below for manual review._",\n        "",\n        "## Raw Findings (writer unavailable)",',
        '        "## Summary",\n        "",\n        f"_Writer agent failed: {reason}. Findings dumped below for manual review._",\n        "",\n        "## Key Findings (writer unavailable)",',
        1,
    ),
    # Stub 2: empty findings case (Synthesizer found nothing)
    (
        '            "## TL;DR\\n\\n"\n            "No findings were extracted for this query. "',
        '            "## Summary\\n\\n"\n            "No findings were extracted for this query. "',
        1,
    ),
]


def main() -> None:
    path = Path("src/agents/writer.py")
    if not path.exists():
        print(f"ERROR: {path} not found. Run this from the project root.")
        return

    content = path.read_text(encoding="utf-8")
    original = content
    total_subs = 0

    for i, (old, new, expected_count) in enumerate(REPLACEMENTS, 1):
        count = content.count(old)
        if count == 0:
            print(f"WARN: replacement #{i} not found — target text may have changed.")
            print(f"  Looking for: {old[:80]!r}...")
            continue
        if count != expected_count:
            print(f"WARN: replacement #{i} found {count} times, expected {expected_count}. Not applied.")
            continue
        content = content.replace(old, new, count)
        total_subs += count
        print(f"OK: applied replacement #{i} ({count} substitution)")

    if content == original:
        print("No changes made.")
        return

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

    print(f"\nOK: patched {path} — {total_subs} total substitutions")
    print(f"File now: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()