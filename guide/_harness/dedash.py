"""Replace prose em/en dashes in the guide, leaving the app's own labels alone.

The UI itself is full of dashes ("5 — Review", "Step 2 — Cbex", "Cje / τB+τC
from 1/(2πfT) vs 1/IC fit"), and a guide that renames them stops matching what
the reader sees on screen. So: skip code fences, inline code, image alt text,
and any "<digit> —" or "Step N —" heading the app owns; rewrite the rest.

    python3 guide/_harness/dedash.py guide/docs [--dry]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

DASH = "[—–]"
PROTECT = [
    re.compile(r"`[^`]*`"),                    # inline code
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),       # images, alt text included
    re.compile(r"\[[^\]]*\]\([^)]*\)"),        # links
    re.compile(r"\*\*[^*]*\*\*"),              # bold UI labels
    re.compile(r"\bStep\s+\d+\s*[—–][^,.;)]*"),
    re.compile(r"(?<![\w])\d+\s*[—–]\s*[A-Z][^,.;)]*"),   # "5 — Review"
    re.compile(r"[τƒ][^\s]*\s*[—–][^,.;)]*"),  # formula-ish labels
]

LIST_HEAD = re.compile(r"^(\s*[-*]\s+\*\*[^*]+\*\*)\s*[—–]\s*")


def mask(line: str):
    holes: list[str] = []

    def stash(m):
        holes.append(m.group(0))
        return f"\x00{len(holes) - 1}\x00"

    for pat in PROTECT:
        line = pat.sub(stash, line)
    return line, holes


def unmask(line: str, holes: list[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: holes[int(m.group(1))], line)


def fix_line(line: str) -> str:
    if not re.search(DASH, line):
        return line
    line = LIST_HEAD.sub(r"\1: ", line)
    body, holes = mask(line)
    body = re.sub(r"\s*[—–]\s*", ", ", body)
    return unmask(body, holes)


def fix_text(text: str) -> str:
    out, fenced = [], False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line)
            continue
        out.append(line if fenced else fix_line(line))
    return "".join(out)


def main() -> int:
    root = Path(sys.argv[1])
    dry = "--dry" in sys.argv
    for p in sorted(root.rglob("*.md")):
        src = p.read_text()
        new = fix_text(src)
        if new == src:
            continue
        left = len(re.findall(DASH, new))
        print(f"{p}: {len(re.findall(DASH, src))} -> {left}")
        if not dry:
            p.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
