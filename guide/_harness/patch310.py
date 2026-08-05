"""Make the shadow copy parseable by Python 3.10.

The repo targets 3.12+, which relaxed f-strings (PEP 701): an expression inside
``{...}`` may span lines and may contain backslashes.  Four files use that.
Nothing here changes behaviour — a multi-line expression is folded onto one
line, and the single backslash case is hoisted into a local.  Run against the
shadow tree only; the real repo is never written to.

    python3 patch310.py /tmp/app
"""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

# The one case line-folding can't fix: a backslash inside the expression.
EXACT = [
    (
        "tools/rf/ssm/custom_model/schematic.py",
        '''                 f'{"" if present else " stroke-dasharray=\\"4 3\\""}/>')''',
        '''                 f'{_dash_attr(present)}/>')''',
    ),
]

HELPER = '''

def _dash_attr(present):
    """Dashed outline for an absent element (hoisted for py3.10 f-strings)."""
    return "" if present else ' stroke-dasharray="4 3"'
'''


def compiles(path: Path) -> str | None:
    try:
        py_compile.compile(str(path), doraise=True, cfile="/tmp/_p310.pyc")
        return None
    except py_compile.PyCompileError as exc:
        return str(exc)
    except SyntaxError as exc:  # pragma: no cover
        return str(exc)


def err_line(path: Path) -> int | None:
    try:
        py_compile.compile(str(path), doraise=True, cfile="/tmp/_p310.pyc")
        return None
    except py_compile.PyCompileError as exc:
        cause = exc.exc_value
        return getattr(cause, "lineno", None)


def fold(path: Path, limit: int = 60) -> bool:
    """Join an f-string's continuation lines until the file parses."""
    for _ in range(limit):
        n = err_line(path)
        if n is None:
            return True
        lines = path.read_text().splitlines(keepends=True)
        i = n - 1
        if i + 1 >= len(lines):
            return False
        head = lines[i].rstrip("\n").rstrip()
        tail = lines[i + 1].lstrip()
        lines[i : i + 2] = [head + " " + tail]
        path.write_text("".join(lines))
    return False


def main() -> int:
    app = Path(sys.argv[1]).resolve()

    for rel, old, new in EXACT:
        f = app / rel
        if not f.exists():
            continue
        src = f.read_text()
        if old in src:
            src = src.replace(old, new)
            # Put the helper after the module docstring / imports block.
            marker = "\ndef "
            at = src.index(marker)
            src = src[:at] + HELPER + src[at:]
            f.write_text(src)

    failed = []
    for p in sorted((app / "tools").rglob("*.py")):
        if compiles(p) is None:
            continue
        if not fold(p):
            failed.append(str(p.relative_to(app)))

    if failed:
        print("patch310: still broken:", *failed, sep="\n  ")
        return 1
    print("patch310: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
