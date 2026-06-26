"""
_i18n.py — Inline bilingual (English / 中文) helper for the custom-model
builder UI.

The builder has far too many one-off strings (section headers, captions, help
text, button labels, many of them with embedded terminal letters) to be worth a
central key registry, so each call site keeps its English + 中文 literals
side-by-side via :func:`tr`.  The active language is owned by the portal's
:mod:`tools.i18n` (the 🌐 Language radio), so switching there re-renders every
``tr(...)`` string here.
"""
from __future__ import annotations

from ...i18n import is_zh


def tr(en: str, zh: str) -> str:
    """Return ``zh`` when the UI language is 中文, else ``en``.

    Pass already-formatted strings (f-strings are fine for both args) so dynamic
    parts — e.g. terminal letters B/C/E vs G/D/S — appear in both languages.
    """
    return zh if is_zh() else en
