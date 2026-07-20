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

from ...i18n import is_zh, tr  # noqa: F401  (re-exported for call sites)

__all__ = ["is_zh", "tr"]
