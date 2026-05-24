"""
models/_shared.py — Module-level helpers shared verbatim across model files.

These were duplicated in cheng.py and xu.py; lifted here so future model
modules can import the same code instead of copy-pasting.
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path


# ════════════════════════════════════════════════════════════════════════════════
# Batched-simulation array helpers — RE-EXPORTED from helpers/_array_utils.
# ════════════════════════════════════════════════════════════════════════════════
# These three functions used to live here, with byte-identical duplicates
# in helpers/deembed_math.py and models/degachi.py.  They moved to
# helpers/_array_utils.py so the helpers package (which can't reach back
# into models without a circular import) can use them too.  This module
# re-exports them so every existing `from ._shared import _b1, _detect_B,
# _stack22` call site keeps working unchanged.
from ..helpers._array_utils import _b1, _detect_B, _stack22  # noqa: F401


# ════════════════════════════════════════════════════════════════════════════════
# Font helpers (used by topology illustration overlays)
# ════════════════════════════════════════════════════════════════════════════════

_FONT_CACHE_DIR = _Path(__file__).parent / "fonts"
_INTER_DOWNLOAD_URLS = (
    "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf",
    "https://github.com/rsms/inter/raw/master/docs/font-files/Inter-Regular.ttf",
)


def _try_download_inter():
    """Attempt to download Inter once and cache it. Returns the cached path or None."""
    target = _FONT_CACHE_DIR / "Inter-Regular.ttf"
    if target.exists():
        return target
    try:
        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        import urllib.request
        for url in _INTER_DOWNLOAD_URLS:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = resp.read()
                if data and len(data) > 10_000:
                    target.write_bytes(data)
                    return target
            except Exception:
                continue
    except Exception:
        pass
    return None


def has_inter():
    for name in ("Inter-Regular.ttf", "Inter.ttf"):
        if _os.path.exists(name):
            return True
    cached = _FONT_CACHE_DIR / "Inter-Regular.ttf"
    if cached.exists():
        return True
    return _try_download_inter() is not None


def _load_font(size: int):
    """Load a TrueType font at the given size, with Inter → Arial → fallback chain."""
    from PIL import ImageFont
    candidates = [
        "Inter-Regular.ttf", "Inter.ttf",
        "arial.ttf", "Arial.ttf",
        "segoeui.ttf", "tahoma.ttf", "calibri.ttf",
    ]
    win_fonts = _os.path.join(_os.environ.get("WINDIR", "C:/Windows"), "Fonts")
    dirs = [
        str(_FONT_CACHE_DIR),
        win_fonts,
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/truetype/liberation",
        "/System/Library/Fonts",
    ]
    for name in candidates:
        for d in dirs:
            path = _os.path.join(d, name)
            if _os.path.exists(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
