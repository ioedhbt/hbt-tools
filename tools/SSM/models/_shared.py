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


# ════════════════════════════════════════════════════════════════════════════════
# Smith-chart overlay for the no-parasitics topology illustration
# ════════════════════════════════════════════════════════════════════════════════
# When a model has *no* parasitics, the topology expander shows a second column:
# the same schematic PNG with the user's CUSTOMIZED matplotlib Smith chart (the
# one from the "🍩 Smith Chart (Matplotlib)" expander) composited at the
# bottom-right.  Tune the two constants below to move / resize that inset so it
# does not overlap the circuit:
#
#   SMITH_OVERLAY_SIZE_FRAC : inset WIDTH as a fraction of the schematic image
#                             width.  Make this SMALLER to shrink the Smith chart
#                             so it stops overlapping the circuit.
#   SMITH_OVERLAY_POS_FRAC  : (x, y) of the inset's TOP-LEFT corner, as fractions
#                             of (image width, image height).  (0, 0) = top-left;
#                             larger x → further right, larger y → further down.
#                             Increase x / y to push the chart toward the corner.
SMITH_OVERLAY_SIZE_FRAC = 0.43
SMITH_OVERLAY_POS_FRAC = (0.57, 0.47)

# ── π-topology (Cheng π and any other Pi model) ─────────────────────────────
# The π schematic has little empty space, so the Smith chart would cover the
# circuit.  For Pi we FIRST extend the canvas with white space, THEN drop the
# Smith chart into that new area.  Tune these THREE constants for the Pi model
# (they do NOT affect the T-topology, which uses the generic constants above):
#
#   PI_SMITH_PAD_FRAC          : (right, bottom) white space to ADD, as fractions
#                                of the ORIGINAL (width, height).  e.g. (0.0, 0.5)
#                                adds 50% more height of white below the circuit.
#                                Increase to make more room for the chart.
#   PI_SMITH_OVERLAY_SIZE_FRAC : inset WIDTH as a fraction of the PADDED image
#                                width.  Smaller → smaller Smith chart.
#   PI_SMITH_OVERLAY_POS_FRAC  : (x, y) top-left corner of the inset, as fractions
#                                of the PADDED image (width, height).
PI_SMITH_PAD_FRAC = (0.3, 0.3)
PI_SMITH_OVERLAY_SIZE_FRAC = 0.4
PI_SMITH_OVERLAY_POS_FRAC = (0.4, 0.45)


def composite_smith_overlay(img, smith_png: bytes, *, pad=(0.0, 0.0),
                            size_frac=None, pos_frac=None):
    """Return a copy of PIL image ``img`` with the user's customized Smith chart
    (``smith_png`` — PNG bytes from ``render_matplotlib_smith(..., return_png=
    True)``) composited near the bottom-right.

    ``pad`` = (right_frac, bottom_frac) extends the canvas with WHITE space
    before compositing (used by Pi topologies that lack empty space).
    ``size_frac`` / ``pos_frac`` override the generic placement constants; when
    None they default to ``SMITH_OVERLAY_SIZE_FRAC`` / ``SMITH_OVERLAY_POS_FRAC``.
    Position fractions are measured on the *padded* image."""
    import io
    from PIL import Image

    size_frac = SMITH_OVERLAY_SIZE_FRAC if size_frac is None else size_frac
    pos_frac  = SMITH_OVERLAY_POS_FRAC  if pos_frac  is None else pos_frac

    img = img.convert("RGB")
    W0, H0 = img.size
    add_w, add_h = int(W0 * pad[0]), int(H0 * pad[1])
    if add_w or add_h:
        canvas = Image.new("RGB", (W0 + add_w, H0 + add_h), "white")
        canvas.paste(img, (0, 0))
        img = canvas

    W, H = img.size
    size_px = max(60, int(W * size_frac))
    inset = Image.open(io.BytesIO(smith_png)).convert("RGBA")
    iw, ih = inset.size
    inset = inset.resize((size_px, max(1, int(ih * size_px / iw))), Image.LANCZOS)

    x = int(W * pos_frac[0])
    y = int(H * pos_frac[1])
    out = img.convert("RGBA").copy()
    out.alpha_composite(inset, (x, y))
    return out.convert("RGB")
