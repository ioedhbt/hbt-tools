"""
models/_shared.py — Module-level helpers shared verbatim across model files.

These were duplicated in cheng.py and xu.py; lifted here so future model
modules can import the same code instead of copy-pasting.
"""
from __future__ import annotations
import os as _os
from pathlib import Path as _Path


# ════════════════════════════════════════════════════════════════════════════════
# Batched-simulation helpers (used by _sim_wrap_batch and _Y_int_*_batch kernels)
# ════════════════════════════════════════════════════════════════════════════════

def _b1(p, key, default, xp, dtype=None):
    """Fetch p[key] (or default) and reshape (B,) → (B,1).  Scalars stay scalar.

    If ``dtype`` is given, the value is coerced to that dtype.  Used by the
    fp32 sweep path so a scalar Python ``float`` constant doesn't promote
    a (B,N) ``float32`` swept tensor back up to ``float64``.
    """
    v = p.get(key, default)
    if dtype is not None:
        a = xp.asarray(v, dtype=dtype)
    else:
        a = xp.asarray(v)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


def _detect_B(p, xp):
    """Determine batch size B from any (B,)-shaped value in p."""
    B = 1
    for v in p.values():
        if isinstance(v, str):
            continue
        try:
            a = xp.asarray(v)
        except Exception:
            continue
        if a.ndim == 1 and a.shape[0] > B:
            B = a.shape[0]
    return B


def _stack22(a00, a01, a10, a11, xp):
    """Stack four (..., ) planes into a (..., 2, 2) tensor.

    Avoids the ``xp.zeros + scatter assignments`` pattern (5 kernel
    launches) — does it in 3 launches via xp.stack and amortises better
    on the GPU.  Inputs may be any broadcastable shapes; the result has
    the broadcast shape with two extra trailing axes.
    """
    return xp.stack(
        [xp.stack([a00, a01], axis=-1),
         xp.stack([a10, a11], axis=-1)],
        axis=-2,
    )


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
