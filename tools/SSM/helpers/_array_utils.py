"""
helpers/_array_utils.py — Tiny shape/broadcast utilities shared by the
batched de-embedding builders (in this package) and the model batched
forward simulators (in `tools.SSM.models`).

Previously each consumer carried its own copy of these three functions:
  - `helpers/deembed_math.py::_b1`
  - `models/_shared.py::_b1` (the canonical superset with `dtype=` support)
  - `models/degachi.py::_b1` / `_detect_B`

The duplicates drifted: the deembed_math copy lacked the `dtype=` arg, so
it silently promoted swept `float32` tensors back to `float64` on the
fp32-sweep path. Consolidating here removes both the drift risk and the
~40 lines of dead code.

Why this lives in `helpers/` and not `models/_shared.py`:
  `models/__init__.py` eagerly imports concrete model modules (cheng,
  xu, kunyang) which themselves import from `helpers/`. Putting these
  utilities under `models/_shared` would force `helpers/deembed_math` to
  reach back into `models/` for an import, creating a load-order cycle.
  Keeping them under `helpers/` lets both packages depend on a leaf
  module without circularity. `models/_shared.py` re-exports the same
  names so existing call sites (`from ._shared import _b1, _detect_B`)
  keep working unchanged.
"""
from __future__ import annotations


def _b1(p, key, default, xp, dtype=None):
    """Fetch p[key] (or default) and reshape (B,) → (B,1). Scalars stay scalar.

    If ``dtype`` is given, the value is coerced to that dtype. Used by the
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
    """Determine batch size B from any (B,)-shaped value in p.

    Walks the param dict and returns the largest leading-axis size found
    on any 1-D array value; falls back to 1 when no batched values
    exist. Skips strings (they'd error in `xp.asarray`).
    """
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
    on the GPU. Inputs may be any broadcastable shapes; the result has
    a trailing (2, 2) appended to their broadcast shape.
    """
    row0 = xp.stack([a00, a01], axis=-1)
    row1 = xp.stack([a10, a11], axis=-1)
    return xp.stack([row0, row1], axis=-2)
