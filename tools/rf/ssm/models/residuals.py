"""
models/residuals.py — S-parameter residual helpers, shared by every model's
tuning / fit-quality code.

Split out of models/base_ui.py (see models/base_ui/__init__.py for the
package-level re-exports that keep `from .base_ui import X` working
unchanged).
"""
from __future__ import annotations
import numpy as np


# ── Residual ──────────────────────────────────────────────────────────────────

def ssm_residual(S_mea: np.ndarray, S_mod: np.ndarray) -> float:
    """RMS relative S-parameter residual across all four ports (%)."""
    total = 0.0
    for r in range(2):
        for c in range(2):
            sm = S_mea[:,r,c]; sk = S_mod[:,r,c]
            denom = np.sum(np.abs(sm)**2)
            if denom > 0:
                total += np.sqrt(np.sum(np.abs(sm-sk)**2) / denom)
    return total / 4.0 * 100.0


# ── Per-port residuals ────────────────────────────────────────────────────────

def _port_residuals(S_mea, S_mod):
    """Return dict with Total, S11, S12, S21, S22 residuals in %."""
    res = {}
    total = 0.0
    for name, (r, c) in [("S11",(0,0)),("S12",(0,1)),("S21",(1,0)),("S22",(1,1))]:
        sm = S_mea[:, r, c]; sk = S_mod[:, r, c]
        denom = np.sum(np.abs(sm)**2)
        if denom > 0:
            val = np.sqrt(np.sum(np.abs(sm - sk)**2) / denom) * 100.0
        else:
            val = 0.0
        res[name] = val
        total += val
    res["Total"] = total / 4.0
    return res


def _port_residuals_batch(S_mea, S_mod_batch, xp):
    """Batched residuals — *fused* across all 4 ports for minimum kernel launches.

    S_mea       : (N, 2, 2)              — measured (already on device)
    S_mod_batch : (B, N, 2, 2)           — model
    Returns dict whose values are (B,) arrays on the *xp* device.

    Old impl ran ~5 ops per port × 4 ports + summation = ~22 kernel launches.
    This impl does ~7 launches total — significant Python/launch overhead saved
    in the GPU hot loop.
    """
    diff   = S_mea[None, :, :, :] - S_mod_batch        # (B, N, 2, 2)
    num    = xp.sum(xp.abs(diff)  ** 2, axis=1)        # (B, 2, 2)
    den    = xp.sum(xp.abs(S_mea) ** 2, axis=0)        # (2, 2)
    den_s  = xp.where(den > 0, den, 1.0)               # avoid div-by-0
    val    = xp.sqrt(num / den_s[None, :, :]) * 100.0  # (B, 2, 2)
    val    = xp.where(den[None, :, :] > 0, val, 0.0)

    s11 = val[:, 0, 0]
    s12 = val[:, 0, 1]
    s21 = val[:, 1, 0]
    s22 = val[:, 1, 1]
    total = (s11 + s12 + s21 + s22) * 0.25
    return {"S11": s11, "S12": s12, "S21": s21, "S22": s22, "Total": total}
