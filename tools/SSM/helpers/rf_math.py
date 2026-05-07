"""
helpers/rf_math.py — Pure RF math utilities (no Streamlit, no plotting).

(Was previously tools/SSM/ssm_core.py — that file has been deleted.)
"""
from __future__ import annotations
import hashlib, json
from functools import lru_cache
import numpy as np


# ── Y / Z / S conversions ─────────────────────────────────────────────────────

def s_to_y(S, z0=50.0):
    s11, s12, s21, s22 = S[:,0,0], S[:,0,1], S[:,1,0], S[:,1,1]
    d = (1+s11)*(1+s22) - s12*s21
    Y = np.zeros_like(S)
    Y[:,0,0] = ((1-s11)*(1+s22)+s12*s21) / (d*z0)
    Y[:,0,1] = -2*s12 / (d*z0)
    Y[:,1,0] = -2*s21 / (d*z0)
    Y[:,1,1] = ((1+s11)*(1-s22)+s12*s21) / (d*z0)
    return Y


def _inv2(M):
    """Batched analytic 2×2 inverse — replaces a per-matrix Python loop."""
    a = M[..., 0, 0]; b = M[..., 0, 1]
    c = M[..., 1, 0]; d = M[..., 1, 1]
    det = a * d - b * c
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_det = np.where(det == 0, np.nan + 0j, 1.0 / det)
    out = np.empty_like(M)
    out[..., 0, 0] =  d * inv_det
    out[..., 0, 1] = -b * inv_det
    out[..., 1, 0] = -c * inv_det
    out[..., 1, 1] =  a * inv_det
    return out

y_to_z = _inv2   # Z = Y⁻¹
z_to_y = _inv2   # Y = Z⁻¹


def y_to_s_single(Y, z0=50.0):
    Yn = Y * z0; I = np.eye(2)
    try:    return np.dot(I - Yn, np.linalg.inv(I + Yn))
    except: return np.full((2,2), np.nan+0j)


def y_to_s_batch(Y, z0=50.0):
    """Vectorised batched Y→S conversion — calls the analytic y_to_s_vec."""
    return y_to_s_vec(Y, z0, np)


def y_to_s_vec(Y, z0=50.0, xp=None):
    """Fully vectorised Y→S conversion for (..., 2, 2) arrays.

    Works with numpy *and* cupy (pass xp=cupy when on GPU).

    Hand-inlined 2×2 algebra — avoids cuSOLVER (xp.linalg.inv) and cuBLAS
    (xp.matmul) entirely.  For 2×2 matrices those generic kernels have huge
    setup cost relative to the 7-op analytic inverse.  Replacement gives
    ~10× speedup on batched 2×2 GPU workloads.
    """
    if xp is None:
        xp = np
    yn00 = Y[..., 0, 0] * z0
    yn01 = Y[..., 0, 1] * z0
    yn10 = Y[..., 1, 0] * z0
    yn11 = Y[..., 1, 1] * z0

    # M = I + Yn ; analytic inverse
    m00 = 1.0 + yn00
    m11 = 1.0 + yn11
    inv_det = 1.0 / (m00 * m11 - yn01 * yn10)
    i00 =  m11 * inv_det
    i01 = -yn01 * inv_det
    i10 = -yn10 * inv_det
    i11 =  m00 * inv_det

    # N = I - Yn ; S = N @ inv(M)
    n00 = 1.0 - yn00
    n11 = 1.0 - yn11
    s00 = n00 * i00 + (-yn01) * i10
    s01 = n00 * i01 + (-yn01) * i11
    s10 = (-yn10) * i00 + n11 * i10
    s11 = (-yn10) * i01 + n11 * i11

    return xp.stack(
        [xp.stack([s00, s01], axis=-1),
         xp.stack([s10, s11], axis=-1)],
        axis=-2,
    )


# ── Analytic 2×2 helpers — bypass cuSOLVER/cuBLAS for tight inner loops ─────

def inv2x2(M, xp=None):
    """Analytic inverse of (..., 2, 2) matrices.

    Replaces xp.linalg.inv on tight 2×2 batched workloads.  cuSOLVER's
    batched LU has launch overhead orders of magnitude larger than the
    7-op analytic adjugate formula needed for a 2×2 matrix.

    Returns a *new* (..., 2, 2) array on the same device as ``M``.
    """
    if xp is None:
        xp = np
    a = M[..., 0, 0]
    b = M[..., 0, 1]
    c = M[..., 1, 0]
    d = M[..., 1, 1]
    inv_det = 1.0 / (a * d - b * c)
    i00 =  d * inv_det
    i01 = -b * inv_det
    i10 = -c * inv_det
    i11 =  a * inv_det
    return xp.stack(
        [xp.stack([i00, i01], axis=-1),
         xp.stack([i10, i11], axis=-1)],
        axis=-2,
    )


def mm2x2(A, B, xp=None):
    """Analytic 2×2 batched matmul: ``A @ B`` for (..., 2, 2) tensors.

    Replaces xp.matmul for hot loops where launching cuBLAS GEMM dominates
    the actual arithmetic.
    """
    if xp is None:
        xp = np
    a00 = A[..., 0, 0]; a01 = A[..., 0, 1]
    a10 = A[..., 1, 0]; a11 = A[..., 1, 1]
    b00 = B[..., 0, 0]; b01 = B[..., 0, 1]
    b10 = B[..., 1, 0]; b11 = B[..., 1, 1]
    c00 = a00 * b00 + a01 * b10
    c01 = a00 * b01 + a01 * b11
    c10 = a10 * b00 + a11 * b10
    c11 = a10 * b01 + a11 * b11
    return xp.stack(
        [xp.stack([c00, c01], axis=-1),
         xp.stack([c10, c11], axis=-1)],
        axis=-2,
    )


# ── Statistics helpers ─────────────────────────────────────────────────────────

def safe_median(arr, n=None):
    a = arr[:n] if n is not None else arr
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if len(a) > 0 else 0.0


def strict_freq_check(f_dut, f_dummy, label):
    if len(f_dut) != len(f_dummy) or not np.allclose(f_dut, f_dummy, rtol=1e-5):
        raise ValueError(f"DUT and {label} frequency grids differ.")


# ── Extended element admittance / impedance ───────────────────────────────────
# Used for Open (pad cap + optional secondary parasitic) and Short (lead + optional Cpar).

def open_elem_Y(C, mode, extra, w):
    """
    Admittance of one pad capacitor at angular frequency w.

    mode="None"       → Y = jωC
    mode="Parallel L" → Y = jωC + 1/(jωL)   [resonance at 1/√LC]
    mode="Series L"   → Y = jωC / (1 - ω²LC)  [series resonance]
    mode="Series R"   → Y = jωC / (1 + jωRC)   [lossy cap, adds Re(Y)]
    """
    if mode == "Parallel L" and extra > 0:
        return 1j*w*C + 1.0/(1j*w*extra + 1e-60)
    if mode == "Series L" and extra > 0:
        denom = 1.0 - w**2 * extra * C
        if abs(denom) < 1e-10: denom = 1e-10
        return 1j*w*C / denom
    if mode == "Series R" and extra > 0:
        return 1j*w*C / (1.0 + 1j*w*extra*C)
    return 1j*w*C


def short_lead_Z(R, L, Cpar, w):
    """
    Impedance of one short-circuit lead at angular frequency w.

    Cpar=0  → Z = R + jωL
    Cpar>0  → Z = (R+jωL) ∥ (1/jωCpar)   [parallel tank]
    """
    Z = R + 1j*w*L
    if Cpar > 0:
        return 1.0 / (1.0/Z + 1j*w*Cpar)
    return Z


# ── Smith chart grid ──────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _smith_grid_xy(max_r=1.0):
    """Cached coordinate arrays for the Smith-chart background.

    Returns a tuple of (x, y, color, width) tuples, one per trace.  The
    Scatter wrappers are built fresh each call (Plotly mutates trace
    `_parent` references on add_trace, so reusing instances is unsafe)
    but the numpy work — np.linspace + sin/cos/sqrt + masking — is reused.
    """
    out = []
    t  = np.linspace(0, 2*np.pi, 500)
    cos_t, sin_t = np.cos(t), np.sin(t)

    for ro in np.arange(1.0, max_r + 0.5, 1.0):
        lw  = 1.6 if ro == 1.0 else 0.9
        col = "rgba(60,60,60,0.85)" if ro == 1.0 else "rgba(170,170,170,0.6)"
        out.append((cos_t * ro, sin_t * ro, col, lw))

    out.append((np.array([-max_r, max_r]), np.array([0., 0.]),
                "rgba(100,100,100,0.6)", 0.8))

    gray = "rgba(155,155,155,0.5)"
    for r in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cx_ = r/(r+1); rad = 1.0/(r+1)
        xc = cx_ + rad*cos_t; yc = rad*sin_t
        mg = np.sqrt(xc**2+yc**2)
        xc = np.where(mg > max_r, np.nan, xc)
        yc = np.where(mg > max_r, np.nan, yc)
        out.append((xc, yc, gray, 0.8))

    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        for sign in [1, -1]:
            xv = sign*x; rad_x = abs(1.0/xv)
            xc = 1.0 + rad_x*cos_t; yc = (1.0/xv) + rad_x*sin_t
            mg = np.sqrt(xc**2+yc**2)
            xc = np.where(mg > max_r, np.nan, xc)
            yc = np.where(mg > max_r, np.nan, yc)
            out.append((xc, yc, gray, 0.8))

    return tuple(out)


def extended_smith_grid(max_r=1.0):
    """Return list of Plotly traces forming a Smith chart background.

    Coordinate computation is cached via _smith_grid_xy; the Scatter
    wrappers are rebuilt fresh on every call so each Plotly figure owns
    independent trace instances.
    """
    import plotly.graph_objects as go
    sk = dict(mode="lines", showlegend=False, hoverinfo="skip")
    return [
        go.Scattergl(x=x, y=y, line=dict(color=col, width=lw), **sk)
        for (x, y, col, lw) in _smith_grid_xy(max_r)
    ]


# ── Misc ──────────────────────────────────────────────────────────────────────

def params_hash(p: dict) -> str:
    try:
        items = {}
        for k, v in p.items():
            try:
                items[k] = round(float(v), 15)
            except (ValueError, TypeError):
                items[k] = str(v)
        return hashlib.md5(
            json.dumps(items, sort_keys=True).encode()
        ).hexdigest()
    except Exception:
        return ""
