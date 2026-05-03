"""
helpers/rf_math.py — Pure RF math utilities (no Streamlit, no plotting).

(Was previously tools/SSM/ssm_core.py — that file has been deleted.)
"""
from __future__ import annotations
import hashlib, json
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
    out = np.zeros_like(M)
    for i in range(len(M)):
        try:    out[i] = np.linalg.inv(M[i])
        except: out[i] = np.full((2,2), np.nan+0j)
    return out

y_to_z = _inv2   # Z = Y⁻¹
z_to_y = _inv2   # Y = Z⁻¹


def y_to_s_single(Y, z0=50.0):
    Yn = Y * z0; I = np.eye(2)
    try:    return np.dot(I - Yn, np.linalg.inv(I + Yn))
    except: return np.full((2,2), np.nan+0j)


def y_to_s_batch(Y, z0=50.0):
    S = np.zeros_like(Y); I = np.eye(2)
    for i in range(len(Y)):
        yn = Y[i] * z0
        try:    S[i] = np.dot(I - yn, np.linalg.inv(I + yn))
        except: S[i] = np.full((2,2), np.nan+0j)
    return S


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

def extended_smith_grid(max_r=1.0):
    """Return list of Plotly traces forming a Smith chart background."""
    import plotly.graph_objects as go
    traces = []; t = np.linspace(0, 2*np.pi, 500)
    sk = dict(mode="lines", showlegend=False, hoverinfo="skip")
    for ro in np.arange(1.0, max_r+0.5, 1.0):
        lw  = 1.6 if ro == 1.0 else 0.9
        col = "rgba(60,60,60,0.85)" if ro == 1.0 else "rgba(170,170,170,0.6)"
        traces.append(go.Scatter(x=np.cos(t)*ro, y=np.sin(t)*ro,
                                 line=dict(color=col, width=lw), **sk))
    traces.append(go.Scatter(x=[-max_r, max_r], y=[0., 0.],
                             line=dict(color="rgba(100,100,100,0.6)", width=0.8), **sk))
    gray = "rgba(155,155,155,0.5)"
    for r in [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]:
        cx_ = r/(r+1); rad = 1.0/(r+1)
        xc = cx_ + rad*np.cos(t); yc = rad*np.sin(t)
        mg = np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
        traces.append(go.Scatter(x=xc, y=yc, line=dict(color=gray, width=0.8), **sk))
    for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
        for sign in [1, -1]:
            xv = sign*x; rad_x = abs(1.0/xv)
            xc = 1.0 + rad_x*np.cos(t); yc = (1.0/xv) + rad_x*np.sin(t)
            mg = np.sqrt(xc**2+yc**2); xc[mg>max_r]=np.nan; yc[mg>max_r]=np.nan
            traces.append(go.Scatter(x=xc, y=yc, line=dict(color=gray, width=0.8), **sk))
    return traces


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
