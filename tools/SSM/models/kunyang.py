"""
models/kunyang.py — Kun-Yang HEMT small-signal model (pi-topology, forward
simulation only — no extraction).

Built inside-out:

  1) Intrinsic three-component fundamental pi-model
       Y_gs : C_gs in series with R_i,  shunt port-1 → ground
       Y_gd : C_gd in series with R_gd, port-1 → port-2
       Y_ds : R_ds ∥ C_ds + transconductance,  shunt port-2 → ground
       gm   : Gm0 · exp(−jωτ)            current source V_gs → drain

  2) Source-side delay network in series with Rs + jωLs:
       Z_delay = R_delay ∥ (1/jωC_delay) = R_delay / (1+jωR_delay·C_delay)
       Z_src   = Rs + jωLs + Z_delay

  3) Add series-lead Z (analogue of HBT base/collector/emitter):
       gate  : Lg, Rg   (re-uses Lb / Rpb storage keys)
       drain : Ld, Rd   (re-uses Lc / Rpc storage keys)
       source: Ls, Rs (+ Z_delay above) (re-uses Le / Rpe storage keys)
     → Z_DUT.

  4) Kun-Yang custom pad / substrate network — in parallel with Z_DUT:
       Cgsp series Rsub1 : port-1 → ground
       Cdsp series Rsub2 : port-2 → ground
       Cgdp              : port-1 → port-2

  The standard open-dummy pad (Cpbe / Cpce / Cpbc) is NOT applied for
  this model — the Kun-Yang substrate network IS the pad layer.

Extraction is intentionally skipped: extract() returns sensible default
parameters and the user fine-tunes them.  Smith-chart fit comparison and
visual / auto tuning all work as for the other registered models.
"""
from __future__ import annotations
from pathlib import Path as _Path
import numpy as np
import streamlit as st

from ..helpers       import (y_to_s_single, y_to_s_vec,
                              params_hash,
                              build_Z_ser, build_Z_ser_vec, build_Z_ser_batch)
from .base_ui        import sync_pad_from_preov, PAD_SPECS, SSMModelTemplate
from ._shared        import _b1, _detect_B, _stack22
from . import AbstractSSMModel


_ILLUS_DIR = _Path(__file__).parent / "illus_template"


def _render_topology_illustration(all_p: dict, fname: str) -> None:
    """Display the static Kun-Yang HEMT schematic JPG (no value overlay)."""
    img_path = _ILLUS_DIR / "KYHEMT_full.jpg"
    if not img_path.exists():
        st.warning(f"Template not found: {img_path}")
        return
    st.image(str(img_path), width="stretch")


# ════════════════════════════════════════════════════════════════════════════════
# Intrinsic pi-model (Y_in) — vectorised forms
# ════════════════════════════════════════════════════════════════════════════════

def _Y_int_KY_vec(p, omega, xp):
    """Vectorised Kun-Yang intrinsic pi-model Y matrix → (N, 2, 2).

    Y_gs = jωC_gs / (1 + jωR_i·C_gs)           (port 1 → ground)
    Y_gd = jωC_gd / (1 + jωR_gd·C_gd)          (port 1 → port 2)
    Y_ds = 1/R_ds + jωC_ds                     (port 2 → ground)
    gm   = Gm0 · exp(−jωτ)                     (transconductance)

    Pi-network Y matrix:
      Y[0,0] = Y_gs + Y_gd
      Y[0,1] = −Y_gd
      Y[1,0] = gm − Y_gd
      Y[1,1] = Y_ds + Y_gd
    """
    Cgs = p["Cgs"]; Ri  = p["Ri"]
    Cgd = p["Cgd"]; Rgd = p["Rgd"]
    Cds = p["Cds"]; Rds = p["Rds"]
    Gm0 = p["Gm0"]; tau = p["tau"]

    jw    = 1j * omega
    Y_gs  = jw * Cgs / (1.0 + Ri  * jw * Cgs)
    Y_gd  = jw * Cgd / (1.0 + Rgd * jw * Cgd)
    Y_ds  = 1.0 / Rds + jw * Cds
    gm    = Gm0 * xp.exp(-jw * tau)

    N = len(omega)
    Y = xp.zeros((N, 2, 2), dtype=complex)
    Y[:, 0, 0] = Y_gs + Y_gd
    Y[:, 0, 1] = -Y_gd
    Y[:, 1, 0] = gm - Y_gd
    Y[:, 1, 1] = Y_ds + Y_gd
    return Y


def _Y_int_KY_batch(p, omega, B, N, xp, cache=None):
    """Batched Kun-Yang intrinsic pi-model Y → 4 (B, N) planes.

    Cache keys (built once by ``_build_intrinsic_static_cache`` when none
    of a sub-network's deps are being swept):

      - ``"KY_int_planes"`` : full (yi00..yi11) tuple
      - ``"Y_gs"`` / ``"Y_gd"`` / ``"Y_ds"`` / ``"gm"`` : per-branch caches
    """
    if cache is not None and "KY_int_planes" in cache:
        return cache["KY_int_planes"]

    cdtype = (cache or {}).get("_cdtype", np.complex128)
    rdtype = np.float32 if cdtype == np.complex64 else np.float64
    J = xp.asarray(1j, dtype=cdtype)

    if cache is not None and "Y_gs" in cache:
        Y_gs = cache["Y_gs"]
    else:
        Cgs = _b1(p, "Cgs", 0.0, xp, rdtype)
        Ri  = _b1(p, "Ri",  0.0, xp, rdtype)
        jwCgs = J * omega * Cgs
        Y_gs  = jwCgs / (1.0 + Ri * jwCgs)

    if cache is not None and "Y_gd" in cache:
        Y_gd = cache["Y_gd"]
    else:
        Cgd = _b1(p, "Cgd", 0.0, xp, rdtype)
        Rgd = _b1(p, "Rgd", 0.0, xp, rdtype)
        jwCgd = J * omega * Cgd
        Y_gd  = jwCgd / (1.0 + Rgd * jwCgd)

    if cache is not None and "Y_ds" in cache:
        Y_ds = cache["Y_ds"]
    else:
        Cds = _b1(p, "Cds", 0.0, xp, rdtype)
        Rds = _b1(p, "Rds", 1.0, xp, rdtype)
        Y_ds = 1.0 / Rds + J * omega * Cds

    if cache is not None and "gm" in cache:
        gm = cache["gm"]
    else:
        Gm0 = _b1(p, "Gm0", 0.0, xp, rdtype)
        tau = _b1(p, "tau", 0.0, xp, rdtype)
        gm  = Gm0 * xp.exp(-J * omega * tau)

    y00 = Y_gs + Y_gd
    y01 = -Y_gd
    y10 = gm - Y_gd
    y11 = Y_ds + Y_gd
    return y00, y01, y10, y11


# ════════════════════════════════════════════════════════════════════════════════
# Kun-Yang custom pad / substrate (parallel network)
# ════════════════════════════════════════════════════════════════════════════════

def _Y_kypad_vec(p, omega, xp):
    """Vectorised Kun-Yang custom pad Y matrix → (N, 2, 2).

    Y_gsp = jωC_gsp / (1 + jωR_sub1·C_gsp)   (port 1 → ground)
    Y_dsp = jωC_dsp / (1 + jωR_sub2·C_dsp)   (port 2 → ground)
    Y_gdp = jωC_gdp                          (port 1 → port 2, shunt cap)

      Y[0,0] = Y_gsp + Y_gdp
      Y[0,1] = −Y_gdp
      Y[1,0] = −Y_gdp
      Y[1,1] = Y_dsp + Y_gdp
    """
    Cgsp  = p.get("Cgsp",  0.0); Rsub1 = p.get("Rsub1", 0.0)
    Cdsp  = p.get("Cdsp",  0.0); Rsub2 = p.get("Rsub2", 0.0)
    Cgdp  = p.get("Cgdp",  0.0)

    jw     = 1j * omega
    jwCgsp = jw * Cgsp
    jwCdsp = jw * Cdsp
    Y_gsp  = jwCgsp / (1.0 + Rsub1 * jwCgsp)
    Y_dsp  = jwCdsp / (1.0 + Rsub2 * jwCdsp)
    Y_gdp  = jw * Cgdp

    N = len(omega)
    Y = xp.zeros((N, 2, 2), dtype=complex)
    Y[:, 0, 0] = Y_gsp + Y_gdp
    Y[:, 0, 1] = -Y_gdp
    Y[:, 1, 0] = -Y_gdp
    Y[:, 1, 1] = Y_dsp + Y_gdp
    return Y


def _Y_kypad_batch(p, omega, B, N, xp, cache=None):
    """Batched Kun-Yang custom pad Y → 4 (B, N) planes."""
    if cache is not None and "KY_pad_planes" in cache:
        return cache["KY_pad_planes"]

    cdtype = (cache or {}).get("_cdtype", np.complex128)
    rdtype = np.float32 if cdtype == np.complex64 else np.float64
    J = xp.asarray(1j, dtype=cdtype)

    Cgsp  = _b1(p, "Cgsp",  0.0, xp, rdtype)
    Rsub1 = _b1(p, "Rsub1", 0.0, xp, rdtype)
    Cdsp  = _b1(p, "Cdsp",  0.0, xp, rdtype)
    Rsub2 = _b1(p, "Rsub2", 0.0, xp, rdtype)
    Cgdp  = _b1(p, "Cgdp",  0.0, xp, rdtype)

    jwCgsp = J * omega * Cgsp
    jwCdsp = J * omega * Cdsp
    Y_gsp  = jwCgsp / (1.0 + Rsub1 * jwCgsp)
    Y_dsp  = jwCdsp / (1.0 + Rsub2 * jwCdsp)
    Y_gdp  = J * omega * Cgdp

    yk00 = Y_gsp + Y_gdp
    yk01 = -Y_gdp
    yk10 = -Y_gdp
    yk11 = Y_dsp + Y_gdp
    return yk00, yk01, yk10, yk11


# ════════════════════════════════════════════════════════════════════════════════
# Forward-simulation wrappers
# ════════════════════════════════════════════════════════════════════════════════

def _sim_wrap(Y_int_fn, p, freq, z0):
    """Per-frequency scalar forward sim.

    Inside → outside:
      Y_in (intrinsic) → Z_DUT = inv(Y_in) + Z_ser → Y_DUT + Y_KYpad → S.
    The standard open-dummy pad (Cpbe / Cpce / Cpbc) is NOT applied —
    the Kun-Yang substrate network IS the pad layer for this model.
    """
    omega = 2.0 * np.pi * freq
    S = np.zeros((len(freq), 2, 2), dtype=complex)
    R_d = p.get("R_delay", 0.0)
    C_d = p.get("C_delay", 0.0)
    for i, w in enumerate(omega):
        Y_in  = Y_int_fn(p, w)
        Z_ser = build_Z_ser(p, w)
        # Source-side delay network in series with Rs+jωLs.  The (indefinite-
        # T) Z_ser matrix already has Z_source in every element, so adding
        # Z_delay to all four entries puts Z_delay in series on the common
        # source path only.
        Z_delay = R_d / (1.0 + 1j * w * R_d * C_d) if R_d != 0.0 else 0.0
        Z_ser = Z_ser + Z_delay
        try:
            Y_dut = np.linalg.inv(np.linalg.inv(Y_in) + Z_ser)
        except np.linalg.LinAlgError:
            Y_dut = np.zeros((2, 2), dtype=complex)

        # Kun-Yang custom pad (parallel)
        Cgsp  = p.get("Cgsp",  0.0); Rsub1 = p.get("Rsub1", 0.0)
        Cdsp  = p.get("Cdsp",  0.0); Rsub2 = p.get("Rsub2", 0.0)
        Cgdp  = p.get("Cgdp",  0.0)
        jwCgsp = 1j * w * Cgsp
        jwCdsp = 1j * w * Cdsp
        Y_gsp = jwCgsp / (1.0 + Rsub1 * jwCgsp)
        Y_dsp = jwCdsp / (1.0 + Rsub2 * jwCdsp)
        Y_gdp = 1j * w * Cgdp
        Y_kp = np.array([[Y_gsp + Y_gdp, -Y_gdp],
                         [-Y_gdp,         Y_dsp + Y_gdp]])

        S[i]  = y_to_s_single(Y_dut + Y_kp, z0)
    return S


def _sim_wrap_vec(Y_int_vec_fn, p, freq, z0, xp):
    """Vectorised forward sim — works for numpy and cupy alike.

    Layers: intrinsic pi → Z_ser → KY substrate pad → S.  No standard
    open-dummy Y_pad layer (KY model uses its own custom pad).
    """
    omega = xp.asarray(2.0 * np.pi * freq, dtype=np.float64)

    # 1) Intrinsic Y
    Y_in = Y_int_vec_fn(p, omega, xp)

    # 2) Z_DUT = inv(Y_in) + Z_ser  → Y_DUT
    Z_ser = build_Z_ser_vec(p, omega, xp)
    # Source-side delay network: Z_delay = R_delay / (1 + jωR_delay·C_delay),
    # in series with Rs+jωLs.  Added to every element of Z_ser since the
    # indefinite-T matrix already carries Z_source in all four entries.
    R_d = p.get("R_delay", 0.0)
    C_d = p.get("C_delay", 0.0)
    Z_delay = R_d / (1.0 + 1j * omega * R_d * C_d)   # (N,) complex
    Z_ser[:, 0, 0] = Z_ser[:, 0, 0] + Z_delay
    Z_ser[:, 0, 1] = Z_ser[:, 0, 1] + Z_delay
    Z_ser[:, 1, 0] = Z_ser[:, 1, 0] + Z_delay
    Z_ser[:, 1, 1] = Z_ser[:, 1, 1] + Z_delay
    Y_dut = xp.linalg.inv(xp.linalg.inv(Y_in) + Z_ser)

    # 3) Add Kun-Yang custom pad (parallel)
    Y_kp = _Y_kypad_vec(p, omega, xp)

    S = y_to_s_vec(Y_dut + Y_kp, z0, xp)
    if xp is not np:
        S = xp.asnumpy(S)
    return S


def _sim_wrap_batch(Y_int_batch_fn, p, freq, z0, xp, cache=None):
    """Batched forward sim over (param_combo × freq).  Hand-inlined 2×2
    algebra throughout — mirrors cheng / xu so the same tuning kernels
    apply.  Returns (B, N, 2, 2) on the *xp* device.
    """
    N = len(freq)
    B = _detect_B(p, xp)
    cdtype = (cache or {}).get("_cdtype", np.complex128)
    rdtype = np.float32 if cdtype == np.complex64 else np.float64
    J = xp.asarray(1j, dtype=cdtype)
    if cache is not None and "omega" in cache:
        omega = cache["omega"]
    else:
        omega = xp.asarray(2.0 * np.pi * freq, dtype=rdtype).reshape(1, N)

    # ── Intrinsic 4 planes ───────────────────────────────────────────
    yi00, yi01, yi10, yi11 = Y_int_batch_fn(p, omega, B, N, xp, cache)

    # ── Z_ser as 4 planes (cache-aware) ─────────────────────────────
    if cache is not None and "Z_ser" in cache:
        zs00, zs01, zs10, zs11 = cache["Z_ser"]
    else:
        zs00, zs01, zs10, zs11 = build_Z_ser_batch(p, omega, B, N, xp)

    # ── Source-side delay network (added to all four Z_ser planes) ──
    if cache is not None and "Z_delay" in cache:
        Z_delay = cache["Z_delay"]
    else:
        R_d = _b1(p, "R_delay", 0.0, xp, rdtype)
        C_d = _b1(p, "C_delay", 0.0, xp, rdtype)
        Z_delay = R_d / (1.0 + J * omega * R_d * C_d)
    zs00 = zs00 + Z_delay
    zs01 = zs01 + Z_delay
    zs10 = zs10 + Z_delay
    zs11 = zs11 + Z_delay

    # ── Z_DUT = inv(Y_in) + Z_ser → Y_DUT (analytic 2×2) ────────────
    inv_det_i = 1.0 / (yi00 * yi11 - yi01 * yi10)
    zi00 =  yi11 * inv_det_i
    zi01 = -yi01 * inv_det_i
    zi10 = -yi10 * inv_det_i
    zi11 =  yi00 * inv_det_i

    zt00 = zi00 + zs00
    zt01 = zi01 + zs01
    zt10 = zi10 + zs10
    zt11 = zi11 + zs11

    inv_det_t = 1.0 / (zt00 * zt11 - zt01 * zt10)
    yd00 =  zt11 * inv_det_t
    yd01 = -zt01 * inv_det_t
    yd10 = -zt10 * inv_det_t
    yd11 =  zt00 * inv_det_t

    # ── Kun-Yang custom pad (cache-aware) — this IS the pad layer for KY,
    # no separate standard open-dummy Y_pad applied.
    yk00, yk01, yk10, yk11 = _Y_kypad_batch(p, omega, B, N, xp, cache)

    # ── Sum into Y_final ────────────────────────────────────────────
    ya00 = yd00 + yk00
    ya01 = yd01 + yk01
    ya10 = yd10 + yk10
    ya11 = yd11 + yk11

    # ── Y → S (fully inlined) ────────────────────────────────────────
    yn00 = ya00 * z0
    yn01 = ya01 * z0
    yn10 = ya10 * z0
    yn11 = ya11 * z0

    m00 = 1.0 + yn00
    m11 = 1.0 + yn11
    inv_det_m = 1.0 / (m00 * m11 - yn01 * yn10)
    mi00 =  m11 * inv_det_m
    mi01 = -yn01 * inv_det_m
    mi10 = -yn10 * inv_det_m
    mi11 =  m00 * inv_det_m

    n00 = 1.0 - yn00
    n11 = 1.0 - yn11
    s00 = n00 * mi00 + (-yn01) * mi10
    s01 = n00 * mi01 + (-yn01) * mi11
    s10 = (-yn10) * mi00 + n11 * mi10
    s11 = (-yn10) * mi01 + n11 * mi11

    return _stack22(s00, s01, s10, s11, xp)


# ════════════════════════════════════════════════════════════════════════════════
# Override-UI specs and pad-spec relabeling (gate/drain/source vs base/c/e)
# ════════════════════════════════════════════════════════════════════════════════

# Custom-pad (Kun-Yang substrate) parameters
_EXT_KY_SPECS = [
    ("Cgsp",  "Cgsp",  1e15, "fF", "%.4f", 0.1),
    ("Rsub1", "Rsub1", 1.0,  "Ω",  "%.3f", 1.0),
    ("Cdsp",  "Cdsp",  1e15, "fF", "%.4f", 0.1),
    ("Rsub2", "Rsub2", 1.0,  "Ω",  "%.3f", 1.0),
    ("Cgdp",  "Cgdp",  1e15, "fF", "%.4f", 0.01),
]

# Intrinsic pi-model parameters
# The last two rows (R_delay, C_delay) form a source-side delay network
# placed in series with Rs + jωLs, between the intrinsic source terminal
# and the source-lead block.  Z_delay = R_delay / (1 + jωR_delay·C_delay).
_INT_KY_SPECS = [
    ("Cgs",     "Cgs",     1e15, "fF", "%.4f", 0.1),
    ("Ri",      "Ri",      1.0,  "Ω",  "%.3f", 0.1),
    ("Cgd",     "Cgd",     1e15, "fF", "%.4f", 0.1),
    ("Rgd",     "Rgd",     1.0,  "Ω",  "%.3f", 0.1),
    ("Cds",     "Cds",     1e15, "fF", "%.4f", 0.1),
    ("Rds",     "Rds",     1.0,  "Ω",  "%.3f", 1.0),
    ("Gm0",     "Gm0",     1e3,  "mS", "%.4f", 0.1),
    ("tau",     "τ",       1e12, "ps", "%.4f", 0.01),
    ("R_delay", "R_delay", 1.0,  "Ω",  "%.3f", 0.1),
    ("C_delay", "C_delay", 1e15, "fF", "%.4f", 0.1),
]


# Pad caps (Cpbe / Cpce / Cpbc in the HBT models) are NOT used by the
# Kun-Yang HEMT — the model defines its own substrate pad network
# (Cgsp / Rsub1, Cdsp / Rsub2, Cgdp) in _EXT_KY_SPECS.  Only the access-
# resistance + lead-inductance entries are kept, relabelled to the HEMT
# terminology.  Key names stay identical so the existing de-embedding
# plumbing (build_Z_ser, peel_parasitics) keeps working.
_KY_PAD_LABEL_OVERRIDES = {
    "Lb":  "Lg",  "Lc":  "Ld",  "Le":  "Ls",
    "Rpb": "Rg",  "Rpc": "Rd",  "Rpe": "Rs",
}
_KY_PAD_DROP_KEYS = {"Cpbe", "Cpce", "Cpbc"}
_KY_PAD_SPECS = [
    (key, _KY_PAD_LABEL_OVERRIDES.get(key, lbl), sc, unit, fmt, step)
    for (key, lbl, sc, unit, fmt, step) in PAD_SPECS
    if key not in _KY_PAD_DROP_KEYS
]


def _override_ui(fname, tK, calc_vals, int_specs, label, ext_specs=_EXT_KY_SPECS):
    """Render the fine-tune override expander for the Kun-Yang HEMT model."""
    all_specs = _KY_PAD_SPECS + ext_specs + int_specs
    sync_pad_from_preov(fname, tK, calc_vals)

    _int_keys = [k for k, *_ in ext_specs + int_specs]
    _sync_hash_key = f"sim_synchash_{tK}_{fname}"
    _sync_hash = params_hash({k: str(round(float(calc_vals.get(k, 0.0)), 15))
                              for k in _int_keys})
    if st.session_state.get(_sync_hash_key) != _sync_hash:
        for key, _, scale, *_ in ext_specs + int_specs:
            st.session_state[f"sim_{tK}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
        st.session_state[_sync_hash_key] = _sync_hash

    with st.expander(f"✏️ Fine-tune {label} intrinsic/extrinsic parameters", expanded=False):
        if st.button(f"↩️ Reset {label} to default values",
                     key=f"rst_sim_{tK}_{fname}"):
            for key, _, scale, *_ in all_specs:
                st.session_state[f"sim_{tK}_{key}_{fname}"] = float(calc_vals.get(key, 0.0)) * scale
            st.rerun()

        # Substrate / custom-pad network FIRST — these caps ARE the pads
        # for the Kun-Yang HEMT (no separate Cpg / Cpd / Cpgd layer).
        st.markdown("**Kun-Yang Custom Pad / Substrate Network**")
        for row_start in range(0, len(ext_specs), 3):
            row = ext_specs[row_start:row_start + 3]
            for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                col_w.number_input(f"{lbl} ({unit})", key=f"sim_{tK}_{key}_{fname}",
                                   format=fmt, step=step)

        # Series-lead + access-resistance row — auto-synced from Step 1.
        st.markdown("**Access Resistance & Lead Inductance** "
                    "*(auto-synced from pre-extraction override)*")
        for row_start in range(0, len(_KY_PAD_SPECS), 3):
            row = _KY_PAD_SPECS[row_start:row_start + 3]
            for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                   key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

        st.markdown("**Intrinsic Pi-Model**")
        for row_start in range(0, len(int_specs), 4):
            row = int_specs[row_start:row_start + 4]
            for col_w, (key, lbl, sc, unit, fmt, step) in zip(st.columns(len(row)), row):
                col_w.number_input(f"{lbl} ({unit})" if unit else lbl,
                                   key=f"sim_{tK}_{key}_{fname}", format=fmt, step=step)

    all_p = {key: st.session_state.get(f"sim_{tK}_{key}_{fname}",
                                       float(calc_vals.get(key, 0.0)) * scale) / scale
             for key, _, scale, *_ in all_specs}
    # Pad caps are not part of the KY model; force them to zero so the
    # forward sim never sees stale values left over in para_eff.  Lead
    # parallel-Cpar values are still passed through for build_Z_ser.
    for ek in ["Cpbe", "Cpce", "Cpbc"]:
        all_p[ek] = 0.0
    for ek in ["Cpbe_mode", "Cpce_mode", "Cpbc_mode"]:
        all_p[ek] = "None"
    for ek in ["Cpbe_extra", "Cpce_extra", "Cpbc_extra",
               "Cpar_Lb", "Cpar_Lc", "Cpar_Le"]:
        all_p[ek] = calc_vals.get(ek, 0.0)
    return all_p


# ════════════════════════════════════════════════════════════════════════════════
# Model class
# ════════════════════════════════════════════════════════════════════════════════

# Default starting values (SI) for forward simulation.  Picked as
# representative GaAs / GaN HEMT mid-band numbers; the user is expected
# to fine-tune them.
_DEFAULT_PARAMS = {
    # Intrinsic pi-model
    "Cgs": 100e-15, "Ri":  3.0,
    "Cgd":  20e-15, "Rgd": 3.0,
    "Cds":  50e-15, "Rds": 200.0,
    "Gm0":  80e-3,  "tau": 1.0e-12,
    # Source-side delay network (R_delay∥C_delay in series with Rs+jωLs).
    # Defaults to R_delay=0 → Z_delay=0 (network effectively disabled).
    "R_delay": 0.0, "C_delay": 0.0,
    # Kun-Yang custom pad
    "Cgsp": 10e-15, "Rsub1": 1000.0,
    "Cdsp": 10e-15, "Rsub2": 1000.0,
    "Cgdp":  2e-15,
}


class KunYangHEMT(SSMModelTemplate, AbstractSSMModel):
    """
    Kun-Yang HEMT pi-topology small-signal model — forward simulation only.

    No measurement-based extraction is implemented; the user tunes parameters
    via the Fine-tune UI and the visual / auto-tuning expanders to fit the
    measured S-parameters on the Smith chart.

    Topology, inside → out:
      intrinsic pi (Cgs/Ri + Cgd/Rgd + Cds||Rds + gm)
        → Z_ser (gate/drain/source leads, re-uses HBT pad-spec keys)
        → Kun-Yang custom pad (Cgsp/Rsub1 + Cdsp/Rsub2 + Cgdp) → S
    The standard open-dummy pad (Cpbe / Cpce / Cpbc) is NOT applied;
    the Kun-Yang substrate network IS the pad layer for this model.
    """
    NAME          = "Kun-Yang HEMT (pi-model)"
    SHORT         = "KY"
    TOPOLOGY_CHAR = "pi"

    SUPPORTS_FP32_SWEEP = True

    # ── Template hooks (see SSMModelTemplate in base_ui.py) ──────────────────
    _INT_SPECS         = _INT_KY_SPECS
    _EXT_SPECS         = _EXT_KY_SPECS
    _Y_INT_VEC_FN      = _Y_int_KY_vec
    _Y_INT_BATCH_FN    = _Y_int_KY_batch
    _SIM_WRAP_VEC_FN   = _sim_wrap_vec
    _SIM_WRAP_BATCH_FN = _sim_wrap_batch
    _TUNING_PAD_SPECS  = _KY_PAD_SPECS   # relabel pads to HEMT terminology

    # Pre-bake truth table: when none of a sub-network's deps are swept the
    # sub-network can be built once and reused per tuning iteration.
    # (No "Y_pad" entry — the KY model does not use the standard pad layer.)
    STATIC_SUBNETWORKS = {
        "Z_ser":         frozenset({"Rpb", "Rpc", "Rpe", "Lb", "Lc", "Le"}),
        "Z_delay":       frozenset({"R_delay", "C_delay"}),
        "Y_gs":          frozenset({"Cgs", "Ri"}),
        "Y_gd":          frozenset({"Cgd", "Rgd"}),
        "Y_ds":          frozenset({"Cds", "Rds"}),
        "gm":            frozenset({"Gm0", "tau"}),
        "KY_int_planes": frozenset({"Cgs", "Ri", "Cgd", "Rgd",
                                     "Cds", "Rds", "Gm0", "tau"}),
        "KY_pad_planes": frozenset({"Cgsp", "Rsub1", "Cdsp", "Rsub2", "Cgdp"}),
    }

    # No PARAM_GROUPS — extraction is skipped, so the interactive
    # per-frequency-array section in the orchestrator is bypassed.

    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):
        """
        No measurement-based extraction is implemented for the Kun-Yang model.
        Returns default starting parameters so the user can tune to fit the
        measured S-parameters on the Smith chart.
        """
        return dict(_DEFAULT_PARAMS), {}

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        """Scalar forward sim (per-frequency loop)."""
        def _Y_int(p, w):
            jw   = 1j * w
            Y_gs = jw * p["Cgs"] / (1.0 + p["Ri"]  * jw * p["Cgs"])
            Y_gd = jw * p["Cgd"] / (1.0 + p["Rgd"] * jw * p["Cgd"])
            Y_ds = 1.0 / p["Rds"] + jw * p["Cds"]
            gm   = p["Gm0"] * np.exp(-jw * p["tau"])
            return np.array([[Y_gs + Y_gd, -Y_gd],
                             [gm   - Y_gd,  Y_ds + Y_gd]])
        return _sim_wrap(_Y_int, params, freq, z0)

    # simulate_vec, simulate_batch — inherited from SSMModelTemplate

    @classmethod
    def render_step_formulas(cls):
        with st.expander("📐 Kun-Yang HEMT formulas — forward simulation (no extraction)",
                         expanded=False):
            st.markdown("**1) Intrinsic pi-model**")
            st.latex(r"Y_{gs}=\frac{j\omega C_{gs}}{1+j\omega R_i C_{gs}},\quad "
                     r"Y_{gd}=\frac{j\omega C_{gd}}{1+j\omega R_{gd} C_{gd}}")
            st.latex(r"Y_{ds}=\frac{1}{R_{ds}}+j\omega C_{ds},\quad "
                     r"g_m=G_{m0}\,e^{-j\omega\tau}")
            st.latex(r"[Y_{in}]=\begin{bmatrix}Y_{gs}+Y_{gd} & -Y_{gd}\\"
                     r"g_m-Y_{gd} & Y_{ds}+Y_{gd}\end{bmatrix}")

            st.markdown("**2) Source-side delay network** "
                        "*(in series with Rs+jωLs)*")
            st.latex(r"Z_{delay}=R_{delay}\;\|\;\tfrac{1}{j\omega C_{delay}}"
                     r"=\frac{R_{delay}}{1+j\omega R_{delay} C_{delay}}")
            st.latex(r"Z_{src}=R_s+j\omega L_s+Z_{delay}")

            st.markdown("**3) Series-lead Z (gate / drain / source)**")
            st.latex(r"[Z_{ser}]=\begin{bmatrix}R_g+j\omega L_g+Z_{src} & Z_{src}\\"
                     r"Z_{src} & R_d+j\omega L_d+Z_{src}\end{bmatrix}")
            st.latex(r"[Y_{DUT}]=([Y_{in}]^{-1}+[Z_{ser}])^{-1}")

            st.markdown("**4) Kun-Yang custom pad / substrate**")
            st.latex(r"Y_{gsp}=\frac{j\omega C_{gsp}}{1+j\omega R_{sub1} C_{gsp}},\quad "
                     r"Y_{dsp}=\frac{j\omega C_{dsp}}{1+j\omega R_{sub2} C_{dsp}},\quad "
                     r"Y_{gdp}=j\omega C_{gdp}")
            st.latex(r"[Y_{KYpad}]=\begin{bmatrix}Y_{gsp}+Y_{gdp} & -Y_{gdp}\\"
                     r"-Y_{gdp} & Y_{dsp}+Y_{gdp}\end{bmatrix}")

            st.markdown("**5) Final Y → S** *(no standard open-dummy pad)*")
            st.latex(r"[Y_{tot}]=[Y_{DUT}]+[Y_{KYpad}]")
            st.latex(r"S=(I-Z_0[Y_{tot}])(I+Z_0[Y_{tot}])^{-1}")

    @classmethod
    def _results_rows(cls, params):
        p = params
        return [
            # Intrinsic
            ("Cgs",  f"{p.get('Cgs',0)*1e15:.4f}", "fF"),
            ("Ri",   f"{p.get('Ri',0):.4f}",        "Ω"),
            ("Cgd",  f"{p.get('Cgd',0)*1e15:.4f}", "fF"),
            ("Rgd",  f"{p.get('Rgd',0):.4f}",       "Ω"),
            ("Cds",  f"{p.get('Cds',0)*1e15:.4f}", "fF"),
            ("Rds",  f"{p.get('Rds',0):.4f}",       "Ω"),
            ("Gm0",  f"{p.get('Gm0',0)*1e3:.4f}",  "mS"),
            ("τ",    f"{p.get('tau',0)*1e12:.4f}", "ps"),
            # Source-side delay network
            ("R_delay", f"{p.get('R_delay',0):.4f}",        "Ω"),
            ("C_delay", f"{p.get('C_delay',0)*1e15:.4f}",   "fF"),
            # Custom pad
            ("Cgsp",  f"{p.get('Cgsp',0)*1e15:.4f}",  "fF"),
            ("Rsub1", f"{p.get('Rsub1',0):.4f}",       "Ω"),
            ("Cdsp",  f"{p.get('Cdsp',0)*1e15:.4f}",  "fF"),
            ("Rsub2", f"{p.get('Rsub2',0):.4f}",       "Ω"),
            ("Cgdp",  f"{p.get('Cgdp',0)*1e15:.4f}",  "fF"),
        ]

    @classmethod
    def _render_results_trace(cls):
        with st.expander("📐 Full formula trace — Kun-Yang HEMT", expanded=False):
            st.markdown("**Inside → out:** intrinsic pi → source-delay → Z_ser → KY-pad → S")
            st.markdown("**Intrinsic pi-model**")
            st.latex(r"Y_{gs}=\frac{j\omega C_{gs}}{1+j\omega R_i C_{gs}}")
            st.latex(r"Y_{gd}=\frac{j\omega C_{gd}}{1+j\omega R_{gd} C_{gd}}")
            st.latex(r"Y_{ds}=\tfrac{1}{R_{ds}}+j\omega C_{ds}")
            st.latex(r"g_m=G_{m0}\,e^{-j\omega\tau}")
            st.latex(r"[Y_{in}]=\begin{bmatrix}Y_{gs}+Y_{gd}&-Y_{gd}\\"
                     r"g_m-Y_{gd}&Y_{ds}+Y_{gd}\end{bmatrix}")
            st.markdown("**Source-side delay network** (parallel R∥C in series with Rs+jωLs)")
            st.latex(r"Z_{delay}=\frac{R_{delay}}{1+j\omega R_{delay} C_{delay}},\;"
                     r"Z_{src}=R_s+j\omega L_s+Z_{delay}")
            st.markdown("**Series-lead wrap** (gate ≡ Lb/Rpb, drain ≡ Lc/Rpc, source ≡ Le/Rpe)")
            st.latex(r"[Z_{ser}]=\begin{bmatrix}R_g+j\omega L_g+Z_{src}&Z_{src}\\"
                     r"Z_{src}&R_d+j\omega L_d+Z_{src}\end{bmatrix}")
            st.latex(r"[Y_{DUT}]=([Y_{in}]^{-1}+[Z_{ser}])^{-1}")
            st.markdown("**Kun-Yang custom pad / substrate** (parallel)")
            st.latex(r"Y_{gsp}=\frac{j\omega C_{gsp}}{1+j\omega R_{sub1} C_{gsp}},\;"
                     r"Y_{dsp}=\frac{j\omega C_{dsp}}{1+j\omega R_{sub2} C_{dsp}},\;"
                     r"Y_{gdp}=j\omega C_{gdp}")
            st.latex(r"[Y_{KYpad}]=\begin{bmatrix}Y_{gsp}+Y_{gdp}&-Y_{gdp}\\"
                     r"-Y_{gdp}&Y_{dsp}+Y_{gdp}\end{bmatrix}")
            st.markdown("**Final Y → S** *(no standard open-dummy pad layer)*")
            st.latex(r"[Y_{tot}]=[Y_{DUT}]+[Y_{KYpad}]")
            st.latex(r"S=(I-Z_0[Y_{tot}])(I+Z_0[Y_{tot}])^{-1}")

    @classmethod
    def _do_override_ui(cls, fname, calc_vals):
        return _override_ui(fname, cls.SHORT, calc_vals,
                            _INT_KY_SPECS, cls.NAME,
                            ext_specs=_EXT_KY_SPECS)

    @classmethod
    def _render_topology(cls, all_p, fname):
        _render_topology_illustration(all_p, fname)

    @classmethod
    def _build_intrinsic_static_cache(cls, p, omega, cache, xp, prebakeable):
        """Pre-bake every intrinsic + KY-pad + Z_delay sub-network whose
        deps are fully constant during the current sweep."""
        if "Z_delay" in prebakeable:
            r_d = float(p.get("R_delay", 0.0))
            c_d = float(p.get("C_delay", 0.0))
            cache["Z_delay"] = r_d / (1.0 + 1j * omega * r_d * c_d)
        if "Y_gs" in prebakeable:
            cgs = float(p.get("Cgs", 0.0)); ri = float(p.get("Ri", 0.0))
            jw_cgs = 1j * omega * cgs
            cache["Y_gs"] = jw_cgs / (1.0 + ri * jw_cgs)
        if "Y_gd" in prebakeable:
            cgd = float(p.get("Cgd", 0.0)); rgd = float(p.get("Rgd", 0.0))
            jw_cgd = 1j * omega * cgd
            cache["Y_gd"] = jw_cgd / (1.0 + rgd * jw_cgd)
        if "Y_ds" in prebakeable:
            cds = float(p.get("Cds", 0.0)); rds = float(p.get("Rds", 1.0))
            cache["Y_ds"] = 1.0 / rds + 1j * omega * cds
        if "gm" in prebakeable:
            gm0 = float(p.get("Gm0", 0.0)); tau = float(p.get("tau", 0.0))
            cache["gm"] = gm0 * xp.exp(-1j * omega * tau)

        if ("KY_int_planes" in prebakeable
                and "Y_gs" in cache and "Y_gd" in cache
                and "Y_ds" in cache and "gm" in cache):
            y_gs = cache["Y_gs"]; y_gd = cache["Y_gd"]
            y_ds = cache["Y_ds"]; gm   = cache["gm"]
            cache["KY_int_planes"] = (
                y_gs + y_gd,
                -y_gd,
                gm   - y_gd,
                y_ds + y_gd,
            )

        if "KY_pad_planes" in prebakeable:
            cgsp  = float(p.get("Cgsp",  0.0)); rsub1 = float(p.get("Rsub1", 0.0))
            cdsp  = float(p.get("Cdsp",  0.0)); rsub2 = float(p.get("Rsub2", 0.0))
            cgdp  = float(p.get("Cgdp",  0.0))
            jw_cgsp = 1j * omega * cgsp
            jw_cdsp = 1j * omega * cdsp
            y_gsp = jw_cgsp / (1.0 + rsub1 * jw_cgsp)
            y_dsp = jw_cdsp / (1.0 + rsub2 * jw_cdsp)
            y_gdp = 1j * omega * cgdp
            cache["KY_pad_planes"] = (
                y_gsp + y_gdp,
                -y_gdp,
                -y_gdp,
                y_dsp + y_gdp,
            )

    @classmethod
    def get_s2p_header_params(cls, params, para_eff):
        """Touchstone .s2p header — Kun-Yang substrate caps, access R / leads,
        intrinsic pi (no standard Cpbe / Cpce / Cpbc layer for this model)."""
        out = {}
        # Lead-equivalent (HEMT labels)
        for k, lbl in [("Rpb","Rg"), ("Rpc","Rd"), ("Rpe","Rs")]:
            out[lbl] = f"{para_eff.get(k,0):.4f} Ω"
        for k, lbl in [("Lb","Lg"), ("Lc","Ld"), ("Le","Ls")]:
            out[lbl] = f"{para_eff.get(k,0)*1e12:.4f} pH"
        # Kun-Yang custom pad
        for k, unit, scale in [("Cgsp","fF",1e15), ("Cdsp","fF",1e15),
                               ("Cgdp","fF",1e15)]:
            out[k] = f"{params.get(k,0)*scale:.4f} {unit}"
        for k in ["Rsub1", "Rsub2"]:
            out[k] = f"{params.get(k,0):.4f} Ω"
        # Intrinsic
        for k, unit, scale in [("Cgs","fF",1e15), ("Cgd","fF",1e15),
                               ("Cds","fF",1e15)]:
            out[k] = f"{params.get(k,0)*scale:.4f} {unit}"
        for k in ["Ri", "Rgd", "Rds"]:
            out[k] = f"{params.get(k,0):.4f} Ω"
        out["Gm0"] = f"{params.get('Gm0',0)*1e3:.4f} mS"
        out["tau"] = f"{params.get('tau',0)*1e12:.4f} ps"
        # Source-side delay network
        out["R_delay"] = f"{params.get('R_delay',0):.4f} Ω"
        out["C_delay"] = f"{params.get('C_delay',0)*1e15:.4f} fF"
        return out
