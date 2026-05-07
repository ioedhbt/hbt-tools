"""
helpers/s2p_io.py — Touchstone .s2p / VNA CSV parsing and writing,
plus forward simulators for Open and Short dummy structures.

Consolidates:
  - ssm_s2p.py   (build_Y_pad, build_Z_ser, write_s2p, parse_s2p_bytes,
                  interpolate_s2f, simulate_open, simulate_short)
  - IOED_HBT_RF_extract.py  (parse_s2p [str], parse_csv, _load_cal)

`parse_s2p` accepts either str or bytes; `parse_s2p_bytes` is kept as a
backwards-compatibility alias for existing imports.
"""
from __future__ import annotations
import io
from typing import Union

import numpy as np
import pandas as pd

from .rf_math import open_elem_Y, short_lead_Z, y_to_s_single, y_to_s_vec, inv2x2

# Optional Streamlit memoisation. No-op decorator outside a Streamlit run.
try:
    import streamlit as _st
    def _cache_data(**kwargs):
        return _st.cache_data(show_spinner=False, **kwargs)
except Exception:
    def _cache_data(**kwargs):
        return lambda f: f


# ── Pad / lead matrix builders (per-frequency, used by simulators) ───────────

def build_Y_pad(p: dict, w: float) -> np.ndarray:
    """
    2×2 admittance matrix for the three pad capacitors at angular freq w.
    Accounts for extended element models stored in p (mode / extra keys).

    Circuit:  Cpbe from B-node to GND,  Cpce from C-node to GND,
              Cpbc from B-node to C-node.
    """
    Ypbe = open_elem_Y(p["Cpbe"], p.get("Cpbe_mode","None"), p.get("Cpbe_extra",0.0), w)
    Ypce = open_elem_Y(p["Cpce"], p.get("Cpce_mode","None"), p.get("Cpce_extra",0.0), w)
    Ypbc = open_elem_Y(p["Cpbc"], p.get("Cpbc_mode","None"), p.get("Cpbc_extra",0.0), w)
    return np.array([[Ypbe+Ypbc, -Ypbc],
                     [-Ypbc,  Ypce+Ypbc]])


def build_Z_ser(p: dict, w: float) -> np.ndarray:
    """
    2×2 impedance matrix for the three series leads at angular freq w.
    Accounts for optional parallel capacitance per lead (Cpar_Lb/Lc/Le).

    Using port notation:  Z11 = Zb+Ze,  Z12=Z21 = Ze,  Z22 = Zc+Ze.
    """
    Zb = short_lead_Z(p["Rpb"], p["Lb"], p.get("Cpar_Lb", 0.0), w)
    Zc = short_lead_Z(p["Rpc"], p["Lc"], p.get("Cpar_Lc", 0.0), w)
    Ze = short_lead_Z(p["Rpe"], p["Le"], p.get("Cpar_Le", 0.0), w)
    return np.array([[Zb+Ze, Ze],
                     [Ze,    Zc+Ze]])


# ── Touchstone write ──────────────────────────────────────────────────────────

def write_s2p(freq_hz: np.ndarray, S: np.ndarray,
              title: str = "", params: dict | None = None) -> bytes:
    """
    Serialize S-parameter data to Touchstone .s2p format.

    Format: # Hz S DB R 50
    Each row: freq  S11_dB S11_ang  S21_dB S21_ang  S12_dB S12_ang  S22_dB S22_ang

    Header comment lines list every key/value from `params`.
    """
    lines = [f"! Forward-simulated: {title}"]
    if params:
        for k, v in params.items():
            lines.append(f"!   {k} = {v}")
    lines.append("# Hz S DB R 50")
    for i, f in enumerate(freq_hz):
        parts = [f"{f:.0f}"]
        for r, c in [(0,0), (1,0), (0,1), (1,1)]:
            s = S[i, r, c]
            db  = 20*np.log10(abs(s) + 1e-30)
            ang = np.degrees(np.angle(s))
            parts += [f"{db:.8f}", f"{ang:.8f}"]
        lines.append(" ".join(parts))
    return "\n".join(lines).encode("utf-8")


# ── Touchstone read ───────────────────────────────────────────────────────────

def parse_s2p(content: Union[str, bytes]):
    """
    Parse a .s2p file. Accepts either str or bytes.
    Returns (freq_hz, S[N,2,2], z0).

    Handles MA, DB, RI formats; Hz/kHz/MHz/GHz frequency units.
    """
    if isinstance(content, (bytes, bytearray)):
        content = content.decode("utf-8", errors="ignore")

    freq_unit, fmt, z0 = "hz", "ma", 50.0
    data_lines = []

    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("!"): continue
        if s.startswith("#"):
            parts = s[1:].lower().split()
            for i, p in enumerate(parts):
                if p in ("hz","khz","mhz","ghz"):   freq_unit = p
                elif p in ("ma","db","ri"):           fmt = p
                elif p == "r" and i+1 < len(parts):
                    try: z0 = float(parts[i+1])
                    except: pass
            continue
        data_lines.append(s)

    vals = np.array([float(x) for x in " ".join(data_lines).split()])
    n = len(vals)//9
    vals = vals[:n*9].reshape(n, 9)
    scale = {"hz":1.0, "khz":1e3, "mhz":1e6, "ghz":1e9}[freq_unit]
    freq  = vals[:,0] * scale

    def to_c(ca, cb):
        a, b = vals[:,ca], vals[:,cb]
        if fmt == "db": return 10**(a/20.0) * np.exp(1j*np.deg2rad(b))
        if fmt == "ma": return a * np.exp(1j*np.deg2rad(b))
        return a + 1j*b   # ri

    S = np.zeros((n, 2, 2), dtype=complex)
    for (r,c),(ca,cb) in zip([(0,0),(1,0),(0,1),(1,1)], [(1,2),(3,4),(5,6),(7,8)]):
        S[:,r,c] = to_c(ca, cb)
    return freq, S, z0


def parse_s2p_bytes(raw: bytes):
    """Backwards-compatibility alias — same as `parse_s2p(raw)`."""
    return parse_s2p(raw)


def parse_csv(content: str, z0: float = 50.0):
    """Parse VNA CSV export (RI format). Expects columns:
    Frequency, Real(S11), Imag(S11), Real(S12), Imag(S12),
    Real(S21), Imag(S21), Real(S22), Imag(S22).
    Frequency must be in Hz."""
    df = pd.read_csv(io.StringIO(content))
    freq = df["Frequency"].values.astype(float)
    n = len(freq)
    S = np.zeros((n, 2, 2), dtype=complex)
    S[:, 0, 0] = df["Real(S11)"].values + 1j * df["Imag(S11)"].values
    S[:, 0, 1] = df["Real(S12)"].values + 1j * df["Imag(S12)"].values
    S[:, 1, 0] = df["Real(S21)"].values + 1j * df["Imag(S21)"].values
    S[:, 1, 1] = df["Real(S22)"].values + 1j * df["Imag(S22)"].values
    return freq, S, z0


def interpolate_s2f(f_src, S_src, f_tgt):
    """Interpolate S-parameter array from f_src grid to f_tgt grid."""
    S_out = np.zeros((len(f_tgt), 2, 2), dtype=complex)
    for r in range(2):
        for c in range(2):
            s = S_src[:,r,c]
            S_out[:,r,c] = (np.interp(f_tgt, f_src, s.real) +
                            1j*np.interp(f_tgt, f_src, s.imag))
    return S_out


# ── Streamlit-aware loader ────────────────────────────────────────────────────

def load_cal(fobj):
    """Load a calibration .s2p uploaded via Streamlit (`st.file_uploader`).

    Returns (freq, S, z0) on success; None if `fobj` is None or if parsing
    fails (in which case an error is shown in the sidebar).
    """
    if fobj is None:
        return None
    try:
        return parse_s2p(fobj.getvalue().decode("utf-8", errors="ignore"))
    except Exception as e:
        # Imported lazily so this module stays importable in non-Streamlit contexts
        import streamlit as st
        st.sidebar.error(f"Parse failed {fobj.name}: {e}")
        return None


# ── Forward simulators for dummy structures ───────────────────────────────────

@_cache_data(max_entries=128)
def simulate_open(p: dict, freq: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """
    Forward-simulate Open dummy S-parameters.
    Circuit: Y_pad only (no series leads, DUT replaced by open).

    p must contain: Cpbe/ce/bc and optional _mode/_extra.
    """
    # Lazy import — circular: deembed_math imports s2p_io, so the vec
    # builders must be fetched at call time.
    from .deembed_math import build_Y_pad_vec
    omega = 2.0 * np.pi * freq
    Y_pad = build_Y_pad_vec(p, omega, np)
    return y_to_s_vec(Y_pad, z0, np)


@_cache_data(max_entries=128)
def simulate_short(p: dict, freq: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """
    Forward-simulate Short dummy S-parameters.
    Circuit: Y_pad + inv(Z_ser)  (DUT terminals shorted → Z_DUT=0).

    p must contain all pad and lead parameters.
    """
    from .deembed_math import build_Y_pad_vec, build_Z_ser_vec
    omega = 2.0 * np.pi * freq
    Y_pad = build_Y_pad_vec(p, omega, np)
    Z_ser = build_Z_ser_vec(p, omega, np)
    Y_ser = inv2x2(Z_ser, np)         # batched analytic 2×2 inverse
    return y_to_s_vec(Y_pad + Y_ser, z0, np)
