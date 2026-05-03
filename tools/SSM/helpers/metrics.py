"""
helpers/metrics.py — Gain figures of merit and frequency-limit extraction.

Consolidates:
  - _compute_h21_U, _find_ft_fmax, extrap_20dbdec  (was in ssm_plots.py)
  - compute_metrics, extract_limit                 (was in IOED_HBT_RF_extract.py)

The leading underscores have been removed from the ssm_plots originals since
these functions are now public API of `tools.SSM.helpers`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .rf_math import s_to_y


# ── h21² / Mason U from S-parameters ──────────────────────────────────────────

def compute_h21_U(S):
    """Compute |h21|² (dB) and Mason's U (dB) from S-parameters.

    Uses z0 = 50 Ω.  Returns (h21_db, U_db) as 1-D arrays of length len(S).
    """
    Y = s_to_y(S, 50.0)
    y11, y12, y21, y22 = Y[:,0,0], Y[:,0,1], Y[:,1,0], Y[:,1,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h21     = -y21 / (y11 + 1e-30)
        h21_db  = 10.0*np.log10(np.abs(h21)**2 + 1e-30)
        num_u   = np.abs(y21 - y12)**2
        den_u   = 4.0*(y11.real*y22.real - y12.real*y21.real)
        U       = np.where(den_u > 0, num_u/den_u, np.nan)
        U_db    = 10.0*np.log10(np.abs(U) + 1e-30)
    return h21_db, U_db


def find_ft_fmax(f_ghz, h21_db, U_db):
    """Linear interpolation to find 0 dB crossing.

    Returns (fT, fmax) — either may be None if no in-band crossing exists.
    """
    def _zero_cross(f, arr):
        arr = np.asarray(arr, dtype=float)
        for i in range(len(arr) - 1):
            if np.isfinite(arr[i]) and np.isfinite(arr[i+1]) and arr[i] > 0 >= arr[i+1]:
                slope = arr[i+1] - arr[i]
                return float(f[i] - arr[i] * (f[i+1] - f[i]) / slope)
        return None
    return _zero_cross(f_ghz, h21_db), _zero_cross(f_ghz, U_db)


def extrap_20dbdec(f_ghz, gain_db, n_pts: int = 60):
    """
    20 dB/decade extrapolation of a gain trace beyond its highest measured frequency.

    If `gain_db` is still above 0 at the last finite point, project the trace forward
    along a -20 dB/dec slope (anchored at that last point) until it crosses 0 dB.

    Returns
    -------
    (f_ext, g_ext, f_zero) :
        f_ext   : ndarray  Frequencies (GHz) of the extrapolated segment, starting at
                           the last measured point and ending where g_ext == 0.
        g_ext   : ndarray  Corresponding gain values (dB).
        f_zero  : float    The 0-dB crossing frequency (GHz) — i.e. fT or fmax.
    or  (None, None, None) if the trace already crosses 0 dB inside the measured band
        or if the data is unusable.
    """
    g = np.asarray(gain_db, dtype=float)
    f = np.asarray(f_ghz, dtype=float)
    m = np.isfinite(g) & np.isfinite(f) & (f > 0)
    if not np.any(m):
        return None, None, None
    fv = f[m]; gv = g[m]
    if gv[-1] <= 0:
        return None, None, None
    f_high = float(fv[-1]); g_high = float(gv[-1])
    f_zero = f_high * 10.0 ** (g_high / 20.0)
    if not np.isfinite(f_zero) or f_zero <= f_high:
        return None, None, None
    f_ext = np.logspace(np.log10(f_high), np.log10(f_zero), n_pts)
    g_ext = g_high - 20.0 * np.log10(f_ext / f_high)
    return f_ext, g_ext, f_zero


# ── Full metrics DataFrame (h21², Mason U, MAG/MSG, K, plateau columns) ──────

def compute_metrics(Y, freq_hz):
    """Build the full metrics DataFrame used by IOED's Bode/plateau plots.

    Columns:
      Freq (GHz), |h21|² (dB), Mason U (dB), MAG/MSG (dB), K Factor,
      fT Plateau (GHz), fmax U Plateau (GHz), fmax MAG Plateau (GHz)
    """
    f = freq_hz*1e-9
    y11,y12,y21,y22 = Y[:,0,0],Y[:,0,1],Y[:,1,0],Y[:,1,1]
    with np.errstate(divide="ignore", invalid="ignore"):
        h21 = -y21/y11
        num_u = np.abs(y21-y12)**2
        den_u = 4.0*(y11.real*y22.real - y12.real*y21.real)
        U = np.where(den_u>0, num_u/den_u, np.nan)
        num_k = 2.0*y11.real*y22.real-(y12*y21).real
        K = num_k/(np.abs(y12*y21)+1e-60)
        MSG = np.abs(y21)/(np.abs(y12)+1e-30)
        MAG = MSG*(K-np.sqrt(np.clip(K**2-1.0,0,None)))
        MAG_MSG = np.where(K>1.0, MAG, MSG)
    return pd.DataFrame({
        "Freq (GHz)":f, "|h21|² (dB)":10*np.log10(np.abs(h21)**2+1e-30),
        "Mason U (dB)":10*np.log10(np.abs(U)+1e-30),
        "MAG/MSG (dB)":10*np.log10(np.abs(MAG_MSG)+1e-30),
        "K Factor":K, "fT Plateau (GHz)":f*np.abs(h21),
        "fmax U Plateau (GHz)":f*np.sqrt(np.abs(U)),
        "fmax MAG Plateau (GHz)":f*np.sqrt(np.abs(MAG_MSG)),
    })


# ── fT/fmax extractor with extrapolation fallback ────────────────────────────

def extract_limit(freq_ghz, gain_db, plateau_arr, n_pts, f_min, f_max):
    """Find a 'genuine' 0 dB crossing in the [f_min, f_max] window.

    Strategy:
      1. Look for a high-frequency 0 dB crossing where the gain stayed above 0
         for at least 10 consecutive points (filters out noise crossings).
      2. If no genuine crossing found but median gain is positive, do a
         log-linear extrapolation of the last `n_pts` points to find where the
         trace would hit 0 dB; report the plateau value as a sanity check.
      3. Otherwise return NaN.

    Returns (v_cross_or_extrap, v_plateau, method_label).
    """
    vm = (freq_ghz>=f_min)&(freq_ghz<=f_max)&~np.isnan(gain_db)
    if not np.any(vm): return np.nan, np.nan, "No Data"
    f_v,g_v,p_v,N = freq_ghz[vm],gain_db[vm],plateau_arr[vm],vm.sum()
    if np.nanmax(g_v)<=0: return np.nan, np.nan, "No Gain"
    above = g_v>=0
    crossings = np.where(above[:-1]&~above[1:])[0]
    genuine_idx = None
    for idx in crossings[::-1]:
        cnt=0
        for j in range(idx,-1,-1):
            if above[j]: cnt+=1
            else: break
        if cnt<10: continue
        if max(0,idx-cnt+1)>int(0.80*N) and cnt<20: continue
        genuine_idx=idx; break
    if genuine_idx is None:
        if np.median(g_v)>0:
            v_plat=np.nanmax(p_v) if not np.isnan(p_v).all() else np.nan
            n_use,v_extrap=min(n_pts,len(f_v)),np.nan
            if len(f_v[-n_use:])>=2:
                with np.errstate(all="ignore"):
                    m,c=np.polyfit(np.log10(f_v[-n_use:]),g_v[-n_use:],1)
                    if m<0: v_extrap=10**(-c/m)
            return v_extrap,v_plat,"Extrap & Plat."
        return np.nan,np.nan,"No Gain"
    idx=genuine_idx
    s,e=max(0,idx-n_pts//2+1),min(N,idx+n_pts//2+1+(n_pts%2))
    if (e-s)<2: s,e=max(0,idx),min(N,idx+2)
    with np.errstate(all="ignore"):
        v_cross=np.polyval(np.polyfit(g_v[s:e],f_v[s:e],min(2,e-s-1)),0.0)
        if v_cross<=0 or v_cross<f_v[s] or v_cross>f_v[e-1]:
            v_cross=f_v[idx]+(0-g_v[idx])*(f_v[idx+1]-f_v[idx])/(g_v[idx+1]-g_v[idx])
    return v_cross,np.nan,"0dB Cross"
