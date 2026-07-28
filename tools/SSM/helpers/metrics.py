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
    """Linear interpolation to find the 0 dB crossing.

    Returns (fT, fmax) — either may be None if no in-band crossing exists.

    Picks the **highest-frequency** crossing that the gain reached after
    staying above 0 dB for a run of consecutive points, which is how
    :func:`extract_limit` (this module, used by the At-a-Glance / bulk-upload
    pages) has always defined a "genuine" crossing.  This function — used by
    the SSM Extraction and Simulation & Fitting pages — previously took
    ``idx[0]``, the *first* sign change, with no run-length filter, so a
    single noise dip below 0 dB anywhere before the real roll-off produced a
    much-too-low fT.  The two pages could therefore report different fT for
    the same device, in the same session, after a handoff.

    Selection can only improve on the old result: when no crossing clears the
    run-length bar it still returns the last crossing rather than None, so no
    caller loses a value it used to get.
    """
    # The earlier implementation looped in Python over every frequency
    # sample calling np.isfinite() per element — that single function ate
    # ~50% of the SSM extraction tab's per-DUT compute (~30 ms of ~60).
    # Everything below stays fully vectorised (<100 µs), including the
    # run-length filter, which uses a running "index of last point at or
    # below 0 dB" rather than a per-crossing backward scan.
    f_arr = np.asarray(f_ghz, dtype=float)

    def _zero_cross(arr):
        arr = np.asarray(arr, dtype=float)
        finite_pair = np.isfinite(arr[:-1]) & np.isfinite(arr[1:])
        crossing = (arr[:-1] > 0) & (arr[1:] <= 0) & finite_pair
        idx = np.flatnonzero(crossing)
        if idx.size == 0:
            return None

        # Consecutive points above 0 dB ending at each sample.
        above = np.isfinite(arr) & (arr > 0)
        pos = np.arange(arr.size)
        last_below = np.maximum.accumulate(np.where(above, -1, pos))
        run_len = pos - last_below

        # extract_limit uses a flat 10 on measured sweeps of several hundred
        # points; scale it down so a short simulated sweep isn't filtered to
        # nothing.
        min_run = min(10, max(2, arr.size // 8))
        genuine = idx[run_len[idx] >= min_run]

        i = int((genuine if genuine.size else idx)[-1])
        slope = arr[i+1] - arr[i]
        if not np.isfinite(slope) or slope == 0.0:
            return float(f_arr[i])
        return float(f_arr[i] - arr[i] * (f_arr[i+1] - f_arr[i]) / slope)

    return _zero_cross(h21_db), _zero_cross(U_db)


def extrap_20dbdec(f_ghz, gain_db, n_pts: int = 60, f_max_target=None):
    """
    20 dB/decade extrapolation of a gain trace beyond its highest measured frequency.

    Anchors a line of slope −20 dB/dec at the last finite (gain, freq) point and
    projects forward.  By default the projection stops at the 0-dB crossing
    (i.e. fT or fmax); pass ``f_max_target`` to extend it past that crossing.

    Returns
    -------
    (f_ext, g_ext, f_zero) :
        f_ext   : ndarray  Frequencies (GHz) of the extrapolated segment.
        g_ext   : ndarray  Corresponding gain values (dB).
        f_zero  : float    The 0-dB crossing frequency (GHz) — i.e. fT or fmax.
    or  (None, None, None) if the trace already crosses 0 dB inside the measured
    band or the data is unusable.
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
    f_end = max(f_zero, float(f_max_target)) if f_max_target else f_zero
    f_ext = np.logspace(np.log10(f_high), np.log10(f_end), n_pts)
    g_ext = g_high - 20.0 * np.log10(f_ext / f_high)
    return f_ext, g_ext, f_zero


def single_pole_extrap(f_ghz, gain_db, idx_lo, idx_hi,
                       n_pts: int = 60, f_max_target=None):
    """
    Single-pole (log-linear) fit over a user-chosen frequency window, projected
    forward to the 0-dB crossing.

    A single-pole transfer function rolls off at −20 dB/dec asymptotically, so
    a log-linear regression of ``gain_db`` vs ``log10(f_ghz)`` on a clean
    high-frequency portion of the trace yields the same fT/fmax as the
    slope-locked −20 dB/dec extrapolation when the device is well-behaved.
    The fitted slope can differ from −20 if the data isn't a clean single pole;
    that disagreement is itself diagnostic.

    Parameters
    ----------
    f_ghz, gain_db : array-like
        Full frequency axis (GHz) and gain (dB).
    idx_lo, idx_hi : int
        Inclusive index window into ``f_ghz`` / ``gain_db`` used for the fit.
    n_pts : int
        Number of points in the projected curve.
    f_max_target : float or None
        Upper-frequency limit (GHz) for the projection.  Defaults to the
        fitted 0-dB crossing; pass a larger value to extend past it.

    Returns
    -------
    (f_ext, g_ext, f_zero, slope, intercept) :
        f_ext     : ndarray (GHz) — frequency axis of the fitted/projected curve,
                    spanning the first window point through ``max(f_zero, f_max_target)``.
        g_ext     : ndarray (dB) — fitted gain values.
        f_zero    : float (GHz) — fitted 0-dB crossing.
        slope     : float (dB per decade of f).
        intercept : float (dB at f = 1 GHz).
    or (None, None, None, slope, intercept) when the fit returns a non-negative
    slope, when the window never reaches 0 dB (no unity gain → no fT/fmax
    exists, and the fitted crossing would sit *behind* the data), or when the
    projection otherwise fails.  ``slope`` / ``intercept`` may still be
    NaN if the window itself is unusable.
    """
    f = np.asarray(f_ghz, dtype=float)
    g = np.asarray(gain_db, dtype=float)
    nan = float("nan")
    if idx_lo < 0 or idx_hi >= len(f) or idx_hi <= idx_lo:
        return None, None, None, nan, nan
    f_w = f[idx_lo:idx_hi + 1]
    g_w = g[idx_lo:idx_hi + 1]
    mask = np.isfinite(f_w) & np.isfinite(g_w) & (f_w > 0)
    if mask.sum() < 2:
        return None, None, None, nan, nan
    log_f = np.log10(f_w[mask])
    with np.errstate(all="ignore"):
        slope, intercept = np.polyfit(log_f, g_w[mask], 1)
    if not np.isfinite(slope) or slope >= 0:
        return None, None, None, float(slope), float(intercept)
    f_zero = 10.0 ** (-intercept / slope)
    if not np.isfinite(f_zero) or f_zero <= 0:
        return None, None, None, float(slope), float(intercept)

    # An fT/fmax only exists if the device actually reaches unity gain. On a
    # window that never rises above 0 dB (an open-base / unbiased DUT, where
    # |h21| < 1 everywhere) the fitted line still crosses 0 dB — but *behind*
    # the data, so the "crossing" is a backward extrapolation, not a transit
    # frequency. Measured case: an open-base 5x10 DUT with max |h21| = -1.06 dB
    # reported f_zero = 0.013 GHz, below the 0.01-5 GHz band it was fitted on.
    # `extrap_20dbdec` already rejects this via its `gv[-1] <= 0` guard; this is
    # the matching guard for the single-pole path.
    if np.nanmax(g_w[mask]) <= 0.0 or f_zero <= float(f_w[mask][0]):
        return None, None, None, float(slope), float(intercept)
    f_start = float(f_w[mask][0])
    f_end = max(f_zero, float(f_max_target)) if f_max_target else f_zero
    if f_end <= f_start:
        return None, None, None, float(slope), float(intercept)
    f_ext = np.logspace(np.log10(f_start), np.log10(f_end), n_pts)
    g_ext = slope * np.log10(f_ext) + intercept
    return f_ext, g_ext, float(f_zero), float(slope), float(intercept)


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
