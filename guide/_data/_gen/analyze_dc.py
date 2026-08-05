"""
guide/_data/_gen/analyze_dc.py — survey dc_data/raw/*.csv (B1500A exports)
and score each file of the four kinds (BC diode, BE diode, Gummel, Family)
for "cleanest / most representative", to pick the DC demo set.

Pure pandas/numpy — no repo imports, run with any Python 3.

    python3 guide/_data/_gen/analyze_dc.py
"""
import glob
import io
import os
import re

import numpy as np
import pandas as pd

RAW_DIR = "/sessions/eloquent-stoic-turing/mnt/hbt-tools/dc_data/raw"


def load_b1500a_csv(path):
    """Parse a Keysight B1500A EasyExpert-style CSV export.

    Structure: metadata lines (``Key, v1, v2, ...``) until a line starting
    with ``DataName`` gives the column headers, followed by ``DataValue``
    data rows, both comma-separated.
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        lines = fh.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DataName"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"no DataName row in {path}")
    cols = [c.strip() for c in lines[header_idx].split(",")[1:]]

    def to_f(p):
        try:
            return float(p)
        except ValueError:
            return np.nan

    rows = []
    for line in lines[header_idx + 1:]:
        if not line.strip().startswith("DataValue"):
            continue
        parts = line.rstrip("\n").split(",")[1:]
        # Ragged rows (trailing blank fields, e.g. an unused Ic_abs column)
        # are padded with NaN rather than dropping the whole row.
        if len(parts) < len(cols):
            parts = parts + [""] * (len(cols) - len(parts))
        rows.append([to_f(p) for p in parts[:len(cols)]])
    df = pd.DataFrame(rows, columns=cols)
    return df


def decades_of_clean_gummel(vb, ic, ib):
    """Rough count of decades over which log(Ic) vs Vb is well-behaved
    (monotonic increasing, positive, not compliance-clipped) — a stand-in
    for 'ideality visible over several decades'."""
    ic = np.asarray(ic, float)
    vb = np.asarray(vb, float)
    order = np.argsort(vb)
    vb, ic = vb[order], ic[order]
    pos = ic > 0
    if pos.sum() < 5:
        return 0.0, np.nan, np.nan
    ic_pos = ic[pos]
    lo, hi = np.nanmin(ic_pos), np.nanmax(ic_pos)
    decades = np.log10(hi / lo) if lo > 0 else np.nan
    # Monotonic fraction: how much of the forward sweep is strictly
    # increasing in log(Ic) (ignores noise-floor wiggle at the bottom).
    dlog = np.diff(np.log10(np.clip(ic_pos, 1e-300, None)))
    mono_frac = float(np.mean(dlog >= -1e-6)) if len(dlog) else np.nan
    return decades, mono_frac, lo


def forward_loglinear_r2(v, i, trim_lo=0.05, trim_hi=0.10):
    """R^2 of log10(I) vs V over the forward branch, after trimming the
    noisiest bottom fraction and the series-resistance-rolloff top fraction
    of the positive-current points (sorted by V). A high R^2 over many
    decades is a direct 'ideal diode / clean Gummel' signal — closer to
    what the app's own ideality_factor() window relies on."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    mask = (v > 0) & (i > 0)
    if mask.sum() < 8:
        return np.nan, np.nan, 0
    vv, ii = v[mask], i[mask]
    order = np.argsort(vv)
    vv, ii = vv[order], ii[order]
    n = len(vv)
    lo = int(n * trim_lo)
    hi = n - int(n * trim_hi)
    if hi - lo < 5:
        lo, hi = 0, n
    vv, ii = vv[lo:hi], ii[lo:hi]
    logi = np.log10(ii)
    if len(vv) < 5 or np.ptp(vv) == 0:
        return np.nan, np.nan, len(vv)
    slope, intercept = np.polyfit(vv, logi, 1)
    pred = slope * vv + intercept
    ss_res = np.sum((logi - pred) ** 2)
    ss_tot = np.sum((logi - logi.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    decades_window = float(logi.max() - logi.min())
    return float(r2), decades_window, len(vv)


def best_ideal_window_decades(v, i, r2_thresh=0.999, min_pts=10):
    """Longest contiguous (in V) run of forward, positive-current points
    whose log10(I) vs V is a near-perfect line (R^2 >= r2_thresh) — the
    span of that run, in decades of I, is a direct 'ideality visible over
    N clean decades' measure (what an ideality-factor window fit is
    actually looking for), independent of whatever the diode/HBT does
    outside that regime (series-R rolloff, leakage, noise floor)."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    mask = (v > 0) & (i > 0)
    if mask.sum() < min_pts:
        return 0.0, np.nan
    vv, ii = v[mask], i[mask]
    order = np.argsort(vv)
    vv = vv[order]
    logi = np.log10(ii[order])
    n = len(vv)
    best_dec, best_r2 = 0.0, np.nan
    for start in range(0, n - min_pts + 1):
        # Grow the window from `start` while R^2 stays above threshold;
        # once it drops, later starts only need to search past the last
        # good end (monotone-ish search, still simple O(n^2) worst case
        # but n <= ~400 here).
        end = start + min_pts
        last_good = None
        while end <= n:
            x = vv[start:end]
            y = logi[start:end]
            if np.ptp(x) == 0:
                end += 1
                continue
            slope, intercept = np.polyfit(x, y, 1)
            pred = slope * x + intercept
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
            if r2 >= r2_thresh:
                last_good = (end, r2)
                end += 1
            else:
                break
        if last_good is not None:
            end_i, r2 = last_good
            dec = logi[end_i - 1] - logi[start]
            if dec > best_dec:
                best_dec, best_r2 = float(dec), float(r2)
    return best_dec, best_r2


def compliance_clip_frac(i, compliance=None):
    """Fraction of samples sitting within 0.5% of the max |I| — a proxy for
    compliance clipping (many identical/near-identical extreme readings)."""
    i = np.asarray(i, float)
    if len(i) == 0:
        return np.nan
    peak = np.nanmax(np.abs(i))
    if peak <= 0:
        return np.nan
    return float(np.mean(np.abs(np.abs(i) - peak) < 0.005 * peak))


def analyze_diode(path, kind):
    df = load_b1500a_csv(path)
    vcol = "Vb"
    # current: BE -> Ie or Ib?; BC -> Ic or Ib? use whichever varying SMU
    # matches the diode current column (largest-magnitude, non-Vb-paired).
    icols = [c for c in df.columns if c.startswith("I")]
    # pick current column with the largest dynamic range in the forward
    # direction (V>0) — the swept diode current.
    best_col, best_range = None, -1
    for c in icols:
        i = df[c].values
        pos = (df[vcol].values > 0) & (i > 0)
        if pos.sum() < 5:
            continue
        rng = np.log10(np.nanmax(i[pos]) / np.nanmin(i[pos])) if np.nanmin(i[pos]) > 0 else -1
        if rng > best_range:
            best_range, best_col = rng, c
    if best_col is None:
        return None
    v = df[vcol].values
    i = df[best_col].values
    decades, mono_frac, ifloor = decades_of_clean_gummel(v, i, None)
    clip = compliance_clip_frac(i)
    r2, dec_win, npts = forward_loglinear_r2(v, i)
    best_dec, best_r2 = best_ideal_window_decades(v, i)
    return dict(
        file=os.path.basename(path), kind=kind, icol=best_col,
        n=len(df), vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)),
        decades=decades, mono_frac=mono_frac, ifloor=ifloor, clip_frac=clip,
        fwd_r2=r2, fwd_decades=dec_win,
        ideal_win_dec=best_dec, ideal_win_r2=best_r2,
    )


def analyze_gummel(path):
    df = load_b1500a_csv(path)
    if not {"Vb", "Ic", "Ib"}.issubset(df.columns):
        return None
    vb, ic, ib = df["Vb"].values, df["Ic"].values, df["Ib"].values
    decades, mono_frac, ifloor = decades_of_clean_gummel(vb, ic, ib)
    clip = compliance_clip_frac(ic)
    beta = np.divide(ic, ib, out=np.full_like(ic, np.nan), where=ib != 0)
    beta_pos = beta[(ib > 1e-9) & np.isfinite(beta)]
    peak_beta = float(np.nanmax(beta_pos)) if len(beta_pos) else np.nan
    r2_ic, dec_win_ic, _ = forward_loglinear_r2(vb, ic)
    r2_ib, dec_win_ib, _ = forward_loglinear_r2(vb, ib)
    ic_ideal_dec, ic_ideal_r2 = best_ideal_window_decades(vb, ic)
    return dict(
        file=os.path.basename(path), n=len(df),
        vb_min=float(np.nanmin(vb)), vb_max=float(np.nanmax(vb)),
        decades=decades, mono_frac=mono_frac, ic_floor=ifloor,
        clip_frac=clip, peak_beta=peak_beta,
        ic_fwd_r2=r2_ic, ic_fwd_dec=dec_win_ic,
        ib_fwd_r2=r2_ib, ib_fwd_dec=dec_win_ib,
        ic_ideal_dec=ic_ideal_dec, ic_ideal_r2=ic_ideal_r2,
    )


def analyze_family(path):
    df = load_b1500a_csv(path)
    if not {"Vc", "Ic", "Ib"}.issubset(df.columns):
        return None
    vc, ic, ib = df["Vc"].values, df["Ic"].values, df["Ib"].values
    n_ib = df["Ib"].nunique()
    clip = compliance_clip_frac(ic)
    # "flatness" of the top curve's saturation region as a clean-family proxy
    order = np.argsort(vc)
    vc_s, ic_s = vc[order], ic[order]
    top_ib = df["Ib"].max()
    sub = df[df["Ib"] == top_ib].sort_values("Vc")
    sat_region = sub[sub["Vc"] > 0.7 * sub["Vc"].max()]
    sat_noise = float(sat_region["Ic"].std() / sat_region["Ic"].mean()) if len(sat_region) > 2 and sat_region["Ic"].mean() != 0 else np.nan
    return dict(
        file=os.path.basename(path), n=len(df),
        vc_min=float(np.nanmin(vc)), vc_max=float(np.nanmax(vc)),
        n_ib_steps=int(n_ib), clip_frac=clip, sat_noise=sat_noise,
        ic_max=float(np.nanmax(ic)),
    )


def main():
    results = {"BC diode": [], "BE diode": [], "Gummel": [], "Family": []}
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.csv"))):
        name = os.path.basename(path)
        try:
            if name.startswith("BC diode"):
                r = analyze_diode(path, "BC diode")
                if r: results["BC diode"].append(r)
            elif name.startswith("BE diode"):
                r = analyze_diode(path, "BE diode")
                if r: results["BE diode"].append(r)
            elif name.startswith("Gummel"):
                r = analyze_gummel(path)
                if r: results["Gummel"].append(r)
            elif name.startswith("Family"):
                r = analyze_family(path)
                if r: results["Family"].append(r)
        except Exception as e:
            print(f"SKIP {name}: {e}")

    for kind, rows in results.items():
        print(f"\n===== {kind} ({len(rows)} files) =====")
        d = pd.DataFrame(rows)
        if d.empty:
            print("no data")
            continue
        pd.set_option("display.width", 200)
        pd.set_option("display.max_rows", 100)
        print(d.sort_values(d.columns[1]).to_string(index=False))


if __name__ == "__main__":
    main()
