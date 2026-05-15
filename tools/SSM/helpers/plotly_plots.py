"""
helpers/plotly_plots.py — Plotly Smith / Bode / Plateau builders.

Originals were inline in IOED_HBT_RF_extract.py:
  PALETTE, _darken, _layout, make_smith, make_bode, make_plateau

The leading underscores have been removed for the public-facing helpers
(_darken → darken, _layout → bode_layout); the rest of the API is unchanged.

NOTE on Step 6 of the migration plan:
  RF_simulator.py currently has its own simpler `_build_smith` / `_build_bode`
  taking a `mults` dict and `freq_hz` rather than a DataFrame. After this
  additive phase, `make_smith` / `make_bode` will be widened (extra optional
  kwargs) to absorb both call sites and `_build_smith`/`_build_bode` will be
  deleted. For now they remain in RF_simulator.py untouched.
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

from .rf_math import extended_smith_grid
from .metrics import extrap_20dbdec, single_pole_extrap


# ── Constants ────────────────────────────────────────────────────────────────

PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]


# ── Small helpers ────────────────────────────────────────────────────────────

def darken(c):
    """Darken a `#rrggbb` color by 45 units per channel.  Returns input on parse failure."""
    try:
        h = c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{max(0,r-45):02x}{max(0,g-45):02x}{max(0,b-45):02x}"
    except Exception:
        return c


def bode_layout(title, ytitle, yr, xr):
    """Plotly layout dict for Bode / plateau plots (log-x frequency axis)."""
    return dict(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(max(xr[0], 1e-4)), np.log10(xr[1])],
                   showgrid=True, gridcolor="#ebebeb", minor_showgrid=True),
        yaxis=dict(title=ytitle, range=list(yr), showgrid=True, gridcolor="#ebebeb"),
        legend=dict(x=1.0, y=1.0, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.88)", bordercolor="#ccc", borderwidth=1),
        plot_bgcolor="white", paper_bgcolor="white", height=500,
        margin=dict(l=55, r=25, t=45, b=50), hovermode="x unified", template="plotly_white")


# ── Smith chart ──────────────────────────────────────────────────────────────

def make_smith(S, f_array, f_min, f_max, toggles, scales, title, max_r=1.0):
    """Plotly Smith chart with up to 4 selectable S-parameter traces.

    Parameters
    ----------
    S        : (N, 2, 2) complex array of S-parameters.
    f_array  : (N,) frequency array in GHz.
    f_min,
    f_max    : float — display only points within [f_min, f_max] GHz.
    toggles  : dict {"S11": bool, "S12": bool, "S21": bool, "S22": bool}
    scales   : dict {"S11": float, ...} — per-trace multiplier (1.0 = identity).
    title    : str — appended after "Smith Chart — ".
    max_r    : float — Smith chart radius (1.0 = standard).
    """
    mask = (f_array >= f_min) & (f_array <= f_max)
    S_p = S[mask].copy(); f_p = f_array[mask]
    fig = go.Figure()
    for tr in extended_smith_grid(max_r):
        fig.add_trace(tr)
    for key, (r, c), color, dash in [
        ("S11", (0, 0), "#1f77b4", "solid"),
        ("S22", (1, 1), "#ff7f0e", "dash"),
        ("S21", (1, 0), "#2ca02c", "dot"),
        ("S12", (0, 1), "#d62728", "dashdot"),
    ]:
        if not toggles.get(key, False):
            continue
        sv = S_p[:, r, c] * scales.get(key, 1.0)
        sv[np.abs(sv) > max_r] = np.nan + 1j*np.nan
        sc = scales.get(key, 1.0)
        lbl = (f"{key} ({f_min:.2g}–{f_max:.2g} GHz)" if sc == 1
               else f"{key} ×{sc:g}")
        hov = [f"f={fv:.3f} GHz<br>Re={rv:.4f}<br>Im={iv:.4f}"
               for fv, rv, iv in zip(f_p, sv.real, sv.imag)]
        fig.add_trace(go.Scattergl(x=sv.real, y=sv.imag, mode="lines",
                                 line=dict(color=color, width=2.2, dash=dash),
                                 name=lbl, text=hov, hoverinfo="text"))
    lim = max_r * 1.05
    fig.update_layout(
        title=dict(text=f"Smith Chart — {title}", font=dict(size=13)),
        xaxis=dict(title="Re(Γ)", range=[-lim, lim], showgrid=False, zeroline=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(title="Im(Γ)", range=[-lim, lim], showgrid=False, zeroline=False),
        plot_bgcolor="white", paper_bgcolor="white", height=540,
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(x=1.02, y=1.0, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc", borderwidth=1),
        hovermode="closest")
    return fig


# ── Bode plot ────────────────────────────────────────────────────────────────

def make_bode(df, title, xr, yr, sh21, su, smag, color, *,
              show_20db: bool = True,
              show_sp: bool = False,
              sp_window_idx=None,
              extrap_f_max=None,
              return_extrap_df: bool = False):
    """
    Individual Bode plot.

    Measured traces use markers (○ for |h21|², □ for Mason U, ◇ for MAG/MSG).
    Two extrapolation overlays are available for |h21|² and Mason U:

      * ``show_20db`` (default ``True``) — slope-locked −20 dB/dec line anchored
        at the last measured point.  Drawn dotted.  Auto-skips when the trace
        already crosses 0 dB inside the measured band.

      * ``show_sp`` (default ``False``) — log-linear (single-pole) fit on a
        user-chosen index window ``sp_window_idx=(idx_lo, idx_hi)``.  The fitted
        slope is free, so this answers "what fT/fmax does the data actually
        imply?" while the −20 dB/dec line gives the textbook value.  Drawn dashed.

    ``extrap_f_max`` (GHz) optionally extends both extrapolation curves past
    their 0-dB crossings so the chart shows where the curves would continue.

    When ``return_extrap_df`` is true, returns ``(fig, extrap_df)`` where
    ``extrap_df`` has columns ``Freq (GHz)`` plus one column per enabled
    (method, gain) pair.  Measured values are used below the highest measured
    frequency; extrapolated values are used above it.  Empty if no extrap is
    enabled or possible.
    """
    fig = go.Figure(); f = df["Freq (GHz)"].values
    hov = "Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"

    f_high_track = float(f[-1]) if len(f) else float(xr[1])
    extrap_used  = False
    extrap_curves: dict = {}  # key -> (f_ext, g_ext, f_zero)

    def _add_20db(y_vals, color_, kind, key):
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = extrap_20dbdec(f, y_vals, f_max_target=extrap_f_max)
        if f_ext is None:
            return
        extrap_used  = True
        f_high_track = max(f_high_track, float(f_ext[-1]))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines",
            name=f"{kind} −20dB/dec (≈{f0:.1f} GHz)",
            line=dict(color=color_, width=1.6, dash="dot"),
            hovertemplate=hov, showlegend=True))
        extrap_curves[key] = (f_ext, g_ext, f0)

    def _add_sp(y_vals, color_, kind, key):
        if sp_window_idx is None:
            return
        nonlocal f_high_track, extrap_used
        idx_lo, idx_hi = sp_window_idx
        f_ext, g_ext, f0, slope, _intc = single_pole_extrap(
            f, y_vals, idx_lo, idx_hi, f_max_target=extrap_f_max)
        if f_ext is None:
            return
        extrap_used  = True
        f_high_track = max(f_high_track, float(f_ext[-1]))
        fig.add_trace(go.Scattergl(
            x=f_ext, y=g_ext, mode="lines",
            name=f"{kind} single-pole ({slope:.1f} dB/dec, ≈{f0:.1f} GHz)",
            line=dict(color=color_, width=1.6, dash="dash"),
            hovertemplate=hov, showlegend=True))
        extrap_curves[key] = (f_ext, g_ext, f0)

    if sh21:
        y = df["|h21|² (dB)"].values
        fig.add_trace(go.Scattergl(
            x=f, y=y, name="|h21|²", mode="lines+markers",
            line=dict(color=color, width=1.4),
            marker=dict(symbol="circle", size=6, color=color),
            hovertemplate=hov))
        if show_20db: _add_20db(y, color, "fT",      "h21_20db")
        if show_sp:   _add_sp  (y, color, "fT",      "h21_sp")
    if su:
        y = df["Mason U (dB)"].values
        col_u = darken(color)
        fig.add_trace(go.Scattergl(
            x=f, y=y, name="Mason U", mode="lines+markers",
            line=dict(color=col_u, width=1.4),
            marker=dict(symbol="square", size=6, color=col_u),
            hovertemplate=hov))
        if show_20db: _add_20db(y, col_u, "fmax(U)", "U_20db")
        if show_sp:   _add_sp  (y, col_u, "fmax(U)", "U_sp")
    if smag:
        y = df["MAG/MSG (dB)"].values
        fig.add_trace(go.Scattergl(
            x=f, y=y, name="MAG/MSG", mode="lines+markers",
            line=dict(color="#2ca02c", width=1.4),
            marker=dict(symbol="diamond", size=6, color="#2ca02c"),
            hovertemplate=hov))

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    # Auto-extend x-range if extrapolation pushes past xr[1]
    xr_eff = (xr[0], max(float(xr[1]), float(f_high_track) * 1.05)) if extrap_used else xr
    fig.update_layout(**bode_layout(f"Bode — {title}", "Gain (dB)", yr, xr_eff))

    if return_extrap_df:
        return fig, _build_bode_extrap_df(df, extrap_curves, sh21, su)
    return fig


def _build_bode_extrap_df(df, extrap_curves: dict, sh21: bool, su: bool):
    """Combine measured + extrapolated values into one DataFrame.

    ``extrap_curves`` maps keys ``"h21_20db"`` / ``"h21_sp"`` / ``"U_20db"`` /
    ``"U_sp"`` to ``(f_ext, g_ext, f_zero)`` tuples.  Returns a DataFrame
    keyed on a unified frequency axis (measured ∪ all extrap freqs).  Below
    the highest measured frequency, columns carry the measured values; above
    it, the corresponding extrapolated curve.
    """
    import pandas as _pd

    if not extrap_curves:
        return _pd.DataFrame()

    f_meas = df["Freq (GHz)"].values
    if len(f_meas) == 0:
        return _pd.DataFrame()
    h21_meas = df["|h21|² (dB)"].values if sh21 else None
    U_meas   = df["Mason U (dB)"].values  if su   else None

    # Unified, sorted, deduplicated freq axis spanning measured + extrap.
    f_parts = [f_meas] + [c[0] for c in extrap_curves.values()]
    f_union = np.unique(np.concatenate(f_parts))
    f_union = f_union[f_union > 0]
    f_high  = float(f_meas[-1])

    out = {"Freq (GHz)": f_union}

    def _stitch(meas_vals, ext_key, label):
        if ext_key not in extrap_curves:
            return
        f_ext, g_ext, _ = extrap_curves[ext_key]
        meas_part = np.interp(f_union, f_meas, meas_vals,
                              left=np.nan, right=np.nan)
        ext_part  = np.interp(f_union, f_ext,  g_ext,
                              left=np.nan, right=np.nan)
        out[label] = np.where(f_union <= f_high, meas_part, ext_part)

    if sh21 and h21_meas is not None:
        _stitch(h21_meas, "h21_20db", "|h21|² −20 dB/dec (dB)")
        _stitch(h21_meas, "h21_sp",   "|h21|² Single-pole (dB)")
    if su and U_meas is not None:
        _stitch(U_meas,   "U_20db",   "Mason U −20 dB/dec (dB)")
        _stitch(U_meas,   "U_sp",     "Mason U Single-pole (dB)")

    return _pd.DataFrame(out)


# ── Plateau plot ─────────────────────────────────────────────────────────────

def make_plateau(df, res, title, xr, sh21, su, smag, color):
    """Plot fT/fmax plateau (gain × bandwidth product) vs frequency.

    `res` is currently unused — kept in the signature for API parity with
    the original IOED definition (it was forwarded but not read).
    """
    cols = []
    if sh21: cols += df["fT Plateau (GHz)"].tolist()
    if su:   cols += df["fmax U Plateau (GHz)"].tolist()
    arr = np.array([v for v in cols if np.isfinite(v) and v > 0])
    ym  = float(np.quantile(arr, 0.97)) * 1.3 if len(arr) else 100
    hov = "Freq:%{x:.4f}GHz<br>GBP:%{y:.4f}GHz<extra></extra>"
    fig = go.Figure()
    if sh21:
        fig.add_trace(go.Scattergl(x=df["Freq (GHz)"], y=df["fT Plateau (GHz)"],
                                 name="fT", line=dict(color=color, width=2.5),
                                 hovertemplate=hov))
    if su:
        fig.add_trace(go.Scattergl(x=df["Freq (GHz)"], y=df["fmax U Plateau (GHz)"],
                                 name="fmax(U)",
                                 line=dict(color=darken(color), width=2.5, dash="dash"),
                                 hovertemplate=hov))
    if smag:
        fig.add_trace(go.Scattergl(x=df["Freq (GHz)"], y=df["fmax MAG Plateau (GHz)"],
                                 name="fmax(MAG)",
                                 line=dict(color="#2ca02c", width=2, dash="dot"),
                                 hovertemplate=hov))
    fig.update_layout(**bode_layout(f"Plateau — {title}", "GBP (GHz)", [0, ym], xr))
    return fig
