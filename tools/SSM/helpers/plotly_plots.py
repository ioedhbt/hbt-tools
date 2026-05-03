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
from .metrics import extrap_20dbdec


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
        fig.add_trace(go.Scatter(x=sv.real, y=sv.imag, mode="lines",
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

def make_bode(df, title, xr, yr, sh21, su, smag, color):
    """
    Individual Bode plot.

    Measured traces use markers (○ for |h21|², □ for Mason U, ◇ for MAG/MSG).
    If |h21|² and/or Mason U are still above 0 dB at the highest measured
    frequency, a 20 dB/dec extrapolation (dotted) is appended and the x-axis
    is auto-extended past the projected fT/fmax crossing.
    """
    fig = go.Figure(); f = df["Freq (GHz)"].values
    hov = "Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"

    f_high_track = float(f[-1]) if len(f) else float(xr[1])
    extrap_used  = False

    def _add_extrap(y_vals, color_, kind):
        """Append a dotted 20 dB/dec extrapolation, return new f_high if any."""
        nonlocal f_high_track, extrap_used
        f_ext, g_ext, f0 = extrap_20dbdec(f, y_vals)
        if f_ext is None:
            return
        extrap_used  = True
        f_high_track = max(f_high_track, f0)
        fig.add_trace(go.Scatter(
            x=f_ext, y=g_ext, mode="lines",
            name=f"{kind} extrap (≈{f0:.1f} GHz)",
            line=dict(color=color_, width=1.6, dash="dot"),
            hovertemplate=hov, showlegend=False))

    if sh21:
        y = df["|h21|² (dB)"].values
        fig.add_trace(go.Scatter(
            x=f, y=y, name="|h21|²", mode="lines+markers",
            line=dict(color=color, width=1.4),
            marker=dict(symbol="circle", size=6, color=color),
            hovertemplate=hov))
        _add_extrap(y, color, "fT")
    if su:
        y = df["Mason U (dB)"].values
        col_u = darken(color)
        fig.add_trace(go.Scatter(
            x=f, y=y, name="Mason U", mode="lines+markers",
            line=dict(color=col_u, width=1.4),
            marker=dict(symbol="square", size=6, color=col_u),
            hovertemplate=hov))
        _add_extrap(y, col_u, "fmax(U)")
    if smag:
        y = df["MAG/MSG (dB)"].values
        fig.add_trace(go.Scatter(
            x=f, y=y, name="MAG/MSG", mode="lines+markers",
            line=dict(color="#2ca02c", width=1.4),
            marker=dict(symbol="diamond", size=6, color="#2ca02c"),
            hovertemplate=hov))

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    # Auto-extend x-range if extrapolation pushes past xr[1]
    xr_eff = (xr[0], max(float(xr[1]), float(f_high_track) * 1.25)) if extrap_used else xr
    fig.update_layout(**bode_layout(f"Bode — {title}", "Gain (dB)", yr, xr_eff))
    return fig


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
        fig.add_trace(go.Scatter(x=df["Freq (GHz)"], y=df["fT Plateau (GHz)"],
                                 name="fT", line=dict(color=color, width=2.5),
                                 hovertemplate=hov))
    if su:
        fig.add_trace(go.Scatter(x=df["Freq (GHz)"], y=df["fmax U Plateau (GHz)"],
                                 name="fmax(U)",
                                 line=dict(color=darken(color), width=2.5, dash="dash"),
                                 hovertemplate=hov))
    if smag:
        fig.add_trace(go.Scatter(x=df["Freq (GHz)"], y=df["fmax MAG Plateau (GHz)"],
                                 name="fmax(MAG)",
                                 line=dict(color="#2ca02c", width=2, dash="dot"),
                                 hovertemplate=hov))
    fig.update_layout(**bode_layout(f"Plateau — {title}", "GBP (GHz)", [0, ym], xr))
    return fig
