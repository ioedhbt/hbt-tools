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
from .metrics import extrap_20dbdec, single_pole_extrap, compute_h21_U


# ── Constants ────────────────────────────────────────────────────────────────

PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
           "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

# Standardised marker symbols for fT / fmax traces.  Every Bode / plateau
# plot in the app (IOED Overlay + Individual, SSM raw-vs-deembedded,
# SSM intrinsic, measured-vs-modeled, RF simulator) uses this same
# mapping so users can identify a metric by shape alone, independent of
# colour.  Modeled traces are intentionally rendered without markers and
# rely on line dash patterns instead.
#   |h21|²  → fT       → circle
#   Mason U → fmax(U)  → square
#   MAG/MSG → fmax(MAG)→ diamond
FT_FMAX_SYMBOLS = {
    "h21": "circle",
    "U":   "square",
    "MAG": "diamond",
}

# Standardised colours for fT / fmax traces.  Used by every Bode plot in
# the app (SSM measured-vs-modeled, RF extraction Overlay / Individual,
# RF simulator) so users can identify a metric by colour alone.
#
#   |h21|²  → fT       → blue   (#1f77b4)
#   Mason U → fmax(U)  → red    (#d62728)
#   MAG/MSG → fmax(MAG)→ red    (same family as Mason U; distinguished
#                                 from U by dash style and marker shape)
#
# Convention for line style within a colour:
#   measured            : solid + markers
#   modeled             : dashed (no markers)
#   extrapolated (20dB
#     / single-pole)    : dotted (no markers)
FT_FMAX_COLORS = {
    "fT":   "#1f77b4",
    "fmax": "#d62728",
}


def thinned_indices(n: int, max_markers: int = 25) -> np.ndarray:
    """Log-spaced sample of indices into a length-`n` array.

    Used by overlay plots to keep marker density manageable while the line
    trace stays at full resolution.  Always includes the first and last
    index so the marker sequence anchors at both endpoints of the line.

    Returning the full range when ``n <= max_markers`` keeps single-trace
    plots (Individual tab) visually identical to dense-marker traces.
    """
    if n <= max_markers:
        return np.arange(n)
    raw = np.geomspace(1, n, max_markers)
    return np.unique(np.round(raw).astype(int).clip(1, n) - 1)


def add_overlay_trace_with_markers(fig, x, y, *, name, color, symbol,
                                     dash=None, line_width=2.5, marker_size=6,
                                     opacity=1.0, hovertemplate=None,
                                     legendgroup=None, max_markers=25,
                                     show_legend=True):
    """Add a Bode trace as (full-resolution line) + (thinned marker overlay).

    Splitting the trace into two halves keeps the line crisp while cutting
    marker WebGL primitives from ~1000 per trace down to ~25.  With 30 files
    × 3 metrics that's a ~40× reduction in marker rendering load, which is
    what actually causes the scroll-sluggishness when many files are loaded.

    The marker trace owns the legend entry (so the legend shows the symbol
    + colour); the line trace is hidden from the legend and grouped with
    the marker via `legendgroup` so toggling visibility from the legend
    affects both.
    """
    from plotly import graph_objects as _go
    x = np.asarray(x)
    y = np.asarray(y)
    lg = legendgroup or name
    fig.add_trace(_go.Scattergl(
        x=x, y=y, mode="lines", name=name,
        line=dict(color=color, width=line_width, dash=dash) if dash
             else dict(color=color, width=line_width),
        opacity=opacity, hovertemplate=hovertemplate,
        legendgroup=lg, showlegend=False))
    idx = thinned_indices(len(x), max_markers)
    fig.add_trace(_go.Scattergl(
        x=x[idx], y=y[idx], mode="markers", name=name,
        marker=dict(symbol=symbol, size=marker_size, color=color),
        opacity=opacity, hovertemplate=hovertemplate,
        legendgroup=lg, showlegend=show_legend))


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
    """Plotly layout dict for Bode / plateau plots (log-x frequency axis).

    Legend placement mirrors the Raw-vs-De-embedded comparison plot in
    :mod:`tools.SSM.ssm_plots._compare_bode_smith`: horizontal, centered
    below the plot area, with a soft white background.  This keeps the
    trace names from squeezing the chart when many DUTs / S-parameters
    are overlaid in the Overlay or Individual tabs.
    """
    return dict(
        title=dict(text=title, font=dict(size=13)),
        xaxis=dict(title="Frequency (GHz)", type="log",
                   range=[np.log10(max(xr[0], 1e-4)), np.log10(xr[1])],
                   showgrid=True, gridcolor="#ebebeb", minor_showgrid=True),
        yaxis=dict(title=ytitle, range=list(yr), showgrid=True, gridcolor="#ebebeb"),
        legend=dict(orientation="h", x=0.5, y=-0.22,
                    xanchor="center", yanchor="top",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor="#ccc",
                    borderwidth=1, font=dict(size=18)),
        plot_bgcolor="white", paper_bgcolor="white", height=560,
        margin=dict(l=55, r=25, t=45, b=160),
        hovermode="x unified", template="plotly_white")


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
              return_extrap_df: bool = False,
              return_excel_bytes: bool = False):
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

    When ``return_excel_bytes`` is true, returns ``(fig, xlsx_bytes)`` using the
    standardised fT/fmax export (:func:`bode_excel_bytes`): a simulated block
    (freq + each gain trace) plus, when extrapolation is in play, a
    side-by-side extrapolated block.  ``None`` when there is nothing to export.
    """
    fig = go.Figure(); f = df["Freq (GHz)"].values
    hov = "Freq:%{x:.4f}GHz<br>Gain:%{y:.4f}dB<extra></extra>"

    f_high_track = float(f[-1]) if len(f) else float(xr[1])
    extrap_used  = False
    extrap_curves: dict = {}  # key -> (f_ext, g_ext, f_zero)
    sim_traces: list = []     # (label, y_db) for the standardised xlsx export

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

    # Standardised colours: fT-producing trace (|h21|²) in blue, every
    # fmax-producing trace (Mason U, MAG/MSG) in red.  Marker shape
    # (circle / square / diamond) still distinguishes within a colour.
    # The `color` argument is preserved for callers that want per-file
    # colouring but is overridden here; pass through `color=PALETTE[...]`
    # at call sites unchanged.
    _c_fT, _c_fmax = FT_FMAX_COLORS["fT"], FT_FMAX_COLORS["fmax"]
    if sh21:
        y = df["|h21|² (dB)"].values
        fig.add_trace(go.Scattergl(
            x=f, y=y, name="|h21|²", mode="lines+markers",
            line=dict(color=_c_fT, width=1.4),
            marker=dict(symbol=FT_FMAX_SYMBOLS["h21"], size=6, color=_c_fT),
            hovertemplate=hov))
        sim_traces.append(("|h21|² (dB)", y))
        if show_20db: _add_20db(y, _c_fT, "fT",      "h21_20db")
        if show_sp:   _add_sp  (y, _c_fT, "fT",      "h21_sp")
    if su:
        y = df["Mason U (dB)"].values
        fig.add_trace(go.Scattergl(
            x=f, y=y, name="Mason U", mode="lines+markers",
            line=dict(color=_c_fmax, width=1.4),
            marker=dict(symbol=FT_FMAX_SYMBOLS["U"], size=6, color=_c_fmax),
            hovertemplate=hov))
        sim_traces.append(("Mason U (dB)", y))
        if show_20db: _add_20db(y, _c_fmax, "fmax(U)", "U_20db")
        if show_sp:   _add_sp  (y, _c_fmax, "fmax(U)", "U_sp")
    if smag:
        y = df["MAG/MSG (dB)"].values
        fig.add_trace(go.Scattergl(
            x=f, y=y, name="MAG/MSG", mode="lines+markers",
            line=dict(color=_c_fmax, width=1.4),
            marker=dict(symbol=FT_FMAX_SYMBOLS["MAG"], size=6, color=_c_fmax),
            hovertemplate=hov))
        sim_traces.append(("MAG/MSG (dB)", y))

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    # Auto-extend x-range if extrapolation pushes past xr[1]
    xr_eff = (xr[0], max(float(xr[1]), float(f_high_track) * 1.05)) if extrap_used else xr
    fig.update_layout(**bode_layout(f"Bode — {title}", "Gain (dB)", yr, xr_eff))

    if return_excel_bytes:
        from .chart_export import bode_excel_bytes
        # Map the internal extrap-curve keys to stable, human-readable column
        # labels for the side-by-side extrapolated block.
        _extrap_labels = {
            "h21_20db": "|h21|² −20 dB/dec (dB)",
            "h21_sp":   "|h21|² Single-pole (dB)",
            "U_20db":   "Mason U −20 dB/dec (dB)",
            "U_sp":     "Mason U Single-pole (dB)",
        }
        extrap_traces = [
            (_extrap_labels.get(k, k), c[0], c[1])
            for k, c in extrap_curves.items()]
        return fig, bode_excel_bytes(f, sim_traces, extrap_traces)

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
    # Standardised colours (see FT_FMAX_COLORS) — fT in blue, fmax in red.
    # Within fmax, U and MAG use the same red but different marker shape
    # + dash pattern for visual differentiation.
    _c_fT, _c_fmax = FT_FMAX_COLORS["fT"], FT_FMAX_COLORS["fmax"]
    fig = go.Figure()
    if sh21:
        fig.add_trace(go.Scattergl(x=df["Freq (GHz)"], y=df["fT Plateau (GHz)"],
                                 name="fT", mode="lines+markers",
                                 line=dict(color=_c_fT, width=2.5),
                                 marker=dict(symbol=FT_FMAX_SYMBOLS["h21"], size=5, color=_c_fT),
                                 hovertemplate=hov))
    if su:
        fig.add_trace(go.Scattergl(x=df["Freq (GHz)"], y=df["fmax U Plateau (GHz)"],
                                 name="fmax(U)", mode="lines+markers",
                                 line=dict(color=_c_fmax, width=2.5, dash="dash"),
                                 marker=dict(symbol=FT_FMAX_SYMBOLS["U"], size=5, color=_c_fmax),
                                 hovertemplate=hov))
    if smag:
        fig.add_trace(go.Scattergl(x=df["Freq (GHz)"], y=df["fmax MAG Plateau (GHz)"],
                                 name="fmax(MAG)", mode="lines+markers",
                                 line=dict(color=_c_fmax, width=2, dash="dot"),
                                 marker=dict(symbol=FT_FMAX_SYMBOLS["MAG"], size=5, color=_c_fmax),
                                 hovertemplate=hov))
    fig.update_layout(**bode_layout(f"Plateau — {title}", "GBP (GHz)", [0, ym], xr))
    return fig


# ── Smith + Bode subplot with a parameter-sweep slider (Plotly frames) ───────

_SMITH_SLIDER_COLORS = {"S11": "#1f77b4", "S22": "#ff7f0e",
                        "S21": "#2ca02c", "S12": "#d62728"}


def make_smith_bode_slider_fig(*, S_batch, freq, model_name: str,
                                slider_specs=None,
                                sweep_disp=None, sweep_label: str = "",
                                sweep_unit: str = "", sweep_fmt: str = "%.4g",
                                S_meas=None, height: int = 560) -> go.Figure:
    """Smith chart + fT/fmax bode side-by-side with one or more Plotly sliders.

    Multi-slider semantics
    ----------------------
    Each entry of ``slider_specs`` is an independent slider that scans its
    own axis with the other parameters held at their current (model-default)
    value.  Frame indices are passed explicitly per slider — there is no
    joint Cartesian sweep (use the live mode for that).

    Parameters
    ----------
    S_batch       : ndarray (B, N_freq, 2, 2) complex.  Flat concatenation
                    of every frame across all sliders.
    freq          : ndarray (N_freq,) in Hz.
    model_name    : str — appears in subplot titles.
    slider_specs  : list[dict] | None.  Each dict:
                      {"label": str, "unit": str, "fmt": str,
                       "values_disp": list, "frame_indices": list[int]}
                    ``frame_indices[i]`` selects which row of ``S_batch``
                    becomes active when the user picks ``values_disp[i]``.
                    If None, fall back to the legacy single-slider kwargs
                    ``sweep_disp / sweep_label / sweep_unit / sweep_fmt`` —
                    in which case every frame in ``S_batch`` is part of one
                    slider.
    S_meas        : optional ndarray (N_freq, 2, 2) measured S — adds static
                    overlay traces for comparison.
    height        : per-subplot pixel height (Smith and Bode share this).
    """
    from plotly.subplots import make_subplots

    S_batch = np.asarray(S_batch)
    if S_batch.ndim != 4 or S_batch.shape[-2:] != (2, 2):
        raise ValueError(f"S_batch must be (B, N, 2, 2); got {S_batch.shape}.")
    B = S_batch.shape[0]
    f_ghz = np.asarray(freq) * 1e-9

    # Legacy single-slider mode — build a slider_specs list automatically.
    if slider_specs is None:
        if sweep_disp is None:
            raise ValueError("Pass slider_specs or sweep_disp.")
        sweep_disp = list(sweep_disp)
        if len(sweep_disp) != B:
            raise ValueError("len(sweep_disp) must match S_batch.shape[0].")
        slider_specs = [dict(label=sweep_label, unit=sweep_unit, fmt=sweep_fmt,
                              values_disp=sweep_disp,
                              frame_indices=list(range(B)))]

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=(f"Smith — {model_name}", f"fT / fmax — {model_name}"),
        column_widths=[1.05, 1.0],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    # ── (1) Static Smith grid (row 1) ─────────────────────────────────────
    for tr in extended_smith_grid(1.0):
        fig.add_trace(tr, row=1, col=1)

    # ── (2) Measured Smith — 4 traces if S_meas provided ─────────────────
    meas_smith_indices: list[int] = []
    if S_meas is not None:
        S_meas = np.asarray(S_meas)
        for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                              ("S21", (1, 0)), ("S12", (0, 1))]:
            col = _SMITH_SLIDER_COLORS[name]
            s = S_meas[:, r, c]
            fig.add_trace(go.Scatter(x=s.real, y=s.imag, mode="markers",
                                      name=f"{name} meas",
                                      marker=dict(color=col, size=5)),
                          row=1, col=1)
            meas_smith_indices.append(len(fig.data) - 1)

    # ── (3) Model Smith — 4 traces (initial frame) ───────────────────────
    model_smith_indices: list[int] = []
    S0 = S_batch[0]
    for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                          ("S21", (1, 0)), ("S12", (0, 1))]:
        col = _SMITH_SLIDER_COLORS[name]
        s = S0[:, r, c]
        fig.add_trace(go.Scatter(x=s.real, y=s.imag, mode="lines",
                                  name=f"{name} model",
                                  line=dict(color=col, width=2, dash="dash")),
                      row=1, col=1)
        model_smith_indices.append(len(fig.data) - 1)

    # ── (4) Bode: measured (optional) ────────────────────────────────────
    bode_meas_indices: list[int] = []
    if S_meas is not None:
        h21_m_db, U_m_db = compute_h21_U(S_meas)
        fig.add_trace(go.Scatter(x=f_ghz, y=h21_m_db,
                                  mode="lines+markers",
                                  name="|h21|² meas",
                                  line=dict(color="#1f77b4", width=1.4),
                                  marker=dict(size=4, symbol="circle")),
                      row=1, col=2)
        bode_meas_indices.append(len(fig.data) - 1)
        fig.add_trace(go.Scatter(x=f_ghz, y=U_m_db,
                                  mode="lines+markers",
                                  name="Mason U meas",
                                  line=dict(color="#1f77b4", width=1.4, dash="dot"),
                                  marker=dict(size=4, symbol="square")),
                      row=1, col=2)
        bode_meas_indices.append(len(fig.data) - 1)

    # ── (5) Bode: model (initial frame) ──────────────────────────────────
    h21_s0_db, U_s0_db = compute_h21_U(S0)
    fig.add_trace(go.Scatter(x=f_ghz, y=h21_s0_db, mode="lines",
                              name="|h21|² model",
                              line=dict(color="#d62728", width=2, dash="dash")),
                  row=1, col=2)
    bode_h21_idx = len(fig.data) - 1
    fig.add_trace(go.Scatter(x=f_ghz, y=U_s0_db, mode="lines",
                              name="Mason U model",
                              line=dict(color="#d62728", width=2, dash="longdash")),
                  row=1, col=2)
    bode_U_idx = len(fig.data) - 1

    # ── Frames: update only model traces (Smith + Bode) ──────────────────
    frame_trace_indices = list(model_smith_indices) + [bode_h21_idx, bode_U_idx]
    frames = []
    for i in range(B):
        Si = S_batch[i]
        h21_db, U_db = compute_h21_U(Si)
        frame_traces = []
        # Smith model — 4 traces
        for _, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                           ("S21", (1, 0)), ("S12", (0, 1))]:
            s = Si[:, r, c]
            frame_traces.append(go.Scatter(x=s.real, y=s.imag))
        # Bode |h21|² + Mason U
        frame_traces.append(go.Scatter(x=f_ghz, y=h21_db))
        frame_traces.append(go.Scatter(x=f_ghz, y=U_db))
        frames.append(go.Frame(name=str(i), data=frame_traces,
                                traces=frame_trace_indices))
    fig.frames = frames

    # ── Build one slider per slider_spec; stack vertically below figure ──
    sliders = []
    n_sliders = len(slider_specs)
    # Position the n sliders at y = -0.10, -0.22, -0.34, …  (relative to plot area)
    base_y = -0.10
    dy     = 0.12
    for slider_idx, sp in enumerate(slider_specs):
        label = sp.get("label", "")
        unit  = sp.get("unit", "")
        fmt   = sp.get("fmt",  "%.4g")
        vals  = list(sp["values_disp"])
        idxs  = list(sp["frame_indices"])
        if len(vals) != len(idxs):
            raise ValueError(f"slider_specs[{slider_idx}]: values_disp and "
                             "frame_indices length mismatch.")
        # Per-step format
        try:
            spec_fmt = fmt.lstrip("%")
            labels = [format(v, spec_fmt) for v in vals]
        except (ValueError, TypeError):
            labels = [str(v) for v in vals]
        steps = [dict(method="animate", label=lbl,
                      args=[[str(idxs[i])],
                            dict(mode="immediate",
                                 frame=dict(duration=0, redraw=True),
                                 transition=dict(duration=0))])
                 for i, lbl in enumerate(labels)]
        unit_suffix = f" {unit}" if unit else ""
        sliders.append(dict(
            active=len(vals) // 2,
            x=0.0, y=base_y - slider_idx * dy,
            xanchor="left", yanchor="top",
            len=1.0,
            currentvalue=dict(prefix=f"{label} = ", suffix=unit_suffix,
                              font=dict(size=12)),
            pad=dict(t=4, b=4),
            steps=steps,
        ))

    fig.update_layout(
        sliders=sliders,
        height=height + 60 + 60 * n_sliders,
        showlegend=True,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="v", x=1.02, y=1.0, xanchor="left",
                    font=dict(size=18)),
        margin=dict(l=50, r=30, t=50, b=80 + 60 * n_sliders),
        hovermode="closest",
    )
    # Smith axes — equal aspect, fixed unit range
    fig.update_xaxes(range=[-1.1, 1.1], showgrid=False, zeroline=False,
                     scaleanchor="y", scaleratio=1, title="Re(Γ)",
                     row=1, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], showgrid=False, zeroline=False,
                     title="Im(Γ)", row=1, col=1)
    # Bode axes — log freq, dB
    fig.update_xaxes(title="Frequency (GHz)", type="log",
                     showgrid=True, gridcolor="#ebebeb", row=1, col=2)
    fig.update_yaxes(title="Gain (dB)", range=[0, 50],
                     showgrid=True, gridcolor="#ebebeb", row=1, col=2)
    return fig


# ── JS-coordinated joint multi-slider Plotly HTML ─────────────────────────────

def _decimate_freq(S_batch, freq, max_points: int = 200):
    """Decimate freq axis to ~max_points for Plotly rendering performance.
    Returns (S_batch_dec, freq_dec).  Pad/lead resonances are usually
    visible at the chosen frequency density; reducing to 200 points keeps
    Smith and Bode looking smooth without overwhelming the browser when
    multiplied by hundreds of frames."""
    S_batch = np.asarray(S_batch)
    n = S_batch.shape[-3] if S_batch.ndim == 4 else S_batch.shape[-3]
    if n <= max_points:
        return S_batch, np.asarray(freq)
    stride = int(np.ceil(n / max_points))
    return S_batch[..., ::stride, :, :], np.asarray(freq)[::stride]


def _compact_json_1d(arr, digits: int = 5) -> str:
    """Compact JSON for a 1-D float array — `%.{digits}g` per number, no
    whitespace. Roughly 3× smaller than ``json.dumps(arr.tolist())`` which
    preserves full float64 precision (~17 sig figs).  For Smith / Bode
    plotting, 5 sig figs is indistinguishable from full precision."""
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return "[]"
    fmt = f"%.{digits}g"
    return "[" + ",".join(fmt % v for v in arr) + "]"


def make_smith_bode_joint_slider_html(*, S_batch_joint, freq, slider_specs,
                                       model_name: str, S_meas=None,
                                       decimate_points: int = 120,
                                       json_digits: int = 5,
                                       smith_mults: dict | None = None) -> str:
    """Smith + Bode figure with HTML range sliders that drive
    ``Plotly.restyle`` directly — no Plotly frame system, no
    ``Plotly.animate`` calls, and no auto-animation when the iframe
    mounts.

    Why this rewrite
    ----------------
    The previous version registered the sweep as Plotly frames and
    called ``Plotly.animate`` on every slider event.  Plotly queues
    animations at ``frame.duration = 500`` ms by default; if the
    figure's mount triggers a chain of internal slider events, those
    queue up and play out one-by-one (≈ 0.5 s × N steps each) before
    the user's drag is even acknowledged.  That's what you saw as
    "21 movements over 11 seconds" of auto-animation.

    The fix is to bypass the animation system entirely:

      • The pre-computed sweep is embedded once as a JS array (compact
        5-sig-fig JSON, optional Bode-x omission since freq is static).
      • One ``<input type="range">`` per swept param sits beneath the
        chart.  On `input` it computes the joint frame index in JS and
        calls ``Plotly.restyle`` to swap trace data — instantaneous,
        no animation, no queue.
      • Figure starts at the *midpoint* frame.  No movement happens
        until the user actually drags.

    Frame ordering convention
    -------------------------
    Frames are laid out row-major across ``slider_specs`` — the first
    spec is the *slowest*-varying axis, last spec is the *fastest*.
    Total frame count must equal ``len(S_batch_joint)``.

    Parameters
    ----------
    S_batch_joint  : ndarray (B_total, N_freq, 2, 2) complex.
    freq           : ndarray (N_freq,) Hz.
    slider_specs   : list of {label, unit, fmt, values_disp} dicts.
    S_meas         : optional measured ndarray (N_freq, 2, 2) — static
                     overlay traces.
    decimate_points: cap freq points per trace.  Default 120; pass
                     ``len(freq)`` for full fidelity.
    json_digits    : sig-figs per float in the embedded JS data.

    Returns
    -------
    HTML string ready for ``st.iframe(html, height=…)`` (or the
    deprecated ``st.components.v1.html`` if you're stuck on an older
    Streamlit; the call signature is identical apart from the missing
    ``scrolling`` arg on ``st.iframe``).
    """
    from plotly.subplots import make_subplots
    import html as _html

    # Validate dims
    dims = [len(sp["values_disp"]) for sp in slider_specs]
    expected = int(np.prod(dims)) if dims else 0
    if S_batch_joint.shape[0] != expected:
        raise ValueError(f"S_batch_joint has {S_batch_joint.shape[0]} frames "
                         f"but cartesian dims imply {expected}.")

    # Decimate freq axis for browser performance.  When decimate_points
    # ≥ len(freq), this is a no-op (full fidelity).
    S_batch_joint, freq = _decimate_freq(S_batch_joint, freq,
                                          max_points=decimate_points)
    f_ghz = np.asarray(freq) * 1e-9
    B = S_batch_joint.shape[0]
    N = len(f_ghz)

    # ── Per-trace Smith multiplier (mirrors the Plotly Smith chart's scale) ──
    # ``smith_mults`` is the same {"S11":…} dict the user edits next to the main
    # Plotly Smith chart; apply it to every Smith trace (measured overlay,
    # initial model, and each embedded frame) so this slider view scales in
    # lock-step with the static chart.
    _sc = {nm: float((smith_mults or {}).get(nm, 1.0))
           for nm in ("S11", "S12", "S21", "S22")}

    def _sc_lbl(nm):
        m = _sc[nm]
        if abs(m - 1.0) < 1e-9:
            return ""
        return f" ×{m:g}" if m >= 1 else f" ÷{1 / m:g}"

    # fT / fmax colours (blue / red) — shared with every other Bode plot.
    _ft_col, _fmax_col = FT_FMAX_COLORS["fT"], FT_FMAX_COLORS["fmax"]

    def _extrap_arr(gain_db, npts=14):
        """20 dB/dec extrapolation of one gain trace → compact (x_json, y_json).

        Returns ("[]", "[]") when the trace already crosses 0 dB in-band or is
        otherwise un-extrapolatable, so the matching slider frame simply draws
        no extrapolation segment.
        """
        fx, gx, _ = extrap_20dbdec(f_ghz, gain_db, n_pts=npts)
        if fx is None:
            return "[]", "[]"
        return (_compact_json_1d(np.asarray(fx), json_digits),
                _compact_json_1d(np.asarray(gx), json_digits))

    # Midpoint joint index — initial display
    midpoints = [d // 2 for d in dims]
    mid_joint = 0
    for k, m in enumerate(midpoints):
        mid_joint = mid_joint * dims[k] + m

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.10,
        subplot_titles=(f"Smith — {model_name}", f"fT / fmax — {model_name}"),
        column_widths=[1.05, 1.0],
        specs=[[{"type": "xy"}, {"type": "xy"}]],
    )

    # Static Smith grid
    for tr in extended_smith_grid(1.0):
        fig.add_trace(tr, row=1, col=1)

    # Optional static measured overlay
    if S_meas is not None:
        S_meas = np.asarray(S_meas)
        if S_meas.shape[0] != N:
            stride = int(np.ceil(S_meas.shape[0] / N))
            S_meas_d = S_meas[::stride][:N]
        else:
            S_meas_d = S_meas
        for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                              ("S21", (1, 0)), ("S12", (0, 1))]:
            col = _SMITH_SLIDER_COLORS[name]
            s = S_meas_d[:, r, c] * _sc[name]
            fig.add_trace(go.Scattergl(x=s.real, y=s.imag, mode="markers",
                                        name=f"{name}{_sc_lbl(name)} meas",
                                        marker=dict(color=col, size=5)),
                          row=1, col=1)
    else:
        S_meas_d = None

    # Initial model Smith traces (midpoint frame)
    model_smith_indices: list[int] = []
    S0 = S_batch_joint[mid_joint]
    for name, (r, c) in [("S11", (0, 0)), ("S22", (1, 1)),
                          ("S21", (1, 0)), ("S12", (0, 1))]:
        col = _SMITH_SLIDER_COLORS[name]
        s = S0[:, r, c] * _sc[name]
        fig.add_trace(go.Scattergl(x=s.real, y=s.imag, mode="lines",
                                    name=f"{name}{_sc_lbl(name)} model",
                                    line=dict(color=col, width=2, dash="dash")),
                      row=1, col=1)
        model_smith_indices.append(len(fig.data) - 1)

    # Optional static measured Bode.  Colour by *quantity*: |h21|² (→ fT) blue,
    # Mason U (→ fmax) red — matched to every other Bode plot.  Measured is
    # distinguished from the model by its markers (model is dashed, no markers).
    if S_meas_d is not None:
        h21_m_db, U_m_db = compute_h21_U(S_meas_d)
        fig.add_trace(go.Scattergl(x=f_ghz, y=h21_m_db,
                                    mode="lines+markers", name="|h21|² meas",
                                    line=dict(color=_ft_col, width=1.4),
                                    marker=dict(size=4, symbol="circle")),
                      row=1, col=2)
        fig.add_trace(go.Scattergl(x=f_ghz, y=U_m_db,
                                    mode="lines+markers", name="Mason U meas",
                                    line=dict(color=_fmax_col, width=1.4,
                                              dash="dot"),
                                    marker=dict(size=4, symbol="square")),
                      row=1, col=2)
        # Static measured extrapolation → 0 dB (fT / fmax).
        _mfx, _mfy, _ = extrap_20dbdec(f_ghz, h21_m_db)
        if _mfx is not None:
            fig.add_trace(go.Scattergl(x=_mfx, y=_mfy, mode="lines",
                                        name="fT meas (extrap)",
                                        line=dict(color=_ft_col, width=1.3,
                                                  dash="dot")),
                          row=1, col=2)
        _mux, _muy, _ = extrap_20dbdec(f_ghz, U_m_db)
        if _mux is not None:
            fig.add_trace(go.Scattergl(x=_mux, y=_muy, mode="lines",
                                        name="fmax meas (extrap)",
                                        line=dict(color=_fmax_col, width=1.3,
                                                  dash="dot")),
                          row=1, col=2)

    # Initial model Bode traces (midpoint frame) — |h21|² blue, Mason U red.
    h21_s0, U_s0 = compute_h21_U(S0)
    fig.add_trace(go.Scattergl(x=f_ghz, y=h21_s0, mode="lines",
                                name="|h21|² model",
                                line=dict(color=_ft_col, width=2, dash="dash")),
                  row=1, col=2)
    bode_h21_idx = len(fig.data) - 1
    fig.add_trace(go.Scattergl(x=f_ghz, y=U_s0, mode="lines",
                                name="Mason U model",
                                line=dict(color=_fmax_col, width=2,
                                          dash="longdash")),
                  row=1, col=2)
    bode_U_idx = len(fig.data) - 1

    # Initial model extrapolation segments (auto, per-frame below) — fT / fmax.
    _e0fx, _e0fy, _ = extrap_20dbdec(f_ghz, h21_s0)
    fig.add_trace(go.Scattergl(
        x=_e0fx if _e0fx is not None else [],
        y=_e0fy if _e0fy is not None else [],
        mode="lines", name="fT model (extrap)",
        line=dict(color=_ft_col, width=1.6, dash="dot")), row=1, col=2)
    ext_h21_idx = len(fig.data) - 1
    _e0ux, _e0uy, _ = extrap_20dbdec(f_ghz, U_s0)
    fig.add_trace(go.Scattergl(
        x=_e0ux if _e0ux is not None else [],
        y=_e0uy if _e0uy is not None else [],
        mode="lines", name="fmax model (extrap)",
        line=dict(color=_fmax_col, width=1.6, dash="dot")), row=1, col=2)
    ext_U_idx = len(fig.data) - 1

    smith_traces = list(model_smith_indices)          # [S11, S22, S21, S12]
    bode_traces  = [bode_h21_idx, bode_U_idx]         # [h21, U]
    extrap_traces = [ext_h21_idx, ext_U_idx]          # [fT extrap, fmax extrap]

    fig.update_layout(
        height=500,
        showlegend=True,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="v", x=1.02, y=1.0, xanchor="left",
                    font=dict(size=18)),
        margin=dict(l=50, r=30, t=50, b=50),
        hovermode="closest",
    )
    fig.update_xaxes(range=[-1.1, 1.1], showgrid=False, zeroline=False,
                     scaleanchor="y", scaleratio=1, title="Re(Γ)",
                     row=1, col=1)
    fig.update_yaxes(range=[-1.1, 1.1], showgrid=False, zeroline=False,
                     title="Im(Γ)", row=1, col=1)
    fig.update_xaxes(title="Frequency (GHz)", type="log",
                     showgrid=True, gridcolor="#ebebeb", row=1, col=2)
    fig.update_yaxes(title="Gain (dB)", range=[0, 50],
                     showgrid=True, gridcolor="#ebebeb", row=1, col=2)

    fig_html = fig.to_html(include_plotlyjs="cdn", full_html=True,
                            div_id="hbtSlPlot")

    # ── Build compact JS data array ───────────────────────────────────
    # Per-frame layout (10 arrays, each length N):
    #   [S11.real, S11.imag, S22.real, S22.imag,
    #    S21.real, S21.imag, S12.real, S12.imag,
    #    h21_dB,   U_dB]
    chunk_parts = []
    _names = ["S11", "S22", "S21", "S12"]
    for i in range(B):
        Si = S_batch_joint[i]
        h21, U = compute_h21_U(Si)
        per_frame = []
        for nm, (r, c) in zip(_names, [(0, 0), (1, 1), (1, 0), (0, 1)]):
            s = Si[:, r, c] * _sc[nm]
            per_frame.append(_compact_json_1d(s.real, json_digits))
            per_frame.append(_compact_json_1d(s.imag, json_digits))
        per_frame.append(_compact_json_1d(h21, json_digits))
        per_frame.append(_compact_json_1d(U,   json_digits))
        # Auto 20 dB/dec extrapolation → fT (from |h21|²) and fmax (from U).
        _hx, _hy = _extrap_arr(h21)
        _ux, _uy = _extrap_arr(U)
        per_frame.extend((_hx, _hy, _ux, _uy))
        chunk_parts.append("[" + ",".join(per_frame) + "]")
    data_js = "[" + ",".join(chunk_parts) + "]"

    # Slider step labels (one list per slider)
    labels_lists = []
    for sp in slider_specs:
        fmt = sp.get("fmt", "%.4g").lstrip("%")
        try:
            lbls = [format(v, fmt) for v in sp["values_disp"]]
        except (ValueError, TypeError):
            lbls = [str(v) for v in sp["values_disp"]]
        labels_lists.append(lbls)
    import json as _json
    labels_js = _json.dumps(labels_lists)
    dims_js   = _json.dumps(dims)
    mids_js   = _json.dumps(midpoints)
    smith_idx_js = _json.dumps(smith_traces)
    bode_idx_js  = _json.dumps(bode_traces)
    extrap_idx_js = _json.dumps(extrap_traces)

    # Slider rows HTML
    sliders_html_parts = []
    for i, sp in enumerate(slider_specs):
        label = _html.escape(str(sp.get("label", "")))
        unit  = _html.escape(str(sp.get("unit",  "")))
        n     = len(sp["values_disp"])
        mid   = midpoints[i]
        init_label = labels_lists[i][mid]
        sliders_html_parts.append(
            f'<div class="hbt-row">'
            f'<div class="hbt-name">{label}</div>'
            f'<div class="hbt-val"><span id="sv{i}">{init_label}</span> {unit}</div>'
            f'<input type="range" id="sl{i}" min="0" max="{n-1}" '
            f'value="{mid}" step="1">'
            f'</div>'
        )
    sliders_html = ('<div id="hbt-sliders">'
                    + "".join(sliders_html_parts) + "</div>")

    inject = f"""
<style>
html, body {{ margin: 0; padding: 0; font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, sans-serif; }}
#hbt-sliders {{
  padding: 12px 60px 14px 60px;
  background: #fafafa;
  border-top: 1px solid #e0e0e0;
}}
.hbt-row {{
  display: flex;
  align-items: center;
  gap: 14px;
  margin: 6px 0;
  font-size: 13px;
}}
.hbt-name {{
  min-width: 70px;
  font-weight: 600;
  color: #222;
}}
.hbt-val {{
  min-width: 140px;
  color: #444;
  font-variant-numeric: tabular-nums;
}}
.hbt-row input[type=range] {{
  flex: 1 1 auto;
  accent-color: #1f77b4;
  height: 24px;
}}
</style>
{sliders_html}
<script>
(function() {{
  var DATA   = {data_js};
  var DIMS   = {dims_js};
  var MIDS   = {mids_js};
  var LBLS   = {labels_js};
  var SMITH  = {smith_idx_js};
  var BODE   = {bode_idx_js};
  var EXTRAP = {extrap_idx_js};

  function jointIndex(pos) {{
    var j = 0;
    for (var k = 0; k < DIMS.length; k++) j = j * DIMS[k] + pos[k];
    return j;
  }}

  function update() {{
    var pos = [];
    for (var i = 0; i < DIMS.length; i++) {{
      var el = document.getElementById('sl' + i);
      pos.push(parseInt(el.value, 10));
      var lbl = document.getElementById('sv' + i);
      if (lbl) lbl.textContent = LBLS[i][pos[i]];
    }}
    var gd = document.getElementById('hbtSlPlot');
    if (!gd || !gd.data) return;
    var frame = DATA[jointIndex(pos)];
    // frame = [S11r, S11i, S22r, S22i, S21r, S21i, S12r, S12i, h21, U,
    //          h21extX, h21extY, UextX, UextY]
    Plotly.restyle(gd, {{
      x: [frame[0], frame[2], frame[4], frame[6]],
      y: [frame[1], frame[3], frame[5], frame[7]]
    }}, SMITH);
    Plotly.restyle(gd, {{
      y: [frame[8], frame[9]]
    }}, BODE);
    Plotly.restyle(gd, {{
      x: [frame[10], frame[12]],
      y: [frame[11], frame[13]]
    }}, EXTRAP);
  }}

  function init() {{
    var gd = document.getElementById('hbtSlPlot');
    if (!gd || !gd.data) {{ setTimeout(init, 80); return; }}
    for (var i = 0; i < DIMS.length; i++) {{
      var s = document.getElementById('sl' + i);
      if (s) s.addEventListener('input', update);
    }}
  }}
  init();
}})();
</script>
"""
    return fig_html.replace("</body>", inject + "</body>")
