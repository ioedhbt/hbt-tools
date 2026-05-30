"""
B1500A_Plot.py — Keysight B1500A output/transfer curve viewer + TLM analysis.

Version is tracked in ``__version__`` below and in ``CHANGELOG.md`` at the
repo root.

Changelog (short):
- v1.1: Parameter extraction for Diode/Gummel/Family — ideality (η) window with
  dashed markers, β(max) at Ic, family hover gain + per-curve offset voltage,
  saturation/output resistance, Early & knee voltage, diode Is / forward /
  leakage / rectification readouts; controls grouped into containers.
- v1.0: Initial version tracking.
"""
__version__ = "1.1"

import re

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =================================================
# Utilities
# =================================================
def find_col_like(df, keywords):
    for c in df.columns:
        cl = c.lower()
        if all(k in cl for k in keywords):
            return c
    return None


def find_cols_starting(df, prefix):
    return [c for c in df.columns if c.lower().startswith(prefix.lower())]


def ideality_factor(v, i, vmin, vmax):
    mask = (v >= vmin) & (v <= vmax) & (i > 0)
    if mask.sum() < 2:
        return np.nan
    slope, _ = np.polyfit(v[mask], np.log(i[mask]), 1)
    q = 1.602e-19
    k = 1.381e-23
    T = 300
    return q / (slope * k * T)


def parse_ib_current(label):
    """Extract the Ib drive current (in amps) from a column label such as
    'Ic at Ib=200uA'. Returns None if no current can be parsed."""
    m = re.search(r"ib\s*=?\s*([\d.]+)\s*([pnumµμ]?)\s*a", label, re.IGNORECASE)
    if not m:
        return None
    scale = {"": 1.0, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6,
             "n": 1e-9, "p": 1e-12}
    return float(m.group(1)) * scale.get(m.group(2).lower(), 1.0)


def fmt_current(a):
    """Human-readable current with engineering units."""
    if a is None or not np.isfinite(a):
        return "—"
    a = float(a)
    for unit, scale in [("A", 1), ("mA", 1e-3), ("µA", 1e-6),
                        ("nA", 1e-9), ("pA", 1e-12)]:
        if abs(a) >= scale:
            return f"{a / scale:.3g} {unit}"
    return f"{a:.3g} A"


def zero_crossing_x(x, y):
    """Linearly interpolate the x where y rises through zero (negative →
    positive). Returns NaN if no such crossing exists."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    for k in range(len(y) - 1):
        if y[k] <= 0 <= y[k + 1] and y[k + 1] != y[k]:
            return x[k] + (0 - y[k]) * (x[k + 1] - x[k]) / (y[k + 1] - y[k])
    return np.nan


def linear_fit_resistance(vce, ic, vmin, vmax):
    """Fit Ic = m·Vce + b over [vmin, vmax]. Returns (R_out, V_early, slope)
    where R_out = dVce/dIc (output resistance) and V_early is the magnitude of
    the extrapolated x-intercept (Early voltage)."""
    vce = np.asarray(vce, float)
    ic = np.asarray(ic, float)
    mask = (vce >= vmin) & (vce <= vmax)
    if mask.sum() < 2:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(vce[mask], ic[mask], 1)
    if m == 0:
        return np.inf, np.nan, 0.0
    return 1.0 / m, abs(-b / m), m


def knee_voltage(vce, ic, frac=0.9):
    """Vce at which Ic first reaches `frac` of its saturation value (mean Ic
    over the top 20 % of the Vce sweep)."""
    vce = np.asarray(vce, float)
    ic = np.asarray(ic, float)
    order = np.argsort(vce)
    vce, ic = vce[order], ic[order]
    sat = ic[vce >= 0.8 * vce.max()].mean()
    if not np.isfinite(sat) or sat <= 0:
        return np.nan
    hit = np.where(ic >= frac * sat)[0]
    return vce[hit[0]] if len(hit) else np.nan


def diode_knee_voltage(v, i):
    """Turn-on (knee) voltage from linear extrapolation of the steep forward
    region back to I = 0 on a linear I–V scale."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    fwd = (v > 0) & (i > 0)
    if fwd.sum() < 2:
        return np.nan
    imax = i[fwd].max()
    mask = fwd & (i >= 0.2 * imax)
    if mask.sum() < 2:
        return np.nan
    m, b = np.polyfit(v[mask], i[mask], 1)
    return -b / m if m != 0 else np.nan


def saturation_current(v, i, vmin, vmax):
    """Reverse-saturation current Is from the y-intercept of the semilog fit."""
    v = np.asarray(v, float)
    i = np.asarray(i, float)
    mask = (v >= vmin) & (v <= vmax) & (i > 0)
    if mask.sum() < 2:
        return np.nan
    _, intercept = np.polyfit(v[mask], np.log(i[mask]), 1)
    return np.exp(intercept)


# =================================================
# Sidebar
# =================================================
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["B1500A Viewer", "TLM Analysis"]
)

# =================================================
# B1500A VIEWER (EXCEL ONLY)
# =================================================
if page == "B1500A Viewer":

    st.title(f"🧪 B1500A Excel Viewer (v{__version__})")

    uploaded = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
    if not uploaded:
        st.stop()

    xls = pd.ExcelFile(uploaded)
    sheet = st.selectbox("Select sheet", xls.sheet_names)
    df = xls.parse(sheet)

    name = f"{uploaded.name} — {sheet}".lower()

    # Auto data type
    if "gummel" in name:
        default = "Gummel"
    elif "family" in name:
        default = "Family"
    else:
        default = "Diode"

    dtype = st.selectbox(
        "Data type",
        ["Diode", "Gummel", "Family"],
        index=["Diode", "Gummel", "Family"].index(default)
    )

    # =================================================
    # DIODE
    # =================================================
    if dtype == "Diode":
        vcol = df.columns[0]
        icol = df.columns[1]

        v = df[vcol].astype(float)
        i_signed = df[icol].astype(float)
        i = i_signed.abs()

        # reserve the plot's slot so it stays on top while its controls
        # live in the results container below
        plot_area = st.container()

        with st.container(border=True):
            st.subheader("Extracted parameters")

            vmin, vmax = st.slider(
                "Ideality factor (η) voltage range (V)",
                float(v.min()), float(v.max()), (0.35, 0.4)
            )

            cv1, cv2 = st.columns(2)
            v_fwd_default = 1.0 if float(v.max()) >= 1.0 else float(v.max())
            v_rev_default = -1.0 if float(v.min()) <= -1.0 else float(v.min())
            v_fwd = cv1.number_input(
                "Forward-current readout voltage (V)",
                float(v.min()), float(v.max()), v_fwd_default
            )
            v_rev = cv2.number_input(
                "Leakage-current readout voltage (V)",
                float(v.min()), float(v.max()), v_rev_default
            )

            # --- extractions ---
            n = ideality_factor(v, i, vmin, vmax)
            i_s = saturation_current(v, i, vmin, vmax)
            knee = diode_knee_voltage(v, i_signed)
            i_fwd = i[(v - v_fwd).abs().idxmin()]
            i_leak = i[(v - v_rev).abs().idxmin()]
            vmag = min(abs(float(v.min())), abs(float(v.max())))
            rect = np.nan
            if vmag > 0:
                f_pt = i[(v - vmag).abs().idxmin()]
                r_pt = i[(v + vmag).abs().idxmin()]
                rect = f_pt / r_pt if r_pt else np.nan

            c1, c2, c3 = st.columns(3)
            c1.metric("Ideality factor, η", f"{n:.2f}")
            c1.metric("Knee / turn-on voltage",
                      f"{knee:.3f} V" if np.isfinite(knee) else "—")
            c2.metric("Saturation current, Is", fmt_current(i_s))
            c2.metric(f"Forward current @ {v_fwd:g} V", fmt_current(i_fwd))
            c3.metric(f"Leakage current @ {v_rev:g} V", fmt_current(i_leak))
            c3.metric("Rectification ratio",
                      f"{rect:.2e}" if np.isfinite(rect) else "—")
            st.caption(
                "η ≈ 1 → diffusion-dominated · η → 2 → recombination-dominated. "
                "Knee from linear extrapolation of the steep forward region; "
                "Is from the semilog-fit intercept over the η window."
            )

        # --- plot (rendered into the reserved area above) ---
        fig = go.Figure(go.Scatter(x=v, y=i, mode="lines+markers", name="|I|"))
        fig.add_vline(x=vmin, line_dash="dash", line_color="gray",
                      annotation_text="η range", annotation_position="top")
        fig.add_vline(x=vmax, line_dash="dash", line_color="gray")
        fig.update_yaxes(type="log", title="Current (A)")
        fig.update_xaxes(title="Voltage (V)")
        fig.update_layout(title="Diode I–V")
        plot_area.plotly_chart(fig, width="stretch")

    # =================================================
    # GUMMEL
    # =================================================
    elif dtype == "Gummel":
        vb_col = find_col_like(df, ["vb"])
        ic_col = find_col_like(df, ["ic"])
        ib_col = find_col_like(df, ["ib"])

        if not all([vb_col, ic_col, ib_col]):
            st.error("Could not identify Vb / Ic / Ib columns")
            st.stop()

        vb = df[vb_col].astype(float)
        ic = df[ic_col].astype(float)
        ib = df[ib_col].astype(float)

        beta = ic / ib.replace(0, np.nan)
        beta_max = beta.max()
        beta_v = vb.loc[beta.idxmax()]
        ic_at_betamax = abs(ic).loc[beta.idxmax()]

        # reserve the plot's slot; its η window slider lives in the
        # ideality container below
        plot_area = st.container()

        with st.container(border=True):
            st.subheader("Ideality factors (η)")
            vmin, vmax = st.slider(
                "Ideality factor (η) voltage range (V)",
                0.0, float(vb.max()), (0.35, 0.4)
            )
            n_ic = ideality_factor(vb, abs(ic), vmin, vmax)
            n_ib = ideality_factor(vb, abs(ib), vmin, vmax)
            c1, c2 = st.columns(2)
            c1.metric("Collector ideality, η(Ic)", f"{n_ic:.2f}")
            c2.metric("Base ideality, η(Ib)", f"{n_ib:.2f}")
            st.caption("Extracted over the dashed Vb window on the plot.")

        with st.container(border=True):
            st.subheader("Current gain (β)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Maximum gain, β(max)", f"{beta_max:.1f}")
            c2.metric("β(max) occurs at Vb", f"{beta_v:.3f} V")
            c3.metric("β(max) occurs at Ic", fmt_current(ic_at_betamax))

        # --- plot (rendered into the reserved area above) ---
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )

        fig.add_trace(go.Scatter(x=vb, y=abs(ic), name="Ic"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=vb, y=abs(ib), name="Ib"),
                      row=1, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(x=vb, y=beta, name="β"),
                      row=1, col=1, secondary_y=True)

        fig.add_vline(x=vmin, line_dash="dash", line_color="gray",
                      annotation_text="η range", annotation_position="top")
        fig.add_vline(x=vmax, line_dash="dash", line_color="gray")

        fig.update_yaxes(type="log", secondary_y=False, title="Current (A)")
        fig.update_yaxes(
            type="linear",
            range=[0.0, 1.5 * beta_max],
            secondary_y=True,
            title="β"
        )
        fig.update_xaxes(title="Vb (V)")
        fig.update_layout(title="Gummel Plot")

        plot_area.plotly_chart(fig, width="stretch")

    # =================================================
    # FAMILY
    # =================================================
    else:
        vc_col = find_col_like(df, ["vc"])
        if not vc_col:
            st.error("Vc column not found")
            st.stop()

        vc = df[vc_col].astype(float)
        ic_cols = [c for c in df.columns if c.lower().startswith("ic")]

        if not ic_cols:
            st.error("No Ic curves found (expected 'Ic at Ib=...')")
            st.stop()

        # default Rsat window = upper 40 % of the Vc sweep (flat region)
        vc_lo, vc_hi = float(vc.min()), float(vc.max())
        default_lo = round(vc_lo + 0.6 * (vc_hi - vc_lo), 3)

        # parse Ib per curve; the largest is the "highest Ib" line
        ib_vals = {c: parse_ib_current(c) for c in ic_cols}
        top_col = max(
            ic_cols,
            key=lambda c: (ib_vals[c] is not None, ib_vals[c] or 0.0)
        )

        # reserve the plot's slot; the Rsat-window slider lives in the
        # output-region container below
        plot_area = st.container()

        with st.container(border=True):
            st.subheader("Output-region parameters")
            st.caption(f"Fitted on the highest-Ib curve: **{top_col}**")
            r_min, r_max = st.slider(
                "Saturation-resistance fit window — Vc (V)",
                vc_lo, vc_hi, (default_lo, round(vc_hi, 3))
            )
            ic_top = df[top_col].astype(float)
            r_out, v_early, _ = linear_fit_resistance(vc, ic_top, r_min, r_max)
            knee = knee_voltage(vc, ic_top)
            c1, c2, c3 = st.columns(3)
            c1.metric("Saturation (output) resistance",
                      f"{r_out:,.0f} Ω" if np.isfinite(r_out) else "—")
            c2.metric("Early voltage, |VA|",
                      f"{v_early:.2f} V" if np.isfinite(v_early) else "—")
            c3.metric("Knee voltage",
                      f"{knee:.3f} V" if np.isfinite(knee) else "—")

        offsets = {}
        fig = go.Figure()
        for c in ic_cols:
            ib_a = ib_vals[c]
            ic_line = df[c].astype(float)
            v_off = zero_crossing_x(vc, ic_line)
            offsets[c] = v_off
            gain = (ic_line / ib_a) if ib_a else pd.Series(
                np.nan, index=ic_line.index)
            customdata = np.column_stack([
                gain.to_numpy(),
                np.full(len(ic_line), v_off),
            ])
            ht = ("Vc=%{x:.3f} V<br>Ic=%{y:.3e} A"
                  "<br>β=%{customdata[0]:.1f}"
                  "<br>V(offset)=%{customdata[1]:.3f} V"
                  "<extra>" + c + "</extra>")
            fig.add_trace(go.Scatter(
                x=vc, y=ic_line, name=c,
                customdata=customdata, hovertemplate=ht
            ))

        all_max = max(float(df[c].max()) for c in ic_cols)
        all_min = min(float(df[c].min()) for c in ic_cols)
        fig.add_vrect(
            x0=r_min, x1=r_max, fillcolor="LightGray", opacity=0.25,
            line_width=0, annotation_text="Rsat fit",
            annotation_position="top left"
        )
        fig.update_xaxes(title="Vc (V)")
        fig.update_yaxes(
            title="Ic (A)",
            range=[min(0.0, all_min) * 1.2, all_max * 1.2]
        )
        fig.update_layout(title="Family I–V", hovermode="closest")
        plot_area.plotly_chart(fig, width="stretch")

        valid_off = [o for o in offsets.values() if np.isfinite(o)]
        off_avg = float(np.mean(valid_off)) if valid_off else np.nan

        with st.container(border=True):
            st.subheader("Offset voltage")
            st.metric(
                "Average offset voltage",
                f"{off_avg * 1e3:.1f} mV" if np.isfinite(off_avg) else "—"
            )
            st.caption(
                "Hover any curve to read its own offset (Ic zero-crossing) and "
                "DC current gain β. The average above is over curves that "
                "actually cross zero."
            )

# =================================================
# TLM ANALYSIS
# =================================================
else:
    st.title(f"📐 TLM Analysis (v{__version__})")

    Z = st.number_input("Pad width Z (µm)", value=80.0)

    R = {
        4: st.number_input("R @ 4 µm (Ω)", value=np.nan),
        8: st.number_input("R @ 8 µm (Ω)", value=np.nan),
        16: st.number_input("R @ 16 µm (Ω)", value=np.nan),
        32: st.number_input("R @ 32 µm (Ω)", value=np.nan),
    }

    x = np.array([k for k, v in R.items() if not np.isnan(v)])
    y = np.array([v for v in R.values() if not np.isnan(v)])

    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        Rc = 0.5 * intercept
        Rsh = slope * Z
        LT = Rc / slope
        rho_c = Rc * LT * Z * 1e-8
        r2 = np.corrcoef(x, y)[0, 1] ** 2

        c1, c2, c3 = st.columns(3)
        c1.metric("Intercept (Ω)", f"{intercept:.2f}")
        c1.metric("Slope (Ω/µm)", f"{slope:.3f}")
        c2.metric("Contact Resistance, Rc (Ω)", f"{Rc:.2f}")
        c2.metric("Sheet Resistance, Rsh (Ω/□)", f"{Rsh:.1f}")
        c3.metric("Transfer Length, LT (µm)", f"{LT:.2f}")
        c3.metric("Specific Contact Resistivity, ρc (Ω·cm²)", f"{rho_c:.2e}")
        st.metric("Goodness, R²", f"{r2:.4f}")

        xs = np.linspace(0, 40, 200)
        ys = slope * xs + intercept

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Measured"))
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(dash="dash"),
            name="Linear fit"
        ))

        fig.update_xaxes(title="Spacing (µm)", range=[0, 40])
        fig.update_yaxes(title="Resistance (Ω)", range=[0, max(y) * 1.2])
        st.plotly_chart(fig, width="stretch")