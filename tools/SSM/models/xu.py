"""
models/xu.py — Dvorak & Bolognesi (2003) CBC/Cpi extraction.

Reference: Dvorak & Bolognesi, IEEE MTT-S 2003.
  Eq. 12: C̃_BC = 1 / (ω · Im{Z22−Z21})
  Eq. 13: C_BC(Z) = Im{1/(ω(Z22−Z21))}  ← recommended
  Eq. 14: C_BC(Y) = Im{Y12}/ω
  Eq. 28: C_π   = Im{Y11+Y12}/ω
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..helpers    import y_to_z, safe_median, params_hash
from .base_ui     import PAD_SPECS
from . import AbstractSSMModel


# ════════════════════════════════════════════════════════════════════════════════
# Extraction
# ════════════════════════════════════════════════════════════════════════════════

def _extract(Y_ex1, freq, n_low):
    omega = 2.0 * np.pi * freq
    Z     = y_to_z(Y_ex1)          # (N,2,2)

    dZ = Z[:, 1, 1] - Z[:, 1, 0]  # Z22 − Z21

    # Eq. 12
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbc_Z1_arr = np.where(
            np.abs(np.imag(dZ) * omega) > 1e-40,
            1.0 / (omega * np.imag(dZ)),
            np.nan,
        )

    # Eq. 13 (recommended)
    with np.errstate(divide="ignore", invalid="ignore"):
        Cbc_Z2_arr = np.where(
            np.abs(dZ) * omega > 1e-40,
            np.imag(1.0 / (omega * dZ)),
            np.nan,
        )

    # Eq. 14
    Cbc_Y_arr = np.imag(Y_ex1[:, 0, 1]) / omega

    # Eq. 28
    Cpi_arr = np.imag(Y_ex1[:, 0, 0] + Y_ex1[:, 0, 1]) / omega

    Cbc_Z1 = safe_median(Cbc_Z1_arr, n_low)
    Cbc_Z2 = safe_median(Cbc_Z2_arr, n_low)
    Cbc_Y  = safe_median(Cbc_Y_arr,  n_low)
    Cpi    = safe_median(Cpi_arr,    n_low)

    params = dict(Cbc_Z1=Cbc_Z1, Cbc_Z2=Cbc_Z2, Cbc_Y=Cbc_Y, Cpi=Cpi)
    arrays = dict(Cbc_Z1_arr=Cbc_Z1_arr, Cbc_Z2_arr=Cbc_Z2_arr,
                  Cbc_Y_arr=Cbc_Y_arr, Cpi_arr=Cpi_arr)
    return params, arrays


# ════════════════════════════════════════════════════════════════════════════════
# XuModel class
# ════════════════════════════════════════════════════════════════════════════════

class XuModel(AbstractSSMModel):
    NAME          = "Xu / Dvorak-Bolognesi (2003) CBC"
    SHORT         = "Xu"
    TOPOLOGY_CHAR = "pi"

    @classmethod
    def extract(cls, Y_ex1, freq, n_low, **kwargs):
        return _extract(Y_ex1, freq, n_low)

    @classmethod
    def simulate(cls, params, freq, z0=50.0):
        # Forward simulation not implemented for this diagnostic model
        return None

    # @classmethod
    # def render_step_formulas(cls):
    #     st.markdown("**Dvorak & Bolognesi (2003) — CBC via three methods + C_π**")
    #     c1, c2, c3, c4 = st.columns(4)
    #     with c1:
    #         st.markdown("**Eq. 12** (Z, Im only)")
    #         st.latex(r"\tilde{C}_{BC}=\frac{1}{\omega\,\mathrm{Im}(Z_{22}-Z_{21})}")
    #     with c2:
    #         st.markdown("**Eq. 13** (Z, recommended)")
    #         st.latex(r"C_{BC}=\mathrm{Im}\!\left\{\frac{1}{\omega(Z_{22}-Z_{21})}\right\}")
    #     with c3:
    #         st.markdown("**Eq. 14** (Y-param)")
    #         st.latex(r"C_{BC}=\frac{\mathrm{Im}(Y_{12})}{\omega}")
    #     with c4:
    #         st.markdown("**Eq. 28** (C_π)")
    #         st.latex(r"C_\pi=\frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}")

    @classmethod
    def render_results_table(cls, params):
        rows = [
            ("Cbc (Eq.12, Z Im)", f"{params['Cbc_Z1']*1e15:.4f}", "fF"),
            ("Cbc (Eq.13, Z rec)", f"{params['Cbc_Z2']*1e15:.4f}", "fF"),
            ("Cbc (Eq.14, Y)",    f"{params['Cbc_Y']*1e15:.4f}",  "fF"),
            ("Cpi (Eq.28)",       f"{params['Cpi']*1e15:.4f}",    "fF"),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Parameter", "Value", "Unit"]),
                     width="stretch", hide_index=True)

    # @classmethod
    # def render_formula_trace(cls):
    #     with st.expander("📐 Full formula trace — Dvorak & Bolognesi (2003)", expanded=False):
    #         st.markdown("**Input:** Y_ex1 (fully de-embedded DUT admittance)")
    #         st.latex(r"[Eq.12]\;\tilde{C}_{BC}=\frac{1}{\omega\,\mathrm{Im}(Z_{22}-Z_{21})}")
    #         st.latex(r"[Eq.13]\;C_{BC}=\mathrm{Im}\!\left\{\frac{1}{\omega(Z_{22}-Z_{21})}\right\}"
    #                  r"\quad\text{(recommended)}")
    #         st.latex(r"[Eq.14]\;C_{BC}=\frac{\mathrm{Im}(Y_{12})}{\omega}")
    #         st.latex(r"[Eq.28]\;C_\pi=\frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}")
    #         st.markdown("No forward simulation implemented — diagnostic / comparison model only.")

    @classmethod
    def render_override_and_smith(cls, fname, S_raw, freq, z0,
                                  para_eff, extract_result, **kwargs):
        params, arrays = extract_result
        f_ghz = freq * 1e-9
        n_low = kwargs.get("n_low", max(3, len(freq) // 10))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Eq. 12** (Z, Im only)")
            st.latex(r"\tilde{C}_{BC}=\frac{1}{\omega\,\mathrm{Im}(Z_{22}-Z_{21})}")
        with c2:
            st.markdown("**Eq. 13** (Z, recommended)")
            st.latex(r"C_{BC}=\mathrm{Im}\!\left\{\frac{1}{\omega(Z_{22}-Z_{21})}\right\}")
        with c3:
            st.markdown("**Eq. 14** (Y-param)")
            st.latex(r"C_{BC}=\frac{\mathrm{Im}(Y_{12})}{\omega}")
        with c4:
            st.markdown("**Eq. 28** (C_π)")
            st.latex(r"C_\pi=\frac{\mathrm{Im}(Y_{11}+Y_{12})}{\omega}")

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        plot_specs = [
            ("Cbc_Z1_arr", f"Cbc Eq.12\n= {params['Cbc_Z1']*1e15:.2f} fF", params["Cbc_Z1"]),
            ("Cbc_Z2_arr", f"Cbc Eq.13\n= {params['Cbc_Z2']*1e15:.2f} fF", params["Cbc_Z2"]),
            ("Cbc_Y_arr",  f"Cbc Eq.14\n= {params['Cbc_Y']*1e15:.2f} fF",  params["Cbc_Y"]),
            ("Cpi_arr",    f"Cpi Eq.28\n= {params['Cpi']*1e15:.2f} fF",    params["Cpi"]),
        ]

        for ax, (key, title, med_val) in zip(axes, plot_specs):
            arr = arrays[key] * 1e15   # → fF
            med_fF = med_val * 1e15
            ax.plot(f_ghz, arr, color="#1f77b4", linewidth=1.0)
            ax.axhline(med_fF, color="red", linestyle="--", linewidth=1.2,
                       label=f"Median {med_fF:.2f} fF")
            # shade the low-freq region used for median
            if n_low < len(f_ghz):
                ax.axvspan(f_ghz[0], f_ghz[n_low - 1],
                           alpha=0.12, color="orange", label="median region")
            ax.set_xlabel("Freq (GHz)")
            ax.set_ylabel("Capacitance (fF)")
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, linewidth=0.4)

        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.info("Forward simulation not available for this model — no Smith chart.")
        return None
