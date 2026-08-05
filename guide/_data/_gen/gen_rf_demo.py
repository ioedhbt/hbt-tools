"""
guide/_data/_gen/gen_rf_demo.py — builds the RF open/short de-embedding demo
set under guide/_data/rf/.

Must be run from /tmp/app (the Python-3.10-safe shadow tree), e.g.:

    cd /tmp/app && python3 guide/_data/_gen/gen_rf_demo.py

Pipeline
--------
Source: deembed_these/deembedded/4x10_left/*.s2p — 7 files that are already
de-embedded (intrinsic device S-parameters, "Y_ex1" in the repo's naming).

We pick one fixed set of pad/lead parasitics and RE-embed every DUT file
with them — i.e. compute what the VNA would have measured before
de-embedding — using the exact inverse of tools/rf/ssm/helpers/deembed_math.py
::peel_parasitics:

    peel_parasitics (forward, measured -> intrinsic):
        Y_dut  = s_to_y(S_raw)
        Z1     = y_to_z(Y_dut - Y_pad)      # remove pad shunt admittance
        Y_ex1  = z_to_y(Z1 - Z_ser)         # remove series lead impedance

    re-embed (inverse, intrinsic -> measured), applied here:
        Z_ex1  = y_to_z(Y_ex1)
        Z1     = Z_ex1 + Z_ser              # add back series lead impedance
        Y1     = z_to_y(Z1)
        Y_dut  = Y1 + Y_pad                 # add back pad shunt admittance
        S_raw  = y_to_s(Y_dut)

open.s2p / short.s2p are produced with the repo's own forward simulators
(s2p_io.simulate_open / simulate_short), which already encode the pad-only /
pad+lead equivalent circuits — so the three files are self-consistent by
construction.
"""
import glob
import os

import numpy as np

from tools.rf.ssm.helpers.s2p_io import (
    parse_s2p, write_s2p, simulate_open, simulate_short,
)
from tools.rf.ssm.helpers.rf_math import s_to_y, y_to_z, z_to_y, y_to_s_vec
from tools.rf.ssm.helpers.deembed_math import build_Y_pad_vec, build_Z_ser_vec

REPO = "/tmp/app"  # shadow tree; guide/ symlinks back to the real repo
DUT_DIR = os.path.join(REPO, "deembed_these/deembedded/4x10_left")
OUT_DIR = os.path.join(REPO, "guide/_data/rf")
RAW_DIR = os.path.join(OUT_DIR, "raw")

# ── The fixed parasitic set quoted in the guide ──────────────────────────────
P = dict(
    Cpbe=22e-15, Cpce=18e-15, Cpbc=5e-15,   # F
    Lb=45e-12, Lc=42e-12, Le=6e-12,          # H
    Rpb=0.0, Rpc=0.0, Rpe=0.0,               # ideal leads — not specified,
                                              # kept at 0 so the quoted C/L
                                              # set is the complete truth.
)


def reembed(freq, S_ex1, z0, p):
    """Exact inverse of deembed_math.peel_parasitics."""
    omega = 2.0 * np.pi * freq
    Y_ex1 = s_to_y(S_ex1, z0)
    Z_ex1 = y_to_z(Y_ex1)
    Z_ser = build_Z_ser_vec(p, omega, np)
    Z1 = Z_ex1 + Z_ser
    Y1 = z_to_y(Z1)
    Y_pad = build_Y_pad_vec(p, omega, np)
    Y_dut = Y1 + Y_pad
    return y_to_s_vec(Y_dut, z0, np)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    dut_files = sorted(glob.glob(os.path.join(DUT_DIR, "*.s2p")))
    assert len(dut_files) == 7, f"expected 7 DUT files, found {len(dut_files)}"

    freq_ref = None
    for path in dut_files:
        name = os.path.basename(path)
        with open(path, "rb") as fh:
            freq, S_ex1, z0 = parse_s2p(fh.read())
        if freq_ref is None:
            freq_ref = freq
        else:
            assert len(freq) == len(freq_ref) and np.allclose(freq, freq_ref), \
                f"{name}: frequency grid differs from the first file"

        S_raw = reembed(freq, S_ex1, z0, P)

        title = f"Re-embedded (raw/measured) — {name}"
        params = {
            "source_deembedded_file": name,
            "Cpbe_fF": P["Cpbe"] * 1e15, "Cpce_fF": P["Cpce"] * 1e15,
            "Cpbc_fF": P["Cpbc"] * 1e15,
            "Lb_pH": P["Lb"] * 1e12, "Lc_pH": P["Lc"] * 1e12,
            "Le_pH": P["Le"] * 1e12,
            "Rpb_ohm": P["Rpb"], "Rpc_ohm": P["Rpc"], "Rpe_ohm": P["Rpe"],
        }
        raw_bytes = write_s2p(freq, S_raw, title=title, params=params)
        out_path = os.path.join(RAW_DIR, name)
        with open(out_path, "wb") as fh:
            fh.write(raw_bytes)
        print(f"wrote {out_path}  ({len(freq)} pts)")

    # ── open.s2p / short.s2p on the same frequency grid ──────────────────────
    z0 = 50.0
    S_open = simulate_open(P, freq_ref, z0)
    S_short = simulate_short(P, freq_ref, z0)

    open_bytes = write_s2p(
        freq_ref, S_open, title="Open dummy (pad caps only)",
        params={"Cpbe_fF": P["Cpbe"] * 1e15, "Cpce_fF": P["Cpce"] * 1e15,
                "Cpbc_fF": P["Cpbc"] * 1e15})
    with open(os.path.join(OUT_DIR, "open.s2p"), "wb") as fh:
        fh.write(open_bytes)
    print(f"wrote {os.path.join(OUT_DIR, 'open.s2p')}  ({len(freq_ref)} pts)")

    short_bytes = write_s2p(
        freq_ref, S_short, title="Short dummy (pads + leads, device shorted)",
        params={"Cpbe_fF": P["Cpbe"] * 1e15, "Cpce_fF": P["Cpce"] * 1e15,
                "Cpbc_fF": P["Cpbc"] * 1e15,
                "Lb_pH": P["Lb"] * 1e12, "Lc_pH": P["Lc"] * 1e12,
                "Le_pH": P["Le"] * 1e12})
    with open(os.path.join(OUT_DIR, "short.s2p"), "wb") as fh:
        fh.write(short_bytes)
    print(f"wrote {os.path.join(OUT_DIR, 'short.s2p')}  ({len(freq_ref)} pts)")


if __name__ == "__main__":
    main()
