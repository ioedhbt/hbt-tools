"""
guide/_data/_gen/verify_rf_demo.py — round-trip check for the RF demo set.

Loads guide/_data/rf/raw/deemb_preext_cold.s2p, guide/_data/rf/open.s2p and
guide/_data/rf/short.s2p exactly as the app would (parse_s2p), runs them
through the app's own extraction (step_open, step_short) and peeling
(peel_parasitics) functions, and compares the result to the original
(pre-re-embedding) de-embedded file. This exercises the actual files on
disk, not just the parasitic constants used to build them.

Run from /tmp/app:
    cd /tmp/app && PYTHONPATH=/tmp/app python3 guide/_data/_gen/verify_rf_demo.py
"""
import os

import numpy as np

from tools.rf.ssm.helpers.s2p_io import parse_s2p
from tools.rf.ssm.helpers.rf_math import y_to_s_batch
from tools.rf.ssm.helpers.deembed_math import step_open, step_short, peel_parasitics

REPO = "/tmp/app"
RF_DIR = os.path.join(REPO, "guide/_data/rf")


def main():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    orig_path = os.path.join(REPO, "deembed_these/deembedded/4x10_left/deemb_preext_cold.s2p")
    raw_path = os.path.join(RF_DIR, "raw/deemb_preext_cold.s2p")
    open_path = os.path.join(RF_DIR, "open.s2p")
    short_path = os.path.join(RF_DIR, "short.s2p")

    with open(orig_path, "rb") as fh:
        f_orig, S_orig, z0_orig = parse_s2p(fh.read())
    with open(raw_path, "rb") as fh:
        f_raw, S_raw, z0_raw = parse_s2p(fh.read())
    with open(open_path, "rb") as fh:
        f_open, S_open, z0_open = parse_s2p(fh.read())
    with open(short_path, "rb") as fh:
        f_short, S_short, z0_short = parse_s2p(fh.read())

    assert np.allclose(f_orig, f_raw) and np.allclose(f_orig, f_open) and np.allclose(f_orig, f_short)
    assert z0_orig == z0_raw == z0_open == z0_short == 50.0

    open_data = (f_open, S_open, z0_open)
    short_data = (f_short, S_short, z0_short)

    # Step 1a: extract pad caps from open.s2p (full-band median — exact
    # analytic data, so any sub-range gives the same answer).
    open_params, _ = step_open(open_data, n0=0, n1=len(f_open), method="Median")
    log("Extracted from open.s2p:")
    for k, v in open_params.items():
        log(f"  {k} = {v*1e15:.6f} fF")

    # Step 1b: extract lead L/R from short.s2p, using the *measured* open
    # (open_data) exactly as the app's default flow does.
    short_params, _ = step_short(
        short_data, f_short,
        open_params["Cpbe"], open_params["Cpce"], open_params["Cpbc"],
        open_data=open_data, n0=0, n1=len(f_short), method="Median",
        measured_open=True,
    )
    log("Extracted from short.s2p:")
    for k, v in short_params.items():
        unit = "pH" if k.startswith("L") else "Ohm"
        scale = 1e12 if k.startswith("L") else 1.0
        log(f"  {k} = {v*scale:.6f} {unit}")

    p_check = dict(
        Cpbe=open_params["Cpbe"], Cpce=open_params["Cpce"], Cpbc=open_params["Cpbc"],
        Lb=short_params["Lb"], Lc=short_params["Lc"], Le=short_params["Le"],
        Rpb=short_params["Rpb"], Rpc=short_params["Rpc"], Rpe=short_params["Rpe"],
    )

    # Step 2: peel the extracted parasitics off the re-embedded raw file.
    Y_ex1_check = peel_parasitics(S_raw, f_raw, z0_raw, p_check)
    S_ex1_check = y_to_s_batch(Y_ex1_check, z0_raw)

    diff = np.abs(S_ex1_check - S_orig)
    max_err = float(diff.max())
    max_err_idx = np.unravel_index(np.argmax(diff), diff.shape)
    rms_err = float(np.sqrt(np.mean(diff**2)))

    log("")
    log(f"max |S_check - S_original| = {max_err:.3e}  (at index {max_err_idx}, "
        f"f = {f_raw[max_err_idx[0]]/1e9:.4f} GHz)")
    log(f"rms |S_check - S_original| = {rms_err:.3e}")
    log(f"PASS (< 1e-6): {max_err < 1e-6}")

    out_path = os.path.join(RF_DIR, "VERIFY.md")
    with open(out_path, "w") as fh:
        fh.write("# RF de-embedding demo — round-trip verification\n\n")
        fh.write(
            "Pipeline: parse `raw/deemb_preext_cold.s2p`, `open.s2p`, "
            "`short.s2p` exactly as the app does (`parse_s2p`), extract pad "
            "capacitances with `step_open` and lead inductances/resistances "
            "with `step_short` (repo's own `tools/rf/ssm/helpers/"
            "deembed_math.py`), peel the extracted parasitics off the raw "
            "file with `peel_parasitics`, and compare the result to the "
            "original (pre-re-embedding) de-embedded file, "
            "`deembed_these/deembedded/4x10_left/deemb_preext_cold.s2p`.\n\n"
        )
        fh.write("Fixed parasitics used to build the demo set:\n\n")
        fh.write(
            "```\nCpbe = 22 fF, Cpbc = 5 fF, Cpce = 18 fF, "
            "Lb = 45 pH, Lc = 42 pH, Le = 6 pH, Rpb = Rpc = Rpe = 0 Ω\n```\n\n"
        )
        fh.write("Extracted back from open.s2p / short.s2p:\n\n```\n")
        for k, v in open_params.items():
            fh.write(f"{k} = {v*1e15:.6f} fF\n")
        for k, v in short_params.items():
            if k.startswith("L"):
                fh.write(f"{k} = {v*1e12:.6f} pH\n")
            else:
                fh.write(f"{k} = {v:.3e} Ohm\n")
        fh.write("```\n\n")
        fh.write("## Result\n\n```\n")
        fh.write("\n".join(lines))
        fh.write("\n```\n")
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
