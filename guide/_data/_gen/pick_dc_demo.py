"""
guide/_data/_gen/pick_dc_demo.py — copies the four chosen dc_data/raw/*.csv
files into guide/_data/dc/ and writes CHOICES.md with the numbers that
justified each pick (see analyze_dc.py for how those numbers were computed:
a sliding-window search for the longest run of forward, positive-current
points whose log10(I) vs V is a near-perfect line, R^2 >= 0.999 — a direct
"ideality visible over N clean decades" measure, plus overall monotonicity
and full-sweep decade count).

Run with any Python 3:  python3 guide/_data/_gen/pick_dc_demo.py
"""
import os
import shutil

RAW_DIR = "/sessions/eloquent-stoic-turing/mnt/hbt-tools/dc_data/raw"
OUT_DIR = "/sessions/eloquent-stoic-turing/mnt/hbt-tools/guide/_data/dc"

CHOICES = {
    "BC diode": "BC diode [40x40 CME(1) ; 3_13_2026 4_08_50 PM].csv",
    "BE diode": "BE diode [60x60 CME(1) _1V_; 3_12_2026 7_46_17 PM].csv",
    "Gummel": "Gummel [60x60 CME(1) _1V_; 3_12_2026 7_43_08 PM].csv",
    "Family": "Family [4x10 CME(4) ; 3_12_2026 8_45_16 PM].csv",
}

NOTES = {
    "BC diode": (
        "**BC diode — `BC diode [40x40 CME(1) ; 3_13_2026 4_08_50 PM].csv`.** "
        "Of the 12 BC-diode sweeps, this is the only one combining perfectly "
        "monotonic Ib across the full ±1 V sweep (mono fraction = 1.000, "
        "vs 0.50–0.99 for the rest — several files have a visible glitch or "
        "sign flip near the noise floor) with the longest clean exponential "
        "run: 2.32 decades at R² = 0.9991 for a straight-line fit of "
        "log10(Ib) vs Vb (the next-best fully-monotonic file, 60x60 CME(1) "
        "\"1V\", manages only 1.77 decades). Total dynamic range is 6.38 "
        "decades of forward current, from ~5.3 pA up."
    ),
    "BE diode": (
        "**BE diode — `BE diode [60x60 CME(1) _1V_; 3_12_2026 7_46_17 PM].csv`.** "
        "Best clean-exponential run of all 14 BE-diode files: 4.38 decades "
        "at R² = 0.9992 (the runner-up, the 40x40 device, reaches 3.74 "
        "decades). It also has the widest total dynamic range in the set, "
        "8.12 decades of |Ie| forward current down to a ~37 fA floor — "
        "consistent with a low-noise sweep rather than a noise-limited one, "
        "over the full ±1 V bias."
    ),
    "Gummel": (
        "**Gummel — `Gummel [60x60 CME(1) _1V_; 3_12_2026 7_43_08 PM].csv`.** "
        "Ideality is visible over 4.07 clean decades of Ic (R² = 0.9992 for "
        "log10(Ic) vs Vb over that window) — more than any other Gummel "
        "file surveyed; the next-best, 60x60 CME(8) at 2.2 V, only reaches "
        "3.84 decades and is materially less monotonic (0.87 vs 0.99 here). "
        "Ic spans 7.92 decades total down to an 86 fA floor over Vb = 0–1 V, "
        "with peak β ≈ 5.6 — a clean, textbook Gummel plot with no visible "
        "compliance clipping (max Ic stays orders of magnitude below the "
        "100 mA SMU compliance set in the test)."
    ),
    "Family": (
        "**Family — `Family [4x10 CME(4) ; 3_12_2026 8_45_16 PM].csv`.** "
        "Widest sweep of all 14 Family files (Vc = 0–4 V, more than double "
        "the next-widest at 3.5 V) while still one of the flattest in "
        "saturation: the top Ib curve's Ic in the Vc > 0.7·Vc_max region "
        "varies only 0.9 % (std/mean = 0.009), beaten only by three "
        "larger-emitter devices with much narrower sweeps (≤2 V). Six clean "
        "Ib steps, Ic up to 2.9 mA, no evidence of compliance clipping. "
        "Also the same 4×10 µm emitter geometry as the RF de-embedding demo "
        "device, so the guide's DC and RF examples describe the same part."
    ),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = ["# DC demo file selection\n",
             "Chosen from `dc_data/raw/*.csv` (67 B1500A exports total) by "
             "actually loading and scoring each file — see "
             "`guide/_data/_gen/analyze_dc.py` for the metrics (forward "
             "monotonicity, total decades of clean current, and a sliding "
             "R² ≥ 0.999 search for the longest clean-exponential window, "
             "which is the direct numeric stand-in for \"ideality visible "
             "over several decades\").\n"]
    for kind, fname in CHOICES.items():
        src = os.path.join(RAW_DIR, fname)
        dst = os.path.join(OUT_DIR, fname)
        shutil.copyfile(src, dst)
        print(f"copied {fname} -> {dst}")
        lines.append("\n" + NOTES[kind] + "\n")
    with open(os.path.join(OUT_DIR, "CHOICES.md"), "w") as fh:
        fh.write("".join(lines))
    print(f"wrote {os.path.join(OUT_DIR, 'CHOICES.md')}")


if __name__ == "__main__":
    main()
