"""
guide/_data/_gen/gen_hp4155a_demo.py — converts the chosen Gummel and Family
B1500A CSVs (guide/_data/dc/*.csv) into the HP/Agilent 4155A SMU-dump format
tools/dc/hp4155a_plot.py parses.

Format required by hp4155a_plot.py::read_table / parse_smu_table
(tools/dc/hp4155a_plot.py):
  - Whitespace- or comma-delimited table, header row first (no preamble —
    pd.read_csv(sep=r"\\s+") takes row 0 as the column names).
  - Column names are the literal, generic SMU channel labels the app's SMU
    Assignment step maps to a physical role: V1/I1, V2/I2, V3/I3, V4/I4
    (tools/dc/hp4155a_plot.py:: smu_cols). A real 4155A "List Display -> Save
    ASCII" dump looks exactly like this when channel names were never
    customized from the instrument defaults.
  - Values written in the classic 4155A scientific-notation style,
    "%.5E" (e.g. "1.23456E-02"), tab-separated.

Run with any Python 3:  python3 guide/_data/_gen/gen_hp4155a_demo.py
"""
import glob
import os

import numpy as np
import pandas as pd

DC_DIR = "/sessions/eloquent-stoic-turing/mnt/hbt-tools/guide/_data/dc"
OUT_DIR = os.path.join(DC_DIR, "hp4155a")


def load_b1500a_csv(path):
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as fh:
        lines = fh.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.strip().startswith("DataName"))
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
        if len(parts) < len(cols):
            parts = parts + [""] * (len(cols) - len(parts))
        rows.append([to_f(p) for p in parts[:len(cols)]])
    return pd.DataFrame(rows, columns=cols)


def write_4155a(path, col_map, title_comment=None):
    """col_map: ordered dict {"V1": series, "I1": series, ...}."""
    header = "\t".join(col_map.keys())
    n = len(next(iter(col_map.values())))
    lines = [header]
    for i in range(n):
        row = "\t".join(f"{col_map[c].iloc[i]:.5E}" for c in col_map)
        lines.append(row)
    with open(path, "w", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    gummel_src = glob.glob(os.path.join(DC_DIR, "Gummel*.csv"))[0]
    family_src = glob.glob(os.path.join(DC_DIR, "Family*.csv"))[0]

    # ── Gummel: base / collector / emitter -> V1I1 / V2I2 / V3I3 ────────────
    g = load_b1500a_csv(gummel_src)
    col_map = {
        "V1": g["Vb"], "I1": g["Ib"],   # Base
        "V2": g["Vc"], "I2": g["Ic"],   # Collector
        "V3": g["Ve"], "I3": g["Ie"],   # Emitter
    }
    # Named for the actual device (60x60 um) rather than 4x10: the Gummel
    # file chosen in step 2 was picked for data quality (ideality visible
    # over 4+ clean decades), and that file happens to be the 60x60 device
    # — see guide/_data/dc/CHOICES.md. Keeping the filename honest.
    out_path = os.path.join(OUT_DIR, "gummel_60x60_hp4155a.txt")
    write_4155a(out_path, col_map)
    print(f"wrote {out_path}  ({len(g)} rows)  <- {os.path.basename(gummel_src)}")

    # ── Family: collector / base (+ emitter, unused by the app but a real
    # instrument dump would still record the common node) -> V1I1/V2I2/V3I3
    f = load_b1500a_csv(family_src)
    col_map = {
        "V1": f["Vc"], "I1": f["Ic"],   # Collector
        "V2": f["Vb"], "I2": f["Ib"],   # Base
        "V3": f["Ve"], "I3": f["Ie"],   # Emitter
    }
    out_path = os.path.join(OUT_DIR, "family_4x10_hp4155a.txt")
    write_4155a(out_path, col_map)
    print(f"wrote {out_path}  ({len(f)} rows)  <- {os.path.basename(family_src)}")


if __name__ == "__main__":
    main()
