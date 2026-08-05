"""
guide/_data/_gen/verify_hp4155a_demo.py — round-trip check for the HP4155A
demo files.

tools/dc/hp4155a_plot.py is a Streamlit *page* (runs top-to-bottom, calls
st.title/st.stop() etc. at module scope), so it can't be imported normally
outside a Streamlit run. read_table() and parse_smu_table() themselves have
no Streamlit dependency, so this pulls those two function definitions
straight out of the source file with `ast` (byte-identical to what ships)
and runs them exactly as the app would, on the actual files.

Run from /tmp/app:
    cd /tmp/app && python3 guide/_data/_gen/verify_hp4155a_demo.py
"""
import ast
import glob
import os

import numpy as np
import pandas as pd

REPO = "/tmp/app"
DC_DIR = os.path.join(REPO, "guide/_data/dc")
HP_DIR = os.path.join(DC_DIR, "hp4155a")
SRC_FILE = os.path.join(REPO, "tools/dc/hp4155a_plot.py")


def load_app_functions(path, names):
    """Extract named top-level function defs from a Streamlit page script
    and exec them into a fresh namespace, without running the page."""
    with open(path) as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    found = {n.name for n in wanted}
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"functions not found in {path}: {missing}")
    module = ast.Module(body=wanted, type_ignores=[])
    ns = {"pd": pd, "np": np, "io": __import__("io")}
    exec(compile(module, path, "exec"), ns)
    return ns


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


def check(name, hp_path, src_path, smu_map, src_cols):
    ns = load_app_functions(SRC_FILE, ["read_table", "parse_smu_table"])
    with open(hp_path) as fh:
        raw = fh.read()
    df = ns["read_table"](raw)
    parsed = ns["parse_smu_table"](df, smu_map)

    src = load_b1500a_csv(src_path)

    lines = [f"## {name}", "", f"Parsed with the app's `read_table`: {df.shape[0]} rows, columns {list(df.columns)}.", ""]
    ok = True
    for role, (vcol_src, icol_src) in src_cols.items():
        if role not in parsed:
            lines.append(f"- FAIL: role '{role}' missing from parsed output")
            ok = False
            continue
        v_app, i_app = parsed[role]
        v_src = src[vcol_src].values
        i_src = src[icol_src].values
        # "%.5E" on write keeps 6 significant digits per value, so round-off
        # is relative-to-each-value, not to the array max — compare
        # per-element relative error (absolute for near-zero values).
        v_err = float(np.max(np.abs(v_app.values - v_src) / np.maximum(np.abs(v_src), 1e-6)))
        i_err = float(np.max(np.abs(i_app.values - i_src) / np.maximum(np.abs(i_src), 1e-15)))
        lines.append(f"- {role}: max relative |V error| = {v_err:.3e}, max relative |I error| = {i_err:.3e}")
        if v_err > 1e-4 or i_err > 1e-4:
            ok = False
    lines.append(f"\nPASS: {ok}\n")
    return ok, lines


def main():
    all_lines = ["# HP4155A conversion — round-trip verification\n"]
    all_ok = True

    gummel_src = glob.glob(os.path.join(DC_DIR, "Gummel*.csv"))[0]
    ok, lines = check(
        "Gummel (gummel_60x60_hp4155a.txt)",
        os.path.join(HP_DIR, "gummel_60x60_hp4155a.txt"),
        gummel_src,
        smu_map={"base": ("V1", "I1"), "collector": ("V2", "I2"), "emitter": ("V3", "I3")},
        src_cols={"base": ("Vb", "Ib"), "collector": ("Vc", "Ic"), "emitter": ("Ve", "Ie")},
    )
    all_ok &= ok
    all_lines += lines

    family_src = glob.glob(os.path.join(DC_DIR, "Family*.csv"))[0]
    ok, lines = check(
        "Family (family_4x10_hp4155a.txt)",
        os.path.join(HP_DIR, "family_4x10_hp4155a.txt"),
        family_src,
        smu_map={"collector": ("V1", "I1"), "base": ("V2", "I2"), "emitter": ("V3", "I3")},
        src_cols={"collector": ("Vc", "Ic"), "base": ("Vb", "Ib"), "emitter": ("Ve", "Ie")},
    )
    all_ok &= ok
    all_lines += lines

    all_lines.append(f"\n**Overall PASS: {all_ok}**\n")
    text = "\n".join(all_lines)
    print(text)
    with open(os.path.join(HP_DIR, "VERIFY.md"), "w") as fh:
        fh.write(text)
    print(f"\nwrote {os.path.join(HP_DIR, 'VERIFY.md')}")


if __name__ == "__main__":
    main()
