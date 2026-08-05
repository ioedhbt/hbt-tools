"""
guide/_data/_gen/verify_gds_demo.py — loads the three hand-written GDS demo
files through the app's own parser, tools.ebeam.gdsii.parser._parse_gds, and
checks cell names, layers, polygon counts and bounding boxes match what was
intended.

Run from /tmp/app:
    cd /tmp/app && PYTHONPATH=/tmp/app python3 guide/_data/_gen/verify_gds_demo.py
"""
import os

from tools.ebeam.gdsii.parser import _parse_gds, _DEFAULT_LIMITS

REPO = "/tmp/app"
GDS_DIR = os.path.join(REPO, "guide/_data/gds")


def describe(path):
    with open(path, "rb") as fh:
        buf = fh.read()
    unit_m, cells = _parse_gds(buf, _DEFAULT_LIMITS)
    lines = [f"### {os.path.basename(path)}", "", f"unit = {unit_m:.3e} m/user-unit "
             f"({'OK 1 um' if abs(unit_m - 1e-6) < 1e-12 else 'MISMATCH'})",
             f"cells = {list(cells.keys())}", ""]
    for name, cell in cells.items():
        lines.append(f"cell `{name}`:")
        for (layer, dt), (cx, cy, starts) in sorted(cell["polys"].items()):
            n_polys = len(starts) - 1
            bbox = (float(cx.min()), float(cy.min()), float(cx.max()), float(cy.max()))
            lines.append(f"  - layer {layer}, datatype {dt}: {n_polys} polygons, "
                         f"bbox (um) = ({bbox[0]:.1f}, {bbox[1]:.1f}) - ({bbox[2]:.1f}, {bbox[3]:.1f})")
        if cell["refs"]:
            lines.append(f"  - refs: {cell['refs']}")
    lines.append("")
    return lines, cells


def main():
    all_lines = ["# GDS demo files — parser verification\n",
                 "Loaded with the app's own `tools.ebeam.gdsii.parser._parse_gds` "
                 "(the function `gdsii/stream.py` itself calls to decode an "
                 "upload).\n"]
    ok = True
    expected = {
        "first_exposure.gds": {(1, 1): 6, (1, 2): 6, (1, 3): 6, (1, 0): 4},
        "second_exposure.gds": {(1, 0): 4, (2, 1): 6, (2, 2): 12},
        "dose_test.gds": {(1, 1): 25, (1, 2): 75, (1, 0): 1},
    }
    for fname in ["first_exposure.gds", "second_exposure.gds", "dose_test.gds"]:
        path = os.path.join(GDS_DIR, fname)
        lines, cells = describe(path)
        all_lines += lines
        top = cells.get("TOP")
        if top is None:
            all_lines.append(f"FAIL: no TOP cell in {fname}\n")
            ok = False
            continue
        got = {k: len(v[2]) - 1 for k, v in top["polys"].items()}
        exp = expected[fname]
        match = got == exp
        all_lines.append(f"expected per-(layer,datatype) polygon counts: {exp}")
        all_lines.append(f"got:                                          {got}")
        all_lines.append(f"MATCH: {match}\n")
        ok &= match

    all_lines.append(f"\n**Overall PASS: {ok}**\n")
    text = "\n".join(all_lines)
    print(text)
    with open(os.path.join(GDS_DIR, "VERIFY.md"), "w") as fh:
        fh.write(text)
    print(f"\nwrote {os.path.join(GDS_DIR, 'VERIFY.md')}")


if __name__ == "__main__":
    main()
