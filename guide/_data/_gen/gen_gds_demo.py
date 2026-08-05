"""
guide/_data/_gen/gen_gds_demo.py — builds the three GDSII demo files for the
EBL guide: first_exposure.gds, second_exposure.gds, dose_test.gds.

All three share one 2000 x 2000 um coordinate system (a small InP HBT test
chip). Geometry is intentionally simple rectangles/crosses — this is a demo
for the app's viewer, not a real mask — but every coordinate is exact and
recorded in guide/_data/gds/GEOMETRY.md so the guide text can quote them.

Run with any Python 3:  python3 guide/_data/_gen/gen_gds_demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gdswrite import GDSWriter, rect, cross  # noqa: E402

OUT_DIR = "/sessions/eloquent-stoic-turing/mnt/hbt-tools/guide/_data/gds"

# ── Shared coordinate system ─────────────────────────────────────────────────
CHIP = (0.0, 0.0, 2000.0, 2000.0)  # (x0, y0, x1, y1) um

# Four alignment marks near the chip corners, 100 um margin.
MARKS = {
    "BL": (100.0, 100.0),
    "BR": (1900.0, 100.0),
    "TL": (100.0, 1900.0),
    "TR": (1900.0, 1900.0),
}
MARK_ARM_LEN = 40.0
MARK_ARM_W = 8.0

# Device array: 2 rows x 3 cols, alternating 4x10 / 5x10 um emitters.
DEVICES = [
    # (cx, cy, emitter_width_um)
    (800.0, 850.0, 4.0),
    (1050.0, 850.0, 5.0),
    (1300.0, 850.0, 4.0),
    (800.0, 1150.0, 5.0),
    (1050.0, 1150.0, 4.0),
    (1300.0, 1150.0, 5.0),
]
EMITTER_LEN = 10.0        # um, fixed
PAD_W, PAD_H = 40.0, 60.0
PAD_GAP = 30.0             # pad inner edge to device center, x-direction

# Dose-test grid: 5x5 blocks, 20x20 um each, 40 um pitch, origin (1300, 200).
DOSE_ORIGIN = (1300.0, 200.0)
DOSE_BLOCK = 20.0
DOSE_PITCH = 40.0
DOSE_N = 5
DOSE_LINE_WIDTHS = [0.5, 1.0, 2.0]
DOSE_MARK_XY = MARKS["BR"]  # reuse the same corner mark, same style/coords


def add_marks(w: GDSWriter, struct: str, layer: int):
    for cx, cy in MARKS.values():
        w.boundary(struct, layer, 0, cross(cx, cy, MARK_ARM_LEN, MARK_ARM_W))


def add_devices_layer1(w: GDSWriter, struct: str, layer: int):
    for cx, cy, we in DEVICES:
        # Base pad (left), emitter finger (center), collector pad (right).
        base_cx = cx - PAD_GAP - PAD_W / 2.0
        coll_cx = cx + PAD_GAP + PAD_W / 2.0
        w.boundary(struct, layer, 1, rect(base_cx, cy, PAD_W, PAD_H))     # base pad, dt=1
        w.boundary(struct, layer, 2, rect(cx, cy, we, EMITTER_LEN))       # emitter, dt=2
        w.boundary(struct, layer, 3, rect(coll_cx, cy, PAD_W, PAD_H))     # collector pad, dt=3


def add_interconnect_layer2(w: GDSWriter, struct: str, layer: int):
    bridge_w, bridge_h = 140.0, 10.0
    post_w = 10.0
    y_off = 40.0  # bridge centerline above device center
    for cx, cy, we in DEVICES:
        w.boundary(struct, layer, 1, rect(cx, cy + y_off, bridge_w, bridge_h))  # metal bridge
        w.boundary(struct, layer, 2, rect(cx - 70.0, cy + y_off, post_w, post_w))  # left post
        w.boundary(struct, layer, 2, rect(cx + 70.0, cy + y_off, post_w, post_w))  # right post


def dose_block(w: GDSWriter, struct: str, layer: int, ox, oy):
    """One 20x20 um dose-test block at local origin (ox, oy) (lower-left)."""
    # 8x8 um pad, lower-left corner at (ox+1, oy+1).
    pad_pts = [(ox + 1, oy + 1), (ox + 9, oy + 1), (ox + 9, oy + 9), (ox + 1, oy + 9)]
    w.boundary(struct, layer, 1, pad_pts)
    x = ox + 11.0
    for lw in DOSE_LINE_WIDTHS:
        pts = [(x, oy + 1), (x + lw, oy + 1), (x + lw, oy + 19), (x, oy + 19)]
        w.boundary(struct, layer, 2, pts)
        x += lw + 1.5  # 1.5 um gap between lines


def build_first_exposure():
    w = GDSWriter(libname="FIRST_EXPOSURE")
    top = w.new_structure("TOP")
    add_marks(w, top, layer=1)
    add_devices_layer1(w, top, layer=1)
    return w


def build_second_exposure():
    w = GDSWriter(libname="SECOND_EXPOSURE")
    top = w.new_structure("TOP")
    add_marks(w, top, layer=1)                 # identical marks, layer 1
    add_interconnect_layer2(w, top, layer=2)    # new pattern, layer 2
    return w


def build_dose_test():
    w = GDSWriter(libname="DOSE_TEST")
    top = w.new_structure("TOP")
    ox0, oy0 = DOSE_ORIGIN
    for i in range(DOSE_N):
        for j in range(DOSE_N):
            dose_block(w, top, layer=1, ox=ox0 + j * DOSE_PITCH, oy=oy0 + i * DOSE_PITCH)
    w.boundary(top, 1, 0, cross(*DOSE_MARK_XY, MARK_ARM_LEN, MARK_ARM_W))
    return w


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    builds = {
        "first_exposure.gds": build_first_exposure(),
        "second_exposure.gds": build_second_exposure(),
        "dose_test.gds": build_dose_test(),
    }
    for fname, w in builds.items():
        path = os.path.join(OUT_DIR, fname)
        w.write(path)
        print(f"wrote {path}  polygons={w.polygon_count}  area={w.total_area_um2:.3f} um^2  "
              f"bytes={os.path.getsize(path)}")


if __name__ == "__main__":
    main()
