"""charts.md — the "Smith Chart (Matplotlib)" expander: drag-labels toggle,
chart appearance (grid), trace styling table (kind/style/color), per-S-param
multiplier table.

Shots:
  00_drag_toggle  — "Drag labels (interactive)" toggle + chart appearance
                    panel (line/grid thickness, grid circles, text size).
  01_trace_table  — Measured/Modeled trace-styling table (Kind/Style/Size/
                    Color/Decimate) + coloring-mode segmented control.
  02_sparam_table — per-S-param Multiplier/Text/x/y/color table.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/deembed_these/deembedded/4x10_left")


def hide_badge(page):
    page.evaluate(
        "() => {"
        "let s = document.getElementById('_hide_badge');"
        "if (!s) { s = document.createElement('style'); s.id = '_hide_badge';"
        "document.head.appendChild(s); }"
        "s.textContent = 'div.st-key-lang_toggle{display:none !important;}';"
        "}"
    )


def run(page, shot):
    from session import settle

    page.set_default_timeout(8000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page, quiet_ms=500)
    hide_badge(page)

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page, quiet_ms=500)
    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=800)
    hide_badge(page)

    mpl = page.get_by_text("Smith Chart (Matplotlib)", exact=False).first
    mpl.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    mpl.click()
    settle(page, quiet_ms=600)
    hide_badge(page)

    drag = page.get_by_text("Drag labels", exact=False).first
    drag.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    db = drag.bounding_box()
    print("DRAG BOX", db)
    shot("00_drag_toggle", clip={"x": 370, "y": max(0, db["y"] - 10),
                                 "width": 1550, "height": 260})

    appear = page.get_by_text("Chart appearance", exact=False).first
    appear.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    ab = appear.bounding_box()
    print("APPEAR BOX", ab)
    shot("00b_appearance", clip={"x": 370, "y": max(0, ab["y"] - 10),
                                 "width": 1550, "height": 220})

    trace = page.get_by_text("Measured", exact=False).first
    trace.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    tb = trace.bounding_box()
    print("TRACE BOX", tb)
    shot("01_trace_table", clip={"x": 370, "y": max(0, tb["y"] - 60),
                                 "width": 1550, "height": 260})

    sp = page.get_by_text("Per-S-parameter trace", exact=False).first
    sp.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    spb = sp.bounding_box()
    print("SP BOX", spb)
    shot("02_sparam_table", clip={"x": 370, "y": max(0, spb["y"] - 10),
                                  "width": 1550, "height": 260})
    print("DONE")
