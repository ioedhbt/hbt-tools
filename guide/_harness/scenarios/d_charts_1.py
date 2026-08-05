"""charts.md — S display scale row, export buttons (xlsx/copy), and the
Bode/Single-pole extrapolation toggle, all in fit mode.

Shots:
  00_scale   — "S display scale" S11..S22 x number_input row above the Smith
               chart.
  01_export  — xlsx + copy buttons under the Smith chart.
  02_extrap  — the -20 dB/dec / Single-pole segmented control on the
               fT/fmax card.
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

    scale = page.get_by_text("S display scale", exact=False).first
    scale.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    sb = scale.bounding_box()
    print("SCALE BOX", sb)
    shot("00_scale", clip={"x": 370, "y": max(0, sb["y"] - 10),
                           "width": 1550, "height": 100})

    xlsx_btn = page.get_by_role("button", name="xlsx", exact=False).first
    xlsx_btn.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    xb = xlsx_btn.bounding_box()
    print("XLSX BOX", xb)
    shot("01_export", clip={"x": 370, "y": max(0, xb["y"] - 70),
                            "width": 1550, "height": 140})

    extrap = page.get_by_text("dB/dec", exact=False).first
    extrap.evaluate("el => el.scrollIntoView({block:'center', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)
    eb = extrap.bounding_box()
    print("EXTRAP BOX", eb)
    shot("02_extrap", clip={"x": 370, "y": max(0, eb["y"] - 60),
                            "width": 1550, "height": 140})
    print("DONE")
