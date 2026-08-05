"""tuning.md — Auto Tuning walkthrough: remove Rbc from "Parameters to fit"
so it holds at its extracted value (58.7816 kOhm) instead of being searched.

Captures the raw (pre-annotation) scope screenshot plus the CSS-pixel bbox
of the Rbc tag *before* it is removed, printed to stdout -- annotate.py runs
afterward (outside the browser) to circle that position on the saved PNG,
since nothing after it shifts left far enough to fill the gap at that x.

Shots:
  00_scope_no_rbc_raw — Full Auto Tune card, Rbc tag removed, unannotated.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf/raw")


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
    settle(page, quiet_ms=700)

    at = page.get_by_text("Auto Tuning for Minimum Residuals", exact=False).first
    at.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    at.click()
    settle(page, quiet_ms=400)
    hide_badge(page)

    scope = page.get_by_text("Full Auto Tune", exact=False).first
    scope.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=300)
    hide_badge(page)

    wrap = page.locator('div[data-testid="stMultiSelect"]').filter(
        has_text="Parameters to fit").first
    tag = wrap.locator('span[data-baseweb="tag"]').filter(
        has_text="Rbc (kΩ)").first
    tb = tag.bounding_box()
    print("RBC TAG BOX", tb)

    icon = tag.locator('svg[title="Delete"]')
    icon.click()
    page.wait_for_timeout(400)
    settle(page, quiet_ms=300)
    hide_badge(page)

    sb = scope.bounding_box()
    print("SCOPE BOX", sb)
    clip = {"x": 370, "y": max(0, sb["y"] - 10), "width": 1550, "height": 260}
    print("CLIP", clip)
    shot("00_scope_no_rbc_raw", clip=clip)
    print("DONE")
