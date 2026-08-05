"""charts.md — the Smith Chart (Matplotlib) control panel, in full-height slices.

d_charts_2.py had the right anchors but clipped 220-260 px tall bands, which
sliced through the tables.  Take full viewport frames at each anchor instead
and crop afterwards.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def hide_badge(page):
    page.evaluate(
        "() => {let s=document.getElementById('_hb');"
        "if(!s){s=document.createElement('style');s.id='_hb';"
        "document.head.appendChild(s);}"
        "s.textContent='div.st-key-lang_toggle{display:none !important;}';}"
    )


def run(page, shot):
    from session import settle

    page.set_default_timeout(8000)
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page, quiet_ms=400)
    hide_badge(page)

    page.get_by_text("Fit to a measured device", exact=False).first.click()
    settle(page, quiet_ms=400)
    page.locator('[data-testid="stMain"] input[type="file"]').first.set_input_files(
        str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=700)
    hide_badge(page)

    mpl = page.get_by_text("Smith Chart (Matplotlib)", exact=False).first
    mpl.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    mpl.click()
    settle(page, quiet_ms=600)
    hide_badge(page)
    shot("00_panel_top")

    page.mouse.wheel(0, 850)
    settle(page, quiet_ms=350)
    hide_badge(page)
    shot("01_panel_mid")

    page.mouse.wheel(0, 850)
    settle(page, quiet_ms=350)
    hide_badge(page)
    shot("02_panel_low")
