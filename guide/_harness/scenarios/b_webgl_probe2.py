"""Probe: does the rz12 Scattergl chart render when it's the FIRST/only
plotly chart touched on the page (minimal prior context creation)?"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")

BIAS_FILES = [
    "deemb_preext_vce3.5_ib80u.s2p",
    "deemb_preext_vce3.5_ib280u.s2p",
]


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1200})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=300)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files([str(DATA / "raw" / f) for f in BIAS_FILES])
    settle(page, quiet_ms=400)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=500)

    page.get_by_text("🍊 Access Resistance Extraction", exact=False).first.click()
    page.get_by_text("Z-Parameter Method — Re(Z₁₂) vs 1/IE", exact=False).first.click()
    settle(page, quiet_ms=400)

    for stem, ie in [("deemb_preext_vce3.5_ib80u", 1.68),
                     ("deemb_preext_vce3.5_ib280u", 8.298)]:
        row = page.get_by_text(stem, exact=False).first \
            .locator("xpath=ancestor::div[contains(@data-testid,'stHorizontalBlock')][1]")
        inp = row.locator('input').last
        inp.fill(str(ie))
        inp.press("Tab")
    settle(page, quiet_ms=600)

    anchor = page.get_by_text("Z-Parameter Method — Re(Z₁₂) vs 1/IE", exact=False).first
    anchor.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=400)
    shot("00_chart", full_page=True)
