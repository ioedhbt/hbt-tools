"""Probe: does a Scattergl chart inside a SINGLE (not double-nested)
expander render on this page (Short dummy Lb/Lc/Le plot)?"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1400})
    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page, quiet_ms=300)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page, quiet_ms=300)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page, quiet_ms=300)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page, quiet_ms=300)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=400)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page, quiet_ms=500)

    page.get_by_text("📌 Open & Short Dummy De-embedding", exact=False).first.click()
    settle(page, quiet_ms=500)

    chart_heading = page.get_by_text("Short — Lead Inductances vs Frequency", exact=False).first
    print("HEADING COUNT", chart_heading.count())
    chart_heading.evaluate("el => el.scrollIntoView({block:'start', behavior:'instant'})")
    settle(page, quiet_ms=500)
    shot("00_short_plot", full_page=True)
