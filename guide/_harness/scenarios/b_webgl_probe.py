"""Probe: does a simple go.Scattergl chart render its data in this sandbox?"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

    page.set_viewport_size({"width": 1600, "height": 1400})
    page.get_by_role("link", name="RF At a Glance").first.click()
    settle(page)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page, quiet_ms=800)
    shot("00_overlay", full_page=True)
