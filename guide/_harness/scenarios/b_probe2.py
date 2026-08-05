"""Probe 2: timing for Run + expander interactions."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page)

    page.get_by_text("Enable device-dummy de-embedding", exact=True).first.click()
    settle(page)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    up.set_input_files(str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page)

    page.get_by_role("button", name="▶ Run SSM Extraction").click()
    settle(page)
    shot("00_after_run", full_page=True)

    exp = page.get_by_text("📌 Open & Short Dummy De-embedding", exact=False)
    print("EXP COUNT", exp.count())
    exp.first.click()
    settle(page)
    shot("01_open_short_expanded", full_page=True)
