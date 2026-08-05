"""Probe scenario: validate locators for the SSM extraction page before
writing the real capture scenarios.  Not part of the shipped guide."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"

DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page)
    shot("00_landed", full_page=True)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    box = up.bounding_box()
    print("MAIN UPLOAD BOX", box)

    # Sidebar toggle
    tog = page.get_by_text("Enable device-dummy de-embedding", exact=True)
    print("TOGGLE COUNT", tog.count())
    tog.first.click()
    settle(page)
    shot("01_sidebar_toggled", full_page=True)

    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    print("SIDEBAR FILE INPUTS", sb_inputs.count())

    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page)
    shot("02_sidebar_files", full_page=True)

    up.set_input_files(str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page)
    shot("03_main_uploaded", full_page=True)

    run_btn = page.get_by_role("button", name="▶ Run SSM Extraction")
    print("RUN BTN COUNT", run_btn.count())
