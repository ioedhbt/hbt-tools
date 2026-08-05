"""ssm-setup.md — upload widgets before running extraction.

Shots:
  00_main_upload   — the DUT bias-file uploader, empty.
  01_sidebar_off   — sidebar "Device Dummy (Open-Short)" toggle, off.
  02_sidebar_files — Open + Short dummy files loaded in the sidebar.
  03_ready_to_run  — DUT file loaded, Run button visible.
"""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"
DATA = Path("/tmp/app/guide/_data/rf")


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="Small Signal Model Extraction by Peeling").first.click()
    settle(page)

    up = page.locator('[data-testid="stMain"] input[type="file"]').first
    box = up.locator("xpath=ancestor::div[@data-testid='stFileUploader'][1]").bounding_box()
    print("MAIN UPLOAD BOX", box)
    shot("00_main_upload")

    tog = page.get_by_text("Enable device-dummy de-embedding", exact=True).first
    tb = tog.bounding_box()
    print("TOGGLE BOX", tb)
    shot("01_sidebar_off")

    tog.click()
    settle(page)
    sb_inputs = page.locator('[data-testid="stSidebar"] input[type="file"]')
    sb_inputs.nth(0).set_input_files(str(DATA / "open.s2p"))
    settle(page)
    sb_inputs.nth(1).set_input_files(str(DATA / "short.s2p"))
    settle(page)
    shot("02_sidebar_files")

    up.set_input_files(str(DATA / "raw" / "deemb_preext_vce3.5_ib200u.s2p"))
    settle(page)
    run_btn = page.get_by_role("button", name="▶ Run SSM Extraction")
    rb = run_btn.bounding_box()
    print("RUN BTN BOX", rb)
    shot("03_ready_to_run")
