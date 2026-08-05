"""Batch De-embed: enable Dev Dummy Open/Short, upload the demo cal + one DUT,
switch to the Batch De-embed tab, and capture the extracted-parameter fields
plus the cap/ind preview plots."""
import time

ENTRY = "IOED_Tool_Web.py"
OPEN_F = "/tmp/app/guide/_data/rf/open.s2p"
SHORT_F = "/tmp/app/guide/_data/rf/short.s2p"
DUT_F = "/tmp/app/guide/_data/rf/raw/deemb_preext_vce3.5_ib280u.s2p"


def run(page, shot):
    from session import settle

    t0 = time.time()

    def mark(label):
        print(f"{label}: {time.time() - t0:.1f}s", flush=True)

    page.get_by_role("link", name="RF At a Glance").first.click()
    page.wait_for_selector('input[type="file"]')
    settle(page)

    # Enable "② Device Dummy (Open-Short)" in the sidebar.
    page.get_by_text("Device Dummy (Open-Short)", exact=False).first.click()
    settle(page)
    mark("toggle enabled")

    dev_open = page.locator('[data-testid="stFileUploader"]:has-text("Dev Open") input[type="file"]').first
    dev_short = page.locator('[data-testid="stFileUploader"]:has-text("Dev Short") input[type="file"]').first
    dev_open.set_input_files(OPEN_F)
    settle(page)
    dev_short.set_input_files(SHORT_F)
    settle(page)
    mark("open/short uploaded")

    shot("00_sidebar_calibration", full_page=True)

    dut_input = page.locator('[data-testid="stFileUploader"]:has-text("Upload DUT") input[type="file"]').first
    dut_input.set_input_files(DUT_F)
    settle(page)
    mark("dut uploaded")

    page.get_by_role("tab", name="Batch De-embed").first.click()
    settle(page)
    page.wait_for_timeout(800)
    shot("01_batch_top", full_page=True)
    mark("batch top shot")

    # Scroll to the override fields (Cpbe/Cpce/Cpbc/Lb/Lc/Le).
    try:
        page.get_by_text("Overrides", exact=False).first.scroll_into_view_if_needed(timeout=5000)
        page.wait_for_timeout(500)
        shot("02_overrides", full_page=True)
        mark("overrides shot")
    except Exception as e:
        print("overrides scroll failed:", str(e)[:200], flush=True)
