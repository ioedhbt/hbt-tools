"""Batch De-embed: override number fields, per-file result, ZIP download, handoff."""
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

    page.get_by_text("Device Dummy (Open-Short)", exact=False).first.click()
    settle(page)

    dev_open = page.locator('[data-testid="stFileUploader"]:has-text("Dev Open") input[type="file"]').first
    dev_short = page.locator('[data-testid="stFileUploader"]:has-text("Dev Short") input[type="file"]').first
    dev_open.set_input_files(OPEN_F)
    settle(page)
    dev_short.set_input_files(SHORT_F)
    settle(page)

    dut_input = page.locator('[data-testid="stFileUploader"]:has-text("Upload DUT") input[type="file"]').first
    dut_input.set_input_files(DUT_F)
    settle(page)
    mark("setup done")

    page.get_by_role("tab", name="Batch De-embed").first.click()
    settle(page)

    # Override number fields (Cpbe/Cpce/Cpbc/Lb/Lc/Le).
    page.get_by_text("Overrides", exact=False).first.scroll_into_view_if_needed()
    page.wait_for_timeout(600)
    shot("01_override_fields", full_page=False)
    mark("override fields shot")

    # Per-file result cards + Bode/Smith for this DUT.
    page.get_by_text("Per-file Results", exact=False).first.scroll_into_view_if_needed()
    page.wait_for_timeout(1000)
    shot("02_per_file_result", full_page=False)
    mark("per-file shot")

    # ZIP download button + handoff section.
    page.get_by_text("Download de-embedded measurement files", exact=False).first.scroll_into_view_if_needed()
    page.wait_for_timeout(600)
    shot("03_download_handoff", full_page=True)
    mark("download/handoff shot")
