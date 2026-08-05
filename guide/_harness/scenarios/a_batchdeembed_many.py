"""Batch De-embed with all 7 demo DUT files against one Open/Short pair."""
import glob
import time

ENTRY = "IOED_Tool_Web.py"
OPEN_F = "/tmp/app/guide/_data/rf/open.s2p"
SHORT_F = "/tmp/app/guide/_data/rf/short.s2p"
DUTS = sorted(glob.glob("/tmp/app/guide/_data/rf/raw/*.s2p"))


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
    dut_input.set_input_files(DUTS)
    settle(page)
    mark(f"setup done, {len(DUTS)} duts")

    page.get_by_role("tab", name="Batch De-embed").first.click()
    settle(page)
    page.wait_for_timeout(1000)

    page.get_by_text("Per-file Results", exact=False).first.scroll_into_view_if_needed()
    page.wait_for_timeout(1200)
    shot("01_per_file_tabs", full_page=False)
    mark("per-file tabs shot")

    page.get_by_text("Download de-embedded measurement files", exact=False).first.scroll_into_view_if_needed()
    page.wait_for_timeout(600)
    shot("02_zip_download", full_page=True)
    mark("zip download shot")
