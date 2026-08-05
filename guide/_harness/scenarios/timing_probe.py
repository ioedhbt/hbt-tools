"""Measure how much of a real workflow fits inside one 45 s bash call."""
import glob
import time

ENTRY = "IOED_Tool_Web.py"
FILES = sorted(glob.glob("/tmp/app/deembed_these/deembedded/4x10_left/*.s2p"))


def run(page, shot):
    t0 = time.time()

    def mark(label):
        print(f"{label}: {time.time() - t0:.1f}s", flush=True)

    mark("page ready")
    page.get_by_role("link", name="RF At a Glance").first.click()
    page.wait_for_selector('input[type="file"]')
    mark("at-a-glance open")

    page.locator('input[type="file"]').first.set_input_files(FILES)
    from session import settle

    settle(page)
    mark(f"uploaded {len(FILES)} files")
    shot("01_uploaded", full_page=True)
    mark("shot 1")

    page.get_by_role("tab", name="Summary").click()
    settle(page)
    shot("02_summary", full_page=True)
    mark("shot 2")
