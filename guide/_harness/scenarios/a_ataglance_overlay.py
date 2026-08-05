"""RF At a Glance: pre-upload state, then Overlay tab (Bode + Plateau).

Uses a single DUT file. Streamlit mounts every st.tabs() branch at once
(including Batch De-embed's per-file sub-tabs), and each Scattergl trace
group opens its own WebGL context; headless Chromium's context cap (~16)
gets exceeded with more than ~1-2 uploaded files, silently blanking the
*first* chart on the page (Overlay's Bode plot) as its context is evicted.
One file keeps the total comfortably under the cap.
"""
import glob
import time

ENTRY = "IOED_Tool_Web.py"
FILES = sorted(glob.glob("/tmp/app/guide/_data/rf/raw/*.s2p"))[1:2]


def run(page, shot):
    from session import settle

    t0 = time.time()

    def mark(label):
        print(f"{label}: {time.time() - t0:.1f}s", flush=True)

    page.get_by_role("link", name="RF At a Glance").first.click()
    page.wait_for_selector('input[type="file"]')
    settle(page)
    shot("00_before_upload", full_page=True)
    mark("before upload")

    page.locator('input[type="file"]').first.set_input_files(FILES)
    settle(page)
    mark(f"uploaded {len(FILES)} file(s)")

    chart = page.locator('[data-testid="stPlotlyChart"]').first
    chart.scroll_into_view_if_needed()
    page.wait_for_timeout(1200)
    shot("01_overlay_bode", full_page=False)
    mark("bode overlay shot")

    page.locator('[data-testid="stPlotlyChart"]').nth(1).scroll_into_view_if_needed()
    page.wait_for_timeout(1000)
    shot("02_overlay_plateau", full_page=False)
    mark("plateau overlay shot")
