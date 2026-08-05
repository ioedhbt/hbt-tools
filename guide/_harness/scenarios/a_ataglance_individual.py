"""RF At a Glance: Individual tab — metric cards, Bode export row, Smith tab, data table."""
import glob
import time

ENTRY = "IOED_Tool_Web.py"
FILES = sorted(glob.glob("/tmp/app/guide/_data/rf/raw/*.s2p"))


def run(page, shot):
    from session import settle

    t0 = time.time()

    def mark(label):
        print(f"{label}: {time.time() - t0:.1f}s", flush=True)

    page.get_by_role("link", name="RF At a Glance").first.click()
    page.wait_for_selector('input[type="file"]')
    page.locator('input[type="file"]').first.set_input_files(FILES)
    settle(page)
    mark(f"uploaded {len(FILES)} files")

    page.get_by_role("tab", name="Individual").first.click()
    settle(page)

    # Switch the Active-file selectbox to the highest-fT device.
    try:
        combo = page.get_by_test_id("stSelectbox").first.get_by_role("combobox")
        combo.click(timeout=4000)
        mark("combo opened")
    except Exception as e:
        print("combo click failed:", str(e)[:200], flush=True)
    try:
        page.get_by_role("option", name="deemb_preext_vce3.5_ib280u").first.click(timeout=4000)
        mark("option clicked")
        settle(page)
    except Exception as e:
        print("option click failed:", str(e)[:200], flush=True)

    shot("01_individual_cards", full_page=True)
    mark("cards shot")

    # Scroll to the Bode chart's export row (xlsx download + copy button).
    page.mouse.wheel(0, 1400)
    page.wait_for_timeout(500)
    shot("02_bode_export", full_page=True)
    mark("export row shot")

    # Smith chart sub-tab.
    try:
        page.get_by_role("tab", name="Smith Chart").first.click()
        settle(page)
        shot("03_smith_tab", full_page=True)
        mark("smith shot")
    except Exception as e:
        print("smith tab failed:", str(e)[:200], flush=True)
