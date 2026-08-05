"""Bounding boxes for Individual-tab and Summary-tab elements, with data loaded."""
import glob

ENTRY = "IOED_Tool_Web.py"
FILES = sorted(glob.glob("/tmp/app/guide/_data/rf/raw/*.s2p"))[1:2]


def run(page, shot):
    from session import settle

    def bb(label, loc):
        try:
            print(label, loc.bounding_box(timeout=3000), flush=True)
        except Exception as e:
            print(label, "FAILED", str(e)[:100], flush=True)

    page.get_by_role("link", name="RF At a Glance").first.click()
    page.wait_for_selector('input[type="file"]')
    settle(page)
    page.locator('input[type="file"]').first.set_input_files(FILES)
    settle(page)

    page.get_by_role("tab", name="Individual").first.click()
    settle(page)

    bb("active_file_select", page.get_by_test_id("stSelectbox").first)
    bb("card_deembed", page.get_by_text("De-embedding", exact=False).first)
    bb("card_fT", page.get_by_text("fT (GHz)", exact=False).first)
    bb("card_fmaxU", page.get_by_text("fmax U", exact=False).first)
    bb("card_fmaxMAG", page.get_by_text("fmax MAG", exact=False).first)
    bb("card_kmin", page.get_by_text("K min", exact=False).first)

    page.mouse.wheel(0, 1400)
    page.wait_for_timeout(800)

    bb("xlsx_btn", page.get_by_role("button", name="xlsx", exact=False).first)
    bb("copy_btn", page.get_by_role("button", name="copy", exact=False).first)
    bb("ssm_extraction_btn", page.get_by_role("button", name="SSM Extraction", exact=False).first)
    bb("sim_fit_btn", page.get_by_role("button", name="Simulation & Fitting", exact=False).first)
    bb("data_table_exp", page.get_by_text("Data Table", exact=False).first)
    shot("00_state", full_page=False)
