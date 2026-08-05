"""B1500A Viewer — BC and BE diode curves, reached via the Multi-Process
hand-off (the four curated guide/_data/dc files, uploaded in an order that
lands on BC diode by default)."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"

_DC = Path(__file__).resolve().parents[3] / "guide" / "_data" / "dc"
FILES = [
    str(_DC / "BC diode [40x40 CME(1) ; 3_13_2026 4_08_50 PM].csv"),
    str(_DC / "BE diode [60x60 CME(1) _1V_; 3_12_2026 7_46_17 PM].csv"),
    str(_DC / "Family [4x10 CME(4) ; 3_12_2026 8_45_16 PM].csv"),
    str(_DC / "Gummel [60x60 CME(1) _1V_; 3_12_2026 7_43_08 PM].csv"),
]


def select(page, label, option):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(300)
    opts = page.get_by_role("option")
    print("n options:", opts.count(), flush=True)
    for i in range(opts.count()):
        print("  opt", i, opts.nth(i).inner_text(), flush=True)
    page.get_by_role("option", name=option, exact=False).first.click(timeout=4000)


def run(page, shot):
    from session import settle

    # The RAM/language badge is position:fixed top-right, so it bleeds into
    # any locator screenshot whose bounding box happens to scroll under it.
    # add_init_script (not add_style_tag) so it survives the full-page
    # navigations st.switch_page / sidebar links trigger.
    page.add_init_script(
        "document.addEventListener('DOMContentLoaded',()=>{"
        "const s=document.createElement('style');"
        "s.textContent='div.st-key-lang_toggle{display:none !important;}';"
        "document.head.appendChild(s);});"
    )

    page.get_by_role("link", name="Measurement Data Multi-Process").first.click()
    page.wait_for_selector('input[type="file"]')
    settle(page)
    page.locator('input[type="file"]').first.set_input_files(FILES)
    settle(page)
    page.get_by_role("button", name="Analyze Data in DC Analysis").click()
    settle(page)

    # Lands on B1500A Viewer, default source = BC_Diode.xlsx (Diode dtype).
    shot("00_landed")

    print("received banner box:", page.get_by_text("Received 4 file", exact=False).bounding_box(timeout=4000), flush=True)
    print("step2 heading box:", page.get_by_text("Choose sheet", exact=False).bounding_box(timeout=4000), flush=True)

    def box(locator, label):
        try:
            b = locator.bounding_box(timeout=4000)
        except Exception as e:
            b = f"ERROR: {e}"
        print(f"{label}:", b, flush=True)

    box(page.get_by_test_id("stSelectbox").filter(has_text="Select file"), "file selectbox box")
    box(page.get_by_test_id("stSelectbox").filter(has_text="Data type"), "dtype selectbox box")

    params = page.get_by_text("Extracted parameters", exact=True).locator(
        "xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    box(params, "params container box (BC)")
    plotchart = page.get_by_test_id("stPlotlyChart").first
    box(plotchart, "plot box (BC)")

    shot("01_bc_params", locator=params)
    shot("02_bc_plot", locator=plotchart)

    select(page, "Select file", "BE_Diode.xlsx")
    settle(page)
    params2 = page.get_by_text("Extracted parameters", exact=True).locator(
        "xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    plotchart2 = page.get_by_test_id("stPlotlyChart").first
    box(params2, "params container box (BE)")
    shot("03_be_params", locator=params2)
    shot("04_be_plot", locator=plotchart2)
