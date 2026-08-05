"""B1500A Viewer — Gummel curve, reached via the Multi-Process hand-off."""
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
    page.get_by_role("option", name=option, exact=False).first.click(timeout=4000)


def hide_badge(page):
    # The RAM/language badge is position:fixed top-right, so it bleeds into
    # any locator screenshot whose bounding box scrolls under it. Direct
    # page.evaluate (not add_init_script/DOMContentLoaded) so it applies
    # deterministically regardless of whether the last transition was a
    # hard navigation or a client-side Streamlit rerun.
    page.evaluate(
        "() => {"
        "let s = document.getElementById('_hide_badge');"
        "if (!s) { s = document.createElement('style'); s.id = '_hide_badge';"
        "document.head.appendChild(s); }"
        "s.textContent = 'div.st-key-lang_toggle{display:none !important;}';"
        "}"
    )


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="Measurement Data Multi-Process").first.click()
    page.wait_for_selector('input[type="file"]')
    settle(page)
    page.locator('input[type="file"]').first.set_input_files(FILES)
    settle(page)
    page.get_by_role("button", name="Analyze Data in DC Analysis").click()
    settle(page)
    hide_badge(page)

    select(page, "Select file", "Gummel.xlsx")
    settle(page)
    hide_badge(page)

    def box(locator, label):
        try:
            b = locator.bounding_box(timeout=4000)
        except Exception as e:
            b = f"ERROR: {e}"
        print(f"{label}:", b, flush=True)

    eta = page.get_by_text("Ideality factors", exact=False).first.locator(
        "xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    beta = page.get_by_text("Current gain", exact=False).first.locator(
        "xpath=ancestor::div[@data-testid='stLayoutWrapper'][1]")
    plotchart = page.get_by_test_id("stPlotlyChart").first

    box(eta, "eta container")
    box(beta, "beta container")
    box(plotchart, "plot")

    shot("01_gummel_eta", locator=eta)
    shot("02_gummel_beta", locator=beta)
    shot("03_gummel_plot", locator=plotchart)
