"""HP4155A Quick Plot — Gummel file (converted from the curated B1500A
Gummel demo, see guide/_data/dc/hp4155a/VERIFY.md).
SMU mapping for this file: V1/I1=Base, V2/I2=Collector, V3/I3=Emitter."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"

_FILE = str(Path(__file__).resolve().parents[3] / "guide" / "_data" / "dc" / "hp4155a" / "gummel_60x60_hp4155a.txt")


def set_role(page, pair_label, role):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=pair_label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(250)
    page.get_by_role("option", name=role, exact=True).first.click(timeout=4000)


def select(page, label, option):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(250)
    page.get_by_role("option", name=option, exact=False).first.click(timeout=4000)


def hide_badge(page):
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

    page.get_by_role("link", name="HP4155A Quick Plot").first.click(timeout=5000)
    settle(page)
    hide_badge(page)

    shot("00_landed")

    print("smu heading box:", page.get_by_role("heading", name="SMU Assignment").bounding_box(timeout=4000), flush=True)
    print("datafile heading box:", page.get_by_role("heading", name="Data File").bounding_box(timeout=4000), flush=True)
    print("upload widget box:", page.get_by_test_id("stFileUploader").first.bounding_box(timeout=4000), flush=True)

    set_role(page, "V1/I1", "Base")
    set_role(page, "V2/I2", "Collector")
    set_role(page, "V3/I3", "Emitter")
    settle(page)
    hide_badge(page)
    shot("01_smu_assigned")

    page.locator('input[type="file"]').first.set_input_files(_FILE)
    settle(page)
    hide_badge(page)
    print("upload widget box (loaded):", page.get_by_test_id("stFileUploader").first.bounding_box(timeout=4000), flush=True)
    print("loaded banner box:", page.get_by_text("Loaded:", exact=False).bounding_box(timeout=4000), flush=True)
    shot("02_file_loaded")

    select(page, "Data type", "Gummel")
    settle(page)
    hide_badge(page)

    page.get_by_role("button", name="Show", exact=False).click(timeout=5000)
    settle(page)
    hide_badge(page)

    shot("03_gummel_result")

    plotchart = page.get_by_test_id("stPlotlyChart").first
    print("plot box:", plotchart.bounding_box(timeout=4000), flush=True)
    shot("04_gummel_plot", locator=plotchart)

    dl_row = page.get_by_text("Download Excel", exact=False).first.locator(
        "xpath=ancestor::div[@data-testid='stHorizontalBlock'][1]")
    print("dl row box:", dl_row.bounding_box(timeout=4000), flush=True)
    shot("05_gummel_export", locator=dl_row)
