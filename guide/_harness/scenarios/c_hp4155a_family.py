"""HP4155A Quick Plot — Family file (converted from the curated B1500A
Family demo, see guide/_data/dc/hp4155a/VERIFY.md).
SMU mapping for this file: V1/I1=Collector, V2/I2=Base, V3/I3=Emitter."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"

_FILE = str(Path(__file__).resolve().parents[3] / "guide" / "_data" / "dc" / "hp4155a" / "family_4x10_hp4155a.txt")


def set_role(page, pair_label, role):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=pair_label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(250)
    page.get_by_role("option", name=role, exact=True).first.click(timeout=4000)


def select(page, label, option, exact=False):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(250)
    page.get_by_role("option", name=option, exact=exact).first.click(timeout=4000)


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

    set_role(page, "V1/I1", "Collector")
    set_role(page, "V2/I2", "Base")
    set_role(page, "V3/I3", "Emitter")
    settle(page)
    hide_badge(page)

    page.locator('input[type="file"]').first.set_input_files(_FILE)
    settle(page)
    hide_badge(page)

    select(page, "Data type", "Family", exact=True)
    settle(page)
    hide_badge(page)

    page.get_by_role("button", name="Show", exact=False).click(timeout=5000)
    settle(page)
    hide_badge(page)

    plotchart = page.get_by_test_id("stPlotlyChart").first
    print("plot box:", plotchart.bounding_box(timeout=4000), flush=True)
    shot("01_family_plot", locator=plotchart)

    dtype_widget = page.get_by_test_id("stSelectbox").filter(has_text="Data type")
    dtype_widget.scroll_into_view_if_needed(timeout=4000)
    print("dtype widget box:", dtype_widget.bounding_box(timeout=4000), flush=True)
    show_btn = page.get_by_role("button", name="Show", exact=False)
    print("show btn box:", show_btn.bounding_box(timeout=4000), flush=True)
    shot("02_settings_viewport")
