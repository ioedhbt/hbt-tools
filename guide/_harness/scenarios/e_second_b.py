"""second-exposure.md part B — replay setup fast, then Second Alignment
Pattern (fix Ny so both marks land inside the grid), Overlayed, Time Calc.

Shots:
  00_sap_warn  — SAP grid auto-fit to the pattern only (4x2): M2 falls
                 outside it, red banner shown.
  01_sap_fixed — Ny bumped to 4 by hand: banner gone, registration mark
                 position panel visible.
  02_overlay   — Overlayed: Mark 1/2 pickers, auto Shift x/y, final plot.
  03_time_in   — Time Calculator inputs before Calculate.
  04_result    — Breakdown + Estimated Time after Calculate.
"""
from pathlib import Path

ENTRY = "tools/process/ebeam/calculator.py"
DATA = Path("/tmp/app/guide/_data/gds")


def select(page, label, option, exact=True):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(150)
    page.get_by_role("option", name=option, exact=exact).first.click(timeout=4000)


def click_radio(page, label, last=False):
    loc = page.get_by_role("radio", name=label, exact=True)
    idx = loc.last if last else loc.first
    idx.scroll_into_view_if_needed(timeout=4000)
    idx.click(timeout=5000)


def fill_card(page, card_text, x_val, y_val):
    anchor = page.get_by_text(card_text, exact=False).first
    anchor.scroll_into_view_if_needed(timeout=4000)
    inputs = anchor.locator("xpath=following::input[@type='number']")
    x_in, y_in = inputs.nth(0), inputs.nth(1)
    x_in.fill(x_val)
    x_in.blur()
    y_in.fill(y_val)
    y_in.blur()


def fill_labeled(page, label, value):
    field = page.get_by_test_id("stNumberInput").filter(has_text=label)
    field.scroll_into_view_if_needed(timeout=4000)
    inp = field.locator("input")
    inp.fill(value)
    inp.blur()


def setup(page, settle):
    page.locator('input[type="file"]').first.set_input_files(
        str(DATA / "second_exposure.gds"))
    settle(page, quiet_ms=250)
    select(page, "Top Cell", "TOP")
    settle(page, quiet_ms=250)
    select(page, "Layer to expose", "L2/D1", exact=False)
    settle(page, quiet_ms=250)

    click_radio(page, "Second Alignment")
    settle(page, quiet_ms=250)
    page.get_by_text("Cross-position preset", exact=False).first.wait_for(
        state="visible", timeout=8000)
    click_radio(page, "Custom", last=True)
    settle(page, quiet_ms=250)

    fill_card(page, "Mark M1", "0.100", "0.100")
    settle(page, quiet_ms=250)
    fill_card(page, "Mark M2", "1.900", "1.900")
    settle(page, quiet_ms=250)

    select(page, "Existing pattern layer", "L1/D0", exact=False)
    settle(page, quiet_ms=250)
    fill_labeled(page, "Target x", "9.8010")
    settle(page, quiet_ms=250)
    fill_labeled(page, "Target y", "9.7995")
    settle(page, quiet_ms=250)


def run(page, shot):
    from session import settle

    setup(page, settle)

    sap_heading = page.get_by_text("Second Alignment Pattern", exact=False).first
    sap_heading.scroll_into_view_if_needed(timeout=5000)
    settle(page, quiet_ms=250)
    shot("00_sap_warn")

    grid_count = page.get_by_text("Grid Count in job1", exact=False).first
    ny_input = grid_count.locator(
        "xpath=following::input[@type='number']").nth(1)
    ny_input.fill("4")
    ny_input.blur()
    settle(page, quiet_ms=250)
    sap_heading.scroll_into_view_if_needed(timeout=5000)
    shot("01_sap_fixed")

    ov_heading = page.get_by_text("Overlayed", exact=False).first
    ov_heading.scroll_into_view_if_needed(timeout=5000)
    settle(page, quiet_ms=250)
    shot("02_overlay")

    dose_field = page.get_by_test_id("stNumberInput").filter(
        has_text="Dose time")
    dose_field.scroll_into_view_if_needed(timeout=5000)
    settle(page, quiet_ms=250)
    shot("03_time_in")

    page.get_by_role("button", name="Calculate time").click(timeout=5000)
    settle(page, quiet_ms=300)
    est = page.get_by_text("Estimated Time", exact=False).first
    est.scroll_into_view_if_needed(timeout=5000)
    page.mouse.wheel(0, -100)
    settle(page, quiet_ms=300)
    shot("04_result")
