"""first-exposure.md — First Exposure workflow with first_exposure.gds.

Shots:
  00_corners   — BL/TR corner fields typed, TL/BR shown disabled/computed.
  01_layer     — cell/layer selectors, L1/D2 (emitters) selected.
  02_fe_inputs — Cel Origin / Grid Count / Shift cards (First Exposure mode).
  03_plot      — chip + grid + mask plot.
  04_time_in   — Time Calculator inputs before Calculate.
  05_result    — Breakdown + Estimated Time after Calculate.
"""
from pathlib import Path

ENTRY = "tools/process/ebeam/calculator.py"
DATA = Path("/tmp/app/guide/_data/gds")


def select(page, label, option, exact=True):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(250)
    page.get_by_role("option", name=option, exact=exact).first.click(timeout=4000)


def click_mode(page, label):
    loc = page.get_by_role("radio", name=label)
    loc.first.scroll_into_view_if_needed(timeout=4000)
    loc.first.click(timeout=5000)


def fill_corner(page, card_text, x_val, y_val):
    anchor = page.get_by_text(card_text, exact=False).first
    anchor.scroll_into_view_if_needed(timeout=4000)
    inputs = anchor.locator("xpath=following::input[@type='number']")
    x_in, y_in = inputs.nth(0), inputs.nth(1)
    x_in.fill(x_val)
    x_in.blur()
    y_in.fill(y_val)
    y_in.blur()


def run(page, shot):
    from session import settle

    fill_corner(page, "Bottom Left (BL)", "105.000", "115.000")
    settle(page)
    fill_corner(page, "Top Right (TR)", "107.000", "117.000")
    settle(page)

    heading = page.get_by_role("heading", name="Corner Positions")
    heading.scroll_into_view_if_needed(timeout=4000)
    shot("00_corners")

    page.locator('input[type="file"]').first.set_input_files(
        str(DATA / "first_exposure.gds"))
    settle(page)
    select(page, "Top Cell", "TOP")
    settle(page)
    select(page, "Layer to expose", "L1/D2", exact=False)
    settle(page)

    cell_w = page.get_by_test_id("stSelectbox").filter(has_text="Top Cell")
    cell_w.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -60)
    shot("01_layer")

    click_mode(page, "First Exposure")
    settle(page)

    page.get_by_text("Cel Origin (mm) in job1", exact=False).first.wait_for(
        state="visible", timeout=8000)
    anchor = page.get_by_text("Cel Origin (mm) in job1", exact=False).first
    anchor.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -60)
    settle(page)
    shot("02_fe_inputs")

    plots = page.get_by_test_id("stPlotlyChart")
    print("N PLOTS", plots.count())
    plot = plots.nth(2)
    plot.scroll_into_view_if_needed(timeout=5000)
    settle(page)
    shot("03_plot", locator=plot)

    dose_in = page.get_by_test_id("stNumberInput").filter(has_text="Dose time")
    dose_in.scroll_into_view_if_needed(timeout=4000)
    settle(page)
    shot("04_time_in")

    page.get_by_role("button", name="Calculate time").click(timeout=5000)
    settle(page)
    est = page.get_by_text("Estimated Time", exact=False).first
    est.scroll_into_view_if_needed(timeout=5000)
    page.mouse.wheel(0, -100)
    settle(page)
    shot("05_result")
