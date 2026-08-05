"""dose-test.md — Dose Time Testing workflow with dose_test.gds.

Shots:
  00_layer     — cell/layer selectors, L1/D1 (pads) selected.
  01_cel       — Cel Origin (job1) x/y typed to bring the block array into
                 the single grid box.
  02_grids     — Single Grid + Chip Position with Grids plots (side by side).
  03_dose_in   — Time Calculator dose-ramp inputs before Calculate.
  04_result    — Breakdown + result plot after Calculate.
"""
from pathlib import Path

ENTRY = "tools/ebeam/calculator.py"
DATA = Path("/tmp/app/guide/_data/gds")


def select(page, label, option, exact=True):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(250)
    page.get_by_role("option", name=option, exact=exact).first.click(timeout=4000)


def click_mode(page, label):
    """Segmented-control mode button: try common ARIA roles, else the
    text node's nearest clickable ancestor."""
    for role in ("radio", "tab", "button"):
        loc = page.get_by_role(role, name=label)
        if loc.count() > 0:
            print("MODE ROLE", role, loc.count())
            loc.first.scroll_into_view_if_needed(timeout=4000)
            loc.first.click(timeout=5000)
            return
    print("MODE: no role match, falling back to ancestor click")
    txt = page.get_by_text(label, exact=True).first
    txt.scroll_into_view_if_needed(timeout=4000)
    anc = txt.locator("xpath=ancestor::*[self::button or self::label][1]")
    if anc.count() > 0:
        anc.first.click(timeout=5000)
    else:
        bb = txt.bounding_box()
        page.mouse.click(bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2)


def number_inputs_after(page, label_text, n=2):
    anchor = page.get_by_text(label_text, exact=False).first
    return anchor.locator("xpath=following::input[@type='number']")


def run(page, shot):
    from session import settle

    page.locator('input[type="file"]').first.set_input_files(
        str(DATA / "dose_test.gds"))
    settle(page)
    select(page, "Top Cell", "TOP")
    settle(page)
    select(page, "Layer to expose", "L1/D1", exact=False)
    settle(page)

    cell_w = page.get_by_test_id("stSelectbox").filter(has_text="Top Cell")
    cell_w.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -60)
    print("CELL BOX", cell_w.bounding_box())
    shot("00_layer")

    click_mode(page, "Dose Time Testing")
    settle(page)

    try:
        page.get_by_text("Cel Origin (mm) in job1", exact=False).first.wait_for(
            state="visible", timeout=8000)
        print("MODE SWITCHED: ok")
    except Exception as exc:
        print("MODE SWITCH FAILED", exc)
        raise

    cel_inputs = number_inputs_after(page, "Cel Origin (mm) in job1")
    print("N CEL INPUTS (following)", cel_inputs.count())
    cel_x = cel_inputs.nth(0)
    cel_y = cel_inputs.nth(1)
    cel_x.scroll_into_view_if_needed(timeout=4000)
    cel_x.fill("8.610")
    cel_x.blur()
    settle(page)
    cel_y.fill("9.710")
    cel_y.blur()
    settle(page)
    print("CEL X BOX", cel_x.bounding_box())
    print("CEL Y BOX", cel_y.bounding_box())
    shot("01_cel")

    plots = page.get_by_test_id("stPlotlyChart")
    print("N PLOTS", plots.count())
    plots.nth(1).scroll_into_view_if_needed(timeout=5000)
    settle(page)
    shot("02_grids")

    dose_init = page.get_by_test_id("stNumberInput").filter(
        has_text="Initial dose")
    dose_init.scroll_into_view_if_needed(timeout=4000)
    print("DOSE INIT BOX", dose_init.bounding_box())
    shot("03_dose_in")

    page.get_by_role("button", name="Calculate time").click(timeout=5000)
    settle(page)
    dose_init.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -40)
    shot("04_result")

    est = page.get_by_text("Estimated Time", exact=False).first
    est.scroll_into_view_if_needed(timeout=5000)
    page.mouse.wheel(0, -100)
    settle(page)
    shot("05_result_full")
