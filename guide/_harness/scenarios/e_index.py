"""index.md — EBL calculator basics: standalone launch, GDS upload, cell/layer
pick, viewer, mode selector.

Shots:
  00_standalone_header — page header + caption, standalone entry (no portal
                          chrome, no sidebar nav).
  01_upload_empty       — the .gds uploader + RAM budget caption, nothing
                          loaded yet.
  02_loaded_line        — "Loaded <name> ..." line + cell/layer selectors
                          after upload.
  03_viewer             — the polygon plot for the selected layer.
  04_mode_selector       — the Workflow mode segmented control.
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


def run(page, shot):
    from session import settle

    shot("00_standalone_header")

    uploader = page.get_by_test_id("stFileUploader").first
    uploader.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -80)  # nudge up so the RAM caption below is in frame
    print("UPLOADER BOX", uploader.bounding_box())
    shot("01_upload_empty")

    page.locator('input[type="file"]').first.set_input_files(
        str(DATA / "first_exposure.gds"))
    settle(page)

    select(page, "Top Cell", "TOP")
    settle(page)
    select(page, "Layer to expose", "L1/D1", exact=False)
    settle(page)

    uploader.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -80)
    cell_w = page.get_by_test_id("stSelectbox").filter(has_text="Top Cell")
    layer_w = page.get_by_test_id("stSelectbox").filter(has_text="Layer to expose")
    print("CELL BOX", cell_w.bounding_box())
    print("LAYER BOX", layer_w.bounding_box())
    shot("02_loaded_line")

    plots = page.get_by_test_id("stPlotlyChart")
    print("N PLOTS", plots.count())
    viewer_plot = plots.nth(1)
    viewer_plot.scroll_into_view_if_needed(timeout=5000)
    print("VIEWER PLOT BOX", viewer_plot.bounding_box(timeout=5000))
    shot("03_viewer", locator=viewer_plot)

    header = page.get_by_role("heading", name="Workflow")
    header.scroll_into_view_if_needed(timeout=4000)
    settle(page)
    print("WORKFLOW HEADER BOX", header.bounding_box())
    shot("04_mode_selector")
