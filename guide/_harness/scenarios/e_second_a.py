"""second-exposure.md part A — layer pick, Custom cross preset, Existing
Pattern (marks layer + target position).

Shots:
  00_layer     — cell/layer selectors, L2/D1 (bridges) selected.
  01_custom    — Cross-position preset = Custom, M1/M2 design positions.
  02_existing  — Existing Pattern: layer = L1/D0 (marks), anchor = M1,
                 Target x/y typed to the position found on the chip.
"""
from pathlib import Path

ENTRY = "tools/ebeam/calculator.py"
DATA = Path("/tmp/app/guide/_data/gds")


def select(page, label, option, exact=True):
    widget = page.get_by_test_id("stSelectbox").filter(has_text=label)
    widget.get_by_role("combobox").scroll_into_view_if_needed(timeout=4000)
    widget.get_by_role("button", name="Open").click(timeout=4000)
    page.wait_for_timeout(200)
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


def run(page, shot):
    from session import settle

    page.locator('input[type="file"]').first.set_input_files(
        str(DATA / "second_exposure.gds"))
    settle(page, quiet_ms=300)
    select(page, "Top Cell", "TOP")
    settle(page, quiet_ms=300)
    select(page, "Layer to expose", "L2/D1", exact=False)
    settle(page, quiet_ms=300)

    cell_w = page.get_by_test_id("stSelectbox").filter(has_text="Top Cell")
    cell_w.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -60)
    shot("00_layer")

    click_radio(page, "Second Alignment")
    settle(page, quiet_ms=300)

    page.get_by_text("Cross-position preset", exact=False).first.wait_for(
        state="visible", timeout=8000)
    click_radio(page, "Custom", last=True)
    settle(page, quiet_ms=300)

    fill_card(page, "Mark M1", "0.100", "0.100")
    settle(page, quiet_ms=300)
    fill_card(page, "Mark M2", "1.900", "1.900")
    settle(page, quiet_ms=300)

    heading = page.get_by_text("Cross-position preset", exact=False).first
    heading.scroll_into_view_if_needed(timeout=4000)
    page.mouse.wheel(0, -60)
    shot("01_custom")

    select(page, "Existing pattern layer", "L1/D0", exact=False)
    settle(page, quiet_ms=300)

    fill_labeled(page, "Target x", "9.8010")
    settle(page, quiet_ms=300)
    fill_labeled(page, "Target y", "9.7995")
    settle(page, quiet_ms=300)

    ep_heading = page.get_by_text("Existing Pattern on Chip", exact=False).first
    ep_heading.scroll_into_view_if_needed(timeout=4000)
    shot("02_existing")
