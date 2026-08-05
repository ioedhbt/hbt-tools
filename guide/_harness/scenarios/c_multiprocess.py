"""Measurement Data Multi-Process — B1500A Smart Batch Tool: upload from the
real dc_data/raw folder, batch-convert, and hand off to DC Analysis."""
from pathlib import Path

ENTRY = "IOED_Tool_Web.py"

_RAW = Path(__file__).resolve().parents[3] / "dc_data" / "raw"
FILES = [
    str(_RAW / "BC diode [20x20 CME(1) ; 3_12_2026 8_08_42 PM].csv"),
    str(_RAW / "BE diode [20x20 CME(1) ; 3_12_2026 7_54_24 PM].csv"),
    str(_RAW / "Family [20x20 CME(1) ; 3_12_2026 7_57_17 PM].csv"),
    str(_RAW / "Gummel [20x20 CME(1) ; 3_13_2026 5_12_44 PM].csv"),
    str(_RAW / "TEST Family [5x10 CME left(1) ; 3_13_2026 11_40_35 AM] - Copy.csv"),
]


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="Measurement Data Multi-Process").first.click()
    page.wait_for_selector('input[type="file"]')
    settle(page)
    shot("01_landing", full_page=True)

    up = page.locator('input[type="file"]').first
    uploader_widget = page.get_by_test_id("stFileUploader").first
    print("uploader widget box:", uploader_widget.bounding_box(), flush=True)
    label_box = page.get_by_text("Upload B1500A CSV files").bounding_box()
    print("uploader label box:", label_box, flush=True)

    up.set_input_files(FILES)
    settle(page)
    shot("02_processed", full_page=True)
    print("uploader widget box (filled):", uploader_widget.bounding_box(), flush=True)

    dl_btn = page.get_by_role("button", name="Download each data as an Excel file")
    print("dl1 box:", dl_btn.first.bounding_box(), flush=True)
    dl_btn2 = page.get_by_role("button", name="Download grouped files by measurement type")
    print("dl2 box:", dl_btn2.first.bounding_box(), flush=True)
    handoff_btn = page.get_by_role("button", name="Analyze Data in DC Analysis")
    print("handoff box:", handoff_btn.first.bounding_box(), flush=True)

    skip_cap = page.get_by_text("skipped", exact=False)
    if skip_cap.count():
        print("skip caption box:", skip_cap.first.bounding_box(), flush=True)
