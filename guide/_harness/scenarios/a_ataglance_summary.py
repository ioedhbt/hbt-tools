"""RF At a Glance: bulk upload, Summary tab, sort by fT, select row, jump to Individual."""
import glob
import time

ENTRY = "IOED_Tool_Web.py"
FILES = sorted(glob.glob("/tmp/app/guide/_data/rf/raw/*.s2p"))


def run(page, shot):
    from session import settle

    t0 = time.time()

    def mark(label):
        print(f"{label}: {time.time() - t0:.1f}s", flush=True)

    page.get_by_role("link", name="RF At a Glance").first.click()
    page.wait_for_selector('input[type="file"]')
    page.locator('input[type="file"]').first.set_input_files(FILES)
    settle(page)
    mark(f"uploaded {len(FILES)} files")

    page.get_by_role("tab", name="Summary").first.click()
    settle(page)
    shot("01_summary_table", full_page=True)
    mark("summary shot")

    df = page.locator('[data-testid="stDataFrame"]:visible').first
    bb = df.bounding_box(timeout=3000)
    print("dataframe bbox:", bb, flush=True)

    # Column edges measured empirically (device px, /2 for css) from
    # 01_summary_table.png: checkbox|File|Ib|fT Cross|fmax U Cross at
    # 40|382|219|235|264 css-px out of the 1140-css-px-wide grid.
    row_h = bb["height"] / 8.0
    header_y = bb["y"] + row_h / 2
    ft_col_x = bb["x"] + bb["width"] * (40 + 382 + 219 + 235 / 2) / 1140.0

    # Click twice: first sorts ascending, second descending (highest fT on top).
    page.mouse.click(ft_col_x, header_y)
    settle(page)
    page.mouse.click(ft_col_x, header_y)
    settle(page)
    shot("02_sorted_by_ft_desc", full_page=True)
    mark("sorted desc")

    # Select the (now top, highest-fT) row via its checkbox.
    row1_y = bb["y"] + row_h * 1.5
    chk_x = bb["x"] + bb["width"] * 20 / 1140.0
    page.mouse.click(chk_x, row1_y)
    settle(page)
    shot("03_row_selected", full_page=True)
    mark("row selected")

    try:
        btn = page.get_by_role("button", name="Individual tab").first
        btn.click()
        settle(page)
        page.wait_for_timeout(600)
        shot("04_open_in_individual", full_page=True)
        mark("opened individual")
    except Exception as e:
        print("open-in-individual failed:", str(e)[:200], flush=True)
