"""Portal orientation captures: home cards, sidebar collapse, language toggle, RAM badge."""
import time

ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    t0 = time.time()

    def mark(label):
        print(f"{label}: {time.time() - t0:.1f}s", flush=True)

    shot("00_home", full_page=True)
    mark("home")

    # Sidebar collapse control — Streamlit's "«" button at the top of the
    # sidebar.  Print bounding boxes so annotate.py can circle them exactly.
    try:
        collapse_btn = page.locator('[data-testid="stSidebarCollapseButton"] button').first
        bb = collapse_btn.bounding_box()
        print("collapse_btn bbox:", bb, flush=True)
    except Exception as e:
        print("collapse_btn lookup failed:", e, flush=True)

    # Language toggle + RAM badge, top-right fixed strip.
    try:
        lang_box = page.locator('div.st-key-lang_toggle').first
        bb = lang_box.bounding_box()
        print("lang_toggle bbox:", bb, flush=True)
    except Exception as e:
        print("lang_toggle lookup failed:", e, flush=True)

    # Sidebar nav column bbox (for the "side panel" crop).
    try:
        sidebar = page.locator('[data-testid="stSidebar"]').first
        bb = sidebar.bounding_box()
        print("sidebar bbox:", bb, flush=True)
    except Exception as e:
        print("sidebar lookup failed:", e, flush=True)

    mark("bboxes printed")

    # Collapse the sidebar, screenshot the resulting hamburger control.
    try:
        page.locator('[data-testid="stSidebarCollapseButton"] button').first.click()
        settle(page)
        shot("01_sidebar_collapsed", full_page=True)
        mark("collapsed")
    except Exception as e:
        print("collapse step failed:", e, flush=True)

    for sel in ['[data-testid="stSidebarCollapsedControl"] button',
                '[data-testid="stSidebarCollapsedControl"]',
                'button:has-text("»")']:
        try:
            loc = page.locator(sel).first
            bb = loc.bounding_box(timeout=2000)
            print(f"opener bbox [{sel}]:", bb, flush=True)
        except Exception as e:
            print(f"opener lookup failed [{sel}]:", str(e)[:120], flush=True)
