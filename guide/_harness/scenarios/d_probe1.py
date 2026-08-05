"""Probe: land on Simulation & Fitting, dump structure."""
ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)
    shot("00_landed", full_page=True)
    print("DONE")
