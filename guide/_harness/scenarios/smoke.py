"""Prove the harness works: land on the portal, open each RF page once."""

ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    shot("00_home", full_page=True)
    page.get_by_role("link", name="RF At a Glance").first.click()
    shot("01_at_a_glance", full_page=True)
