"""Portal language toggle: switch to Chinese and confirm the whole page follows."""
ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    page.get_by_text("中文", exact=True).first.click()
    settle(page)
    shot("00_zh", full_page=True)
