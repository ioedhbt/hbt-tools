"""Probe exact bounding boxes for model bar + fit expander."""
ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    page.set_default_timeout(3000)
    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)

    for name in ["Cheng's T", "Cheng's π", "Xu T", "Kun-Yang HEMT",
                 "🧩 Custom model", "Open and Short Pad"]:
        try:
            loc = page.get_by_role("radio", name=name).first
            print("RADIO", name, loc.bounding_box(timeout=3000))
        except Exception as e:
            print("RADIO", name, "ERR", type(e).__name__)

    # Try a few role guesses for the expander header.
    for role in ["button", "tab", "heading"]:
        try:
            loc = page.get_by_role(role, name="Fit to a measured device", exact=False).first
            n = loc.count()
            print("ROLE", role, "COUNT", n)
            if n:
                print("ROLE", role, "BOX", loc.bounding_box(timeout=3000))
        except Exception as e:
            print("ROLE", role, "ERR", type(e).__name__)

    print("DONE")
