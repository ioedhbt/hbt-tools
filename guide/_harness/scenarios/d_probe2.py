"""Probe bounding boxes on Simulation & Fitting landing view."""
ENTRY = "IOED_Tool_Web.py"


def run(page, shot):
    from session import settle

    page.get_by_role("link", name="SSM Simulation & Fitting").first.click()
    settle(page)

    try:
        loc = page.get_by_text("Model", exact=True).first
        print("MODEL LABEL", loc.bounding_box())
    except Exception as e:
        print("MODEL LABEL ERR", e)

    try:
        loc = page.get_by_role("radio", name="Open and Short Pad").first
        print("LAST OPT", loc.bounding_box())
    except Exception as e:
        print("LAST OPT ERR", e)

    try:
        exp = page.get_by_text("Fit to a measured device (optional)", exact=False).first
        print("FIT EXPANDER", exp.bounding_box())
    except Exception as e:
        print("FIT EXP ERR", e)

    shot("00_top", clip={"x": 0, "y": 0, "width": 1600, "height": 420})
    print("DONE")
