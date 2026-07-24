"""dc_handoff.py — cross-page bus to send processed DC measurement DataFrames
from the Multi-Process page to the DC Analysis (B1500A) viewer without
re-uploading. Payload travels through st.session_state (preserved across
st.switch_page within a session)."""
from __future__ import annotations

import streamlit as st

PAGE_DC_ANALYSIS = "tools/B1500A_Plot.py"   # matches the st.Page path in i18n TOOLS
_KEY = "_dc_handoff_files"


def send(files: list[dict]) -> None:
    """Stash a list of type-group workbooks, one per measurement type:
    {"name": str, "sheets": {sheet_name: DataFrame}, "dtype": "Family"|"Gummel"|"Diode"}.
    In DC Analysis the file dropdown lists ``name`` (the type) and the sheet
    picker lists the measurements (``sheets`` keys) within it."""
    st.session_state[_KEY] = files


def peek():
    return st.session_state.get(_KEY)


def take():
    return st.session_state.pop(_KEY, None)
