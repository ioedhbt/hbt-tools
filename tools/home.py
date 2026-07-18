"""
home.py — Landing page for the IOED Lab Portal.

A bilingual orientation screen: a short intro, then one card per tool grouped
by measurement domain.  Each card shows the tool name (with a hover tooltip
listing key features), a one-line description, and an Open button, so the
home page stays compact while full feature details remain accessible via the
"?" icon.
"""
from __future__ import annotations

import streamlit as st

from tools import i18n


st.title(i18n.t("home_title"))
st.caption(i18n.t("home_intro"))
st.divider()

# One row of cards per domain group, in canonical order.
for group_key in i18n.GROUP_ORDER:
    tools_in_group = [k for k, m in i18n.TOOLS.items() if m["group"] == group_key]
    if not tools_in_group:
        continue

    st.subheader(i18n.group_label(group_key))
    cols = st.columns(len(tools_in_group), gap="medium")
    for col, tool_key in zip(cols, tools_in_group):
        meta = i18n.TOOLS[tool_key]
        with col:
            with st.container(border=True):
                feats = i18n.tool_features(tool_key)
                st.subheader(
                    f"{meta['icon']} {i18n.tool_name(tool_key)}",
                    help="\n".join(f"- {f}" for f in feats) if feats else None,
                    anchor=False,
                )
                st.caption(i18n.tool_desc(tool_key))
                st.page_link(
                    meta["path"],
                    label=i18n.t("open_tool"),
                    icon="➡️",
                    width="stretch",
                )
    st.write("")
