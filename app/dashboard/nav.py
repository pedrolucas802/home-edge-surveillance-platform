"""Page registry, so pages can link to each other with st.page_link."""

from __future__ import annotations

import streamlit as st

PAGES: dict[str, st.Page] = {}


def link(key: str, label: str, icon: str | None = None) -> None:
    page = PAGES.get(key)
    if page is not None:
        st.page_link(page, label=label, icon=icon, width="stretch")
