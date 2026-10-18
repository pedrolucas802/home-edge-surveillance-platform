"""Home Edge Surveillance dashboard: a home page plus one page per device kind.

Run with: streamlit run app/dashboard/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.dashboard import nav
from app.dashboard.styles import inject_styles
from app.dashboard.views import cameras, configuration, devices, home, radars, tripwires


def main() -> None:
    st.set_page_config(
        page_title="Home Edge Surveillance",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    nav.PAGES.update(
        {
            "home": st.Page(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
            "cameras": st.Page(cameras.render, title="Cameras", icon=":material/videocam:", url_path="cameras"),
            "radars": st.Page(radars.render, title="Radars", icon=":material/radar:", url_path="radars"),
            "tripwires": st.Page(tripwires.render, title="Tripwires", icon=":material/sensors:", url_path="tripwires"),
            "devices": st.Page(devices.render, title="All devices", icon=":material/devices:", url_path="devices"),
            "configuration": st.Page(
                configuration.render, title="Configuration", icon=":material/settings:", url_path="configuration"
            ),
        }
    )
    pages = nav.PAGES
    page = st.navigation(
        {
            "": [pages["home"]],
            "Live": [pages["cameras"], pages["radars"], pages["tripwires"]],
            "System": [pages["devices"], pages["configuration"]],
        }
    )
    try:
        page.run()
    except ValueError as exc:
        # Usually a broken devices file; say so instead of a stack trace.
        st.error(f"Configuration error: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()
