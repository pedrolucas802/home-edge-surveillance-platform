"""Dashboard CSS. Colours use translucent tints so light and dark themes both work."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
    --hes-accent: #2f6fed;
    --hes-accent-2: #12b5a5;
    --hes-line: rgba(127, 140, 160, 0.28);
    --hes-surface: rgba(127, 140, 160, 0.07);
    --hes-muted: rgba(127, 140, 160, 1);
    --hes-ok: #14a35a;
    --hes-warn: #d98a04;
    --hes-bad: #d64545;
}
.block-container { padding-top: 3.2rem; max-width: 1280px; }
[data-testid="stSidebarNav"] a span { font-size: 0.95rem; }

/* Page header */
.page-header { display: flex; gap: 0.9rem; align-items: center; margin-bottom: 1.1rem; }
.page-header h1 { font-size: 1.75rem; margin: 0; padding: 0; line-height: 1.2; }
.page-header p { margin: 0.15rem 0 0 0; color: var(--hes-muted); font-size: 0.95rem; }
.page-icon {
    width: 3rem; height: 3rem; border-radius: 14px; display: flex; align-items: center;
    justify-content: center; font-size: 1.5rem; background: var(--hes-surface);
    border: 1px solid var(--hes-line);
}

/* Home hero */
.hero {
    border-radius: 24px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem; color: #f4f7ff;
    background:
        radial-gradient(1200px 400px at 85% -20%, rgba(18, 181, 165, 0.55), transparent 60%),
        radial-gradient(900px 500px at -10% 120%, rgba(47, 111, 237, 0.65), transparent 60%),
        linear-gradient(135deg, #0d1b33 0%, #13284d 55%, #0f3346 100%);
    box-shadow: 0 18px 40px rgba(13, 27, 51, 0.25);
}
.hero-kicker { text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.75rem; opacity: 0.75; }
.hero h1 { color: #ffffff; font-size: 2.3rem; margin: 0.35rem 0 0.4rem 0; padding: 0; line-height: 1.15; }
.hero p { font-size: 1.05rem; max-width: 46rem; opacity: 0.9; margin: 0; }
.hero-chips { margin-top: 1.2rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.hero-chip {
    background: rgba(255, 255, 255, 0.12); border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 999px; padding: 0.3rem 0.8rem; font-size: 0.85rem;
}
.hero-chip strong { color: #ffffff; }

/* Stat tiles */
.stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 1.2rem; }
.stat-tile {
    border: 1px solid var(--hes-line); background: var(--hes-surface); border-radius: 16px;
    padding: 0.8rem 1rem;
}
.stat-label { font-size: 0.78rem; color: var(--hes-muted); text-transform: uppercase; letter-spacing: 0.06em; }
.stat-value { font-size: 1.7rem; font-weight: 700; line-height: 1.25; }
.stat-hint { font-size: 0.8rem; color: var(--hes-muted); }

/* Home link cards */
.link-card-icon { font-size: 2rem; line-height: 1; }
.link-card-title { font-size: 1.15rem; font-weight: 700; margin-top: 0.5rem; }
.link-card-text { color: var(--hes-muted); font-size: 0.9rem; min-height: 2.6rem; }
.link-card-stat { font-size: 0.9rem; margin: 0.45rem 0 0.2rem 0; }
.feature { padding: 0.2rem 0.2rem 0.6rem 0; }
.feature h4 { margin: 0.2rem 0 0.25rem 0; padding: 0; font-size: 1rem; }
.feature p { color: var(--hes-muted); font-size: 0.9rem; margin: 0; }
.section-title { font-size: 1.1rem; font-weight: 700; margin: 1.4rem 0 0.6rem 0; }

/* Device cards */
.hub-card {
    border: 1px solid var(--hes-line); border-radius: 18px; padding: 0.9rem 1rem 0.8rem 1rem;
    background: var(--hes-surface); margin-bottom: 0.9rem;
}
.hub-card.card-alert { border-color: rgba(214, 69, 69, 0.55); box-shadow: 0 0 0 1px rgba(214, 69, 69, 0.25) inset; }
.hub-card.card-offline { opacity: 0.72; }
.hub-card-header { display: flex; justify-content: space-between; align-items: center; gap: 0.75rem; margin-bottom: 0.55rem; }
.hub-card-title { font-size: 1.05rem; font-weight: 700; }
.hub-card-subtitle { font-size: 0.8rem; color: var(--hes-muted); }
.kv-row { display: flex; justify-content: space-between; font-size: 0.86rem; padding: 0.12rem 0; gap: 0.5rem; }
.kv-row span { color: var(--hes-muted); }
.card-alert-text {
    background: rgba(214, 69, 69, 0.12); color: var(--hes-bad); border-radius: 10px;
    padding: 0.3rem 0.6rem; margin-bottom: 0.45rem; font-weight: 700; font-size: 0.86rem;
}
.status-badge { padding: 0.22rem 0.65rem; border-radius: 999px; font-size: 0.76rem; font-weight: 700; white-space: nowrap; }
.badge-live { background: rgba(20, 163, 90, 0.15); color: var(--hes-ok); }
.badge-stale { background: rgba(217, 138, 4, 0.15); color: var(--hes-warn); }
.badge-offline { background: rgba(214, 69, 69, 0.15); color: var(--hes-bad); }
.badge-muted { background: rgba(127, 140, 160, 0.18); color: var(--hes-muted); }
.pill {
    display: inline-block; padding: 0.16rem 0.55rem; margin: 0.15rem 0.2rem 0.1rem 0;
    border-radius: 999px; background: rgba(47, 111, 237, 0.14); color: var(--hes-accent);
    font-size: 0.8rem; font-weight: 600;
}
.pill-muted { background: rgba(127, 140, 160, 0.15); color: var(--hes-muted); }

/* Radar map */
.radar-map { display: block; margin: 0.1rem 0 0.4rem 0; }
.radar-cone { fill: rgba(47, 111, 237, 0.10); stroke: rgba(47, 111, 237, 0.45); }
.radar-ring { fill: none; stroke: rgba(47, 111, 237, 0.3); stroke-dasharray: 3 3; }
.radar-limit { fill: none; stroke: var(--hes-warn); stroke-width: 1.5; stroke-dasharray: 6 3; }
.radar-text { font-size: 8px; fill: var(--hes-muted); }
.radar-origin { fill: var(--hes-accent); }
.radar-target-moving { fill: #e8590c; fill-opacity: 0.95; }
.radar-target-still { fill: #f59f00; fill-opacity: 0.95; }

/* Tripwire beam */
.beam-svg { display: block; margin: 0.2rem 0 0.5rem 0; }
.beam-post { fill: rgba(127, 140, 160, 0.55); }
.beam-ok { stroke: var(--hes-ok); stroke-width: 4; stroke-linecap: round; }
.beam-broken { stroke: var(--hes-bad); stroke-width: 4; stroke-linecap: round; }
.beam-fault { stroke: var(--hes-warn); stroke-width: 4; stroke-linecap: round; }
.beam-gap { stroke: var(--hes-bad); stroke-width: 14; stroke-opacity: 0.18; }
.signal-bar { position: relative; height: 10px; border-radius: 999px; background: rgba(127, 140, 160, 0.2); margin: 0.3rem 0 0.4rem 0; }
.signal-fill { height: 100%; border-radius: 999px; }
.signal-good { background: var(--hes-ok); }
.signal-weak { background: var(--hes-bad); }
.signal-mark { position: absolute; top: -3px; width: 2px; height: 16px; background: var(--hes-muted); }

/* Event feed */
.feed-row {
    display: flex; justify-content: space-between; gap: 0.8rem; padding: 0.45rem 0.1rem;
    border-bottom: 1px solid var(--hes-line); font-size: 0.9rem;
}
.feed-row:last-child { border-bottom: none; }
.feed-time { color: var(--hes-muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
