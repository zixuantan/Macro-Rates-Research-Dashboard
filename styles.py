from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


def configure_plotly_theme() -> None:
    """Give every Plotly chart the same restrained research-terminal style."""
    template = go.layout.Template(pio.templates["plotly_white"])
    template.layout.update(
        font={"family": "Inter, ui-sans-serif, system-ui, sans-serif", "color": "#243247", "size": 12},
        colorway=["#3157D5", "#0F8A83", "#D97706", "#7C3AED", "#DC5A5A", "#64748B"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"font": {"color": "#14213D", "size": 17}, "x": 0.01, "xanchor": "left"},
        hoverlabel={"bgcolor": "#14213D", "font": {"color": "#FFFFFF"}, "bordercolor": "#14213D"},
        legend={"bgcolor": "rgba(255,255,255,0)", "font": {"color": "#475569"}},
        margin={"l": 48, "r": 24, "t": 64, "b": 44},
    )
    template.layout.xaxis.update(
        gridcolor="#E7EBF2",
        linecolor="#CBD3E1",
        tickcolor="#CBD3E1",
        zerolinecolor="#CBD3E1",
        title_font={"color": "#64748B", "size": 12},
        tickfont={"color": "#64748B", "size": 11},
    )
    template.layout.yaxis.update(
        gridcolor="#E7EBF2",
        linecolor="#CBD3E1",
        tickcolor="#CBD3E1",
        zerolinecolor="#CBD3E1",
        title_font={"color": "#64748B", "size": 12},
        tickfont={"color": "#64748B", "size": 11},
    )
    # Existing panels explicitly request plotly_white. Updating that registered
    # template improves all charts without duplicating layout code in each panel.
    pio.templates["plotly_white"] = template


def apply_app_styles() -> None:
    """Apply the dashboard-wide visual system."""
    st.html(
        """
        <style>
            :root {
                --rm-ink: #14213d;
                --rm-text: #334155;
                --rm-muted: #64748b;
                --rm-blue: #3157d5;
                --rm-blue-soft: #edf2ff;
                --rm-teal: #0f8a83;
                --rm-surface: rgba(255, 255, 255, .92);
                --rm-border: #dfe5ee;
                --rm-shadow: 0 8px 28px rgba(30, 47, 78, .06);
            }

            html, body, [class*="css"] {
                font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                    "Segoe UI", sans-serif;
            }

            .stApp {
                color: var(--rm-text);
                background:
                    radial-gradient(circle at 12% 0%, rgba(49, 87, 213, .055), transparent 28rem),
                    linear-gradient(180deg, #f9fbfe 0%, #f5f7fb 100%);
            }

            header[data-testid="stHeader"] {
                background: rgba(249, 251, 254, .82);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid rgba(223, 229, 238, .7);
            }

            .main .block-container {
                max-width: 1480px;
                padding: 2rem 2.5rem 4rem;
            }

            .rm-masthead {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 2rem;
                margin: .3rem 0 1.5rem;
                padding: 0 0 1.35rem;
                border-bottom: 1px solid var(--rm-border);
            }

            .rm-eyebrow {
                margin-bottom: .45rem;
                color: var(--rm-blue);
                font-size: .7rem;
                font-weight: 750;
                letter-spacing: .14em;
                text-transform: uppercase;
            }

            .rm-masthead h1 {
                margin: 0;
                color: var(--rm-ink);
                font-size: clamp(1.85rem, 3vw, 2.65rem);
                font-weight: 720;
                letter-spacing: -.045em;
                line-height: 1.05;
            }

            .rm-masthead p {
                max-width: 720px;
                margin: .55rem 0 0;
                color: var(--rm-muted);
                font-size: .94rem;
                line-height: 1.5;
            }

            .rm-status {
                display: inline-flex;
                align-items: center;
                gap: .45rem;
                flex: 0 0 auto;
                margin-bottom: .18rem;
                padding: .44rem .7rem;
                border: 1px solid #cfe3df;
                border-radius: 999px;
                color: #0b6b65;
                background: #eef8f6;
                font-size: .72rem;
                font-weight: 700;
            }

            .rm-status::before {
                width: .46rem;
                height: .46rem;
                border-radius: 50%;
                background: var(--rm-teal);
                box-shadow: 0 0 0 3px rgba(15, 138, 131, .12);
                content: "";
            }

            h1, h2, h3, h4 { color: var(--rm-ink); letter-spacing: -.025em; }
            h2 { padding-top: .25rem; font-size: 1.55rem !important; }
            h3 { margin-top: 1.85rem !important; font-size: 1.18rem !important; }
            h4 { font-size: .98rem !important; }
            p, li { line-height: 1.58; }
            a { color: var(--rm-blue); }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f1f4f9 0%, #edf1f7 100%);
                border-right: 1px solid #dbe2ec;
            }

            [data-testid="stSidebar"] > div:first-child { padding-top: 1.55rem; }
            [data-testid="stSidebar"] h2 {
                padding: 0 !important;
                color: var(--rm-ink);
                font-size: 1.05rem !important;
            }

            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
                color: #718096;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: .28rem;
                padding: .3rem;
                border: 1px solid var(--rm-border);
                border-radius: .8rem;
                background: rgba(255, 255, 255, .68);
                box-shadow: 0 3px 14px rgba(30, 47, 78, .035);
            }

            .stTabs [data-baseweb="tab"] {
                height: 2.65rem;
                padding: 0 .86rem;
                border-radius: .56rem;
                color: #64748b;
                font-size: .8rem;
                font-weight: 650;
            }

            .stTabs [aria-selected="true"] {
                color: var(--rm-blue) !important;
                background: var(--rm-blue-soft) !important;
            }

            .stTabs [data-baseweb="tab-highlight"] { display: none; }
            .stTabs [data-baseweb="tab-border"] { display: none; }

            div[data-testid="stMetric"] {
                min-height: 7.4rem;
                padding: 1rem 1.05rem .9rem;
                overflow: visible;
                align-items: flex-start;
                border: 1px solid var(--rm-border);
                border-radius: .8rem;
                background: var(--rm-surface);
                box-shadow: var(--rm-shadow);
            }

            div[data-testid="stMetricLabel"],
            div[data-testid="stMetricLabel"] p {
                margin: 0 !important;
                overflow: visible !important;
                color: var(--rm-muted);
                font-size: .76rem;
                font-weight: 700;
                letter-spacing: .015em;
                line-height: 1.25;
                text-overflow: unset !important;
                white-space: normal !important;
                word-break: break-word !important;
            }

            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] p {
                margin: .15rem 0 0 !important;
                overflow: visible !important;
                color: var(--rm-ink);
                font-size: 1.42rem;
                font-weight: 710;
                letter-spacing: -.025em;
                line-height: 1.15;
                text-overflow: unset !important;
                white-space: normal !important;
                word-break: break-word !important;
            }

            div[data-testid="stMetricDelta"],
            div[data-testid="stMetricDelta"] p {
                overflow: visible !important;
                font-size: .72rem;
                line-height: 1.25;
                text-overflow: unset !important;
                white-space: normal !important;
                word-break: break-word !important;
            }

            [data-testid="stPlotlyChart"] {
                overflow: hidden;
                border: 1px solid var(--rm-border);
                border-radius: .9rem;
                background: var(--rm-surface);
                box-shadow: var(--rm-shadow);
            }

            [data-testid="stDataFrame"],
            [data-testid="stTable"] {
                overflow: hidden;
                border: 1px solid var(--rm-border);
                border-radius: .75rem;
                background: #fff;
                box-shadow: 0 5px 18px rgba(30, 47, 78, .04);
            }

            [data-testid="stAlert"] {
                border-radius: .72rem;
                border-width: 1px;
                box-shadow: 0 4px 16px rgba(30, 47, 78, .035);
            }

            [data-testid="stExpander"] {
                overflow: hidden;
                border: 1px solid var(--rm-border);
                border-radius: .75rem;
                background: rgba(255, 255, 255, .72);
            }

            [data-testid="stExpander"] summary { font-weight: 650; }

            .stButton > button,
            .stDownloadButton > button,
            [data-testid="stFormSubmitButton"] > button {
                min-height: 2.55rem;
                border-radius: .6rem;
                border-color: #cbd5e1;
                font-weight: 650;
                transition: transform .12s ease, box-shadow .12s ease, border-color .12s ease;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                transform: translateY(-1px);
                border-color: var(--rm-blue);
                box-shadow: 0 6px 16px rgba(49, 87, 213, .12);
            }

            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-baseweb="select"] > div,
            [data-baseweb="input"] > div {
                border-color: #cfd7e3 !important;
                border-radius: .6rem !important;
                background: rgba(255, 255, 255, .9) !important;
            }

            [data-testid="stSegmentedControl"] [data-baseweb="button-group"] {
                padding: .22rem;
                border: 1px solid var(--rm-border);
                border-radius: .68rem;
                background: #f1f4f8;
            }

            [data-testid="stSegmentedControl"] button { border-radius: .48rem !important; }
            [data-testid="stCaptionContainer"] { color: var(--rm-muted); }
            hr { border-color: var(--rm-border); }

            @media (max-width: 900px) {
                .main .block-container { padding: 1.25rem 1rem 3rem; }
                .rm-masthead { align-items: flex-start; flex-direction: column; gap: .8rem; }
                .rm-status { margin-bottom: 0; }
                .stTabs [data-baseweb="tab-list"] { overflow-x: auto; }
                div[data-testid="stMetric"] { min-height: 6.7rem; }
            }
        </style>
        """
    )


def render_masthead() -> None:
    st.html(
        """
        <div class="rm-masthead">
            <div>
                <div class="rm-eyebrow">Macro strategy workspace</div>
                <h1>Rates Research Dashboard</h1>
            </div>
        </div>
        """
    )
