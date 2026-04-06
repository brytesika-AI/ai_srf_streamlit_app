"""
AI-SRF · Streamlit Application
AI-Driven Strategic Resilience Framework for South African Corporate Digitalisation

By: Bright Sikazwe, PhD Candidate
University of Johannesburg — College of Business and Economics
Department of Information and Knowledge Management
"""

import os
import json
import logging
import threading
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="AI-SRF · Strategic Resilience Framework",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    GROQ_API_KEY, SA_CORPUS, COMPETITOR_CASES, PRESET_SCENARIOS,
    RISK_COLORS, LAYER_COLORS, SCOPE_KEYWORDS, RAG_DATA_DIR,
    CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
)
from rag_engine import SAKnowledgeBase
from agents import run_full_pipeline
from mcp_bridge import get_configured_servers, probe_mcp_servers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Custom CSS (South African warmth, Playfair Display, DM Sans) ─────────────
if False:
    st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
    /* ── Base ── */
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main { background: #FAF5EC; }
    .block-container { padding: 1.2rem 2rem 2rem; max-width: 1400px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background: #1C1917 !important; }
    [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #F97316 !important; font-family: 'Playfair Display', serif !important; }
    [data-testid="stSidebar"] .stMarkdown p { font-size: 12px; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.1) !important; }

    /* ── Typography ── */
    h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #1C1917 !important; }
    .logo-text   { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 900; color: #F97316; letter-spacing: 1px; }
    .brand-sub   { font-family: 'Space Mono', monospace; font-size: 9px; color: rgba(255,255,255,0.45); letter-spacing: 2px; text-transform: uppercase; }
    .ubuntu-tag  { font-family: 'DM Sans', sans-serif; font-size: 10px; color: rgba(255,255,255,0.3); font-style: italic; }

    /* ── Agent cards ── */
    .agent-card { border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; animation: fadeIn 0.4s ease; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

    /* ── Risk badge ── */
    .risk-badge { display: inline-flex; align-items: center; gap: 8px; border-radius: 8px; padding: 8px 14px; font-family: 'Playfair Display', serif; font-weight: 700; font-size: 16px; }

    /* ── Verdict chips ── */
    .verdict-proceed  { background: #F0FDF4; color: #166534; border: 1px solid #BBF7D0; border-radius: 12px; padding: 2px 12px; font-size: 11px; font-weight: 700; font-family: 'Space Mono', monospace; }
    .verdict-mod      { background: #FFFBEB; color: #78350F; border: 1px solid #FDE68A; border-radius: 12px; padding: 2px 12px; font-size: 11px; font-weight: 700; font-family: 'Space Mono', monospace; }
    .verdict-defer    { background: #FEF2F2; color: #7F1D1D; border: 1px solid #FECACA; border-radius: 12px; padding: 2px 12px; font-size: 11px; font-weight: 700; font-family: 'Space Mono', monospace; }
    .unbypassable     { background: #7F1D1D; color: #FEE2E2; border-radius: 10px; padding: 2px 10px; font-size: 9px; font-weight: 700; font-family: 'Space Mono', monospace; }

    /* ── Chat bubbles ── */
    .user-bubble      { background: #1C1917; color: white; border-radius: 14px 14px 0 14px; padding: 12px 18px; margin: 6px 0 6px auto; max-width: 75%; font-size: 14px; line-height: 1.55; }
    .synthesis-card   { background: white; border-left: 4px solid #16A34A; border-radius: 10px; padding: 16px 20px; margin: 12px 0; }
    .guardrail-card   { background: #FEF9EE; border: 1.5px solid #F59E0B; border-radius: 8px; padding: 12px 16px; margin: 10px 0; }

    /* ── Layer pills ── */
    .layer-1 { background: #C2410C; color: white; border-radius: 10px; padding: 2px 10px; font-size: 10px; font-weight: 700; font-family: 'Space Mono', monospace; }
    .layer-2 { background: #1E3A5F; color: white; border-radius: 10px; padding: 2px 10px; font-size: 10px; font-weight: 700; font-family: 'Space Mono', monospace; }
    .layer-3 { background: #166534; color: white; border-radius: 10px; padding: 2px 10px; font-size: 10px; font-weight: 700; font-family: 'Space Mono', monospace; }

    /* ── Tab buttons ── */
    .stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #F97316 !important; border-bottom: 2px solid #F97316 !important; }

    /* ── Input ── */
    .stTextArea textarea { background: #FFFDF9 !important; border: 1.5px solid #E5D9C8 !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; }

    /* ── Buttons ── */
    .stButton > button { background: #F97316 !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; font-family: 'DM Sans', sans-serif !important; }
    .stButton > button:hover { background: #EA580C !important; }

    /* ── Progress bar ── */
    .stProgress > div > div { background: linear-gradient(90deg, #F97316, #FBBF24) !important; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] { background: white; border-radius: 10px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
    [data-testid="stMetricValue"] { color: #F97316 !important; font-family: 'Playfair Display', serif !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=DM+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg: #FAF5EC;
  --panel: #FFFFFF;
  --ink: #1C1917;
  --muted: #6B7280;
  --line: #E5D9C8;
  --accent: #F97316;
  --accent-dark: #EA580C;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: 'DM Sans', sans-serif !important;
}

.main { background: var(--bg); }
.block-container { max-width: 1380px !important; padding: .3rem 1.75rem 1.75rem !important; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: var(--ink) !important; letter-spacing: -0.02em; }

[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #fcf7ef 0%, #f8f1e6 100%) !important;
}
[data-testid="stHeader"] {
  background: rgba(250,245,236,0.94) !important;
  height: 2.2rem !important;
  border-bottom: 1px solid rgba(229,217,200,.35);
}
[data-testid="stToolbar"] { top: .15rem !important; }
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] {
  background: #1C1917 !important;
  border-right: 1px solid rgba(255,255,255,.08);
  min-width: 252px;
  max-width: 252px;
}
[data-testid="stSidebarContent"] {
  background: #1C1917 !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebarContent"] * {
  color: rgba(255,255,255,0.86) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebarContent"] p,
[data-testid="stSidebarContent"] li,
[data-testid="stSidebarContent"] span,
[data-testid="stSidebarContent"] label,
[data-testid="stSidebarContent"] div {
  color: rgba(255,255,255,0.82) !important;
}
[data-testid="stSidebar"] hr,
[data-testid="stSidebarContent"] hr {
  border-color: rgba(255,255,255,.08) !important;
  margin-top: 1rem !important;
  margin-bottom: 1rem !important;
}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebarContent"] .stButton > button {
  background: rgba(255,255,255,.08) !important;
  border: 1px solid rgba(255,255,255,.14) !important;
  color: rgba(255,255,255,.78) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 10px !important;
  font-weight: 700 !important;
}

.stApp a, .stApp p, .stApp li, .stApp label, .stApp span {
  color: var(--ink);
}

.logo-text { font-family: 'Playfair Display', serif; font-size: 28px; font-weight: 900; color: #F97316; letter-spacing: .03em; }
.brand-sub { font-family: 'Space Mono', monospace; font-size: 10px; color: rgba(255,255,255,0.62) !important; letter-spacing: 2px; text-transform: uppercase; }
.ubuntu-tag { font-family: 'DM Sans', sans-serif; font-size: 11px; color: rgba(255,255,255,0.52) !important; font-style: italic; }

.agent-card { border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; animation: fadeIn .4s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.risk-badge { display: inline-flex; align-items: center; gap: 8px; border-radius: 8px; padding: 8px 14px; font-family: 'Playfair Display', serif; font-weight: 700; font-size: 16px; }

.verdict-proceed { background: #F0FDF4; color: #166534; border: 1px solid #BBF7D0; border-radius: 12px; padding: 2px 12px; font-size: 11px; font-weight: 700; font-family: 'Space Mono', monospace; }
.verdict-mod { background: #FFFBEB; color: #78350F; border: 1px solid #FDE68A; border-radius: 12px; padding: 2px 12px; font-size: 11px; font-weight: 700; font-family: 'Space Mono', monospace; }
.verdict-defer { background: #FEF2F2; color: #7F1D1D; border: 1px solid #FECACA; border-radius: 12px; padding: 2px 12px; font-size: 11px; font-weight: 700; font-family: 'Space Mono', monospace; }
.unbypassable { background: #7F1D1D; color: #FEE2E2; border-radius: 10px; padding: 2px 10px; font-size: 9px; font-weight: 700; font-family: 'Space Mono', monospace; }

.user-bubble { background: #1C1917; color: white; border-radius: 14px 14px 0 14px; padding: 12px 18px; margin: 6px 0 6px auto; max-width: 75%; font-size: 14px; line-height: 1.55; }
.synthesis-card { background: white; border-left: 4px solid #16A34A; border-radius: 10px; padding: 16px 20px; margin: 12px 0; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
.guardrail-card { background: #FEF9EE; border: 1.5px solid #F59E0B; border-radius: 8px; padding: 12px 16px; margin: 10px 0; }

.layer-1 { background: #C2410C; color: white; border-radius: 10px; padding: 2px 10px; font-size: 10px; font-weight: 700; font-family: 'Space Mono', monospace; }
.layer-2 { background: #1E3A5F; color: white; border-radius: 10px; padding: 2px 10px; font-size: 10px; font-weight: 700; font-family: 'Space Mono', monospace; }
.layer-3 { background: #166534; color: white; border-radius: 10px; padding: 2px 10px; font-size: 10px; font-weight: 700; font-family: 'Space Mono', monospace; }

.stTabs { margin-top: -.2rem; }
.stTabs [data-baseweb="tab-list"] {
  gap: .2rem;
  border-bottom: 1px solid #e7d8c2;
  background: rgba(255,255,255,.92);
  backdrop-filter: blur(8px);
  padding: .15rem .95rem 0;
  border-top-left-radius: 18px;
  border-top-right-radius: 18px;
  box-shadow: 0 10px 24px rgba(152,98,23,.06);
}
.stTabs [data-baseweb="tab"] {
  color: #5e6573 !important;
  padding: .95rem .95rem .75rem !important;
  border-bottom: 2px solid transparent;
  font-family: 'DM Sans', sans-serif;
  font-weight: 600;
  font-size: .98rem;
}
.stTabs [data-baseweb="tab"]:hover { color: #1f1b17 !important; }
.stTabs [aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
  font-weight: 700 !important;
}

.stTextArea textarea { background: #FFFDF9 !important; border: 1.5px solid #E5D9C8 !important; border-radius: 8px !important; font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; line-height: 1.55 !important; }
.stButton > button { background: #F97316 !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; font-family: 'DM Sans', sans-serif !important; }
.stButton > button:hover { background: #EA580C !important; }
.stProgress > div > div { background: linear-gradient(90deg, #F97316, #FBBF24) !important; }

[data-testid="stMetric"] { background: white; border-radius: 10px; padding: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
[data-testid="stMetricValue"] { color: #F97316 !important; font-family: 'Playfair Display', serif !important; }
[data-testid="stExpander"] details { background: white; border: 1px solid #E5D9C8; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,.04); overflow: hidden; margin-bottom: 10px; }
[data-testid="stExpander"] summary { background: #FFFDF9; padding-top: .65rem !important; padding-bottom: .65rem !important; }

.welcome-card { background: white; border-radius: 12px; padding: 18px 22px; border-left: 3px solid var(--accent); box-shadow: 0 1px 4px rgba(0,0,0,.06); margin-bottom: 18px; }
.section-intro { font-size: 13px; color: #6B7280; margin-bottom: 20px; line-height: 1.6; }
.top-hero {
  background: linear-gradient(180deg, rgba(255,255,255,.92) 0%, rgba(255,252,247,.98) 100%);
  border: 1px solid #eadcc8;
  border-radius: 18px;
  box-shadow: 0 14px 30px rgba(122,84,29,.08);
  padding: .45rem .5rem 0;
  margin: .1rem 0 1rem;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

if False:
    st.markdown(
    """
    <style>
    :root {
      --bg: #FAF5EC;
      --panel: #FFFFFF;
      --panel-soft: #FFFDF9;
      --ink: #1C1917;
      --muted: #6B7280;
      --line: #E5D9C8;
      --accent: #F97316;
      --accent-dark: #EA580C;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
      background: var(--bg) !important;
      color: var(--ink) !important;
      font-family: 'DM Sans', sans-serif !important;
    }
    .block-container { max-width: 1380px !important; padding: 1rem 1.75rem 1.75rem !important; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stSidebar"] {
      background: #1C1917 !important;
      border-right: 1px solid rgba(255,255,255,.08);
      min-width: 252px;
      max-width: 252px;
    }
    [data-testid="stSidebar"] hr {
      border-color: rgba(255,255,255,.08) !important;
      margin-top: 1rem !important;
      margin-bottom: 1rem !important;
    }
    [data-testid="stSidebar"] .stButton > button {
      background: rgba(255,255,255,.08) !important;
      border: 1px solid rgba(255,255,255,.12) !important;
      color: rgba(255,255,255,.72) !important;
      font-family: 'Space Mono', monospace !important;
      font-size: 10px !important;
      font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
      gap: .15rem;
      border-bottom: 1px solid var(--line);
      background: white;
      padding: 0 .85rem;
    }
    .stTabs [data-baseweb="tab"] {
      color: var(--muted) !important;
      padding: .8rem .9rem;
      border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
      color: var(--accent) !important;
      border-bottom: 2px solid var(--accent) !important;
    }
    [data-testid="stExpander"] details {
      background: white;
      border: 1px solid #E5D9C8;
      border-radius: 10px;
      box-shadow: 0 1px 3px rgba(0,0,0,.04);
      overflow: hidden;
      margin-bottom: 10px;
    }
    [data-testid="stExpander"] summary {
      background: #FFFDF9;
      padding-top: .65rem !important;
      padding-bottom: .65rem !important;
    }
    .welcome-card {
      background: white;
      border-radius: 12px;
      padding: 18px 22px;
      border-left: 3px solid var(--accent);
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
      margin-bottom: 18px;
    }
    .section-intro {
      font-size: 13px;
      color: #6B7280;
      margin-bottom: 20px;
      line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Session state initialisation ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cycle_results" not in st.session_state:
    st.session_state.cycle_results = {}
if "ror_values" not in st.session_state:
    st.session_state.ror_values = {}
if "risk_state" not in st.session_state:
    st.session_state.risk_state = "Nominal"
if "risk_signals" not in st.session_state:
    st.session_state.risk_signals = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "runtime_warnings" not in st.session_state:
    st.session_state.runtime_warnings = []
if "mcp_health" not in st.session_state:
    st.session_state.mcp_health = None
if "governance_trace" not in st.session_state:
    st.session_state.governance_trace = []
if "rag_package" not in st.session_state:
    st.session_state.rag_package = {}


# ─── Initialise RAG engine (once per session) ────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag_engine() -> SAKnowledgeBase:
    kb = SAKnowledgeBase(
        groq_api_key=GROQ_API_KEY,
        persist_dir=CHROMA_PERSIST_DIR,
        collection=CHROMA_COLLECTION,
    )
    # Index built-in SA corpus
    kb.index_corpus(SA_CORPUS)
    # Index PDFs from RAW_DATA_RAG folder
    os.makedirs(RAG_DATA_DIR, exist_ok=True)
    kb.index_pdfs(RAG_DATA_DIR)
    return kb


# ─── Guardrail ────────────────────────────────────────────────────────────────
def is_in_scope(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SCOPE_KEYWORDS) or len(text) < 25


# ─── ROR chart ────────────────────────────────────────────────────────────────
def make_ror_bar_chart(options: list) -> go.Figure:
    def pct(s):
        try:
            return int(str(s).replace("%", "").replace("+", "").strip())
        except Exception:
            return 0

    rows = []
    for o in options:
        imp = o.get("estimated_ror_impact", {})
        rows.append({"Option": o.get("type", "?"), "DLR": pct(imp.get("dlr")), "DA": pct(imp.get("da")), "IAR": pct(imp.get("iar")), "ASY": pct(imp.get("asy"))})
    df = pd.DataFrame(rows)

    fig = go.Figure()
    colours = {"DLR": "#F97316", "DA": "#1E3A5F", "IAR": "#166534", "ASY": "#7C2D12"}
    for col, color in colours.items():
        fig.add_trace(go.Bar(name=col, x=df["Option"], y=df[col], marker_color=color, text=df[col].astype(str) + "%", textposition="outside"))
    fig.update_layout(
        barmode="group", plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="DM Sans"), height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="% Improvement", ticksuffix="%", gridcolor="#F3F4F6"),
        xaxis=dict(title=""),
    )
    return fig


def parse_pct_value(value, default: int = 0) -> int:
    st.caption("Controls: POPIA ingestion gate Â· regulatory anchor injection Â· provenance audit Â· MCP-style function tools")
    try:
        cleaned = str(value).replace("+", "").replace("%", "").replace("â‰¥", "").strip()
        return int(float(cleaned.split()[0]))
    except Exception:
        return default


def make_ror_radar(ror_vals: dict) -> go.Figure:
    cats  = ["DLR", "DA", "IAR", "ASY"]
    vals  = [ror_vals.get(c, ror_vals.get(c.lower(), 0)) for c in cats]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=cats + [cats[0]],
        fill="toself",
        line_color="#F97316",
        fillcolor="rgba(249,115,22,0.18)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False, height=260,
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


# Helper function to generate agent keys, moved above the sidebar so Streamlit can find it
def _agent_key(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="logo-text">AI-SRF</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Strategic Resilience Framework</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:9px;color:rgba(255,255,255,0.35);margin-top:2px;">South Africa · Corporate Digitalisation</div>', unsafe_allow_html=True)
    st.markdown('<div class="ubuntu-tag" style="margin-top:6px;"><em>"Umuntu ngumuntu ngabantu"</em></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px;color:rgba(255,255,255,0.4);margin-top:4px;">By: Bright Sikazwe, PhD Candidate<br>University of Johannesburg</div>', unsafe_allow_html=True)
    st.divider()

    # Risk State
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;color:rgba(255,255,255,0.45);letter-spacing:1.5px;margin-bottom:6px;">LAYER 1 · RISK STATE</div>', unsafe_allow_html=True)
    rs = st.session_state.risk_state
    rc = RISK_COLORS.get(rs, RISK_COLORS["Nominal"])
    st.markdown(
        f'<div style="background:rgba(255,255,255,.05);border-radius:8px;padding:10px;display:flex;align-items:center;gap:10px;">'
        f'<span style="font-size:22px;">{rc["icon"]}</span>'
        f'<div><div style="font-weight:800;color:{rc["hex"]};font-family:Playfair Display,serif;font-size:15px;">{rs}</div>'
        f'<div style="font-size:9px;color:rgba(255,255,255,0.3);">Bayesian classification</div></div></div>',
        unsafe_allow_html=True,
    )
    for sig in st.session_state.risk_signals[:3]:
        st.markdown(f'<div style="font-size:10px;color:rgba(255,255,255,0.45);margin-top:4px;">↳ <strong style="color:rgba(255,255,255,0.7);">{sig.get("signal_source","")}</strong>: {sig.get("current_status","")[:38]}</div>', unsafe_allow_html=True)

    st.divider()

    # Agent pipeline status
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;color:rgba(255,255,255,0.45);letter-spacing:1.5px;margin-bottom:6px;">7-AGENT PIPELINE</div>', unsafe_allow_html=True)
    AGENTS_LIST = [
        ("env", "Environmental Monitor", 1, "#F97316"),
        ("socratic", "Socratic Partner", 2, "#818CF8"),
        ("forensic", "Forensic Analyst", 2, "#818CF8"),
        ("catalyst", "Creative Catalyst", 2, "#818CF8"),
        ("devils", "Devil's Advocate", 2, "#818CF8"),
        ("scaffold", "Implementation Scaffolding", 3, "#34D399"),
        ("monitor", "Monitoring Agent", 3, "#34D399"),
    ]
    for stage_key, name, layer, color in AGENTS_LIST:
        is_active = st.session_state.active_agent == name
        completed = st.session_state.cycle_results.get(stage_key)
        dot_style = f"animation:pulse 1s infinite;" if is_active else ""
        dot_color = "#F97316" if is_active else ("#10B981" if completed else color)
        label_style = f"color:#F97316;font-weight:700;" if is_active else (f"color:rgba(255,255,255,0.8);" if completed else "color:rgba(255,255,255,0.45);")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:7px;margin-bottom:4px;padding:3px 6px;border-radius:4px;background:{"rgba(249,115,22,.15)" if is_active else "transparent"};">'
            f'<div style="width:6px;height:6px;border-radius:50%;background:{dot_color};flex-shrink:0;{dot_style}"></div>'
            f'<div style="font-size:11px;{label_style}">{name}</div>'
            f'{"<span style=\"margin-left:auto;font-size:8px;color:#F97316;font-family:Space Mono,monospace;\">ACTIVE</span>" if is_active else ""}'
            f'{"<span style=\"margin-left:auto;font-size:8px;color:#10B981;\">✓</span>" if (completed and not is_active) else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ROR metrics
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;color:rgba(255,255,255,0.45);letter-spacing:1.5px;margin-bottom:6px;">RETURN ON RESILIENCE</div>', unsafe_allow_html=True)
    ror_v = st.session_state.ror_values
    for k, label in [("dlr", "Decision Latency"), ("da", "Decision Alpha"), ("iar", "Infrastructure Autonomy"), ("asy", "Algorithmic Sovereignty")]:
        val = ror_v.get(k, 0)
        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:10px;margin-bottom:2px;">'
            f'<span style="color:rgba(255,255,255,0.6);">{label}</span>'
            f'<span style="color:#F97316;font-weight:700;font-family:Space Mono,monospace;">{f"+{val}%" if val else "—"}</span></div>'
            f'<div style="background:rgba(255,255,255,0.08);border-radius:2px;height:3px;">'
            f'<div style="background:linear-gradient(90deg,#F97316,#FBBF24);border-radius:2px;height:3px;width:{val}%;transition:width 1.2s ease;"></div></div></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # RAG status
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;color:rgba(255,255,255,0.45);letter-spacing:1.5px;margin-bottom:6px;">RAG KNOWLEDGE BASE</div>', unsafe_allow_html=True)
    try:
        kb = load_rag_engine()
        n_docs = kb.collection_size()
        st.markdown(f'<div style="font-size:11px;color:rgba(255,255,255,0.7);">✓ ChromaDB active · {n_docs} chunks</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:10px;color:rgba(255,255,255,0.4);">Model: nomic-embed-text-v1_5 (Groq)</div>', unsafe_allow_html=True)
        pdf_count = len(list(Path(RAG_DATA_DIR).glob("*.pdf")))
        st.markdown(f'<div style="font-size:10px;color:rgba(255,255,255,0.4);">PDFs in RAW_DATA_RAG: {pdf_count}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div style="font-size:10px;color:#FCA5A5;">RAG init error: {str(e)[:50]}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;color:rgba(255,255,255,0.45);letter-spacing:1.5px;margin-bottom:6px;">MCP LIVE SEARCH</div>', unsafe_allow_html=True)
    configured_mcp = sorted(get_configured_servers().keys())
    if configured_mcp:
        st.markdown(
            f'<div style="font-size:11px;color:rgba(255,255,255,0.72);">Configured: {", ".join(configured_mcp)}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Probe MCP Health", key="probe_mcp_health", use_container_width=True):
            st.session_state.mcp_health = probe_mcp_servers()
            st.rerun()
        health = st.session_state.mcp_health
        if health:
            st.markdown(
                f'<div style="font-size:10px;color:rgba(255,255,255,0.58);margin-top:6px;">Healthy: {health.get("healthy_count", 0)}/{health.get("configured_count", 0)}</div>',
                unsafe_allow_html=True,
            )
            for item in health.get("servers", [])[:5]:
                color = "#34D399" if item.get("status") == "ok" else "#FCA5A5"
                detail = f'{item.get("tool_count", 0)} tools' if item.get("status") == "ok" else item.get("error", "unreachable")[:44]
                st.markdown(
                    f'<div style="font-size:10px;color:{color};margin-top:3px;">{item.get("server")}: {detail}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div style="font-size:10px;color:#FCA5A5;">No MCP servers configured</div>', unsafe_allow_html=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
def _render_message(msg: dict):
    """Render a single message of any type."""
    role = msg.get("role")

    if role == "user":
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin:10px 0;"><div class="user-bubble">{msg["text"]}</div></div>',
            unsafe_allow_html=True,
        )
    elif role == "agent":
        _render_agent_card(msg)
    elif role == "synthesis":
        st.markdown(
            f'<div class="synthesis-card"><div style="font-family:Space Mono,monospace;font-size:9px;color:#16A34A;font-weight:800;margin-bottom:8px;letter-spacing:1px;">AI-SRF GOVERNANCE CYCLE COMPLETE</div><div style="font-size:14px;color:#1C1917;line-height:1.7;white-space:pre-wrap;">{msg["text"]}</div></div>',
            unsafe_allow_html=True,
        )


def _render_agent_card(msg: dict):
    stage = msg.get("stage", "")
    data = msg.get("data", {})
    stage_brief = msg.get("stage_brief", "")
    stage_meta = {
        "env": ("Environmental Monitor", 1, "#C2410C"),
        "socratic": ("Socratic Partner", 2, "#1E3A5F"),
        "forensic": ("Forensic Analyst", 2, "#1E3A5F"),
        "catalyst": ("Creative Catalyst", 2, "#1E3A5F"),
        "devils": ("Devil's Advocate", 2, "#7F1D1D"),
        "scaffold": ("Implementation Scaffolding", 3, "#166534"),
        "monitor": ("Monitoring Agent", 3, "#166534"),
    }
    name, layer, color = stage_meta.get(stage, ("Agent", 2, "#555"))

    with st.expander(f"**L{layer} · {name}**", expanded=True):
        col_badge, col_trace = st.columns([5, 1])
        with col_badge:
            st.markdown(f'<span class="layer-{layer}">Layer {layer}</span>&nbsp;<span style="font-size:13px;font-weight:700;color:{color};">{name}</span>', unsafe_allow_html=True)
        with col_trace:
            if st.checkbox("Show JSON trace", key=f"trace_{stage}_{id(msg)}", value=False):
                st.json(data, expanded=False)

        if stage_brief:
            st.markdown(
                f'<div style="background:#FFFDF9;border:1px solid #EADCC8;border-radius:10px;padding:12px 14px;margin:10px 0 12px;">'
                f'<div style="font-size:10px;color:#9A6B35;font-family:Space Mono,monospace;font-weight:700;letter-spacing:.08em;margin-bottom:6px;">STAGE BRIEF</div>'
                f'<div style="font-size:13px;color:#374151;line-height:1.65;">{stage_brief}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if stage == "env":
            rs = data.get("current_risk_state", "Nominal")
            rc = RISK_COLORS.get(rs, RISK_COLORS["Nominal"])
            st.markdown(
                f'<div style="background:{rc["bg"]};border-radius:8px;padding:10px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;"><span style="font-size:24px;">{rc["icon"]}</span><div><div style="font-weight:800;color:{rc["hex"]};font-family:Playfair Display,serif;font-size:16px;">{rs} Risk State</div><div style="font-size:11px;color:#6B7280;">Bayesian classification - Eskom / Transnet / ZAR / Broadband</div></div></div>',
                unsafe_allow_html=True,
            )
            for sig in data.get("triggering_signals", []):
                st.markdown(f'- **{sig.get("signal_source","")}**: {sig.get("current_status","")} `{sig.get("latency_or_downtime_metric","")}`')
            if data.get("contingency_templates_activated"):
                st.caption("Templates: " + " · ".join(data["contingency_templates_activated"]))
        elif stage == "socratic":
            if data.get("identified_blind_spots"):
                st.caption("Blind spots: " + " · ".join(data["identified_blind_spots"]))
            for q in data.get("socratic_questions", []):
                st.markdown(f'<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:6px;padding:10px 14px;margin-bottom:7px;"><div style="font-size:10px;color:#C2410C;font-weight:700;margin-bottom:3px;">{q.get("tied_to_signal","")}</div><div style="font-size:13px;color:#1C1917;line-height:1.5;">{q.get("question","")}</div></div>', unsafe_allow_html=True)
        elif stage == "forensic":
            rs = data.get("risk_summary", {})
            sev = rs.get("severity", "High")
            sev_col = "#DC2626" if sev == "High" else "#CA8A04"
            st.markdown(f'<span style="background:{"#FEF2F2" if sev=="High" else "#FEFCE8"};color:{sev_col};border-radius:10px;padding:2px 12px;font-size:11px;font-weight:700;">{sev} Severity</span>&nbsp;<span style="background:#F3F4F6;color:#6B7280;border-radius:10px;padding:2px 12px;font-size:11px;">{rs.get("reversibility","")}</span>', unsafe_allow_html=True)
            st.markdown(f"**Executive summary:** {rs.get('executive_summary','')}")
            if data.get("distributional_audit_and_informal_economy"):
                st.markdown(f"**Distributional audit:** {data['distributional_audit_and_informal_economy'][:280]}...")
            for exp in data.get("regulatory_exposure", []):
                st.error(exp)
        elif stage == "catalyst":
            opts = data.get("strategic_options", [])
            if opts:
                cols = st.columns(len(opts))
                palettes = [("#14532D", "#F0FDF4"), ("#7C2D12", "#FFF7ED"), ("#1E3A5F", "#EFF6FF")]
                for i, (opt, col) in enumerate(zip(opts, cols)):
                    border_color, bg_color = palettes[i % len(palettes)]
                    chips = "".join([f'<span style="background:white;padding:1px 6px;border-radius:4px;display:inline-block;margin:2px 4px 0 0;"><strong style="color:{border_color};">{k.upper()}</strong> {v}</span>' for k, v in opt.get("estimated_ror_impact", {}).items()])
                    with col:
                        st.markdown(f'<div style="border-radius:10px;padding:14px;background:{bg_color};border:1px solid {border_color}33;height:100%;"><div style="font-size:9px;font-weight:800;color:{border_color};font-family:Space Mono,monospace;text-transform:uppercase;margin-bottom:4px;">{opt.get("type","")}</div><div style="font-size:13px;font-weight:700;color:{border_color};margin-bottom:7px;">{opt.get("title","")}</div><div style="font-size:12px;color:#374151;margin-bottom:10px;line-height:1.55;">{opt.get("strategy_description","")[:140]}...</div><div style="font-size:10px;color:#6B7280;">{chips}</div></div>', unsafe_allow_html=True)
        elif stage == "devils":
            st.markdown('<span class="unbypassable">UNBYPASSABLE CHECKPOINT</span>', unsafe_allow_html=True)
            for report in data.get("stress_test_report", []):
                verdict = report.get("verdict", {})
                rating = verdict.get("rating", "PROCEED_WITH_MODIFICATION")
                cls = "verdict-proceed" if rating == "PROCEED" else ("verdict-defer" if rating == "DEFER" else "verdict-mod")
                st.markdown(f'<div style="margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid #E5E7EB;"><span style="font-size:12px;font-weight:700;color:#111827;">{report.get("option_title","")}</span>&nbsp;<span class="{cls}">{rating}</span></div>', unsafe_allow_html=True)
                for flaw in report.get("fatal_flaws", [])[:2]:
                    st.markdown(f"- {flaw}")
                if verdict.get("mandatory_conditions"):
                    st.info(f"Mandatory: {verdict['mandatory_conditions']}")
        elif stage == "scaffold":
            plan = data if isinstance(data, dict) else {}
            tiers = [("tier_1_native_execution", "Tier 1 - Native Execution", "#14532D", "#F0FDF4", "task_name"), ("tier_2_ai_augmented_scaffolding", "Tier 2 - AI-Augmented Scaffolding", "#7C2D12", "#FFF7ED", "workflow"), ("tier_3_capability_development", "Tier 3 - Capability Development", "#1E3A5F", "#EFF6FF", "prerequisite")]
            for key, label, tc, tbg, field in tiers:
                items = plan.get(key, [])
                if items:
                    st.markdown(f'<div style="font-weight:700;color:{tc};font-size:12px;background:{tbg};border-radius:4px;padding:3px 10px;margin-bottom:5px;">{label}</div>', unsafe_allow_html=True)
                    for item in items:
                        st.markdown(f"- {item.get(field, item.get('task_name', str(item))[:80])}")
        elif stage == "monitor":
            metrics = data.get("ror_tracking_metrics", {})
            if metrics:
                st.markdown("**ROR Tracking Targets:**")
                for key, value in metrics.items():
                    st.markdown(f'`{key.replace("_target","").replace("_"," ").upper()}` - {value}')
            flags = data.get("behavioral_audit_flags", [])
            if flags:
                first_flag = flags[0]
                st.warning(f"Behavioural Audit Flag: {first_flag.get('agent_name','')} - {first_flag.get('description','')}")
            triggers = data.get("layer_1_rescan_triggers", [])
            if triggers:
                st.markdown("**Layer 1 Rescan Triggers:**")
                for trigger in triggers[:4]:
                    st.markdown(f"- {trigger}")


tab_chat, tab_ror, tab_rag, tab_peer, tab_syscard = st.tabs([
    "💬 Strategic Dialogue",
    "📊 ROR Dashboard",
    "📚 Evidence Base",
    "🌍 Peer Analysis",
    "📋 System Card",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STRATEGIC DIALOGUE
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    if st.session_state.runtime_warnings:
        st.warning(
            "Live Groq or external tool access was unavailable during the last run. "
            "The AI-SRF cycle continued with governed fallback outputs.\n\n"
            + "\n".join(f"- {w}" for w in st.session_state.runtime_warnings[:5])
        )

    # Welcome card (show only when no messages)
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="background:white;border-radius:14px;padding:22px 26px;border-left:4px solid #F97316;box-shadow:0 1px 6px rgba(0,0,0,.06);margin-bottom:20px;">
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
              <div style="width:44px;height:44px;border-radius:50%;background:linear-gradient(135deg,#F97316,#1C1917);display:flex;align-items:center;justify-content:center;color:white;font-weight:800;font-size:15px;font-family:Playfair Display,serif;">AI</div>
              <div>
                <div style="font-weight:800;color:#1C1917;font-family:Playfair Display,serif;font-size:16px;">AI-SRF · Strategic Resilience Partner</div>
                <div style="font-size:11px;color:#9CA3AF;">South Africa · 7-Agent Governance Cycle · RAG-grounded · Ubuntu-anchored</div>
              </div>
            </div>
            <div style="font-size:14px;color:#374151;line-height:1.7;">
              <strong>Sawubona.</strong> I see you — and I see the complexity you are navigating.<br><br>
              I am your AI-SRF: a governance partner designed <em>exclusively</em> for South African corporate digitalisation.
              Submit your strategic challenge and I will run the full 7-agent governance cycle — sensing your environment,
              surfacing blind spots, mapping hidden risks, generating strategic options, stress-testing every assumption,
              scaffolding implementation, and establishing measurement baselines.<br><br>
              <em style="color:#6B7280;">Grounded in your RAW_DATA_RAG corpus · Groq nomic-embed-text-v1_5 · ChromaDB · llama-3.3-70b-versatile</em>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Preset scenario buttons
        st.markdown("**Try a preset disruption scenario:**")
        cols = st.columns(2)
        for i, s in enumerate(PRESET_SCENARIOS):
            with cols[i % 2]:
                if st.button(s["label"], key=f"preset_{i}", use_container_width=True):
                    st.session_state._pending_query = s["query"]
                    st.rerun()

    # Render existing messages
    for msg in st.session_state.messages:
        _render_message(msg)

    # ── Input area ──────────────────────────────────────────────────────────
    st.divider()
    col_in, col_btn = st.columns([6, 1])
    with col_in:
        query_input = st.text_area(
            "Your digitalisation challenge:",
            value=st.session_state.get("_pending_query", ""),
            height=90,
            placeholder="Describe your SA corporate digitalisation challenge…",
            label_visibility="collapsed",
            key="query_textarea",
        )
    with col_btn:
        submit = st.button("→ Run", use_container_width=True, disabled=st.session_state.active_agent is not None)

    st.caption("🛡 Guardrail active: SA corporate digitalisation only · Groq llama-3.3-70b-versatile · nomic-embed-text-v1_5 · ChromaDB")

    # ── Handle pending preset ────────────────────────────────────────────────
    if "_pending_query" in st.session_state:
        query_input = st.session_state.pop("_pending_query")
        submit = True

    # ── Execute pipeline ─────────────────────────────────────────────────────
    if submit and query_input and query_input.strip():
        query = query_input.strip()

        # Guardrail check
        if not is_in_scope(query):
            st.markdown(
                f'<div class="guardrail-card">🛡 <strong>Guardrail Active</strong> — '
                f'This platform is restricted to digitalisation strategy for South African corporate organisations. '
                f'Please reframe your query to cover AI deployment, digital transformation, cloud strategy, '
                f'POPIA/King IV/B-BBEE compliance, or infrastructure resilience in the SA corporate context.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        # Reset state for new cycle
        st.session_state.messages.append({"role": "user", "text": query})
        st.session_state.cycle_results = {}
        st.session_state.ror_values = {}
        st.session_state.governance_trace = []
        st.session_state.runtime_warnings = []

        # RAG retrieval
        kb = load_rag_engine()
        rag_package = kb.build_governance_context_package(query, k=5)
        docs = rag_package.get("documents", [])
        rag_ctx = rag_package.get("context", "")
        st.session_state.rag_package = rag_package

        # Progress placeholder
        status_ph = st.empty()
        progress_ph = st.empty()
        agent_ph = st.empty()

        AGENT_STEPS = [
            ("env",      "Environmental Monitor", 1,  14),
            ("socratic", "Socratic Partner",      2,  28),
            ("forensic", "Forensic Analyst",      3,  42),
            ("catalyst", "Creative Catalyst",     4,  57),
            ("devils",   "Devil's Advocate",      5,  71),
            ("scaffold", "Implementation Scaffolding", 6, 85),
            ("monitor",  "Monitoring Agent",      7,  100),
        ]

        def _progress_cb(agent: str, msg: str):
            st.session_state.active_agent = agent
            status_ph.info(f"**{agent}** — {msg}")

        # Stream pipeline
        for key, agent_name, step_n, pct in AGENT_STEPS:
            progress_ph.progress(pct / 100)

        all_results = {}
        for stage, result in run_full_pipeline(query, rag_ctx, rag_package=rag_package, progress_callback=_progress_cb):
            if isinstance(result, dict) and result.get("_governance_trace"):
                st.session_state.governance_trace = result["_governance_trace"]
            if isinstance(result, dict) and result.get("_runtime_error"):
                st.session_state.runtime_warnings.append(
                    f"{stage}: {result.get('_runtime_error')}"
                )
            stage_brief = result.get("_stage_brief", "") if isinstance(result, dict) else ""
            cleaned_result = {k: v for k, v in result.items() if not str(k).startswith("_")} if isinstance(result, dict) else result
            all_results[stage] = cleaned_result
            st.session_state.cycle_results[stage] = cleaned_result

            # Update risk state from env
            if stage == "env":
                st.session_state.risk_state = cleaned_result.get("current_risk_state", "Elevated")
                st.session_state.risk_signals = cleaned_result.get("triggering_signals", [])

            # Update ROR from monitor
            if stage == "monitor":
                def _pct(s):
                    try:
                        return int(str(s).replace("≥", "").replace("+", "").replace("%", "").strip().split())
                    except Exception:
                        return 0
                m = cleaned_result.get("ror_tracking_metrics", {})
                st.session_state.ror_values = {
                    "dlr": parse_pct_value(m.get("decision_latency_reduction_target", "40"), 40),
                    "da":  parse_pct_value(m.get("decision_alpha_target", "52"), 52),
                    "iar": parse_pct_value(m.get("infrastructure_autonomy_ratio_target", "91"), 91),
                    "asy": parse_pct_value(m.get("algorithmic_sovereignty_yield_target", "85"), 85),
                }

            # Record ROR from catalyst
            if stage == "catalyst":
                opts = cleaned_result.get("strategic_options", [])
                if opts:
                    def pp(s):
                        try: return int(str(s).replace("+","").replace("%",""))
                        except: return 0
                    best = max(opts, key=lambda o: pp(o.get("estimated_ror_impact", {}).get("iar", "0")))
                    imp = best.get("estimated_ror_impact", {})
                    st.session_state.ror_values.update({
                        "dlr": parse_pct_value(imp.get("dlr", "40"), 40),
                        "da":  parse_pct_value(imp.get("da", "52"), 52),
                        "iar": parse_pct_value(imp.get("iar", "91"), 91),
                        "asy": parse_pct_value(imp.get("asy", "61"), 61),
                    })

            # Append result message
            st.session_state.messages.append({
                "role": "agent", "stage": stage, "data": cleaned_result, "stage_brief": stage_brief,
            })

        # Synthesis
        st.session_state.messages.append({"role": "synthesis", "text": all_results.get("synthesis", "Cycle complete.")})
        st.session_state.active_agent = None
        if st.session_state.runtime_warnings:
            status_ph.warning(
                "Live LLM/tool access degraded. The cycle completed using governed fallback outputs for: "
                + "; ".join(st.session_state.runtime_warnings[:4])
            )
        else:
            status_ph.empty()
        progress_ph.empty()
        st.rerun()


def _render_message(msg: dict):
    """Render a single message of any type."""
    role = msg.get("role")

    if role == "user":
        st.markdown(
            f'<div style="display:flex;justify-content:flex-end;margin:10px 0;">'
            f'<div class="user-bubble">{msg["text"]}</div></div>',
            unsafe_allow_html=True,
        )

    elif role == "agent":
        _render_agent_card(msg)

    elif role == "synthesis":
        st.markdown(
            f'<div class="synthesis-card">'
            f'<div style="font-family:Space Mono,monospace;font-size:9px;color:#16A34A;font-weight:800;margin-bottom:8px;letter-spacing:1px;">✓ AI-SRF GOVERNANCE CYCLE COMPLETE</div>'
            f'<div style="font-size:14px;color:#1C1917;line-height:1.7;white-space:pre-wrap;">{msg["text"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_agent_card(msg: dict):
    stage = msg.get("stage", "")
    data = msg.get("data", {})

    STAGE_META = {
        "env":      ("Environmental Monitor", 1, "#C2410C"),
        "socratic": ("Socratic Partner",      2, "#1E3A5F"),
        "forensic": ("Forensic Analyst",      2, "#1E3A5F"),
        "catalyst": ("Creative Catalyst",     2, "#1E3A5F"),
        "devils":   ("Devil's Advocate",      2, "#7F1D1D"),
        "scaffold": ("Implementation Scaffolding", 3, "#166534"),
        "monitor":  ("Monitoring Agent",      3, "#166534"),
    }
    name, layer, color = STAGE_META.get(stage, ("Agent", 2, "#555"))

    with st.expander(f"**L{layer} · {name}**", expanded=True):
        # Header
        col_badge, col_trace = st.columns([5, 1])
        with col_badge:
            st.markdown(f'<span class="layer-{layer}">Layer {layer}</span>&nbsp;<span style="font-size:13px;font-weight:700;color:{color};">{name}</span>', unsafe_allow_html=True)
        with col_trace:
            if st.checkbox("Show JSON trace", key=f"trace_{stage}_{id(msg)}", value=False):
                st.json(data, expanded=False)

        # ─── Environmental Monitor ───────────────────────────────────────
        if stage == "env":
            rs = data.get("current_risk_state", "Nominal")
            rc = RISK_COLORS.get(rs, RISK_COLORS["Nominal"])
            st.markdown(
                f'<div style="background:{rc["bg"]};border-radius:8px;padding:10px 14px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:24px;">{rc["icon"]}</span>'
                f'<div><div style="font-weight:800;color:{rc["hex"]};font-family:Playfair Display,serif;font-size:16px;">{rs} Risk State</div>'
                f'<div style="font-size:11px;color:#6B7280;">Bayesian classification — Eskom / Transnet / ZAR / Broadband</div></div></div>',
                unsafe_allow_html=True,
            )
            for sig in data.get("triggering_signals", []):
                st.markdown(f'↳ **{sig.get("signal_source","")}** — {sig.get("current_status","")} `{sig.get("latency_or_downtime_metric","")}`')
            if data.get("contingency_templates_activated"):
                st.caption("Templates: " + " · ".join(data["contingency_templates_activated"]))

        # ─── Socratic Partner ────────────────────────────────────────────
        elif stage == "socratic":
            if data.get("identified_blind_spots"):
                st.caption("Blind spots: " + " · ".join(data["identified_blind_spots"]))
            for q in data.get("socratic_questions", []):
                st.markdown(
                    f'<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:6px;padding:10px 14px;margin-bottom:7px;">'
                    f'<div style="font-size:10px;color:#C2410C;font-weight:700;margin-bottom:3px;">↳ {q.get("tied_to_signal","")}</div>'
                    f'<div style="font-size:13px;color:#1C1917;line-height:1.5;">{q.get("question","")}</div></div>',
                    unsafe_allow_html=True,
                )

        # ─── Forensic Analyst ────────────────────────────────────────────
        elif stage == "forensic":
            rs = data.get("risk_summary", {})
            sev = rs.get("severity", "High")
            sev_col = "#DC2626" if sev == "High" else "#CA8A04"
            st.markdown(
                f'<span style="background:{"#FEF2F2" if sev=="High" else "#FEFCE8"};color:{sev_col};'
                f'border-radius:10px;padding:2px 12px;font-size:11px;font-weight:700;">{sev} Severity</span>&nbsp;'
                f'<span style="background:#F3F4F6;color:#6B7280;border-radius:10px;padding:2px 12px;font-size:11px;">{rs.get("reversibility","")}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Executive summary:** {rs.get('executive_summary','')}")
            if data.get("distributional_audit_and_informal_economy"):
                st.markdown(f"**Distributional audit:** {data['distributional_audit_and_informal_economy'][:280]}…")
            for exp in data.get("regulatory_exposure", []):
                st.error(f"⚠ {exp}")

        # ─── Creative Catalyst ───────────────────────────────────────────
        elif stage == "catalyst":
            opts = data.get("strategic_options", [])
            if opts:
                opt_cols = st.columns(len(opts))
                OCFG = [("#14532D","#F0FDF4"), ("#7C2D12","#FFF7ED"), ("#1E3A5F","#EFF6FF")]
                for i, (o, col) in enumerate(zip(opts, opt_cols)):
                    border_color, bg_color = OCFG[i % len(OCFG)]
                    with col:
                        impact_chips = "".join(
                            [
                                f'<span style="background:white;padding:1px 6px;border-radius:4px;display:inline-block;margin:2px 4px 0 0;">'
                                f'<strong style="color:{border_color};">{k2.upper()}</strong> {v}</span>'
                                for k2, v in o.get("estimated_ror_impact", {}).items()
                            ]
                        )
                        st.markdown(
                            f'<div style="border-radius:10px;padding:14px;background:{bg_color};border:1px solid {border_color}33;height:100%;">'
                            f'<div style="font-size:9px;font-weight:800;color:{border_color};font-family:Space Mono,monospace;text-transform:uppercase;margin-bottom:4px;">{o.get("type","")}</div>'
                            f'<div style="font-size:13px;font-weight:700;color:{border_color};margin-bottom:7px;">{o.get("title","")}</div>'
                            f'<div style="font-size:12px;color:#374151;margin-bottom:10px;line-height:1.55;">{o.get("strategy_description","")[:140]}...</div>'
                            f'<div style="font-size:10px;color:#6B7280;">{impact_chips}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        # ─── Devil's Advocate ────────────────────────────────────────────
        elif stage == "devils":
            st.markdown('<span class="unbypassable">⚠ UNBYPASSABLE CHECKPOINT</span>', unsafe_allow_html=True)
            for r in data.get("stress_test_report", []):
                v = r.get("verdict", {})
                rating = v.get("rating", "PROCEED_WITH_MODIFICATION")
                vcls = "verdict-proceed" if rating == "PROCEED" else ("verdict-defer" if rating == "DEFER" else "verdict-mod")
                st.markdown(
                    f'<div style="margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid #E5E7EB;">'
                    f'<span style="font-size:12px;font-weight:700;color:#111827;">{r.get("option_title","")}</span>&nbsp;'
                    f'<span class="{vcls}">{rating}</span></div>',
                    unsafe_allow_html=True,
                )
                for flaw in r.get("fatal_flaws", [])[:2]:
                    st.markdown(f'✗ {flaw}')
                if v.get("mandatory_conditions"):
                    st.info(f"**Mandatory:** {v['mandatory_conditions']}")

        # ─── Implementation Scaffolding ──────────────────────────────────
        elif stage == "scaffold":
            plan = data if isinstance(data, dict) else {}
            TIERS = [
                ("tier_1_native_execution",    "Tier 1 — Native Execution",    "#14532D", "#F0FDF4", "task_name"),
                ("tier_2_ai_augmented_scaffolding", "Tier 2 — AI-Augmented Scaffolding", "#7C2D12", "#FFF7ED", "workflow"),
                ("tier_3_capability_development",   "Tier 3 — Capability Development",  "#1E3A5F", "#EFF6FF", "prerequisite"),
            ]
            for key, label, tc, tbg, field in TIERS:
                items = plan.get(key, [])
                if items:
                    st.markdown(f'<div style="font-weight:700;color:{tc};font-size:12px;background:{tbg};border-radius:4px;padding:3px 10px;margin-bottom:5px;">{label}</div>', unsafe_allow_html=True)
                    for it in items:
                        st.markdown(f'· {it.get(field, it.get("task_name", str(it))[:80])}')

        # ─── Monitoring Agent ─────────────────────────────────────────────
        elif stage == "monitor":
            met = data.get("ror_tracking_metrics", {})
            if met:
                st.markdown("**ROR Tracking Targets:**")
                for k, v in met.items():
                    st.markdown(f'`{k.replace("_target","").replace("_"," ").upper()}` — {v}')
            flags = data.get("behavioral_audit_flags", [])
            if flags:
                first_flag = flags[0]
                st.warning(f"**Behavioural Audit Flag:** {first_flag.get('agent_name','')} — {first_flag.get('description','')}")
            triggers = data.get("layer_1_rescan_triggers", [])
            if triggers:
                st.markdown("**Layer 1 Rescan Triggers:**")
                for t in triggers[:4]:
                    st.markdown(f'⟳ {t}')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_ror:
    st.markdown('<h2 style="font-family:Playfair Display,serif;">Return on Resilience Dashboard</h2>', unsafe_allow_html=True)
    st.markdown("Four indicators measuring the sustainable value generated by the AI-SRF governance cycle.")

    ror = st.session_state.ror_values
    c1, c2, c3, c4 = st.columns(4)
    for col, k, label, desc, c in [
        (c1, "dlr", "DLR", "Decision Latency Reduction", "#F97316"),
        (c2, "da",  "DA",  "Decision Alpha",             "#1E3A5F"),
        (c3, "iar", "IAR", "Infrastructure Autonomy",    "#166534"),
        (c4, "asy", "ASY", "Algorithmic Sovereignty",    "#7C2D12"),
    ]:
        with col:
            val = ror.get(k, 0)
            st.markdown(
                f'<div style="background:white;border-radius:12px;padding:20px;border-top:3px solid {c};box-shadow:0 1px 4px rgba(0,0,0,.06);">'
                f'<div style="font-size:38px;font-weight:800;color:{c};font-family:Playfair Display,serif;">{f"{val}%" if val else "—"}</div>'
                f'<div style="font-size:12px;font-weight:700;color:#1C1917;margin-top:4px;">{label}</div>'
                f'<div style="font-size:10px;color:#6B7280;margin-top:3px;">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    opts = st.session_state.cycle_results.get("catalyst", {}).get("strategic_options", [])
    if opts:
        col_bar, col_radar = st.columns(2)
        with col_bar:
            st.plotly_chart(make_ror_bar_chart(opts), use_container_width=True)
        with col_radar:
            if any(v > 0 for v in ror.values()):
                st.plotly_chart(make_ror_radar(ror), use_container_width=True)
    else:
        st.info("Submit a digitalisation challenge in the Strategic Dialogue tab to generate ROR projections.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVIDENCE BASE
# ══════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.markdown('<h2 style="font-family:Playfair Display,serif;">SA Institutional Evidence Base</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-intro">Five-stage sovereign RAG pipeline: POPIA-gated ingestion, chunking and embeddings, semantic retrieval, regulatory context enforcement, and governance monitoring.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        **Embedding Model:** Groq `nomic-embed-text-v1_5` (OpenAI-compatible endpoint) ·
        **Vector DB:** ChromaDB persistent (`{CHROMA_PERSIST_DIR}`) ·
        **Chunking:** RecursiveCharacterTextSplitter (1000/200) ·
        **Sources:** RAW_DATA_RAG PDFs · World Bank Open Data · Stats SA · Eskom EskomSePush · Information Regulator SA
        """
    )

    try:
        kb = load_rag_engine()
        n = kb.collection_size()
        st.success(f"✓ ChromaDB active — {n} indexed chunks · Groq nomic-embed-text-v1_5 embeddings")
    except Exception as e:
        st.error(f"RAG engine error: {e}")

    if st.session_state.rag_package:
        pkg = st.session_state.rag_package
        c_pkg1, c_pkg2, c_pkg3 = st.columns(3)
        c_pkg1.metric("Retrieved Chunks", pkg.get("retrieved_count", 0))
        c_pkg2.metric("Enforced Chunks", pkg.get("enforced_count", 0))
        c_pkg3.metric("ASY Hint", f"{pkg.get('algorithmic_sovereignty_yield_hint', 0)}%")

        with st.expander("Current Governance Context Package", expanded=False):
            st.markdown("**Regulatory anchors injected into the context window:**")
            for anchor in pkg.get("regulatory_anchors", []):
                st.markdown(f"- {anchor}")
            st.markdown("**Compliance notes:**")
            for note in pkg.get("compliance_notes", []):
                st.markdown(f"- {note}")

    # PDF uploader
    st.subheader("Add Documents to RAW_DATA_RAG")
    uploaded = st.file_uploader("Upload PDF documents to expand the knowledge base", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        os.makedirs(RAG_DATA_DIR, exist_ok=True)
        for f in uploaded:
            dest = Path(RAG_DATA_DIR) / f.name
            dest.write_bytes(f.read())
        st.success(f"Uploaded {len(uploaded)} PDF(s) to RAW_DATA_RAG. Reload to index.")
        if st.button("Re-index now"):
            st.cache_resource.clear()
            st.rerun()

    st.subheader("Indexed Knowledge Documents")
    retrieved_ids = {d["id"] for d in (kb.retrieve("South Africa digital strategy", k=20) if "kb" in dir() else [])}
    for doc in SA_CORPUS:
        is_ret = doc["id"] in retrieved_ids
        with st.expander(f"{'🔶 ' if is_ret else ''}{doc['title']}", expanded=False):
            st.markdown(f"**Source:** {doc['source']}")
            st.markdown(doc["content"])
            if is_ret:
                st.success("Retrieved in last query")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PEER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_peer:
    st.markdown('<h2 style="font-family:Playfair Display,serif;">Emerging Market Peer Analysis</h2>', unsafe_allow_html=True)
    st.markdown("How comparable organisations in infrastructure-constrained markets addressed similar digitalisation challenges — and what SA corporates can extract.")

    for c in COMPETITOR_CASES:
        with st.expander(f"**{c['company']}** — {c['market']}", expanded=False):
            col_info, col_ror = st.columns(2)
            with col_info:
                st.markdown(f"📍 **{c['market']}**")
                st.markdown(f"**Challenge:** {c['challenge']}")
                st.markdown(f"**Approach:** {c['approach']}")
                st.markdown(f"**Outcome:** {c['outcome']}")
                st.markdown(
                    f'<div style="background:#FFF7ED;border-left:3px solid #F97316;border-radius:6px;padding:10px 14px;margin-top:8px;">'
                    f'<strong style="color:#C2410C;">SA Relevance:</strong> {c["sa_relevance"]}</div>',
                    unsafe_allow_html=True,
                )
            with col_ror:
                st.markdown("**ROR Outcomes:**")
                for k, v in c["ror"].items():
                    st.metric(label=k.upper(), value=v)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SYSTEM CARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_syscard:
    st.markdown('<h2 style="font-family:Playfair Display,serif;">AI System Card</h2>', unsafe_allow_html=True)
    st.markdown(
        "Formal transparency artifact for the AI-SRF in accordance with **King IV Principle 12** (IoDSA 2016) "
        "and Mitchell et al. (2019). Auditable by regulatory bodies and adopting organisations."
    )

    SYSTEM_CARD_SECTIONS = [
        ("Section 6 Implementation Mapping", (
            "Executable prototype objective: instantiate the AI-SRF as a working socio-technical artefact for Silicon Sampling and Delphi validation.\n\n"
            "Orchestration: LangGraph state machine coordinating seven prompt-governed specialist agents.\n\n"
            "Inference and tooling: Groq reasoning and embedding endpoints, MCP-style governance server profile, structured function tools, and sovereign RAG package delivery.\n\n"
            "Auditability: each node emits governance-trace metadata with prompt and output digests suitable for board-facing review."
        )),
        ("🎯 Intended Use & Scope Restriction", (
            "Corporate digitalisation strategy advisory for South African organisations **exclusively**. "
            "Guardrail blocks all non-SA-corporate-digitalisation queries at input layer.\n\n"
            "**Permitted:** AI deployment governance · Digital transformation strategy · Cloud architecture · "
            "POPIA/King IV/B-BBEE compliance · Infrastructure resilience · ROR measurement · "
            "Informal economy integration\n\n"
            "**Not permitted:** Personal financial advice · Legal practice · Medical applications · "
            "Non-SA corporate contexts · Non-digitalisation queries"
        )),
        ("🔒 Content Safety Policies (Hard-Coded)", (
            "1. **POPIA (Act 4 of 2013):** Processes minimum PII necessary. Zero cross-session data retention. "
            "No cross-border PII transfer guidance without TIA requirement.\n"
            "2. **Employment Equity Act:** All outputs audited for discriminatory proxy variables via "
            "Forensic Analyst distributional audit (mandatory, non-skippable).\n"
            "3. **Governance override:** Cannot generate strategies that circumvent embedded governance safeguards. "
            "Devil's Advocate is architecturally unbypassable.\n"
            "4. **Scope guardrail:** All non-SA-corporate-digitalisation queries rejected at input layer."
        )),
        ("⚙️ Model & Infrastructure Stack", (
            "**LLM inference:** Groq `llama-3.3-70b-versatile` (ADK-style 7-agent orchestration with function calling)\n"
            "**Embedding:** Groq `nomic-embed-text-v1_5` (OpenAI-compatible endpoint, 768-dim, 8192-token context)\n"
            "**Vector DB:** ChromaDB persistent client (cosine similarity, HNSW index)\n"
            "**RAG pipeline:** LangChain `PyPDFDirectoryLoader` + `RecursiveCharacterTextSplitter` (1000/200)\n"
            "**Data sources:** RAW_DATA_RAG corpus · World Bank Open Data API · Stats SA · "
            "Eskom EskomSePush API · Information Regulator SA\n"
            "**Tools (function calling):** World Bank indicator API · SA infrastructure signals · "
            "Data provenance audit · ROR baseline estimator · B-BBEE compliance checker\n"
            "**UI:** Streamlit with custom CSS (DM Sans / Playfair Display / Space Mono)"
        )),
        ("🏗️ 3-Layer Governance Architecture", (
            "**Layer 1 (Sensing):** Environmental Monitor Agent — Bayesian updating on Eskom/Transnet/ZAR/broadband. "
            "Four risk states: Nominal/Elevated/Compound/Critical.\n\n"
            "**Layer 2 (Reasoning):**\n"
            "- Socratic Partner → diagnostic framing, executive blind spots\n"
            "- Forensic Analyst → dependency mapping + distributional audit + regulatory scan\n"
            "- Creative Catalyst → Hedge/Exploit/Defer options with ROR projections\n"
            "- Devil's Advocate → **UNBYPASSABLE** adversarial stress-test\n\n"
            "**Layer 3 (Alignment):**\n"
            "- Implementation Scaffolding → 3-tier capability decomposition\n"
            "- Monitoring Agent → ROR tracking + behavioural audit + Layer 1 rescan triggers\n\n"
            "**Defence-in-depth:** No single point of failure can generate an unchecked AI-driven strategic error."
        )),
        ("📐 ROR Measurement Methodology", (
            "**DLR (Decision Latency Reduction):** Time from disruption signal to board-validated response. "
            "Baseline: 72hrs. Target: ≤43hrs. Instrument: automated timestamp comparison.\n\n"
            "**DA (Decision Alpha):** Quality delta vs baseline — 7-pt Delphi Likert scale across "
            "viability + compliance + risk mitigation dimensions. Baseline: 4.1. Target: ≥5.2.\n\n"
            "**IAR (Infrastructure Autonomy Ratio):** Edge AI uptime % independent of national grid. "
            "Baseline: 61%. Target: ≥91%.\n\n"
            "**ASY (Algorithmic Sovereignty Yield):** % recommendations grounded in SA-local institutional "
            "data vs unconstrained global training priors. Target: ≥85%."
        )),
        ("⚖️ Regulatory Compliance", (
            "| Regulation | How Addressed |\n"
            "|---|---|\n"
            "| POPIA Act 4/2013 | Data minimisation · No cross-session retention · TIA required for cross-border processing |\n"
            "| King IV (IoDSA 2016) | Transparent reasoning traces · System Card · Board-auditable JSON outputs |\n"
            "| Employment Equity Act | Forensic Analyst distributional audit · Proxy variable detection |\n"
            "| B-BBEE Codes 2015 | Workforce impact assessment in Creative Catalyst · Skills Development SLA |\n"
            "| Zondo Commission 2022 | Data provenance audit tool · State capture contamination flags |"
        )),
        ("⚠️ Known Limitations & Validation Status", (
            "1. Real-time Eskom/Transnet API integration uses best-available public endpoints — accuracy depends on API uptime.\n"
            "2. ROR baselines require 6-month operational data for organisation-specific calibration.\n"
            "3. Silicon Sampling validation (9 personas × 4 disruption scenarios) pending Delphi expert panel (Phase 2).\n"
            "4. Synchronous behavioural review (pre-execution blocking) is on architectural roadmap — currently asynchronous.\n"
            "5. Competitor case studies are illustrative; SA-specific competitive intelligence requires proprietary data integration.\n"
            "6. Voice interface (speech-to-text/TTS) is optional and depends on browser Web Speech API support."
        )),
    ]

    for section_title, section_body in SYSTEM_CARD_SECTIONS:
        with st.expander(section_title, expanded=False):
            st.markdown(section_body)

    if st.session_state.governance_trace:
        with st.expander("Live Governance Trace", expanded=False):
            st.caption("Prompt and output digests captured per agent node for board-auditable review.")
            st.json(st.session_state.governance_trace, expanded=False)

    with st.expander("MCP Health", expanded=False):
        st.caption("Runtime probe of configured MCP servers from the current machine.")
        if st.button("Run MCP Health Probe", key="run_mcp_health_panel"):
            st.session_state.mcp_health = probe_mcp_servers()
            st.rerun()
        health = st.session_state.mcp_health
        if not health:
            st.info("No MCP health probe has been run yet.")
        else:
            a, b = st.columns(2)
            a.metric("Configured Servers", health.get("configured_count", 0))
            b.metric("Healthy Servers", health.get("healthy_count", 0))
            for item in health.get("servers", []):
                label = "OK" if item.get("status") == "ok" else "ERROR"
                with st.expander(f"{item.get('server')} · {label}", expanded=False):
                    st.write(f"Command: `{item.get('command')}`")
                    st.write(f"Args: `{item.get('args', [])}`")
                    if item.get("status") == "ok":
                        st.write(f"Tools discovered: {item.get('tool_count', 0)}")
                        if item.get("tools"):
                            st.write(", ".join(item.get("tools", [])))
                    else:
                        st.error(item.get("error", "Unknown error"))

    st.divider()
    st.markdown(
        '<div style="text-align:center;font-size:12px;color:#9CA3AF;font-family:Space Mono,monospace;">'
        'AI-SRF v33 · By: Bright Sikazwe, PhD Candidate · University of Johannesburg · '
        'College of Business and Economics · Department of Information and Knowledge Management<br>'
        '"Umuntu ngumuntu ngabantu" — AI governance in service of African institutional contexts'
        '</div>',
        unsafe_allow_html=True,
    )
