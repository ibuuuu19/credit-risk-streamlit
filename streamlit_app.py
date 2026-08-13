"""
CreditScore Pro v4.0 — Application Streamlit Premium
======================================================
• Page Diagnostic : scoring de risque de crédit (Spark ML / Random Forest)
• Page Dashboard  : historique persistant + KPIs + graphiques + export PDF
• Mode Dark/Light avec persistance et transition fluide

v4.0 — UI Premium + Theme System + Charts adaptatifs
"""

import json
import os
import time
import uuid
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, DoubleType, StringType
)
from pyspark.ml import PipelineModel

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# --------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------
MODEL_PATH   = "credit_risk_pipeline_model"
HISTORY_FILE = "historique_analyses.json"

SEUIL_MODERE = 0.20
SEUIL_ELEVE  = 0.40

COULEURS = {
    "Favorable":   "#0FA36B",
    "Modéré":      "#E8A13A",
    "Défavorable": "#D64545",
}

def couleur_risque(v):
    if v < 20:  return COULEURS["Favorable"]
    if v < 40:  return COULEURS["Modéré"]
    return COULEURS["Défavorable"]

st.set_page_config(
    page_title="CreditScore Pro — Risque de Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------
# Initialisation du thème (session_state)
# --------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# --------------------------------------------------------------------
# Design system CSS — Thème dynamique
# --------------------------------------------------------------------
THEME = st.session_state.theme

# Palette selon le thème
if THEME == "dark":
    C = {
        "bg_0": "#0A0E1A",
        "bg_1": "#0F1524",
        "bg_2": "#161D30",
        "bg_card": "rgba(22,29,48,.72)",
        "bg_card_solid": "#161D30",
        "border": "rgba(255,255,255,.08)",
        "border_2": "rgba(255,255,255,.14)",
        "border_hover": "rgba(212,175,55,.45)",
        "text": "#F5F7FB",
        "text_2": "#A8B2C9",
        "text_3": "#6B7594",
        "text_4": "#3F4764",
        "navy": "#D4AF37",         # Gold en dark pour contraste
        "navy_2": "#1B3B6F",
        "hero_g1": "#0B1F3A",
        "hero_g2": "#1B3B6F",
        "hero_g3": "#27498B",
        "sidebar_bg": "rgba(15,21,36,.92)",
        "shadow": "0 12px 36px rgba(0,0,0,.4)",
        "shadow_glow": "0 0 40px rgba(212,175,55,.15)",
        "glass": "rgba(255,255,255,.04)",
        "glass_border": "rgba(255,255,255,.1)",
        "blue": "#2E6DB4",         # ✅ Correction KeyError 'blue'
        "gold": "#D4AF37",         # ✅ Correction KeyError 'gold'
    }
else:
    C = {
        "bg_0": "#F4F6FB",
        "bg_1": "#E9EEF7",
        "bg_2": "#FFFFFF",
        "bg_card": "rgba(255,255,255,.95)",
        "bg_card_solid": "#FFFFFF",
        "border": "rgba(18,36,74,.08)",
        "border_2": "rgba(18,36,74,.14)",
        "border_hover": "rgba(184,148,31,.45)",
        "text": "#0F1A2E",
        "text_2": "#4A5572",
        "text_3": "#7A8699",
        "text_4": "#B5BCCC",
        "navy": "#12244A",
        "navy_2": "#1B3B6F",
        "hero_g1": "#0B1F3A",
        "hero_g2": "#1B3B6F",
        "hero_g3": "#27498B",
        "sidebar_bg": "rgba(255,255,255,.88)",
        "shadow": "0 10px 28px rgba(18,36,74,.08)",
        "shadow_glow": "0 0 40px rgba(184,148,31,.12)",
        "glass": "rgba(255,255,255,.6)",
        "glass_border": "rgba(18,36,74,.08)",
        "blue": "#2E6DB4",         # ✅ Correction KeyError 'blue'
        "gold": "#D4AF37",         # ✅ Correction KeyError 'gold'
    }

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {{
    --bg-0: {C['bg_0']};
    --bg-1: {C['bg_1']};
    --bg-2: {C['bg_2']};
    --bg-card: {C['bg_card']};
    --border: {C['border']};
    --border-2: {C['border_2']};
    --border-hover: {C['border_hover']};
    --text: {C['text']};
    --text-2: {C['text_2']};
    --text-3: {C['text_3']};
    --text-4: {C['text_4']};
    --navy: {C['navy']};
    --navy-2: {C['navy_2']};
    --hero-g1: {C['hero_g1']};
    --hero-g2: {C['hero_g2']};
    --hero-g3: {C['hero_g3']};
    --shadow: {C['shadow']};
    --shadow-glow: {C['shadow_glow']};
    --glass: {C['glass']};
    --glass-border: {C['glass_border']};
    --green: #0FA36B;
    --red: #D64545;
    --amber: #E8A13A;
    --blue: #2E6DB4;
    --gold: #D4AF37;
    --gold-d: #B8941F;
}}

html, body, .stApp {{
    background: var(--bg-0);
    font-family: 'Manrope', sans-serif;
    color: var(--text);
    transition: background .4s ease, color .4s ease;
}}
#MainMenu {{visibility: hidden;}}
footer    {{visibility: hidden;}}
.block-container {{
    padding-top: 4.5rem;
    max-width: 1280px;
    background: linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 100%);
}}
h1,h2,h3,h4 {{
    font-family: 'Sora', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -.02em;
}}
p, span, div, label {{
    transition: color .3s ease, background .3s ease, border-color .3s ease;
}}

/* ════════════ SIDEBAR ════════════ */
section[data-testid="stSidebar"] {{
    background: var(--sidebar-bg) !important;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,.08);
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 1.5rem;
}}
section[data-testid="stSidebar"] h2 {{
    font-family: 'Sora' !important;
    font-size: 1.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -.02em !important;
    background: linear-gradient(135deg, var(--gold) 0%, #F0D488 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1.5rem !important;
}}

.theme-toggle-wrap {{
    display: flex;
    justify-content: center;
    margin: 1rem 0 1.5rem;
}}
.theme-toggle {{
    display: inline-flex; align-items: center; gap: 10px;
    padding: 10px 18px;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 30px;
    backdrop-filter: blur(10px);
    transition: all .3s ease;
    font-family: 'Manrope', sans-serif;
    font-weight: 600; font-size: .85rem;
    color: var(--text);
    cursor: pointer;
    width: 100%;
    justify-content: center;
}}
.theme-toggle:hover {{
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}}
.theme-toggle .icon {{
    font-size: 1.1rem;
    transition: transform .4s ease;
}}
.theme-toggle:hover .icon {{
    transform: rotate(20deg) scale(1.15);
}}

section[data-testid="stSidebar"] .stRadio > div {{
    display: flex; flex-direction: column; gap: 8px;
    margin-top: 1rem;
}}
section[data-testid="stSidebar"] .stRadio label {{
    display: flex; align-items: center; gap: 12px;
    padding: 14px 18px; border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--glass);
    backdrop-filter: blur(10px);
    cursor: pointer; font-weight: 600; font-size: .9rem;
    transition: all .3s ease; margin: 0;
    color: var(--text-2);
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    border-color: var(--border-hover);
    transform: translateX(4px);
    color: var(--text);
}}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: linear-gradient(135deg, var(--hero-g1), var(--hero-g2));
    color: #fff;
    border-color: transparent;
    box-shadow: 0 8px 24px rgba(18,36,74,.35);
}}
section[data-testid="stSidebar"] .stRadio input[type="radio"] {{ display: none; }}

div[data-testid="stMetric"] {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 14px 16px;
    backdrop-filter: blur(10px);
    transition: all .3s ease;
}}
div[data-testid="stMetric"]:hover {{
    border-color: var(--border-hover);
    transform: translateY(-2px);
    box-shadow: var(--shadow-glow);
}}
div[data-testid="stMetric"] label {{
    color: var(--text-3) !important;
    font-weight: 600 !important;
    font-size: .72rem !important;
    letter-spacing: .5px !important;
    text-transform: uppercase !important;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: var(--text) !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 800 !important;
}}

.hero {{
    background: linear-gradient(135deg, var(--hero-g1) 0%, var(--hero-g2) 55%, var(--hero-g3) 100%);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 1rem;
    box-shadow: 0 16px 42px rgba(11,31,58,.35), inset 0 1px 0 rgba(255,255,255,.08);
    position: relative; overflow: hidden;
    margin: .5rem 0 1.8rem;
    animation: fadeUp .6s ease both;
    border: 1px solid rgba(255,255,255,.06);
}}
.hero::before {{
    content: ''; position: absolute; inset: 0;
    background:
      radial-gradient(circle at 85% 20%, rgba(212,175,55,.22) 0%, transparent 45%),
      radial-gradient(circle at 10% 90%, rgba(255,255,255,.1) 0%, transparent 40%),
      radial-gradient(circle at 50% 120%, rgba(46,109,180,.15) 0%, transparent 55%);
}}
.hero::after {{
    content: ''; position: absolute; inset: 0;
    background-image:
      repeating-linear-gradient(45deg, rgba(255,255,255,.015) 0, rgba(255,255,255,.015) 1px, transparent 1px, transparent 20px),
      repeating-linear-gradient(-45deg, rgba(255,255,255,.01) 0, rgba(255,255,255,.01) 1px, transparent 1px, transparent 20px);
    opacity: .6;
}}
.hero h1 {{
    color: #fff !important;
    font-size: 1.9rem !important;
    margin: 0 !important;
    letter-spacing: -.025em !important;
    position: relative;
    line-height: 1.15 !important;
    font-weight: 800 !important;
}}
.hero h1 span {{
    background: linear-gradient(90deg, #D4AF37 0%, #F0D488 50%, #D4AF37 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    font-style: italic;
}}
.hero p {{
    color: rgba(255,255,255,.82);
    margin: .35rem 0 0 !important;
    font-size: .92rem;
    position: relative;
    font-weight: 400;
}}
.hero-badges {{
    display: flex; gap: 8px; flex-wrap: wrap;
    position: relative;
}}
.hbadge {{
    background: rgba(255,255,255,.1);
    border: 1px solid rgba(255,255,255,.2);
    backdrop-filter: blur(12px);
    color: #fff;
    padding: 6px 14px; border-radius: 30px;
    font-size: .74rem; font-weight: 600;
    letter-spacing: .02em;
    transition: all .3s ease;
    display: inline-flex; align-items: center; gap: 6px;
}}
.hbadge:hover {{
    background: rgba(255,255,255,.18);
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0,0,0,.2);
}}
.hbadge.gold {{
    background: linear-gradient(135deg, rgba(212,175,55,.25), rgba(212,175,55,.1));
    border-color: rgba(212,175,55,.6);
    color: #F0D488;
    box-shadow: 0 0 20px rgba(212,175,55,.2);
}}

.kpi {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    display: flex; gap: 16px; align-items: center;
    box-shadow: var(--shadow);
    border-left: 4px solid var(--navy);
    transition: all .4s cubic-bezier(.16,1,.3,1);
    margin-bottom: 14px;
    animation: fadeUp .5s ease both;
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}}
.kpi::before {{
    content: '';
    position: absolute; top: 0; right: 0;
    width: 80px; height: 80px;
    background: radial-gradient(circle, rgba(212,175,55,.08) 0%, transparent 70%);
    transform: translate(20px, -20px);
    transition: transform .4s ease;
}}
.kpi:hover {{
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,.2);
    border-color: var(--border-hover);
}}
.kpi:hover::before {{
    transform: translate(10px, -10px) scale(1.3);
}}
.kpi-icon {{
    width: 52px; height: 52px;
    border-radius: 14px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    transition: transform .4s cubic-bezier(.34,1.56,.64,1);
    position: relative;
    z-index: 1;
}}
.kpi:hover .kpi-icon {{
    transform: scale(1.1) rotate(-5deg);
}}
.kpi-label {{
    font-size: .7rem; text-transform: uppercase;
    letter-spacing: .8px; color: var(--text-3);
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}}
.kpi-value {{
    font-family: 'Sora', sans-serif;
    font-size: 1.65rem; font-weight: 800;
    color: var(--text);
    line-height: 1.1;
    letter-spacing: -.02em;
    margin-top: 2px;
}}
.kpi-sub {{
    font-size: .75rem; color: var(--text-3);
    font-weight: 500; margin-top: 3px;
}}
.tone-green .kpi-icon {{ background: rgba(15,163,107,.12); color: var(--green); }}
.tone-green {{ border-left-color: var(--green); }}
.tone-red   .kpi-icon {{ background: rgba(214,69,69,.12); color: var(--red); }}
.tone-red   {{ border-left-color: var(--red); }}
.tone-amber .kpi-icon {{ background: rgba(232,161,58,.12); color: var(--amber); }}
.tone-amber {{ border-left-color: var(--amber); }}
.tone-gold  .kpi-icon {{ background: rgba(212,175,55,.12); color: var(--gold); }}
.tone-gold  {{ border-left-color: var(--gold); }}
.tone-blue  .kpi-icon {{ background: rgba(46,109,180,.12); color: var(--blue); }}
.tone-blue  {{ border-left-color: var(--blue); }}
.tone-navy  .kpi-icon {{ background: rgba(27,59,111,.12); color: var(--navy-2); }}
.tone-navy  {{ border-left-color: var(--navy-2); }}

.section-title {{
    display: flex; gap: 14px; align-items: center;
    margin: 2.5rem 0 1.2rem;
    position: relative;
}}
.section-title .st-icon {{
    width: 46px; height: 46px; border-radius: 14px;
    font-size: 1.3rem;
    background: linear-gradient(135deg, var(--hero-g1), var(--hero-g2));
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 20px rgba(18,36,74,.28), inset 0 1px 0 rgba(255,255,255,.1);
    color: #fff;
    flex-shrink: 0;
    transition: transform .3s cubic-bezier(.34,1.56,.64,1);
}}
.section-title:hover .st-icon {{
    transform: rotate(-6deg) scale(1.08);
}}
.section-title h3 {{
    margin: 0 !important;
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    letter-spacing: -.02em !important;
}}
.section-title p {{
    margin: 3px 0 0 !important;
    font-size: .85rem !important;
    color: var(--text-3) !important;
    font-weight: 500 !important;
}}

.chart-head {{
    margin: 1.5rem 0 .75rem;
    padding-left: 2px;
    border-left: 3px solid var(--gold);
    padding-left: 14px;
}}
.chart-head h4 {{
    margin: 0 !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    letter-spacing: -.01em !important;
}}
.chart-head p {{
    margin: 3px 0 0 !important;
    font-size: .82rem !important;
    color: var(--text-3) !important;
}}

.insight {{
    display: flex; gap: 14px; align-items: flex-start;
    border-radius: 16px; padding: 16px 18px;
    margin: 12px 0 20px;
    font-size: .88rem; line-height: 1.65;
    border: 1px solid;
    animation: fadeUp .5s ease both;
    position: relative;
    overflow: hidden;
}}
.insight::before {{
    content: '';
    position: absolute; top: 0; left: 0;
    width: 3px; height: 100%;
}}
.insight .i-icon {{
    font-size: 1.3rem;
    flex-shrink: 0;
    margin-top: 2px;
}}
.insight-info {{
    background: rgba(46,109,180,.08);
    border-color: rgba(46,109,180,.25);
    color: var(--text);
}}
.insight-info::before {{ background: var(--blue); }}
.insight-success {{
    background: rgba(15,163,107,.08);
    border-color: rgba(15,163,107,.25);
    color: var(--text);
}}
.insight-success::before {{ background: var(--green); }}
.insight-warning {{
    background: rgba(232,161,58,.08);
    border-color: rgba(232,161,58,.25);
    color: var(--text);
}}
.insight-warning::before {{ background: var(--amber); }}
.insight-danger {{
    background: rgba(214,69,69,.08);
    border-color: rgba(214,69,69,.25);
    color: var(--text);
}}
.insight-danger::before {{ background: var(--red); }}

.result-card {{
    border-radius: 22px; text-align: center;
    padding: 2.5rem 1.5rem;
    color: #fff;
    margin: 1.4rem 0;
    position: relative; overflow: hidden;
    animation: fadeUp .55s ease both;
    border: 1px solid rgba(255,255,255,.08);
}}
.result-card::before {{
    content: ''; position: absolute; inset: 0;
    background:
      radial-gradient(circle at 80% 10%, rgba(255,255,255,.2), transparent 50%),
      radial-gradient(circle at 20% 90%, rgba(255,255,255,.08), transparent 50%);
}}
.result-card::after {{
    content: ''; position: absolute; inset: 0;
    background-image:
      repeating-linear-gradient(45deg, rgba(255,255,255,.02) 0, rgba(255,255,255,.02) 1px, transparent 1px, transparent 15px);
}}
.result-low  {{
    background: linear-gradient(135deg, #0FA36B 0%, #0B8558 50%, #066843 100%);
    box-shadow: 0 18px 44px rgba(15,163,107,.38), inset 0 1px 0 rgba(255,255,255,.15);
}}
.result-mid  {{
    background: linear-gradient(135deg, #E8A13A 0%, #C77F1B 50%, #A96815 100%);
    box-shadow: 0 18px 44px rgba(232,161,58,.38), inset 0 1px 0 rgba(255,255,255,.15);
}}
.result-high {{
    background: linear-gradient(135deg, #D64545 0%, #A93030 50%, #851F1F 100%);
    box-shadow: 0 18px 44px rgba(214,69,69,.38), inset 0 1px 0 rgba(255,255,255,.15);
}}
.result-score {{
    font-family: 'Sora', sans-serif;
    font-size: 4.2rem; font-weight: 800;
    margin: .5rem 0;
    letter-spacing: -.03em;
    text-shadow: 0 2px 20px rgba(0,0,0,.2);
    position: relative;
}}
.result-label {{
    font-size: 1.1rem; font-weight: 700;
    letter-spacing: -.01em;
    position: relative;
}}
.result-detail {{
    display: inline-block;
    background: rgba(255,255,255,.14);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,.2);
    padding: 7px 20px; border-radius: 30px;
    font-size: .85rem; margin-top: .7rem;
    font-weight: 600;
    position: relative;
}}

.stButton > button {{
    background: linear-gradient(135deg, var(--hero-g1), var(--hero-g2)) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: .8rem 1.6rem !important;
    transition: all .35s cubic-bezier(.16,1,.3,1) !important;
    box-shadow: 0 8px 22px rgba(18,36,74,.3) !important;
    position: relative !important;
    overflow: hidden !important;
    letter-spacing: .01em !important;
}}
.stButton > button::before {{
    content: ''; position: absolute;
    top: 0; left: -100%; width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.2), transparent);
    transition: left .6s ease;
}}
.stButton > button:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 14px 32px rgba(18,36,74,.45) !important;
}}
.stButton > button:hover::before {{
    left: 100%;
}}
.stButton > button:active {{
    transform: translateY(-1px) !important;
}}

.stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
.stTextInput input, textarea {{
    border-radius: 12px !important;
    border: 1.5px solid var(--border-2) !important;
    background: var(--glass) !important;
    color: var(--text) !important;
    transition: all .25s ease !important;
}}
.stNumberInput input:focus, .stSelectbox div[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus, textarea:focus {{
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(212,175,55,.15) !important;
}}

div[data-testid="stDataFrame"] {{
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}}

.empty-state {{
    background: var(--bg-card);
    border: 2px dashed var(--border-2);
    border-radius: 24px;
    text-align: center;
    padding: 5rem 2rem;
    color: var(--text-3);
    backdrop-filter: blur(12px);
    position: relative;
    overflow: hidden;
}}
.empty-state::before {{
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% 50%, rgba(212,175,55,.06) 0%, transparent 60%);
}}
.empty-state .big {{
    font-size: 4rem;
    display: block;
    margin-bottom: .5rem;
    animation: float 3s ease-in-out infinite;
    position: relative;
}}
.empty-state h3 {{
    color: var(--text) !important;
    margin: .8rem 0 .5rem !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    position: relative;
}}
.empty-state p {{
    position: relative;
    max-width: 420px;
    margin: 0 auto;
    line-height: 1.7;
}}

.footer {{
    text-align: center; color: var(--text-3);
    font-size: .82rem;
    padding: 2.5rem 0 1.5rem;
    border-top: 1px solid var(--border);
    margin-top: 4rem;
    position: relative;
}}
.footer::before {{
    content: ''; position: absolute;
    top: -1px; left: 50%; transform: translateX(-50%);
    width: 60px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}}
.footer strong {{
    color: var(--text);
    font-weight: 700;
}}
.footer a {{
    color: var(--gold);
    text-decoration: none;
    transition: color .25s ease;
}}
.footer a:hover {{
    color: var(--gold-d);
}}

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); filter: blur(4px); }}
    to   {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
}}
@keyframes shimmer {{
    0% {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-10px); }}
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: .7; transform: scale(1.08); }}
}}

.stAlert {{
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: blur(10px);
}}
hr {{
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}}
[data-testid="stSidebar"] hr {{
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
    opacity: .5;
}}

.streamlit-expanderHeader {{
    font-weight: 700 !important;
    color: var(--text) !important;
}}

div[data-testid="stColumn"] {{
    transition: all .3s ease;
}}

@media (max-width: 768px) {{
    .hero {{ padding: 1.25rem 1.4rem; }}
    .hero h1 {{ font-size: 1.4rem !important; }}
    .result-score {{ font-size: 3rem; }}
}}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--bg-1); }}
::-webkit-scrollbar-thumb {{
    background: linear-gradient(180deg, var(--hero-g2), var(--hero-g3));
    border-radius: 10px;
    border: 2px solid var(--bg-1);
}}
::-webkit-scrollbar-thumb:hover {{
    background: linear-gradient(180deg, var(--gold), var(--gold-d));
}}

:focus-visible {{
    outline: 2px solid var(--gold);
    outline-offset: 2px;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Helpers HTML
# --------------------------------------------------------------------
def hero(titre, sous, badges):
    b = "".join(f'<span class="hbadge{" gold" if g else ""}">{t}</span>' for t, g in badges)
    st.markdown(f"""
    <div class="hero">
        <div>
            <h1>🏦 CreditScore <span>Pro</span></h1>
            <p>{sous}</p>
        </div>
        <div class="hero-badges">{b}</div>
    </div>""", unsafe_allow_html=True)

def kpi_card(icone, label, valeur, sous="", tone="navy"):
    st.markdown(f"""
    <div class="kpi tone-{tone}">
        <div class="kpi-icon">{icone}</div>
        <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{valeur}</div>
            <div class="kpi-sub">{sous}</div>
        </div>
    </div>""", unsafe_allow_html=True)

def section_title(icon, titre, sous=None):
    s = f'<div class="section-title"><span class="st-icon">{icon}</span><div><h3>{titre}</h3>'
    if sous:
        s += f"<p>{sous}</p>"
    s += "</div></div>"
    st.markdown(s, unsafe_allow_html=True)

def chart_title(icon, titre, sous):
    st.markdown(f'<div class="chart-head"><h4>{icon} {titre}</h4><p>{sous}</p></div>',
                unsafe_allow_html=True)

def insight_box(icon, titre, texte, tone="info"):
    st.markdown(f"""
    <div class="insight insight-{tone}">
        <span class="i-icon">{icon}</span>
        <div><strong>{titre}</strong> — {texte}</div>
    </div>""", unsafe_allow_html=True)

def style_fig(fig, height=320, legend=False):
    text_color = C["text"]
    grid_color = C["border"]
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=text_color, size=12),
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=legend,
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        hoverlabel=dict(
            bgcolor=C["bg_card_solid"],
            font_size=12,
            font_family="Manrope",
            font_color=text_color,
            bordercolor=C["border_2"],
        ),
    )
    return fig

def afficher_fig(fig):
    """Affichage sécurisé."""
    try:
        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": False,
            "responsive": True,
        })
    except Exception as e:
        st.warning(f"⚠️ Graphique indisponible ({type(e).__name__}) : {e}")

# --------------------------------------------------------------------
# Historique (persistance JSON)
# --------------------------------------------------------------------
def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_history(records):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Impossible de sauvegarder l'historique : {e}")

if "historique" not in st.session_state:
    st.session_state.historique = load_history()

def decision_depuis_proba(p):
    if p < SEUIL_MODERE: return "Favorable"
    if p < SEUIL_ELEVE:  return "Modéré"
    return "Défavorable"

# --------------------------------------------------------------------
# Génération PDF — RAPPORT COMPLET ENRICHI
# --------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generer_graphiques_pdf(df):
    """Génère tous les graphiques matplotlib pour le PDF"""
    graphiques = []
    NAVY = "#12244A"
    GRID = "#DCE3EF"
    TXT = "#26324A"

    # 1. Répartition des décisions (Donut)
    try:
        counts = df.decision.value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        cols = [COULEURS.get(c, "#888888") for c in counts.index]
        wedges, texts, autotexts = ax.pie(
            counts.values, labels=counts.index, colors=cols,
            autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
            textprops=dict(fontsize=11, color=TXT, weight="bold"),
        )
        for t in autotexts:
            t.set_color("white"); t.set_fontweight("bold"); t.set_fontsize(10)
        ax.set_title("Répartition des Décisions de Crédit",
                     fontsize=14, fontweight="bold", color=NAVY, pad=20)
        graphiques.append((_fig_to_png(fig), "Répartition des décisions de crédit"))
    except Exception:
        pass

    # 2. Distribution des probabilités (Histogramme)
    try:
        bins = pd.cut(df["proba_pct"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
        hist = bins.value_counts().sort_index()
        labels = [f"{int(i.left)}-{int(i.right)}%" for i in hist.index]
        cols = [couleur_risque(i.left) for i in hist.index]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, hist.values, color=cols, edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Nombre d'analyses", fontsize=11, color=TXT, weight="bold")
        ax.set_xlabel("Probabilité de défaut", fontsize=11, color=TXT, weight="bold")
        ax.set_title("Distribution des Probabilités de Défaut",
                     fontsize=14, fontweight="bold", color=NAVY, pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TXT)
        for i, v in enumerate(hist.values):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold', fontsize=10)
        graphiques.append((_fig_to_png(fig), "Distribution des probabilités de défaut"))
    except Exception:
        pass

    # 3. Risque par objet de prêt
    try:
        par_objet = df.groupby("objet")["proba_pct"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        cols = [couleur_risque(v) for v in par_objet.values]
        ax.barh(par_objet.index, par_objet.values, color=cols,
                edgecolor="white", linewidth=1.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Risque moyen (%)", fontsize=11, color=TXT, weight="bold")
        ax.set_title("Risque Moyen par Objet de Prêt",
                     fontsize=14, fontweight="bold", color=NAVY, pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TXT)
        for i, v in enumerate(par_objet.values):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold', fontsize=10)
        graphiques.append((_fig_to_png(fig), "Risque moyen par objet de prêt"))
    except Exception:
        pass

    # 4. Répartition par statut de logement
    try:
        logement_counts = df.logement.value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors_log = ["#2E6DB4", "#1B3B6F", "#0FA36B", "#E8A13A"]
        ax.bar(logement_counts.index, logement_counts.values, color=colors_log[:len(logement_counts)],
               edgecolor="white", linewidth=1.5)
        ax.set_ylabel("Nombre de dossiers", fontsize=11, color=TXT, weight="bold")
        ax.set_title("Répartition par Statut de Logement",
                     fontsize=14, fontweight="bold", color=NAVY, pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TXT)
        for i, v in enumerate(logement_counts.values):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold', fontsize=10)
        graphiques.append((_fig_to_png(fig), "Répartition par statut de logement"))
    except Exception:
        pass

    # 5. Scatter Revenu vs Montant (avec couleur par risque)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(df.revenu, df.montant, c=df.proba_pct, cmap='RdYlGn_r',
                            s=60, alpha=0.7, edgecolors='white', linewidth=1.5)
        ax.set_xlabel("Revenu annuel ($)", fontsize=11, color=TXT, weight="bold")
        ax.set_ylabel("Montant du prêt ($)", fontsize=11, color=TXT, weight="bold")
        ax.set_title("Revenu vs Montant du Prêt (couleur = risque)",
                     fontsize=14, fontweight="bold", color=NAVY, pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TXT)
        cbar = plt.colorbar(scatter, ax=ax, label="Probabilité de défaut (%)")
        cbar.set_label("Probabilité de défaut (%)", fontsize=10, color=TXT, weight="bold")
        graphiques.append((_fig_to_png(fig), "Revenu vs montant du prêt"))
    except Exception:
        pass

    # 6. Évolution temporelle
    try:
        df_temp = df.copy()
        df_temp["date_only"] = df_temp.date.dt.date
        daily = df_temp.groupby("date_only").size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(daily.date_only.astype(str), daily.count, color="#2E6DB4",
               edgecolor="white", linewidth=1.5)
        ax.set_xlabel("Date", fontsize=11, color=TXT, weight="bold")
        ax.set_ylabel("Nombre d'analyses", fontsize=11, color=TXT, weight="bold")
        ax.set_title("Activité Journalière des Analyses",
                     fontsize=14, fontweight="bold", color=NAVY, pad=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TXT, rotation=45)
        plt.tight_layout()
        graphiques.append((_fig_to_png(fig), "Activité journalière"))
    except Exception:
        pass

    return graphiques
def generer_rapport_pdf(df, inclure_graphiques=True):
    """Génère un rapport PDF complet et professionnel"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(name='TitrePrincipal', parent=styles['Heading1'],
        fontSize=28, textColor=colors.HexColor('#12244A'),
        spaceAfter=20, alignment=TA_CENTER, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SousTitre', parent=styles['Heading2'],
        fontSize=16, textColor=colors.HexColor('#1B3B6F'),
        spaceAfter=15, spaceBefore=25, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='TitreSection', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor('#0B1F3A'),
        spaceAfter=12, spaceBefore=20, fontName='Helvetica-Bold',
        borderWidth=2, borderColor=colors.HexColor('#D4AF37'), borderPadding=5))
    styles.add(ParagraphStyle(name='CorpsTexte', parent=styles['Normal'],
        fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=10))
    styles.add(ParagraphStyle(name='InsightBox', parent=styles['Normal'],
        fontSize=9, leading=12, backColor=colors.HexColor('#EAF1FB'),
        borderColor=colors.HexColor('#C9DCF5'),
        borderWidth=1, borderPadding=8, spaceAfter=12))
    styles.add(ParagraphStyle(name='Caption', parent=styles['Normal'],
        fontSize=9, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=15))
    styles.add(ParagraphStyle(name='Note', parent=styles['Normal'],
        fontSize=8, textColor=colors.grey, spaceAfter=8))
    styles.add(ParagraphStyle(name='BulletPoint', parent=styles['Normal'],
        fontSize=10, leading=14, leftIndent=20, bulletIndent=10, spaceAfter=6))

    story = []

    # ========== PAGE 1 : COUVERTURE ==========
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("CreditScore Pro", styles['TitrePrincipal']))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Rapport d'Analyse de Risque de Crédit", styles['SousTitre']))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Évaluation Intelligente du Risque — Spark ML & Random Forest",
                          ParagraphStyle('sub', parent=styles['Normal'], fontSize=11,
                                        textColor=colors.HexColor('#4A5572'), alignment=TA_CENTER)))
    story.append(Spacer(1, 2*cm))

    total = len(df)
    taux_fav = (df.decision == "Favorable").mean() * 100
    taux_mod = (df.decision == "Modéré").mean() * 100
    taux_def = (df.decision == "Défavorable").mean() * 100
    proba_moy = df.proba_pct.mean()
    montant_moy = df.montant.mean()
    dti_moy = df.dti.mean() * 100

    donnees_couverture = [
        ['Indicateur', 'Valeur', 'Interprétation'],
        ["Nombre d'analyses", str(total), "Volume total de dossiers évalués"],
        ["Taux d'avis favorables", f'{taux_fav:.1f}%', "Dossiers à faible risque"],
        ["Taux d'avis modérés", f'{taux_mod:.1f}%', "Dossiers nécessitant examen"],
        ["Taux d'avis défavorables", f'{taux_def:.1f}%', "Dossiers à risque élevé"],
        ['Probabilité de défaut moyenne', f'{proba_moy:.1f}%', "Risque global du portefeuille"],
        ['Montant moyen demandé', f'{montant_moy:,.0f} $', "Valeur moyenne des prêts"],
        ['DTI moyen', f'{dti_moy:.1f}%', "Ratio endettement/revenu"],
        ['Date du rapport', datetime.now().strftime('%d/%m/%Y %H:%M'), "Heure de génération"],
    ]
    table_couverture = Table(donnees_couverture, colWidths=[6*cm, 3*cm, 8*cm])
    table_couverture.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#12244A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFD')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(table_couverture)
    story.append(PageBreak())

    # ========== PAGE 2 : OBJECTIF ET MÉTHODOLOGIE ==========
    story.append(Paragraph("1. Objectif du Modèle", styles['TitreSection']))
    story.append(Paragraph(
        "CreditScore Pro est un système d'évaluation du risque de crédit basé sur l'apprentissage automatique. "
        "Son objectif principal est de prédire la probabilité qu'un emprunteur fasse défaut sur son prêt, "
        "permettant ainsi aux institutions financières de prendre des décisions éclairées et de minimiser les pertes.",
        styles['CorpsTexte']
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("<b>Variables Clés Analysées :</b>", styles['CorpsTexte']))
    variables = [
        "• <b>Âge et Revenu</b> : Profil financier de base de l'emprunteur",
        "• <b>Ancienneté professionnelle</b> : Stabilité de l'emploi et des revenus",
        "• <b>Montant du prêt et Taux d'intérêt</b> : Caractéristiques du crédit demandé",
        "• <b>Statut de logement</b> : Propriétaire, locataire ou hypothéqué",
        "• <b>Objet du prêt</b> : Finalité (éducation, médical, personnel, etc.)",
        "• <b>Historique de crédit</b> : Longueur et qualité de l'historique bancaire",
        "• <b>Antécédents de défaut</b> : Événements passés de non-paiement",
    ]
    for var in variables:
        story.append(Paragraph(var, styles['BulletPoint']))

    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("<b>Méthodologie :</b>", styles['CorpsTexte']))
    story.append(Paragraph(
        "Le modèle utilise un algorithme Random Forest entraîné sur des données historiques de crédit. "
        "Il analyse les relations complexes entre les variables pour prédire la probabilité de défaut. "
        "Les décisions sont classées en trois catégories : Favorable (< 20%), Modéré (20-40%), et "
        "Défavorable (> 40%), avec des recommandations adaptées à chaque niveau de risque.",
        styles['CorpsTexte']
    ))
    story.append(PageBreak())

    # ========== PAGE 3 : GRAPHIQUES PRINCIPAUX ==========
    story.append(Paragraph("2. Analyse Visuelle du Portefeuille", styles['TitreSection']))

    if inclure_graphiques:
        graphiques = generer_graphiques_pdf(df)

        for i in range(0, len(graphiques), 2):
            for j in range(2):
                if i + j < len(graphiques):
                    img_data, titre = graphiques[i + j]
                    img = Image(io.BytesIO(img_data), width=15*cm, height=9*cm)
                    story.append(img)
                    story.append(Paragraph(f"Figure {i+j+1} : {titre}", styles['Caption']))
                    story.append(Spacer(1, 0.3*cm))

            if i + 2 < len(graphiques):
                story.append(PageBreak())
                story.append(Paragraph("2. Analyse Visuelle du Portefeuille (suite)", styles['TitreSection']))

    story.append(PageBreak())

    # ========== PAGE 4 : ANALYSE DÉTAILLÉE ==========
    story.append(Paragraph("3. Analyse Détaillée et Insights", styles['TitreSection']))

    story.append(Paragraph("<b>3.1 Risque par Objet de Prêt</b>", styles['SousTitre']))
    par_objet = df.groupby("objet").agg({
        "proba_pct": "mean",
        "montant": "mean",
        "decision": lambda x: (x == "Favorable").sum() / len(x) * 100
    }).sort_values("proba_pct")

    for objet in par_objet.index:
        risque = par_objet.loc[objet, "proba_pct"]
        montant = par_objet.loc[objet, "montant"]
        taux_fav_obj = par_objet.loc[objet, "decision"]
        couleur_risque_txt = "faible" if risque < 20 else ("modéré" if risque < 40 else "élevé")

        story.append(Paragraph(
            f"<b>{objet}</b> : Risque moyen de {risque:.1f}% ({couleur_risque_txt}), "
            f"montant moyen {montant:,.0f}$, taux d'acceptation {taux_fav_obj:.1f}%",
            styles['CorpsTexte']
        ))

    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("<b>3.2 Risque par Statut de Logement</b>", styles['SousTitre']))
    par_logement = df.groupby("logement").agg({
        "proba_pct": "mean",
        "montant": "mean"
    }).sort_values("proba_pct")

    for logement in par_logement.index:
        risque = par_logement.loc[logement, "proba_pct"]
        montant = par_logement.loc[logement, "montant"]

        story.append(Paragraph(
            f"<b>{logement}</b> : Risque moyen de {risque:.1f}%, montant moyen {montant:,.0f}$",
            styles['CorpsTexte']
        ))

    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("<b>3.3 Corrélations Importantes</b>", styles['SousTitre']))

    if len(df) > 2:
        corr_dti = df["dti"].corr(df["proba_defaut"])
        corr_revenu = df["revenu"].corr(df["proba_defaut"])
        corr_age = df["age"].corr(df["proba_defaut"])

        story.append(Paragraph(
            f"• <b>DTI vs Risque</b> : corrélation de {corr_dti:.2f} — "
            f"{'Forte' if abs(corr_dti) > 0.6 else 'Modérée' if abs(corr_dti) > 0.3 else 'Faible'} "
            f"relation positive (plus le DTI est élevé, plus le risque augmente)",
            styles['BulletPoint']
        ))
        story.append(Paragraph(
            f"• <b>Revenu vs Risque</b> : corrélation de {corr_revenu:.2f} — "
            f"{'Forte' if abs(corr_revenu) > 0.6 else 'Modérée' if abs(corr_revenu) > 0.3 else 'Faible'} "
            f"relation (des revenus plus élevés tendent à réduire le risque)",
            styles['BulletPoint']
        ))
        story.append(Paragraph(
            f"• <b>Âge vs Risque</b> : corrélation de {corr_age:.2f} — "
            f"{'Forte' if abs(corr_age) > 0.6 else 'Modérée' if abs(corr_age) > 0.3 else 'Faible'} "
            f"relation",
            styles['BulletPoint']
        ))

    story.append(PageBreak())

    # ========== PAGE 5 : RECOMMANDATIONS ==========
    story.append(Paragraph("4. Recommandations Stratégiques", styles['TitreSection']))

    story.append(Paragraph("<b>4.1 Recommandations Générales</b>", styles['SousTitre']))

    if taux_def > 30:
        story.append(Paragraph(
            "⚠️ <b>Alerte</b> : Le taux de refus dépasse 30%, indiquant un portefeuille à risque élevé. "
            "Un ciblage plus strict en amont des demandes est recommandé.",
            styles['InsightBox']
        ))
    elif taux_fav > 70:
        story.append(Paragraph(
            "✅ <b>Excellent</b> : Le portefeuille présente un profil globalement sain avec un taux "
            "d'acceptation élevé. Maintenir les critères actuels.",
            styles['InsightBox']
        ))

    recommandations = [
        "• <b>Pour les dossiers Favorables</b> : Accorder rapidement, proposer des services complémentaires, "
        "fidéliser le client avec des offres privilégiées.",
        "• <b>Pour les dossiers Modérés</b> : Exiger des garanties proportionnelles, majorer légèrement "
        "le taux d'intérêt, recommander une assurance crédit.",
        "• <b>Pour les dossiers Défavorables</b> : Refuser ou exiger des garanties substantielles, "
        "proposer un montant réduit, recommander une amélioration de la situation financière.",
    ]
    for reco in recommandations:
        story.append(Paragraph(reco, styles['BulletPoint']))

    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph("<b>4.2 Recommandations Spécifiques</b>", styles['SousTitre']))

    if len(par_objet) > 0:
        pire_objet = par_objet.index[-1]
        meilleur_objet = par_objet.index[0]

        story.append(Paragraph(
            f"• Les prêts <b>{pire_objet}</b> présentent le risque le plus élevé ({par_objet.iloc[-1]['proba_pct']:.1f}%). "
            f"Renforcer les critères d'acceptation pour cette catégorie.",
            styles['BulletPoint']
        ))
        story.append(Paragraph(
            f"• Les prêts <b>{meilleur_objet}</b> sont les plus sûrs ({par_objet.iloc[0]['proba_pct']:.1f}%). "
            f"Encourager ce type de demande avec des conditions avantageuses.",
            styles['BulletPoint']
        ))

    if dti_moy > 35:
        story.append(Paragraph(
            f"• <b>Attention DTI</b> : Le DTI moyen de {dti_moy:.1f}% dépasse le seuil critique de 35%. "
            "Mettre en place un contrôle automatique pour les demandes avec DTI > 40%.",
            styles['BulletPoint']
        ))

    story.append(PageBreak())

    # ========== PAGE 6 : HISTORIQUE DÉTAILLÉ ==========
    story.append(Paragraph("5. Historique Détaillé des Analyses", styles['TitreSection']))
    story.append(Paragraph(
        f"Tableau complet des {total} analyses enregistrées dans le système, triées par date décroissante.",
        styles['CorpsTexte']
    ))
    story.append(Spacer(1, 0.3*cm))

    # ✅ CORRECTION : créer TOUTES les colonnes d'affichage (les minuscules n'existaient pas sous ce nom)
    df_hist = df.copy()
    df_hist["Date"] = df_hist["date"].dt.strftime("%d/%m/%Y %H:%M")
    df_hist["Âge"] = df_hist["age"].astype(int)
    df_hist["Revenu"] = df_hist["revenu"].apply(lambda x: f"{x:,.0f}$")
    df_hist["Prêt"] = df_hist["montant"].apply(lambda x: f"{x:,.0f}$")
    df_hist["Risque"] = df_hist["proba_pct"].apply(lambda x: f"{x:.1f}%")
    df_hist["Score"] = df_hist["score"].astype(int)
    df_hist["Objet"] = df_hist["objet"].astype(str)          # ✅ était manquant
    df_hist["Logement"] = df_hist["logement"].astype(str)    # ✅ était manquant
    df_hist["Décision"] = df_hist["decision"].astype(str)    # ✅ était manquant

    colonnes = ["Date", "Âge", "Revenu", "Prêt", "Objet", "Logement", "Risque", "Score", "Décision"]
    df_table = df_hist[colonnes].sort_values("Date", ascending=False).head(20)

    donnees_table = [colonnes] + df_table.values.tolist()
    table_hist = Table(donnees_table, colWidths=[2.2*cm, 1.2*cm, 2*cm, 2*cm, 2.5*cm, 2*cm, 1.5*cm, 1.3*cm, 2.3*cm])
    table_hist.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#12244A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFD')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(table_hist)

    if total > 20:
        story.append(Paragraph(
            f"<i>Tableau limité aux 20 analyses les plus récentes sur {total} au total.</i>",
            styles['Note']
        ))

    story.append(PageBreak())

    # ========== PAGE 7 : CONCLUSION ==========
    story.append(Paragraph("6. Conclusion", styles['TitreSection']))
    story.append(Paragraph(
        f"Le rapport analyse {total} dossiers de crédit avec une probabilité de défaut moyenne de "
        f"{proba_moy:.1f}%. Le modèle CreditScore Pro identifie efficacement les profils à risque et "
        "fournit des recommandations adaptées à chaque niveau de risque.",
        styles['CorpsTexte']
    ))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(
        "Les analyses montrent que les variables les plus déterminantes sont le taux d'endettement (DTI), "
        "le revenu annuel, et l'objet du prêt. Une surveillance continue et des ajustements périodiques "
        "des seuils de décision permettront d'optimiser la performance du portefeuille.",
        styles['CorpsTexte']
    ))
    story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(
        f"<i>Rapport généré automatiquement par CreditScore Pro v4.0 le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</i>",
        styles['Note']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# --------------------------------------------------------------------
# Spark & modèle
# --------------------------------------------------------------------
MODEL_SCHEMA = StructType([
    StructField("person_age", DoubleType(), True),
    StructField("person_income", DoubleType(), True),
    StructField("person_emp_length", DoubleType(), True),
    StructField("loan_int_rate", DoubleType(), True),
    StructField("person_home_ownership", StringType(), True),
    StructField("loan_intent", StringType(), True),
    StructField("cb_person_default_on_file", StringType(), True),
    StructField("loan_amnt", DoubleType(), True),
    StructField("cb_person_cred_hist_length", DoubleType(), True),
])

@st.cache_resource
def get_spark():
    return (SparkSession.builder
            .appName("credit_risk_streamlit")
            .master("local[*]")
            .config("spark.driver.memory", "2g")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())

@st.cache_resource
def load_model_safe(_spark):
    try:
        return PipelineModel.load(MODEL_PATH)
    except Exception:
        return None

spark = get_spark()
model = load_model_safe(spark)

# --------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 CreditScore Pro")

    icon = "☀️" if THEME == "dark" else "🌙"
    label = "Mode Clair" if THEME == "dark" else "Mode Sombre"
    st.button(
        f"{icon}  {label}",
        on_click=toggle_theme,
        use_container_width=True,
        key="theme_btn",
    )
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["diagnostic", "dashboard"],
        format_func=lambda p: "🏠  Diagnostic" if p == "diagnostic" else "📊  Dashboard analytique",
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("### 📊 Performance du modèle")
    c1, c2 = st.columns(2)
    c1.metric("AUC ROC", "0.92", "▲ +0.02")
    c2.metric("Précision", "88%", "▲ +3%")
    st.markdown("---")

    st.markdown("""
    **🔍 Variables clés**
    - Âge & revenu : profil financier
    - Endettement : ratio prêt / revenu
    - Historique : emploi & crédit
    - Antécédents de défaut
    """)
    st.caption(f"v4.0 • Random Forest • {len(st.session_state.historique)} analyse(s) • Thème : {THEME}")

# ====================================================================
# PAGE 1 — DIAGNOSTIC
# ====================================================================
def render_diagnostic():
    hero("CreditScore Pro", "Évaluation intelligente du risque de crédit — Spark ML",
         [("🔒 Analyse sécurisée", False), ("⚡ Instantané", False), ("🏆 AUC 0.92", True)])

    if model is None:
        st.error(f"""
        ❌ **Modèle introuvable** (`{MODEL_PATH}`)

        Vérifiez que le pipeline a bien été sauvegardé à la racine de l'app.
        """)
        return

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        section_title("📋", "Profil du demandeur", "Renseignez les informations du dossier")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.number_input("Âge", 18, 100, 35)
            revenu = st.number_input("Revenu annuel ($)", 0, 10_000_000, 55_000, step=1000)
        with r1c2:
            emp_length = st.number_input("Ancienneté pro (années)", 0.0, 60.0, 5.0, step=0.5)
            loan_amnt = st.number_input("Montant du prêt ($)", 0, 10_000_000, 15_000, step=1000)
        with r1c3:
            int_rate = st.number_input("Taux d'intérêt (%)", 0.0, 40.0, 10.5, step=0.1)
            cred_hist = st.number_input("Ancienneté crédit (années)", 0, 60, 8)

        st.markdown("#### 🏠 Informations complémentaires")
        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            home = st.selectbox("Logement", ["RENT", "MORTGAGE", "OWN", "OTHER"])
        with r2c2:
            intent = st.selectbox("Objet du prêt", ["EDUCATION", "MEDICAL", "VENTURE",
                                  "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
        with r2c3:
            default = st.selectbox("Défaut passé ?", ["N", "Y"])

    with col_right:
        section_title("📊", "Indicateurs calculés", "Ratios et vigilance automatiques")

        dti = (loan_amnt / revenu) if revenu > 0 else 0.0
        niveau_dti = "Élevé" if dti > 0.35 else "Modéré" if dti > 0.20 else "Faible"
        tone_dti = "red" if dti > 0.35 else "amber" if dti > 0.20 else "green"
        kpi_card("📈", "Taux d'endettement", f"{dti:.1%}", niveau_dti, tone_dti)

        coverage = (revenu / loan_amnt) if loan_amnt > 0 else 0
        tone_cov = "green" if coverage > 3 else "amber" if coverage > 1.5 else "red"
        kpi_card("🛡️", "Ratio couverture", f"{coverage:.1f}x",
                 "Revenu / Prêt", tone_cov)

        score_prelim = max(0, min(100, int(
            100
            - (dti * 150)
            + (min(age, 50) - 20) * 0.5
            + min(emp_length, 15) * 1.5
            + (5 if default == "N" else -15)
            + min(cred_hist, 20) * 0.8
        )))
        tone_score = "green" if score_prelim >= 70 else "amber" if score_prelim >= 50 else "red"
        kpi_card("🎯", "Score préliminaire", f"{score_prelim}/100",
                 "Estimation heuristique", tone_score)

        alertes = []
        if age < 25:       alertes.append("Âge inférieur à 25 ans")
        if emp_length < 2: alertes.append("Ancienneté professionnelle < 2 ans")
        if dti > 0.35:     alertes.append("Endettement élevé (> 35 %)")
        if default == "Y": alertes.append("Antécédent de défaut de paiement")

        if alertes:
            for a in alertes:
                insight_box("⚠️", "Vigilance", a, "warning")
        else:
            insight_box("✅", "Profil favorable", "Aucun point de vigilance détecté.", "success")

    st.markdown("")
    _, cbtn, _ = st.columns([1, 2, 1])
    with cbtn:
        predict_clicked = st.button("🚀 Analyser le risque de crédit", use_container_width=True)

    if not predict_clicked:
        return

    ligne = [(float(age), float(revenu), float(emp_length), float(int_rate),
              home, intent, default, float(loan_amnt), float(cred_hist))]
    df_input = spark.createDataFrame(ligne, schema=MODEL_SCHEMA)

    try:
        with st.spinner("Analyse du dossier en cours…"):
            time.sleep(0.6)
            res = model.transform(df_input).select("prediction", "probability").collect()[0]

        proba_defaut = float(res["probability"].toArray()[1])
        decision = decision_depuis_proba(proba_defaut)
        couleur = COULEURS[decision]

        record = {
            "id": str(uuid.uuid4())[:8],
            "horodatage": datetime.now().isoformat(),
            "age": age, "revenu": revenu, "anciennete": emp_length,
            "taux": int_rate, "logement": home, "objet": intent,
            "defaut_passe": default, "montant": loan_amnt,
            "hist_credit": cred_hist, "dti": round(dti, 4),
            "proba_defaut": round(proba_defaut, 4),
            "decision": decision,
            "score": int(round((1 - proba_defaut) * 100)),
        }
        st.session_state.historique.append(record)
        save_history(st.session_state.historique)
        st.toast("✅ Analyse enregistrée dans le Dashboard", icon="📊")

        st.markdown("---")
        section_title("📊", "Résultats de l'analyse", "Score, niveau de risque et recommandations")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📈 Risque de défaut", f"{proba_defaut:.1%}")
        m2.metric("✅ Solvabilité", f"{1 - proba_defaut:.1%}")
        m3.metric("🎯 Niveau", decision)
        m4.metric("📊 Score client", f"{record['score']}/100")

        if decision == "Défavorable":
            cls, icon, msg, det = "result-high", "🔴", "Risque ÉLEVÉ — Décision défavorable recommandée", "Plusieurs facteurs de risque identifiés"
        elif decision == "Modéré":
            cls, icon, msg, det = "result-mid", "🟡", "Risque MODÉRÉ — Analyse complémentaire recommandée", "Examen de garanties supplémentaires conseillé"
        else:
            cls, icon, msg, det = "result-low", "🟢", "Risque FAIBLE — Décision favorable recommandée", "Profil solvable avec historique satisfaisant"

        st.markdown(f"""
        <div class="result-card {cls}">
            <div class="result-label">{icon} {msg}</div>
            <div class="result-score">{proba_defaut:.1%}</div>
            <div style="opacity:.88; font-size:.92rem; position:relative;">
                Probabilité estimée de défaut de paiement
            </div>
            <div class="result-detail">{det}</div>
        </div>""", unsafe_allow_html=True)

        g1, g2 = st.columns([1, 2], gap="large")

        with g1:
            chart_title("🎯", "Jauge de risque", "Position du dossier sur l'échelle de risque")
            try:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(proba_defaut * 100, 1),
                    number=dict(suffix="%", font=dict(size=40, family="Sora", color=couleur)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickfont=dict(color=C["text_2"])),
                        bar=dict(color=couleur, thickness=0.55),
                        bgcolor=C["glass"],
                        steps=[
                            dict(range=[0, 20], color="rgba(15,163,107,0.14)"),
                            dict(range=[20, 40], color="rgba(232,161,58,0.14)"),
                            dict(range=[40, 100], color="rgba(214,69,69,0.14)"),
                        ],
                    ),
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10),
                                        paper_bgcolor="rgba(0,0,0,0)")
                afficher_fig(fig_gauge)
            except Exception as e:
                st.warning(f"⚠️ Jauge indisponible : {e}")

        with g2:
            chart_title("📊", "Décomposition des facteurs", "Score par dimension du dossier")
            try:
                facteurs = {
                    "Âge": max(0, min(100, (age - 18) / 82 * 100)),
                    "Revenu": max(0, min(100, revenu / 100_000 * 100)),
                    "Ancienneté": max(0, min(100, emp_length / 20 * 100)),
                    "Endettement": max(0, min(100, (1 - dti) * 100)),
                    "Historique": max(0, min(100, cred_hist / 20 * 100)),
                }
                fig = go.Figure(go.Bar(
                    x=list(facteurs.keys()), y=list(facteurs.values()),
                    marker=dict(
                        color=[couleur_risque(100 - v) if k == "Endettement" else couleur_risque(100 - v)
                               for k, v in facteurs.items()],
                        line=dict(color="rgba(255,255,255,.2)", width=1),
                    ),
                    text=[f"{v:.0f}" for v in facteurs.values()], textposition="outside",
                    textfont=dict(color=C["text"], family="Sora"),
                ))
                fig.update_layout(
                    height=260, showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Manrope", color=C["text"]),
                    yaxis=dict(range=[0, 115], gridcolor=C["border"], zerolinecolor=C["border"]),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(color=C["text_2"])),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                afficher_fig(fig)
            except Exception as e:
                st.warning(f"⚠️ Graphique indisponible : {e}")

        st.markdown("#### 💡 Recommandations")
        if decision == "Défavorable":
            st.warning("""
            1. Refuser le crédit ou exiger des garanties supplémentaires
            2. Demander un garant solvable
            3. Proposer un montant réduit avec taux adapté
            4. Recommander une amélioration de la situation financière
            """)
        elif decision == "Modéré":
            st.info("""
            1. Proposer un prêt avec garanties proportionnelles
            2. Majorer légèrement le taux d'intérêt
            3. Recommander une assurance crédit
            4. Mettre en place un suivi renforcé
            """)
        else:
            st.success("""
            1. Accorder le crédit aux conditions proposées
            2. Proposer des services bancaires complémentaires
            3. Fidéliser le client avec des offres privilégiées
            4. Appliquer le processus d'approbation simplifié
            """)

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

# ====================================================================
# PAGE 2 — DASHBOARD
# ====================================================================
def render_dashboard():
    n = len(st.session_state.historique)
    hero("Dashboard analytique", "Historique, tendances et explications des analyses",
         [("🗄️ Persistance JSON", False), (f"📈 {n} analyse(s)", True), ("📄 Export PDF", True)])

    records = st.session_state.historique
    if not records:
        st.markdown("""
        <div class="empty-state">
            <div class="big">🗂️</div>
            <h3>Aucune analyse pour le moment</h3>
            <p>Lancez votre premier diagnostic depuis la page <strong>🏠 Diagnostic</strong>.<br>
            Chaque analyse sera automatiquement enregistrée et analysée ici.</p>
        </div>""", unsafe_allow_html=True)
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["horodatage"])
    df["proba_pct"] = (df["proba_defaut"] * 100).round(1)

    # ---------- KPIs ----------
    section_title("🎯", "Indicateurs clés", "Vue synthétique de l'activité de scoring")
    total = len(df)
    taux_fav = (df.decision == "Favorable").mean() * 100
    taux_ref = (df.decision == "Défavorable").mean() * 100
    proba_moy = df.proba_pct.mean()
    montant_moy = df.montant.mean()
    dti_moy = df.dti.mean() * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("🧮", "Analyses", f"{total}", "dossiers traités", "blue")
    with k2:
        kpi_card("✅", "Favorables", f"{taux_fav:.0f}%", "dossiers acceptables", "green")
    with k3:
        kpi_card("📉", "Risque moyen", f"{proba_moy:.1f}%", "probabilité de défaut", "amber")
    with k4:
        kpi_card("🔴", "Refus", f"{taux_ref:.0f}%", "dossiers à risque", "red")

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("💰", "Montant moyen", f"{montant_moy:,.0f} $", "par dossier", "gold")
    with k2:
        kpi_card("⚖️", "DTI moyen", f"{dti_moy:.1f}%", "ratio prêt / revenu", "navy")
    with k3:
        score_moy = df["score"].mean()
        tone_s = "green" if score_moy >= 70 else "amber" if score_moy >= 50 else "red"
        kpi_card("🏆", "Score moyen", f"{score_moy:.0f}/100", "solvabilité globale", tone_s)

    section_title("📈", "Graphiques & explications", "Interprétation automatique des données")

    # ---------- 1. Donut décisions ----------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_title("🍩", "Répartition des décisions", "Part de chaque avis rendu")
        try:
            counts = df.decision.value_counts()
            fig = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values), hole=0.68,
                marker=dict(colors=[COULEURS.get(c, "#888888") for c in counts.index],
                            line=dict(color=C["bg_card_solid"], width=3)),
                textinfo="percent+label",
                textfont=dict(family="Sora", size=13, color=C["text"]),
                hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value} dossiers<extra></extra>",
            ))
            fig.update_layout(legend=dict(orientation="h", y=-0.08,
                                          font=dict(color=C["text_2"])))
            afficher_fig(style_fig(fig, 320, legend=True))
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
        tone = "success" if taux_fav >= 50 else "danger"
        insight_box("💡", "Interprétation",
                    f"Sur {total} analyse(s), {taux_fav:.0f}% ont reçu un avis favorable et {taux_ref:.0f}% un avis défavorable. "
                    + ("Le flux de dossiers soumis présente un profil globalement sain."
                       if taux_fav >= 50 else
                       "La majorité des dossiers soumis est risquée : un ciblage en amont des demandes est conseillé."),
                    tone)

    # ---------- 2. Histogramme risque ----------
    with c2:
        chart_title("📊", "Distribution du risque", "Histogramme des probabilités de défaut")
        try:
            bins = pd.cut(df["proba_pct"], bins=[0, 20, 40, 60, 80, 100], include_lowest=True)
            hist = bins.value_counts().sort_index()
            labels = [f"{int(i.left)}–{int(i.right)}%" for i in hist.index]
            colors_list = [couleur_risque(i.left) for i in hist.index]

            fig = go.Figure(go.Bar(
                x=labels,
                y=list(hist.values),
                marker=dict(
                    color=colors_list,
                    line=dict(color="rgba(255,255,255,.3)", width=1),
                ),
                text=list(hist.values),
                textposition="outside",
                textfont=dict(color=C["text"], family="Sora"),
                hovertemplate="<b>%{x}</b><br>%{y} dossier(s)<extra></extra>",
            ))
            afficher_fig(style_fig(fig, 320))
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")

        insight_box("🔎", "Interprétation",
                    f"Le risque moyen s'établit à {proba_moy:.1f}% (médiane : {df.proba_pct.median():.1f}%). "
                    + ("La distribution est concentrée sur les risques faibles."
                       if proba_moy < 30 else
                       "Une part importante des dossiers se situe en zone de risque élevé."),
                    "info")

    # ---------- 3. Activité temporelle ----------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_title("📅", "Activité dans le temps", "Nombre d'analyses par jour")
        try:
            act = df.groupby(df.date.dt.date).size().reset_index(name="n")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=act["date"].astype(str), y=act.n,
                marker=dict(
                    color="rgba(46,109,180,.7)",
                    line=dict(color=C["blue"], width=1.5),
                ),
                name="Analyses",
            ))
            fig.add_trace(go.Scatter(
                x=act["date"].astype(str), y=act.n,
                mode="lines+markers",
                line=dict(color=C["gold"], width=2.5),
                marker=dict(size=8, color=C["gold"]),
                name="Tendance",
            ))
            fig.update_layout(barmode="overlay", hovermode="x unified")
            afficher_fig(style_fig(fig, 320))
            pic = act.loc[act.n.idxmax()]
            txt = (f"Le pic d'activité ({int(pic.n)} analyse(s)) a été enregistré le {pic.date}. "
                   + ("L'usage est régulier sur la période." if len(act) > 1 else "Première journée d'utilisation du scoring."))
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
            txt = "Données temporelles insuffisantes pour une interprétation."
        insight_box("📅", "Interprétation", txt, "info")

    # ---------- 4. Risque par objet ----------
    with c2:
        chart_title("🎯", "Risque moyen par objet de prêt", "Quelles finalités sont les plus risquées ?")
        try:
            par_objet = df.groupby("objet")["proba_pct"].mean().sort_values().reset_index()
            fig = go.Figure(go.Bar(
                y=list(par_objet.objet), x=list(par_objet.proba_pct), orientation="h",
                marker=dict(
                    color=[couleur_risque(v) for v in par_objet.proba_pct],
                    line=dict(color="rgba(255,255,255,.2)", width=1),
                ),
                text=[f"{v:.0f}%" for v in par_objet.proba_pct],
                textposition="outside",
                textfont=dict(color=C["text"], family="Sora"),
                hovertemplate="<b>%{y}</b><br>Risque moyen : %{x:.1f}%<extra></extra>",
            ))
            fig.update_layout(xaxis=dict(range=[0, 110]))
            afficher_fig(style_fig(fig, 320))
            pire, meilleur = par_objet.objet.iloc[-1], par_objet.objet.iloc[0]
            txt = (f"Les prêts « {pire} » présentent le risque moyen le plus élevé, tandis que « {meilleur} » "
                   "sont les plus sûrs. Adaptez les conditions selon la finalité.")
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
            txt = "Données insuffisantes pour comparer les objets de prêt."
        insight_box("⚠️", "Interprétation", txt, "warning")

    # ---------- 5. Scatter revenu / prêt (bulles) ----------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_title("🔵", "Revenu vs montant du prêt", "Taille de bulle = probabilité de défaut")
        try:
            fig = go.Figure()
            for dec in ["Favorable", "Modéré", "Défavorable"]:
                sub = df[df.decision == dec]
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=list(sub.revenu), y=list(sub.montant), mode="markers", name=dec,
                    marker=dict(
                        size=list(sub.proba_pct / 3 + 8),
                        color=COULEURS[dec],
                        opacity=0.78,
                        line=dict(width=1.5, color="rgba(255,255,255,.6)"),
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Revenu : %{x:,.0f} $<br>"
                        "Prêt : %{y:,.0f} $<br>"
                        "Risque : %{customdata[1]:.1f}%<extra></extra>"
                    ),
                    customdata=list(zip(sub.objet, sub.proba_pct)),
                ))
            fig.update_layout(
                xaxis=dict(title=dict(text="Revenu annuel ($)", font=dict(color=C["text_2"])),
                           gridcolor=C["border"]),
                yaxis=dict(title=dict(text="Montant du prêt ($)", font=dict(color=C["text_2"])),
                           gridcolor=C["border"]),
                legend=dict(orientation="h", y=-0.18,
                            font=dict(color=C["text_2"])),
                hovermode="closest",
            )
            afficher_fig(style_fig(fig, 340, legend=True))
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
        try:
            corr = df["dti"].corr(df["proba_defaut"]) if len(df) > 2 else float("nan")
            if pd.isna(corr):
                txt = "Corrélation non calculable avec si peu d'analyses (minimum 3 requises)."
            else:
                force = "forte" if abs(corr) > 0.6 else "modérée" if abs(corr) > 0.3 else "faible"
                txt = (f"La corrélation entre endettement (DTI) et probabilité de défaut est {force} (r = {corr:.2f}). "
                       "Les bulles rouges se concentrent chez les demandeurs à fort endettement.")
        except Exception:
            txt = "Corrélation non calculable."
        insight_box("🔗", "Interprétation", txt, "info")

    # ---------- 6. Radar chart profil moyen ----------
    with c2:
        chart_title("🎭", "Profil moyen des dossiers", "Radar des caractéristiques")
        try:
            age_norm = (df["age"].mean() - 18) / (80 - 18) * 100
            rev_norm = min(100, df["revenu"].mean() / 150_000 * 100)
            emp_norm = min(100, df["anciennete"].mean() / 20 * 100)
            cred_norm = min(100, df["hist_credit"].mean() / 20 * 100)
            dti_norm = max(0, 100 - df["dti"].mean() * 100 * 2)
            solv_norm = 100 - df["proba_pct"].mean()

            categories = ["Âge", "Revenu", "Ancienneté", "Hist. Crédit", "Solvabilité", "Faible DTI"]
            values = [age_norm, rev_norm, emp_norm, cred_norm, solv_norm, dti_norm]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(212,175,55,.2)",
                line=dict(color=C["gold"], width=2.5),
                marker=dict(size=8, color=C["gold"]),
                name="Profil moyen",
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        gridcolor=C["border"],
                        tickfont=dict(color=C["text_3"], size=9),
                    ),
                    angularaxis=dict(
                        gridcolor=C["border"],
                        tickfont=dict(color=C["text_2"], size=11),
                    ),
                ),
                showlegend=False,
            )
            afficher_fig(style_fig(fig, 340))
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
        insight_box("🎯", "Interprétation",
                    "Le radar révèle les forces et faiblesses moyennes du portefeuille. "
                    "Les axes extérieurs indiquent des profils plus solides.",
                    "info")

    # ---------- Historique détaillé ----------
    section_title("🗄️", "Historique détaillé", "Toutes les analyses enregistrées")

    df_aff = df.copy()
    df_aff["Date"] = df_aff["date"].dt.strftime("%d/%m/%Y %H:%M")
    df_aff = df_aff.rename(columns={
        "age": "Âge", "revenu": "Revenu ($)", "montant": "Prêt ($)",
        "objet": "Objet", "logement": "Logement", "decision": "Décision",
        "score": "Score",
    })
    df_aff = df_aff[["Date", "Âge", "Revenu ($)", "Prêt ($)", "Objet", "Logement",
                     "proba_pct", "Score", "Décision"]]

    try:
        st.dataframe(
            df_aff.sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "proba_pct": st.column_config.ProgressColumn(
                    "Risque", min_value=0, max_value=100, format="%.1f"
                ),
            },
        )
    except Exception:
        st.dataframe(df_aff.sort_values("Date", ascending=False),
                     use_container_width=True, hide_index=True)

    # ---------- Boutons d'export ----------
    cexp, cpdf, cdel, _ = st.columns([1, 1, 1, 2])

    csv = df.to_csv(index=False).encode("utf-8")
    cexp.download_button("⬇️ Exporter CSV", data=csv,
                         file_name="historique_analyses.csv", mime="text/csv",
                         use_container_width=True)

    if cpdf.button("📄 Générer PDF", use_container_width=True):
        with st.spinner("Génération du rapport PDF..."):
            try:
                pdf_bytes = generer_rapport_pdf(df, inclure_graphiques=True)
                st.download_button(
                    label="⬇️ Télécharger le rapport",
                    data=pdf_bytes,
                    file_name=f"rapport_credit_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("✅ Rapport PDF généré avec succès !")
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération : {e}")

    with cdel:
        if st.session_state.get("confirm_clear"):
            ca, cb = st.columns(2)
            if ca.button("✅ Confirmer", type="primary"):
                st.session_state.historique = []
                save_history([])
                st.session_state.confirm_clear = False
                st.rerun()
            if cb.button("❌ Annuler"):
                st.session_state.confirm_clear = False
                st.rerun()
        else:
            if st.button("🗑️ Vider l'historique", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()

# --------------------------------------------------------------------
def render_footer():
    st.markdown(f"""
    <div class="footer">
        <strong>CreditScore Pro v4.0</strong> • Modèle Random Forest • Spark ML & Streamlit
        <br>🔒 Données traitées de manière confidentielle • Thème actuel : <strong>{THEME}</strong>
        <br><a href="https://ibuuuu19.github.io/credit-risk-streamlit/privacy/" target="_blank">
        Politique de confidentialité</a>
    </div>""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Routage
# --------------------------------------------------------------------
if page == "diagnostic":
    render_diagnostic()
else:
    render_dashboard()

render_footer()