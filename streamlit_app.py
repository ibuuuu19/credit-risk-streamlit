"""
CreditScore Pro v3.0 — Application Streamlit professionnelle
=============================================================
• Page Diagnostic  : scoring de risque de crédit (Spark ML / Random Forest)
• Page Dashboard   : historique persistant des analyses + KPIs + graphiques
                     Plotly + explications interprétatives automatiques

Corrections apportées :
• `import straeamlit` -> `import streamlit` (faute de frappe)
• Balises HTML non fermées corrigées
• Historique persisté en JSON (survit aux redémarrages)
"""

import json
import os
import time
import uuid
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, DoubleType, StringType
)
from pyspark.ml import PipelineModel

# --------------------------------------------------------------------
# Constantes & configuration
# --------------------------------------------------------------------
MODEL_PATH   = "credit_risk_pipeline_model"
HISTORY_FILE = "historique_analyses.json"

SEUIL_MODERE = 0.20   # en dessous : risque faible
SEUIL_ELEVE  = 0.40   # au dessus  : risque élevé

COULEURS = {
    "Favorable":   "#0FA36B",
    "Modéré":      "#E8A13A",
    "Défavorable": "#D64545",
}

st.set_page_config(
    page_title="CreditScore Pro — Risque de Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------
# Design system — CSS premium
# --------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/* ============ BASE ============ */
html, body, .stApp {
    background: #F4F6FB;
    font-family: 'Manrope', sans-serif;
    color: #26324A;
}
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
.block-container { padding-top: 1.5rem; max-width: 1220px; }

h1,h2,h3,h4 { font-family: 'Sora', sans-serif !important; color: #12244A !important; }

/* ============ SIDEBAR ============ */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E5EAF3;
}
section[data-testid="stSidebar"] .stRadio > div {
    display: flex; flex-direction: column; gap: 8px;
}
section[data-testid="stSidebar"] .stRadio label {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; border-radius: 12px;
    border: 1px solid #E5EAF3; background: #F8FAFD;
    cursor: pointer; font-weight: 600; font-size: .92rem;
    transition: all .25s ease; margin: 0;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    border-color: #1B3B6F; transform: translateX(3px);
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(135deg, #12244A, #1B3B6F);
    color: #fff; border-color: transparent;
    box-shadow: 0 6px 18px rgba(18,36,74,.28);
}
section[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }

/* ============ HERO ============ */
.hero {
    background: linear-gradient(135deg, #0B1F3A 0%, #1B3B6F 55%, #27498B 100%);
    border-radius: 20px; padding: 2.2rem 2.4rem;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 1rem;
    box-shadow: 0 12px 34px rgba(11,31,58,.28);
    position: relative; overflow: hidden; margin-bottom: 1.8rem;
    animation: fadeUp .6s ease both;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background:
      radial-gradient(circle at 85% 20%, rgba(212,175,55,.18) 0%, transparent 45%),
      radial-gradient(circle at 10% 90%, rgba(255,255,255,.08) 0%, transparent 40%);
}
.hero h1 {
    color: #fff !important; font-size: 2.3rem; margin: 0;
    letter-spacing: -1px; position: relative;
}
.hero h1 span { color: #D4AF37; }
.hero p {
    color: rgba(255,255,255,.78); margin: .4rem 0 0;
    font-size: .98rem; position: relative;
}
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; position: relative; }
.hbadge {
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
    backdrop-filter: blur(8px); color: #fff;
    padding: 6px 14px; border-radius: 30px; font-size: .78rem; font-weight: 600;
}
.hbadge.gold { background: rgba(212,175,55,.22); border-color: rgba(212,175,55,.5); color: #F0D488; }

/* ============ KPI CARDS ============ */
.kpi {
    background: #fff; border: 1px solid #E5EAF3; border-radius: 16px;
    padding: 1.1rem 1.2rem; display: flex; gap: 14px; align-items: center;
    box-shadow: 0 2px 10px rgba(18,36,74,.05);
    border-left: 4px solid #1B3B6F;
    transition: all .3s ease; margin-bottom: 14px;
    animation: fadeUp .5s ease both;
}
.kpi:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(18,36,74,.12); }
.kpi-icon {
    width: 46px; height: 46px; border-radius: 12px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 1.35rem;
    background: #EEF3FB;
}
.kpi-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .6px; color: #7A8699; font-weight: 700; }
.kpi-value { font-family: 'Sora'; font-size: 1.55rem; font-weight: 800; color: #12244A; line-height: 1.2; }
.kpi-sub   { font-size: .75rem; color: #93A0B4; font-weight: 500; }
.tone-green .kpi-icon { background: #E6F7F0; } .tone-green { border-left-color: #0FA36B; }
.tone-red   .kpi-icon { background: #FBECEC; } .tone-red   { border-left-color: #D64545; }
.tone-amber .kpi-icon { background: #FDF4E5; } .tone-amber { border-left-color: #E8A13A; }
.tone-gold  .kpi-icon { background: #FAF3DF; } .tone-gold  { border-left-color: #C9A227; }
.tone-blue  .kpi-icon { background: #EAF1FB; } .tone-blue  { border-left-color: #2E6DB4; }

/* ============ SECTIONS & INSIGHTS ============ */
.section-title { display: flex; gap: 12px; align-items: center; margin: 2rem 0 1rem; }
.section-title .st-icon {
    width: 40px; height: 40px; border-radius: 12px; font-size: 1.2rem;
    background: linear-gradient(135deg,#12244A,#1B3B6F);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(18,36,74,.25);
}
.section-title h3 { margin: 0; font-size: 1.15rem; }
.section-title p  { margin: 2px 0 0; font-size: .82rem; color: #7A8699; }

.chart-head {
    background: #fff; border: 1px solid #E5EAF3; border-bottom: none;
    border-radius: 16px 16px 0 0; padding: 1rem 1.2rem .6rem;
}
.chart-head h4 { margin: 0; font-size: .95rem; }
.chart-head p  { margin: 2px 0 0; font-size: .78rem; color: #7A8699; }
.chart-body {
    background: #fff; border: 1px solid #E5EAF3; border-top: none;
    border-radius: 0 0 16px 16px; padding: .5rem .8rem 1rem;
    margin-bottom: 14px;
}

.insight {
    display: flex; gap: 12px; align-items: flex-start;
    border-radius: 14px; padding: 14px 16px; margin: 10px 0 18px;
    font-size: .86rem; line-height: 1.6; border: 1px solid;
    animation: fadeUp .5s ease both;
}
.insight .i-icon { font-size: 1.15rem; }
.insight-info    { background: #EAF1FB; border-color: #C9DCF5; color: #23477A; }
.insight-success { background: #E6F7F0; border-color: #BEE8D6; color: #0B6B47; }
.insight-warning { background: #FDF4E5; border-color: #F3DDB4; color: #8A5A14; }
.insight-danger  { background: #FBECEC; border-color: #F2C9C9; color: #8C2B2B; }

/* ============ RÉSULTAT DIAGNOSTIC ============ */
.result-card {
    border-radius: 18px; text-align: center; padding: 2rem 1.5rem;
    color: #fff; margin: 1.2rem 0; position: relative; overflow: hidden;
    animation: fadeUp .55s ease both;
}
.result-card::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(circle at 80% 10%, rgba(255,255,255,.18), transparent 50%);
}
.result-low  { background: linear-gradient(135deg, #0FA36B, #0B8558); box-shadow: 0 12px 30px rgba(15,163,107,.35); }
.result-mid  { background: linear-gradient(135deg, #E8A13A, #C77F1B); box-shadow: 0 12px 30px rgba(232,161,58,.35); }
.result-high { background: linear-gradient(135deg, #D64545, #A93030); box-shadow: 0 12px 30px rgba(214,69,69,.35); }
.result-score { font-family: 'Sora'; font-size: 3.6rem; font-weight: 800; margin: .3rem 0; }
.result-label { font-size: 1.05rem; font-weight: 700; }
.result-detail {
    display: inline-block; background: rgba(255,255,255,.16);
    backdrop-filter: blur(6px);
    padding: 6px 18px; border-radius: 30px; font-size: .85rem; margin-top: .6rem;
}

/* ============ WIDGETS ============ */
.stButton > button {
    background: linear-gradient(135deg, #12244A, #1B3B6F) !important;
    color: #fff !important; font-weight: 700 !important;
    border: none !important; border-radius: 12px !important;
    padding: .7rem 1.5rem !important;
    transition: all .3s ease !important;
    box-shadow: 0 6px 18px rgba(18,36,74,.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 26px rgba(18,36,74,.4) !important;
}
.stButton > button[kind="secondary"] { background: #fff !important; }

div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #E5EAF3; border-radius: 14px;
    padding: 12px 16px; box-shadow: 0 2px 10px rgba(18,36,74,.05);
}
div[data-testid="stMetric"] label { color: #7A8699 !important; font-weight: 600; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #12244A !important; font-family: 'Sora'; }

.stNumberInput input, .stSelectbox select {
    border-radius: 10px !important; border: 1.5px solid #DCE3EF !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #1B3B6F !important; box-shadow: 0 0 0 3px rgba(27,59,111,.12) !important;
}

/* ============ DIVERS ============ */
.empty-state {
    background: #fff; border: 2px dashed #C9D4E6; border-radius: 20px;
    text-align: center; padding: 4rem 2rem; color: #7A8699;
}
.empty-state .big { font-size: 3.4rem; }
.empty-state h3 { color: #12244A !important; margin: .8rem 0 .4rem; }

.footer {
    text-align: center; color: #7A8699; font-size: .8rem;
    padding: 2rem 0 1rem; border-top: 1px solid #E5EAF3; margin-top: 3rem;
}
.footer a { color: #1B3B6F; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Helpers HTML
# --------------------------------------------------------------------
def hero(titre, sous, badges):
    b = "".join(
        f'<span class="hbadge{" gold" if g else ""}>{t}</span>'
        for t, g in badges
    )
    st.markdown(f"""
    <div class="hero">
        <div><h1>🏦 CreditScore <span>Pro</span></h1><p>{sous}</p></div>
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

def chart_head(icon, titre, sous):
    st.markdown(f"""
    <div class="chart-head"><h4>{icon} {titre}</h4><p>{sous}</p></div>
    <div class="chart-body">""", unsafe_allow_html=True)

def chart_foot():
    st.markdown("</div>", unsafe_allow_html=True)

def insight_box(icon, titre, texte, tone="info"):
    st.markdown(f"""
    <div class="insight insight-{tone}">
        <span class="i-icon">{icon}</span>
        <div><strong>{titre}</strong> — {texte}</div>
    </div>""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Historique des analyses (persistance JSON)
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
    if p < SEUIL_MODERE:  return "Favorable"
    if p < SEUIL_ELEVE:   return "Modéré"
    return "Défavorable"

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
# Sidebar : navigation + infos
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏦 CreditScore Pro")
    page = st.sidebar.radio(
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
    st.caption(f"v3.0 • Random Forest • {len(st.session_state.historique)} analyse(s) stockée(s)")

# ====================================================================
# PAGE 1 — DIAGNOSTIC
# ====================================================================
def render_diagnostic():
    hero(
        "CreditScore Pro",
        "Évaluation intelligente du risque de crédit — Spark ML",
        [("🔒 Analyse sécurisée", False), ("⚡ Instantané", False), ("🏆 AUC 0.92", True)],
    )

    if model is None:
        st.error(f"""
        ❌ **Modèle introuvable** (`{MODEL_PATH}`)

        Vérifiez que le pipeline a bien été sauvegardé à la racine de l'app
        (`ml_save()` côté R / `PipelineModel.write()` côté Python).
        Le **Dashboard** reste consultable.
        """)
        return

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        section_title("📋", "Profil du demandeur", "Renseignez les informations du dossier de crédit")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            age = st.number_input("Âge", 18, 100, 35, help="Âge du demandeur")
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
        section_title("📊", "Indicateurs calculés", "Ratios et points de vigilance automatiques")

        dti = (loan_amnt / revenu) if revenu > 0 else 0.0
        niveau_dti = "Élevé" if dti > 0.35 else "Modéré" if dti > 0.20 else "Faible"
        tone_dti = "red" if dti > 0.35 else "amber" if dti > 0.20 else "green"
        kpi_card("📈", "Taux d'endettement", f"{dti:.1%}", niveau_dti, tone_dti)

        alertes = []
        if age < 25:            alertes.append("Âge inférieur à 25 ans")
        if emp_length < 2:      alertes.append("Ancienneté professionnelle < 2 ans")
        if dti > 0.35:          alertes.append("Endettement élevé (> 35 %)")
        if default == "Y":      alertes.append("Antécédent de défaut de paiement")

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

        # ---- Enregistrement dans l'historique ----
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
        st.toast("✅ Analyse enregistrée — retrouvez-la dans le Dashboard", icon="📊")

        # ---- Résultats ----
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
            <div style="opacity:.85; font-size:.9rem;">Probabilité estimée de défaut de paiement</div>
            <div class="result-detail">{det}</div>
        </div>""", unsafe_allow_html=True)

        # ---- Jauge + facteurs ----
        g1, g2 = st.columns([1, 2], gap="large")

        with g1:
            chart_head("🎯", "Jauge de risque", "Positionnement du dossier sur l'échelle de risque")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(proba_defaut * 100, 1),
                number=dict(suffix="%", font=dict(size=40, family="Sora", color=couleur)),
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=couleur, thickness=0.55),
                    bgcolor="rgba(18,36,74,0.06)",
                    steps=[
                        dict(range=[0, 20], color="rgba(15,163,107,0.14)"),
                        dict(range=[20, 40], color="rgba(232,161,58,0.14)"),
                        dict(range=[40, 100], color="rgba(214,69,69,0.14)"),
                    ],
                ),
            ))
            fig_gauge.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
            chart_foot()

        with g2:
            chart_head("📊", "Décomposition des facteurs", "Niveau de risque par dimension du dossier")
            facteurs = {
                "Âge": max(0, min(100, (age - 18) / 82 * 100)),
                "Revenu": max(0, min(100, revenu / 100_000 * 100)),
                "Ancienneté": max(0, min(100, emp_length / 20 * 100)),
                "Endettement": max(0, min(100, dti * 100)),
                "Historique": max(0, min(100, cred_hist / 20 * 100)),
            }
            fig = go.Figure(go.Bar(
                x=list(facteurs.keys()), y=list(facteurs.values()),
                marker_color=["#0FA36B" if v < 50 else "#E8A13A" if v < 70 else "#D64545"
                              for v in facteurs.values()],
                text=[f"{v:.0f}%" for v in facteurs.values()], textposition="outside",
                marker=dict(cornerradius=6),
            ))
            fig.update_layout(
                height=240, showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Manrope"),
                yaxis=dict(range=[0, 110], gridcolor="#E5EAF3"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            chart_foot()

        # ---- Recommandations ----
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
# PAGE 2 — DASHBOARD ANALYTIQUE
# ====================================================================
def render_dashboard():
    hero(
        "Dashboard analytique",
        "Historique, tendances et explications des analyses effectuées",
        [("🗄️ Persistance JSON", False), ("📈 " + str(len(st.session_state.historique)) + " analyses", True)],
    )

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

    # ---------------- KPIs ----------------
    section_title("🎯", "Indicateurs clés", "Vue synthétique de l'activité de scoring")
    total = len(df)
    taux_fav = (df.decision == "Favorable").mean() * 100
    taux_ref = (df.decision == "Défavorable").mean() * 100
    proba_moy = df.proba_pct.mean()
    montant_moy = df.montant.mean()
    dti_moy = df.dti.mean() * 100

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("🧮", "Analyses effectuées", f"{total}", "dossiers traités", "blue")
        kpi_card("💰", "Montant moyen demandé", f"{montant_moy:,.0f} $", "par dossier", "gold")
    with k2:
        kpi_card("✅", "Taux d'avis favorables", f"{taux_fav:.0f}%", "dossiers acceptables", "green")
        kpi_card("📉", "Risque moyen", f"{proba_moy:.1f}%", "probabilité de défaut", "amber")
    with k3:
        kpi_card("🔴", "Taux de refus", f"{taux_ref:.0f}%", "dossiers à risque élevé", "red")
        kpi_card("⚖️", "Endettement moyen", f"{dti_moy:.1f}%", "ratio prêt / revenu", "navy")

    # ---------------- Graphique 1 : décisions ----------------
    section_title("📈", "Graphiques & explications", "Interprétation automatique des données historiques")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        chart_head("🍩", "Répartition des décisions", "Part de chaque avis rendu par le modèle")
        counts = df.decision.value_counts()
        fig = go.Figure(go.Pie(
            labels=counts.index, values=counts.values, hole=0.62,
            marker=dict(colors=[COULEURS.get(c, "#888") for c in counts.index]),
            textinfo="percent", textfont=dict(size=13, family="Manrope"),
        ))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Manrope"),
                          showlegend=True, legend=dict(orientation="h", y=-0.08))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_foot()
        tone = "success" if taux_fav >= 50 else "danger"
        insight_box("💡", "Lecture",
                    f"Sur {total} analyses, {taux_fav:.0f}% ont reçu un avis favorable et {taux_ref:.0f}% un avis défavorable. "
                    + ("Le flux de dossiers soumis présente un profil globalement sain."
                       if taux_fav >= 50 else
                       "La majorité des dossiers soumis est risquée : un ciblage en amont des demandes est conseillé."),
                    tone)

    with c2:
        chart_head("📊", "Distribution du risque", "Histogramme des probabilités de défaut")
        bins = pd.cut(df.proba_pct, bins=range(0, 101, 10))
        hist = df.groupby(bins, observed=True).size().reset_index(name="n")
        hist["label"] = hist["proba_pct"].apply(lambda i: f"{i.left}-{i.right}")
        fig = go.Figure(go.Bar(
            x=hist.label, y=hist.n,
            marker_color=[COULEURS["Favorable"] if i.left < 20 else
                          COULEURS["Modéré"] if i.left < 40 else
                          COULEURS["Défavorable"] for i in hist["proba_pct"]],
            marker=dict(cornerradius=5),
        ))
        fig.update_layout(height=300, showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Manrope"),
                          yaxis=dict(gridcolor="#E5EAF3"), xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_foot()
        insight_box("🔎", "Lecture",
                    f"Le risque moyen s'établit à {proba_moy:.1f}% (médiane : {df.proba_pct.median():.1f}%). "
                    + ("La distribution est concentrée sur les risques faibles : le modèle discrimine surtout des profils solvables."
                       if proba_moy < 30 else
                       "Une part importante des dossiers se situe en zone de risque élevé."),
                    "info")

    # ---------------- Graphique 2 : activité + objets ----------------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_head("📅", "Activité dans le temps", "Nombre d'analyses par jour")
        act = df.groupby(df.date.dt.date).size().reset_index(name="n")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=act["date"].astype(str), y=act.n,
                             marker_color="#2E6DB4", marker=dict(cornerradius=5)))
        fig.add_trace(go.Scatter(x=act["date"].astype(str), y=act.n,
                                 mode="lines+markers", line=dict(color="#D4AF37", width=2.5)))
        fig.update_layout(height=300, showlegend=False, barmode="overlay",
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Manrope"),
                          yaxis=dict(gridcolor="#E5EAF3"), xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_foot()
        pic = act.loc[act.n.idxmax()]
        insight_box("📅", "Lecture",
                    f"Le pic d'activité ({int(pic.n)} analyses) a été enregistré le {pic.date}. "
                    + ("L'usage est régulier sur la période." if len(act) > 1 else "Première journée d'utilisation du scoring."),
                    "info")

    with c2:
        chart_head("🎯", "Risque moyen par objet de prêt", "Quelles finalités présentent le plus de risque ?")
        par_objet = df.groupby("objet").proba_pct.mean().sort_values(ascending=True).reset_index()
        fig = go.Figure(go.Bar(
            y=par_objet.objet, x=par_objet.proba_pct, orientation="h",
            marker_color=[COULEURS["Favorable"] if v < 20 else
                          COULEURS["Modéré"] if v < 40 else
                          COULEURS["Défavorable"] for v in par_objet.proba_pct],
            marker=dict(cornerradius=5),
            text=[f"{v:.0f}%" for v in par_objet.proba_pct], textposition="outside",
        ))
        fig.update_layout(height=300, showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Manrope"),
                          xaxis=dict(gridcolor="#E5EAF3", range=[0, 100]), yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_foot()
        pire, meilleur = par_objet.objet.iloc[-1], par_objet.objet.iloc[0]
        insight_box("⚠️", "Lecture",
                    f"Les prêts « {pire} » présentent le risque moyen le plus élevé, tandis que « {meilleur} » "
                    "sont les plus sûrs. Adaptez les conditions (taux, garanties) selon la finalité.",
                    "warning")

    # ---------------- Graphique 3 : scatter + logement ----------------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_head("🔵", "Revenu vs montant du prêt", "Taille de la bulle = probabilité de défaut")
        fig = go.Figure()
        for dec in ["Favorable", "Modéré", "Défavorable"]:
            sub = df[df.decision == dec]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub.revenu, y=sub.montant, mode="markers", name=dec,
                marker=dict(size=sub.proba_pct / 4 + 6, color=COULEURS[dec],
                            opacity=0.75, line=dict(width=1, color="white")),
            ))
        fig.update_layout(height=320,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Manrope"),
                          xaxis=dict(title="Revenu annuel ($)", gridcolor="#E5EAF3"),
                          yaxis=dict(title="Montant du prêt ($)", gridcolor="#E5EAF3"),
                          legend=dict(orientation="h", y=-0.18),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_foot()
        corr = df["dti"].corr(df["proba_defaut"])
        force = "forte" if abs(corr) > 0.6 else "modérée" if abs(corr) > 0.3 else "faible"
        insight_box("🔗", "Lecture",
                    f"La corrélation entre endettement (DTI) et probabilité de défaut est {force} "
                    f"(r = {corr:.2f}). Les bulles rouges (dossiers risqués) se concentrent chez les "
                    "demandeurs peu revenus relativement au montant emprunté.",
                    "info")

    with c2:
        chart_head("🏠", "Risque par situation de logement", "Probabilité de défaut moyenne par statut")
        par_home = df.groupby("logement").agg(n=("proba_pct", "size"), moy=("proba_pct", "mean")).reset_index()
        fig = go.Figure(go.Bar(
            x=par_home.logement, y=par_home.moy,
            marker_color=[COULEURS["Favorable"] if v < 20 else
                          COULEURS["Modéré"] if v < 40 else
                          COULEURS["Défavorable"] for v in par_home.moy],
            marker=dict(cornerradius=6),
            text=[f"{v:.0f}%<br>({int(n)})" for v, n in zip(par_home.moy, par_home.n)],
            textposition="outside",
        ))
        fig.update_layout(height=320, showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(family="Manrope"),
                          yaxis=dict(gridcolor="#E5EAF3", range=[0, 100]), xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        chart_foot()
        top = par_home.loc[par_home.moy.idxmax()]
        insight_box("🏠", "Lecture",
                    f"Le statut « {top.logement} » concentre le risque moyen le plus élevé ({top.moy:.0f}%). "
                    "Les propriétaires (OWN/MORTGAGE) présentent historiquement des défauts plus faibles.",
                    "warning")

    # ---------------- Historique détaillé ----------------
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

    st.dataframe(
        df_aff.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "proba_pct": st.column_config.ProgressColumn(
                "Risque", min_value=0, max_value=100, format="%.1f"),
            "Score": st.column_config.NumberColumn("Score", format="%d/100"),
        },
    )

    cexp, cdel, _ = st.columns([1, 1, 2])
    csv = df.to_csv(index=False).encode("utf-8")
    cexp.download_button("⬇️ Exporter CSV", data=csv,
                         file_name="historique_analyses.csv", mime="text/csv")
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
            if st.button("🗑️ Vider l'historique"):
                st.session_state.confirm_clear = True
                st.rerun()

# --------------------------------------------------------------------
# Footer commun
# --------------------------------------------------------------------
def render_footer():
    st.markdown(f"""
    <div class="footer">
        <strong>CreditScore Pro</strong> • Modèle Random Forest • Spark ML & Streamlit
        <br>🔒 Données traitées de manière confidentielle • Aucune donnée persistée hors historique local
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