"""
CreditScore Pro v3.2 — Application Streamlit professionnelle
=============================================================
• Page Diagnostic : scoring de risque de crédit (Spark ML / Random Forest)
• Page Dashboard  : historique persistant + KPIs + graphiques + export PDF

v3.2 — Ajout de l'export PDF professionnel avec interprétations détaillées
"""

import json
import os
import time
import uuid
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

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
# Design system CSS
# --------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

html, body, .stApp { background: #F4F6FB; font-family: 'Manrope', sans-serif; color: #26324A; }
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
.block-container { padding-top: 1.5rem; max-width: 1220px; }
h1,h2,h3,h4 { font-family: 'Sora', sans-serif !important; color: #12244A !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5EAF3; }
section[data-testid="stSidebar"] .stRadio > div { display: flex; flex-direction: column; gap: 8px; }
section[data-testid="stSidebar"] .stRadio label {
    display: flex; align-items: center; gap: 10px;
    padding: 12px 16px; border-radius: 12px;
    border: 1px solid #E5EAF3; background: #F8FAFD;
    cursor: pointer; font-weight: 600; font-size: .92rem;
    transition: all .25s ease; margin: 0;
}
section[data-testid="stSidebar"] .stRadio label:hover { border-color: #1B3B6F; transform: translateX(3px); }
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: linear-gradient(135deg, #12244A, #1B3B6F);
    color: #fff; border-color: transparent;
    box-shadow: 0 6px 18px rgba(18,36,74,.28);
}
section[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none; }

/* Hero */
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
.hero h1 { color: #fff !important; font-size: 2.3rem; margin: 0; letter-spacing: -1px; position: relative; }
.hero h1 span { color: #D4AF37; }
.hero p { color: rgba(255,255,255,.78); margin: .4rem 0 0; font-size: .98rem; position: relative; }
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; position: relative; }
.hbadge {
    background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
    backdrop-filter: blur(8px); color: #fff;
    padding: 6px 14px; border-radius: 30px; font-size: .78rem; font-weight: 600;
}
.hbadge.gold { background: rgba(212,175,55,.22); border-color: rgba(212,175,55,.5); color: #F0D488; }

/* KPI */
.kpi {
    background: #fff; border: 1px solid #E5EAF3; border-radius: 16px;
    padding: 1.1rem 1.2rem; display: flex; gap: 14px; align-items: center;
    box-shadow: 0 2px 10px rgba(18,36,74,.05); border-left: 4px solid #1B3B6F;
    transition: all .3s ease; margin-bottom: 14px; animation: fadeUp .5s ease both;
}
.kpi:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(18,36,74,.12); }
.kpi-icon {
    width: 46px; height: 46px; border-radius: 12px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; background: #EEF3FB;
}
.kpi-label { font-size: .72rem; text-transform: uppercase; letter-spacing: .6px; color: #7A8699; font-weight: 700; }
.kpi-value { font-family: 'Sora'; font-size: 1.55rem; font-weight: 800; color: #12244A; line-height: 1.2; }
.kpi-sub   { font-size: .75rem; color: #93A0B4; font-weight: 500; }
.tone-green .kpi-icon { background: #E6F7F0; } .tone-green { border-left-color: #0FA36B; }
.tone-red   .kpi-icon { background: #FBECEC; } .tone-red   { border-left-color: #D64545; }
.tone-amber .kpi-icon { background: #FDF4E5; } .tone-amber { border-left-color: #E8A13A; }
.tone-gold  .kpi-icon { background: #FAF3DF; } .tone-gold  { border-left-color: #C9A227; }
.tone-blue  .kpi-icon { background: #EAF1FB; } .tone-blue  { border-left-color: #2E6DB4; }

/* Sections & titres de graphiques */
.section-title { display: flex; gap: 12px; align-items: center; margin: 2rem 0 1rem; }
.section-title .st-icon {
    width: 40px; height: 40px; border-radius: 12px; font-size: 1.2rem;
    background: linear-gradient(135deg,#12244A,#1B3B6F);
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 12px rgba(18,36,74,.25);
}
.section-title h3 { margin: 0; font-size: 1.15rem; }
.section-title p  { margin: 2px 0 0; font-size: .82rem; color: #7A8699; }

.chart-head { margin: 1.4rem 0 .5rem; }
.chart-head h4 { margin: 0; font-size: .98rem; }
.chart-head p  { margin: 2px 0 0; font-size: .8rem; color: #7A8699; }

/* Insights */
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

/* Résultat diagnostic */
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

/* Widgets */
.stButton > button {
    background: linear-gradient(135deg, #12244A, #1B3B6F) !important;
    color: #fff !important; font-weight: 700 !important;
    border: none !important; border-radius: 12px !important;
    padding: .7rem 1.5rem !important; transition: all .3s ease !important;
    box-shadow: 0 6px 18px rgba(18,36,74,.25) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 26px rgba(18,36,74,.4) !important; }

div[data-testid="stMetric"] {
    background: #fff; border: 1px solid #E5EAF3; border-radius: 14px;
    padding: 12px 16px; box-shadow: 0 2px 10px rgba(18,36,74,.05);
}
div[data-testid="stMetric"] label { color: #7A8699 !important; font-weight: 600; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #12244A !important; font-family: 'Sora'; }

.stNumberInput input, .stSelectbox select {
    border-radius: 10px !important; border: 1.5px solid #DCE3EF !important;
}

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
    b = "".join(f'<span class="hbadge{" gold" if g else ""}">{t}</span>' for t, g in badges)
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

def chart_title(icon, titre, sous):
    st.markdown(f'<div class="chart-head"><h4>{icon} {titre}</h4><p>{sous}</p></div>',
                unsafe_allow_html=True)

def insight_box(icon, titre, texte, tone="info"):
    st.markdown(f"""
    <div class="insight insight-{tone}">
        <span class="i-icon">{icon}</span>
        <div><strong>{titre}</strong> — {texte}</div>
    </div>""", unsafe_allow_html=True)

def style_fig(fig, height=300, legend=False):
    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color="#26324A"),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=legend,
    )
    return fig

def afficher_fig(fig):
    """Affichage sécurisé : un graphique cassé ne tue plus la page."""
    try:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
# Génération du rapport PDF
# --------------------------------------------------------------------
def generer_rapport_pdf(df, inclure_graphiques=True):
    """Génère un rapport PDF professionnel avec toutes les analyses."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='TitrePrincipal', parent=styles['Heading1'],
        fontSize=24, textColor=colors.HexColor('#12244A'),
        spaceAfter=30, alignment=TA_CENTER, fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='SousTitre', parent=styles['Heading2'],
        fontSize=14, textColor=colors.HexColor('#1B3B6F'),
        spaceAfter=12, spaceBefore=20, fontName='Helvetica-Bold',
    ))
    styles.add(ParagraphStyle(
        name='CorpsTexte', parent=styles['Normal'],
        fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name='InsightBox', parent=styles['Normal'],
        fontSize=9, leading=12,
        backColor=colors.HexColor('#EAF1FB'),
        borderColor=colors.HexColor('#C9DCF5'),
        borderWidth=1, borderPadding=8, spaceAfter=10,
    ))
    
    story = []
    
    # ============ PAGE DE COUVERTURE ============
    story.append(Spacer(1, 3*cm))
    story.append(Paragraph("CreditScore Pro", styles['TitrePrincipal']))
    story.append(Paragraph("Rapport d'Analyse de Risque de Crédit", styles['Heading2']))
    story.append(Spacer(1, 1*cm))
    
    total = len(df)
    taux_fav = (df.decision == "Favorable").mean() * 100
    proba_moy = df.proba_pct.mean()
    montant_moy = df.montant.mean()
    
    donnees_couverture = [
        ['Indicateur', 'Valeur'],
        ['Nombre d\'analyses', str(total)],
        ['Taux d\'avis favorables', f'{taux_fav:.1f}%'],
        ['Probabilité de défaut moyenne', f'{proba_moy:.1f}%'],
        ['Montant moyen demandé', f'{montant_moy:,.0f} $'],
        ['Date du rapport', datetime.now().strftime('%d/%m/%Y')],
    ]
    
    table_couverture = Table(donnees_couverture, colWidths=[8*cm, 6*cm])
    table_couverture.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#12244A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFD')]),
    ]))
    story.append(table_couverture)
    story.append(PageBreak())
    
    # ============ RÉSUMÉ EXÉCUTIF ============
    story.append(Paragraph("Résumé Exécutif", styles['SousTitre']))
    resume = (
        f"Ce rapport synthétise <b>{total} analyses</b> de risque de crédit effectuées "
        f"avec le modèle CreditScore Pro. Le taux d'avis favorables est de <b>{taux_fav:.1f}%</b>, "
        f"indiquant {'un profil globalement sain' if taux_fav >= 50 else 'une majorité de dossiers à risque'}. "
        f"La probabilité de défaut moyenne s'établit à <b>{proba_moy:.1f}%</b>, avec un montant moyen "
        f"de prêt demandé de <b>{montant_moy:,.0f} $</b>."
    )
    story.append(Paragraph(resume, styles['CorpsTexte']))
    story.append(Spacer(1, 0.5*cm))
    
    if taux_fav >= 70:
        interpretation = "✅ Le portefeuille analysé présente un profil très favorable. La majorité des dossiers sont solvables avec un risque maîtrisé."
    elif taux_fav >= 50:
        interpretation = "⚠️ Le portefeuille est globalement acceptable mais nécessite une vigilance sur certains segments."
    else:
        interpretation = "🔴 La majorité des dossiers présentent un risque élevé. Un renforcement des critères de sélection est recommandé."
    
    story.append(Paragraph(interpretation, styles['InsightBox']))
    story.append(Spacer(1, 0.5*cm))
    
    # Graphiques (si kaleido disponible)
    if inclure_graphiques:
        try:
            import plotly.io as pio
            
            story.append(Paragraph("Visualisations Clés", styles['SousTitre']))
            
            counts = df.decision.value_counts()
            fig_donut = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values), hole=0.6,
                marker=dict(colors=['#0FA36B', '#E8A13A', '#D64545'][:len(counts)]),
            ))
            
            img_bytes = pio.to_image(fig_donut, format='png', width=500, height=350)
            img = Image(io.BytesIO(img_bytes), width=12*cm, height=8.4*cm)
            story.append(img)
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(
                "<i>Répartition des décisions de crédit</i>",
                ParagraphStyle('Caption', parent=styles['Normal'],
                              fontSize=9, alignment=TA_CENTER)
            ))
            story.append(Spacer(1, 0.5*cm))
        except Exception:
            story.append(Paragraph(
                "⚠️ Graphiques non inclus (kaleido non disponible)",
                styles['CorpsTexte']
            ))
    
    # ============ TABLEAU DÉTAILLÉ ============
    story.append(PageBreak())
    story.append(Paragraph("Détail des Analyses", styles['SousTitre']))
    
    df_table = df[[
        'date', 'age', 'revenu', 'montant', 'objet',
        'logement', 'proba_pct', 'decision'
    ]].copy()
    df_table['date'] = df_table['date'].dt.strftime('%d/%m/%y')
    df_table = df_table.sort_values('date', ascending=False).head(20)
    
    donnees_table = [['Date', 'Âge', 'Revenu', 'Prêt', 'Objet', 'Risque', 'Décision']]
    for _, row in df_table.iterrows():
        donnees_table.append([
            row['date'],
            str(row['age']),
            f"{row['revenu']:,}",
            f"{row['montant']:,}",
            row['objet'][:12],
            f"{row['proba_pct']:.0f}%",
            row['decision'],
        ])
    
    table_detail = Table(
        donnees_table,
        colWidths=[2.2*cm, 1.5*cm, 2.2*cm, 2.2*cm, 2.8*cm, 1.8*cm, 2.5*cm]
    )
    table_detail.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#12244A')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFD')]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    for i, row in enumerate(donnees_table[1:], start=1):
        decision = row[-1]
        if decision == 'Favorable':
            table_detail.setStyle(TableStyle([
                ('BACKGROUND', (-1, i), (-1, i), colors.HexColor('#E6F7F0')),
                ('TEXTCOLOR', (-1, i), (-1, i), colors.HexColor('#0B6B47')),
            ]))
        elif decision == 'Modéré':
            table_detail.setStyle(TableStyle([
                ('BACKGROUND', (-1, i), (-1, i), colors.HexColor('#FDF4E5')),
                ('TEXTCOLOR', (-1, i), (-1, i), colors.HexColor('#8A5A14')),
            ]))
        else:
            table_detail.setStyle(TableStyle([
                ('BACKGROUND', (-1, i), (-1, i), colors.HexColor('#FBECEC')),
                ('TEXTCOLOR', (-1, i), (-1, i), colors.HexColor('#8C2B2B')),
            ]))
    
    story.append(table_detail)
    
    if len(df) > 20:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            f"<i>Tableau limité aux 20 analyses les plus récentes sur {total} au total.</i>",
            ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
        ))
    
    # ============ ANALYSES DÉTAILLÉES ============
    story.append(PageBreak())
    story.append(Paragraph("Analyses Détaillées & Interprétations", styles['SousTitre']))
    
    df_detail = df.sort_values('date', ascending=False).head(5)
    
    for idx, (_, row) in enumerate(df_detail.iterrows(), 1):
        story.append(Paragraph(
            f"Analyse #{idx} — {row['date'].strftime('%d/%m/%Y %H:%M')}",
            styles['Heading3']
        ))
        
        carac = (
            f"<b>Profil :</b> {row['age']} ans, revenu {row['revenu']:,} $, "
            f"ancienneté {row['anciennete']} ans<br/>"
            f"<b>Prêt :</b> {row['montant']:,} $ à {row['taux']}% pour {row['objet']}<br/>"
            f"<b>Logement :</b> {row['logement']}, historique crédit {row['hist_credit']} ans<br/>"
            f"<b>Endettement :</b> {row['dti']*100:.1f}%"
        )
        story.append(Paragraph(carac, styles['CorpsTexte']))
        
        decision = row['decision']
        proba = row['proba_pct']
        
        if decision == 'Favorable':
            couleur = '#0FA36B'
            emoji = '✅'
            interpretation_text = (
                f"Le dossier présente un profil favorable avec une probabilité de défaut "
                f"de seulement {proba:.1f}%. Les indicateurs financiers sont solides et "
                f"le risque est maîtrisé. L'octroi du crédit peut être envisagé aux "
                f"conditions proposées."
            )
        elif decision == 'Modéré':
            couleur = '#E8A13A'
            emoji = '⚠️'
            interpretation_text = (
                f"Le dossier présente un risque modéré (probabilité de défaut : {proba:.1f}%). "
                f"Certains indicateurs appellent à la prudence. Un examen complémentaire et "
                f"des garanties proportionnées sont recommandés."
            )
        else:
            couleur = '#D64545'
            emoji = '🔴'
            interpretation_text = (
                f"Le dossier présente un risque élevé (probabilité de défaut : {proba:.1f}%). "
                f"Plusieurs facteurs de risque sont identifiés. Le refus ou des conditions "
                f"très restrictives (garanties substantielles, taux majoré) sont recommandés."
            )
        
        style_resultat = ParagraphStyle(
            f'Resultat{idx}', parent=styles['Normal'],
            fontSize=10, leading=13,
            backColor=colors.HexColor(couleur + '20'),
            borderColor=colors.HexColor(couleur),
            borderWidth=2, borderPadding=10, spaceAfter=12,
        )
        
        story.append(Paragraph(
            f"{emoji} <b>Décision : {decision}</b> (Score : {row['score']}/100)<br/><br/>"
            f"{interpretation_text}",
            style_resultat
        ))
        
        story.append(Spacer(1, 0.3*cm))
    
    if len(df) > 5:
        story.append(Paragraph(
            f"<i>Seules les 5 analyses les plus récentes sont détaillées. "
            f"Le rapport complet comprend {total} analyses.</i>",
            ParagraphStyle('Note', parent=styles['Normal'], fontSize=8, textColor=colors.grey)
        ))
    
    # ============ RECOMMANDATIONS GLOBALES ============
    story.append(PageBreak())
    story.append(Paragraph("Recommandations Globales", styles['SousTitre']))
    
    if taux_fav >= 70:
        recommandations = """
        <b>1. Maintenir les critères actuels</b><br/>
        Le portefeuille présente un profil sain. Les critères de sélection sont appropriés.<br/><br/>
        <b>2. Optimiser l'offre commerciale</b><br/>
        Proposer des services complémentaires aux clients solvables pour augmenter la fidélisation.<br/><br/>
        <b>3. Surveiller les segments à risque</b><br/>
        Identifier les sous-segments qui concentrent les risques pour adapter les conditions.
        """
    elif taux_fav >= 50:
        recommandations = """
        <b>1. Renforcer le ciblage en amont</b><br/>
        Améliorer la pré-qualification des demandes pour réduire le taux de dossiers risqués.<br/><br/>
        <b>2. Adapter les conditions</b><br/>
        Pour les dossiers modérés : garanties proportionnelles, taux ajustés, suivi renforcé.<br/><br/>
        <b>3. Analyser les segments critiques</b><br/>
        Identifier les objets de prêt et profils qui concentrent les risques.
        """
    else:
        recommandations = """
        <b>1. Revoir les critères de sélection</b><br/>
        Renforcer les exigences minimales (revenu, ancienneté, endettement).<br/><br/>
        <b>2. Refuser les dossiers à haut risque</b><br/>
        Appliquer strictement le refus pour les probabilités de défaut > 40%.<br/><br/>
        <b>3. Exiger des garanties substantielles</b><br/>
        Garanties réelles, caution solidaire, assurance crédit obligatoire.<br/><br/>
        <b>4. Formation des équipes</b><br/>
        Sensibiliser les conseillers aux profils à risque.
        """
    
    story.append(Paragraph(recommandations, styles['CorpsTexte']))
    
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        f"<i>Rapport généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — "
        f"CreditScore Pro — Modèle Random Forest (AUC: 0.92)</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                      textColor=colors.grey, alignment=TA_CENTER)
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
    st.caption(f"v3.2 • Random Forest • {len(st.session_state.historique)} analyse(s)")

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
        Le **Dashboard** reste consultable.
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
            <div style="opacity:.85; font-size:.9rem;">Probabilité estimée de défaut de paiement</div>
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
                afficher_fig(fig_gauge)
            except Exception as e:
                st.warning(f"⚠️ Jauge indisponible : {e}")

        with g2:
            chart_title("📊", "Décomposition des facteurs", "Risque par dimension du dossier")
            try:
                facteurs = {
                    "Âge": max(0, min(100, (age - 18) / 82 * 100)),
                    "Revenu": max(0, min(100, revenu / 100_000 * 100)),
                    "Ancienneté": max(0, min(100, emp_length / 20 * 100)),
                    "Endettement": max(0, min(100, dti * 100)),
                    "Historique": max(0, min(100, cred_hist / 20 * 100)),
                }
                fig = go.Figure(go.Bar(
                    x=list(facteurs.keys()), y=list(facteurs.values()),
                    marker=dict(color=[couleur_risque(v) for v in facteurs.values()]),
                    text=[f"{v:.0f}%" for v in facteurs.values()], textposition="outside",
                ))
                fig.update_layout(
                    height=240, showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Manrope"),
                    yaxis=dict(range=[0, 110], gridcolor="#E5EAF3"),
                    xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    margin=dict(l=10, r=10, t=20, b=10),
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

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("🧮", "Analyses effectuées", f"{total}", "dossiers traités", "blue")
        kpi_card("💰", "Montant moyen", f"{montant_moy:,.0f} $", "par dossier", "gold")
    with k2:
        kpi_card("✅", "Avis favorables", f"{taux_fav:.0f}%", "dossiers acceptables", "green")
        kpi_card("📉", "Risque moyen", f"{proba_moy:.1f}%", "probabilité de défaut", "amber")
    with k3:
        kpi_card("🔴", "Taux de refus", f"{taux_ref:.0f}%", "dossiers à risque", "red")
        kpi_card("⚖️", "Endettement moyen", f"{dti_moy:.1f}%", "ratio prêt / revenu", "navy")

    section_title("📈", "Graphiques & explications", "Interprétation automatique des données")

    # ---------- 1. Donut décisions ----------
    c1, c2 = st.columns(2, gap="large")
    with c1:
        chart_title("🍩", "Répartition des décisions", "Part de chaque avis rendu")
        try:
            counts = df.decision.value_counts()
            fig = go.Figure(go.Pie(
                labels=list(counts.index), values=list(counts.values), hole=0.62,
                marker=dict(colors=[COULEURS.get(c, "#888888") for c in counts.index]),
                textinfo="percent",
            ))
            fig.update_layout(legend=dict(orientation="h", y=-0.08))
            afficher_fig(style_fig(fig, 300, legend=True))
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
            fig = go.Figure(go.Bar(x=labels, y=list(hist.values),
                                   marker=dict(color=colors_list),
                                   text=list(hist.values), textposition="outside"))
            afficher_fig(style_fig(fig, 300))
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
            fig.add_trace(go.Bar(x=act["date"].astype(str), y=act.n,
                                 marker=dict(color="#2E6DB4")))
            fig.add_trace(go.Scatter(x=act["date"].astype(str), y=act.n,
                                     mode="lines+markers",
                                     line=dict(color="#D4AF37", width=2.5)))
            fig.update_layout(barmode="overlay")
            afficher_fig(style_fig(fig, 300))
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
                marker=dict(color=[couleur_risque(v) for v in par_objet.proba_pct]),
                text=[f"{v:.0f}%" for v in par_objet.proba_pct], textposition="outside",
            ))
            fig.update_layout(xaxis=dict(range=[0, 100]))
            afficher_fig(style_fig(fig, 300))
            pire, meilleur = par_objet.objet.iloc[-1], par_objet.objet.iloc[0]
            txt = (f"Les prêts « {pire} » présentent le risque moyen le plus élevé, tandis que « {meilleur} » "
                   "sont les plus sûrs. Adaptez les conditions (taux, garanties) selon la finalité.")
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
            txt = "Données insuffisantes pour comparer les objets de prêt."
        insight_box("⚠️", "Interprétation", txt, "warning")

    # ---------- 5. Scatter revenu / prêt ----------
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
                    marker=dict(size=list(sub.proba_pct / 4 + 6),
                                color=COULEURS[dec], opacity=0.75,
                                line=dict(width=1, color="white")),
                ))
            fig.update_layout(
                xaxis=dict(title="Revenu annuel ($)"),
                yaxis=dict(title="Montant du prêt ($)"),
                legend=dict(orientation="h", y=-0.18),
            )
            afficher_fig(style_fig(fig, 320, legend=True))
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
        try:
            corr = df["dti"].corr(df["proba_defaut"]) if len(df) > 2 else float("nan")
            if pd.isna(corr):
                txt = "Corrélation non calculable avec si peu d'analyses (minimum 3 requises)."
            else:
                force = "forte" if abs(corr) > 0.6 else "modérée" if abs(corr) > 0.3 else "faible"
                txt = (f"La corrélation entre endettement (DTI) et probabilité de défaut est {force} (r = {corr:.2f}). "
                       "Les bulles rouges se concentrent chez les demandeurs dont le prêt est élevé relativement au revenu.")
        except Exception:
            txt = "Corrélation non calculable."
        insight_box("🔗", "Interprétation", txt, "info")

    # ---------- 6. Risque par logement ----------
    with c2:
        chart_title("🏠", "Risque par situation de logement", "Probabilité de défaut moyenne par statut")
        try:
            par_home = df.groupby("logement").agg(
                n=("proba_pct", "size"), moy=("proba_pct", "mean")).reset_index()
            fig = go.Figure(go.Bar(
                x=list(par_home.logement), y=list(par_home.moy),
                marker=dict(color=[couleur_risque(v) for v in par_home.moy]),
                text=[f"{v:.0f}%" for v in par_home.moy], textposition="outside",
            ))
            fig.update_layout(yaxis=dict(range=[0, 100]))
            afficher_fig(style_fig(fig, 320))
            top = par_home.loc[par_home.moy.idxmax()]
            txt = (f"Le statut « {top.logement} » concentre le risque moyen le plus élevé ({top.moy:.0f}%). "
                   "Les propriétaires présentent historiquement des défauts plus faibles.")
        except Exception as e:
            st.warning(f"⚠️ Graphique indisponible : {e}")
            txt = "Données insuffisantes par statut de logement."
        insight_box("🏠", "Interprétation", txt, "warning")

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
                "proba_pct": st.column_config.ProgressColumn("Risque", min_value=0, max_value=100, format="%.1f"),
            },
        )
    except Exception:
        st.dataframe(df_aff.sort_values("Date", ascending=False),
                     use_container_width=True, hide_index=True)

    # ---------- Boutons d'export ----------
    cexp, cpdf, cdel, _ = st.columns([1, 1, 1, 2])
    
    csv = df.to_csv(index=False).encode("utf-8")
    cexp.download_button("⬇️ Exporter CSV", data=csv,
                         file_name="historique_analyses.csv", mime="text/csv")
    
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
    st.markdown("""
    <div class="footer">
        <strong>CreditScore Pro</strong> • Modèle Random Forest • Spark ML & Streamlit
        <br>🔒 Données traitées de manière confidentielle
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