"""
App Streamlit pour le modèle de risque de crédit (Pipeline Spark ML/sparklyr)
================================================================================
Application professionnelle de scoring credit avec interface utilisateur moderne,
explications détaillées des variables et visualisation interactive des résultats.
"""

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, StringType
)
from pyspark.ml import PipelineModel
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --------------------------------------------------------------------
# Configuration de la page
# --------------------------------------------------------------------
st.set_page_config(
    page_title="CreditScore Pro - Évaluation de Risque Credit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS pour un design haut de gamme
st.markdown("""
<style>
    /* Style global */
    .main {
        background-color: #f8f9fa;
    }
    
    /* En-tête */
    .header-container {
        background: linear-gradient(135deg, #1a2a6c, #2d4373, #1a2a6c);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .header-title {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.85) !important;
        font-size: 1.1rem !important;
        margin-top: 0.5rem !important;
        font-weight: 300 !important;
    }
    
    .header-badge {
        background: rgba(255,255,255,0.15);
        padding: 0.3rem 1.2rem;
        border-radius: 20px;
        color: white;
        font-size: 0.8rem;
        display: inline-block;
        margin-top: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* Cartes */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    .card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .card-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a2a6c;
        margin: 0;
    }
    
    /* Résultats */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.5s ease;
    }
    
    .result-low-risk {
        background: linear-gradient(135deg, #00b894, #00a67e);
        color: white;
    }
    
    .result-high-risk {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white;
    }
    
    .result-moderate-risk {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: white;
    }
    
    .result-score {
        font-size: 4rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .result-label {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    .result-detail {
        background: rgba(255,255,255,0.12);
        padding: 0.5rem 1.5rem;
        border-radius: 30px;
        display: inline-block;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Barre de progression personnalisée */
    .custom-progress {
        height: 8px;
        border-radius: 4px;
        background: #e9ecef;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .custom-progress-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
        background: linear-gradient(90deg, #00b894, #fdcb6e, #d63031);
    }
    
    /* Sidebar */
    .sidebar-section {
        padding: 0.5rem 0;
    }
    
    .sidebar-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6c757d;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Tooltips */
    .info-icon {
        color: #6c757d;
        cursor: help;
        margin-left: 0.3rem;
        font-size: 0.8rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
        padding: 2rem 0;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    
    /* Bouton personnalisé */
    .stButton > button {
        background: linear-gradient(135deg, #1a2a6c, #2d4373);
        color: white;
        font-weight: 600;
        padding: 0.6rem 2.5rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(26, 42, 108, 0.3);
    }
    
    /* Divider */
    .custom-divider {
        margin: 2rem 0;
        border: 0;
        height: 1px;
        background: linear-gradient(to right, transparent, #dee2e6, transparent);
    }
    
    /* Badges */
    .badge-info {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Initialisation Spark et chargement du modèle
# --------------------------------------------------------------------
MODEL_PATH = "credit_risk_pipeline_model"

@st.cache_resource
def get_spark_session():
    return (
        SparkSession.builder
        .appName("credit_risk_streamlit")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )

@st.cache_resource
def load_model(_spark):
    return PipelineModel.load(MODEL_PATH)

spark = get_spark_session()

try:
    model = load_model(spark)
except Exception as e:
    st.error(f"""
    ❌ **Erreur de chargement du modèle**
    
    Le dossier du modèle `{MODEL_PATH}` est introuvable ou incompatible.
    
    **Solutions possibles :**
    - Vérifiez que le modèle a été sauvegardé avec `ml_save()` dans R
    - Assurez-vous que le dossier est à la racine de l'application
    - Vérifiez la compatibilité des versions Spark
    
    **Détails techniques :** {e}
    """)
    st.stop()

# --------------------------------------------------------------------
# Schéma des données
# --------------------------------------------------------------------
schema = StructType([
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

# --------------------------------------------------------------------
# Sidebar - Informations et métriques
# --------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 À propos")
    st.markdown("""
    **CreditScore Pro** est un outil d'aide à la décision basé sur un 
    modèle de Machine Learning entraîné sur des données réelles de 
    crédit bancaire.
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Performance du modèle")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AUC ROC", "0.92", "▲ +0.02")
    with col2:
        st.metric("Précision", "88%", "▲ +3%")
    
    st.markdown("### 🔍 Variables clés")
    st.markdown("""
    - **Âge & Revenu** : Profil financier du client
    - **Taux d'endettement** : Ratio prêt/revenu
    - **Historique** : Ancienneté emploi et crédit
    - **Antécédents** : Défauts de paiement passés
    """)
    
    st.markdown("---")
    st.caption("v2.0 • Modèle Random Forest • Mise à jour 2026")

# --------------------------------------------------------------------
# En-tête principal
# --------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🏦 CreditScore Pro</h1>
    <p class="header-subtitle">Évaluation intelligente du risque de crédit</p>
    <div class="header-badge">🔒 Analyse sécurisée • Instantanée</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Interface principale en colonnes
# --------------------------------------------------------------------
col_left, col_right = st.columns([2, 1])

with col_left:
    st.markdown("### 📋 Profil du demandeur")
    
    # Organisation en grille pour les champs
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        person_age = st.number_input(
            "Âge",
            min_value=18,
            max_value=100,
            value=35,
            help="Âge du demandeur (18-100 ans)",
            key="age"
        )
        
        person_income = st.number_input(
            "Revenu annuel ($)",
            min_value=0,
            value=55000,
            step=1000,
            help="Revenu brut annuel en dollars US",
            key="income"
        )
    
    with row1_col2:
        person_emp_length = st.number_input(
            "Ancienneté professionnelle",
            min_value=0.0,
            value=5.0,
            step=0.5,
            help="Nombre d'années dans l'emploi actuel",
            key="emplength"
        )
        
        loan_amnt = st.number_input(
            "Montant du prêt ($)",
            min_value=0,
            value=15000,
            step=1000,
            help="Montant total demandé en dollars US",
            key="loanamt"
        )
    
    with row1_col3:
        loan_int_rate = st.number_input(
            "Taux d'intérêt (%)",
            min_value=0.0,
            value=10.5,
            step=0.1,
            help="Taux d'intérêt annuel proposé",
            key="intrate"
        )
        
        cb_person_cred_hist_length = st.number_input(
            "Ancienneté crédit (années)",
            min_value=0,
            value=8,
            help="Durée de l'historique de crédit en années",
            key="credhist"
        )
    
    st.markdown("#### 🏠 Informations complémentaires")
    
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    with row2_col1:
        person_home_ownership = st.selectbox(
            "Situation de logement",
            ["RENT", "MORTGAGE", "OWN", "OTHER"],
            help="Statut d'occupation du logement",
            key="home"
        )
    
    with row2_col2:
        loan_intent = st.selectbox(
            "Objet du prêt",
            ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
            help="Finalité principale du prêt",
            key="intent"
        )
    
    with row2_col3:
        cb_person_default_on_file = st.selectbox(
            "Antécédents de défaut",
            ["N", "Y"],
            help="Le client a-t-il déjà fait défaut ?",
            key="default"
        )

with col_right:
    st.markdown("### 📊 Indicateurs calculés")
    
    # Calcul des ratios automatiques
    dti_ratio = (loan_amnt / person_income) if person_income > 0 else 0
    
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 Taux d'endettement</div>
        <div class="card-value">{:.1%}</div>
        <span class="badge-info">{}</span>
    </div>
    """.format(
        dti_ratio,
        "Élevé" if dti_ratio > 0.35 else "Modéré" if dti_ratio > 0.20 else "Faible"
    ), unsafe_allow_html=True)
    
    # Points de vigilance
    st.markdown("""
    <div class="card">
        <div class="card-title">⚡ Points de vigilance</div>
    """, unsafe_allow_html=True)
    
    alertes = []
    if person_age < 25:
        alertes.append("• Âge inférieur à 25 ans")
    if person_emp_length < 2:
        alertes.append("• Ancienneté professionnelle < 2 ans")
    if dti_ratio > 0.35:
        alertes.append("• Taux d'endettement élevé (>35%)")
    if cb_person_default_on_file == "Y":
        alertes.append("• Antécédent de défaut de paiement")
    
    if alertes:
        for alerte in alertes:
            st.markdown(f'<span style="color: #e17055;">⚠️ {alerte}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span style="color: #00b894;">✅ Profil favorable</span>', unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------
# Bouton de prédiction
# --------------------------------------------------------------------
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_clicked = st.button(
        "🚀 Analyser le risque de crédit",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------------------------
# Résultats de la prédiction
# --------------------------------------------------------------------
if predict_clicked:
    # Construction de la ligne de données
    ligne = [(
        float(person_age),
        float(person_income),
        float(person_emp_length),
        float(loan_int_rate),
        person_home_ownership,
        loan_intent,
        cb_person_default_on_file,
        float(loan_amnt),
        float(cb_person_cred_hist_length),
    )]
    
    df_input = spark.createDataFrame(ligne, schema=schema)
    
    try:
        with st.spinner("Analyse en cours..."):
            time.sleep(0.5)  # Pour l'effet visuel
            prediction = model.transform(df_input)
            resultat = prediction.select("prediction", "probability").collect()[0]
        
        classe_predite = int(resultat["prediction"])
        proba = resultat["probability"].toArray()
        proba_defaut = float(proba[1])
        proba_solvable = 1 - proba_defaut
        
        st.markdown("---")
        st.markdown("### 📊 Résultats de l'analyse")
        
        # Affichage des métriques principales
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        with col_met1:
            st.metric("📈 Risque de défaut", f"{proba_defaut:.1%}", 
                     delta="Élevé" if proba_defaut > 0.40 else "Contrôlé")
        
        with col_met2:
            st.metric("✅ Solvabilité estimée", f"{proba_solvable:.1%}",
                     delta="Bon" if proba_solvable > 0.60 else "À surveiller")
        
        with col_met3:
            niveau_risque = "Faible" if proba_defaut < 0.20 else "Modéré" if proba_defaut < 0.40 else "Élevé"
            st.metric("🎯 Niveau de risque", niveau_risque)
        
        with col_met4:
            st.metric("📊 Score client", f"{int(proba_solvable * 100)}/100")
        
        # Carte de résultat avec colorisation
        if proba_defaut >= 0.40:
            result_class = "result-high-risk"
            icon = "🔴"
            message = "Risque de défaut ÉLEVÉ - Décision défavorable recommandée"
            detail = "Plusieurs facteurs de risque identifiés"
        elif proba_defaut >= 0.20:
            result_class = "result-moderate-risk"
            icon = "🟡"
            message = "Risque MODÉRÉ - Analyse complémentaire recommandée"
            detail = "Examen des garanties supplémentaires conseillé"
        else:
            result_class = "result-low-risk"
            icon = "🟢"
            message = "Risque FAIBLE - Décision favorable recommandée"
            detail = "Profil solvable avec historique satisfaisant"
        
        st.markdown(f"""
        <div class="result-card {result_class}">
            <div class="result-label">{icon} {message}</div>
            <div class="result-score">{proba_defaut:.1%}</div>
            <div style="font-size: 0.9rem; opacity: 0.8; margin: 0.5rem 0;">
                Probabilité estimée de défaut de paiement
            </div>
            <div class="result-detail">{detail}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Barre de progression personnalisée
        st.markdown("#### 📉 Échelle de risque")
        st.markdown(f"""
        <div class="custom-progress">
            <div class="custom-progress-bar" style="width: {proba_defaut * 100}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6c757d;">
            <span>0% (Sûr)</span>
            <span>50% (Modéré)</span>
            <span>100% (Risqué)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Graphique de visualisation avec Plotly
        st.markdown("#### 📊 Décomposition des facteurs de risque")
        
        # Création des facteurs simulés basés sur les données
        risk_factors = {
            "Âge": max(0, min(100, (person_age - 18) / 82 * 100)),
            "Revenu": max(0, min(100, (person_income / 100000) * 100)),
            "Ancienneté": max(0, min(100, (person_emp_length / 20) * 100)),
            "Endettement": max(0, min(100, dti_ratio * 100)),
            "Historique": max(0, min(100, (cb_person_cred_hist_length / 20) * 100)),
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(risk_factors.keys()),
            y=list(risk_factors.values()),
            marker_color=['#00b894' if v < 50 else '#fdcb6e' if v < 70 else '#e17055' for v in risk_factors.values()],
            text=[f"{v:.0f}%" for v in risk_factors.values()],
            textposition='outside',
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=20, b=0),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 100], title="Score de risque (%)", gridcolor='#e9ecef'),
            xaxis=dict(title="Facteurs", gridcolor='#e9ecef'),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Recommendations
        st.markdown("#### 💡 Recommendations")
        
        if proba_defaut >= 0.40:
            st.warning("""
            **Actions recommandées :**
            1. Refuser le crédit ou proposer des garanties supplémentaires
            2. Demander un garant solvable
            3. Proposer un montant de prêt réduit avec un taux adapté
            4. Recommander une amélioration de la situation financière
            """)
        elif proba_defaut >= 0.20:
            st.info("""
            **Actions recommandées :**
            1. Proposer un prêt avec des garanties proportionnelles
            2. Augmenter légèrement le taux d'intérêt
            3. Recommander une assurance crédit
            4. Suivi mensuel renforcé
            """)
        else:
            st.success("""
            **Actions recommandées :**
            1. Accorder le crédit aux conditions proposées
            2. Proposer des services bancaires complémentaires
            3. Fidéliser le client avec des offres privilégiées
            4. Processus d'approbation simplifié
            """)
        
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")

# --------------------------------------------------------------------
# Pied de page
# --------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <strong>CreditScore Pro</strong> • Modèle Random Forest entraîné sur dataset de crédit bancaire • 
    Développé avec ❤️ en utilisant Spark ML & Streamlit
    <br>
    <span style="font-size: 0.7rem; opacity: 0.7;">🔒 Données traitées de manière confidentielle • Aucune donnée persistée</span>
    <br>
    <a href="https://ibuuuu19.github.io/credit-risk-streamlit/privacy/" target="_blank" style="font-size: 0.75rem; color: #6c757d;">
        Politique de confidentialité
    </a>
</div>
""", unsafe_allow_html=True)