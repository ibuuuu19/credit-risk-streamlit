"""
App Streamlit pour le modele de risque de credit (pipeline Spark ML/sparklyr)
================================================================================
Charge un PipelineModel Spark entraine en R et sauvegarde avec ml_save(),
et l'utilise pour predire le risque de defaut d'un client via une interface web.
"""

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, StringType
)
from pyspark.ml import PipelineModel

# --------------------------------------------------------------------
# Config page
# --------------------------------------------------------------------
st.set_page_config(page_title="Score de risque de credit", page_icon="💳", layout="centered")

MODEL_PATH = "credit_risk_pipeline_model"  # dossier copie depuis R (ml_save)

# --------------------------------------------------------------------
# Session Spark + chargement du modele (mis en cache pour ne charger qu'une fois)
# --------------------------------------------------------------------
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
    st.error(
        "Impossible de charger le modele. Verifie que le dossier "
        f"'{MODEL_PATH}' existe bien a la racine de l'app et provient "
        "d'un ml_save() compatible avec cette version de Spark.\n\n"
        f"Detail : {e}"
    )
    st.stop()

# --------------------------------------------------------------------
# Schema attendu par le pipeline (colonnes brutes utilisees en entree
# de ft_imputer / ft_string_indexer / ft_vector_assembler)
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
# Interface utilisateur
# --------------------------------------------------------------------
st.title("💳 Score de risque de credit")
st.caption("Pipeline Random Forest (Spark ML) entraine en R / sparklyr")

st.markdown("### Informations sur le client")

col1, col2 = st.columns(2)

with col1:
    person_age = st.number_input("Age", min_value=18, max_value=100, value=30)
    person_income = st.number_input("Revenu annuel ($)", min_value=0, value=50000, step=1000)
    person_emp_length = st.number_input("Anciennete emploi (annees)", min_value=0.0, value=5.0, step=0.5)
    person_home_ownership = st.selectbox(
        "Statut logement",
        ["RENT", "MORTGAGE", "OWN", "OTHER"],
    )
    loan_amnt = st.number_input("Montant du pret ($)", min_value=0, value=10000, step=500)

with col2:
    loan_intent = st.selectbox(
        "Objet du pret",
        ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"],
    )
    loan_int_rate = st.number_input("Taux d'interet (%)", min_value=0.0, value=11.0, step=0.1)
    cb_person_default_on_file = st.selectbox("Defaut de paiement historique ?", ["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Anciennete historique credit (annees)", min_value=0, value=5)

st.warning(
    "⚠️ Les valeurs des listes deroulantes ci-dessus (statut logement, objet du pret, "
    "defaut historique) doivent correspondre EXACTEMENT aux categories vues pendant "
    "l'entrainement (memes libelles, meme casse). Verifie-les avec "
    "`df %>% distinct(person_home_ownership)` etc. dans ton notebook R et ajuste "
    "les listes dans ce fichier si besoin."
)

# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------
if st.button("Calculer le score", type="primary"):
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
        prediction = model.transform(df_input)
        resultat = prediction.select("prediction", "probability").collect()[0]

        classe_predite = int(resultat["prediction"])
        proba = resultat["probability"].toArray()
        proba_defaut = float(proba[1])  # probabilite de la classe 1 (defaut)

        st.markdown("---")
        st.markdown("### Resultat")

        if classe_predite == 1:
            st.error(f"⚠️ Risque de defaut eleve — probabilite estimee : {proba_defaut:.1%}")
        else:
            st.success(f"✅ Risque de defaut faible — probabilite estimee : {proba_defaut:.1%}")

        st.progress(proba_defaut)

    except Exception as e:
        st.error(f"Erreur lors de la prediction : {e}")

st.markdown("---")
st.caption(
    "Modele : Random Forest optimise par validation croisee (5 folds), "
    "AUC ~ 0.90 sur le jeu de test."
)