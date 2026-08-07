# Déploiement du modèle de risque de crédit sur Streamlit

## Étape 1 — Sauvegarder le modèle depuis R

Dans ta session RStudio/sparklyr où `Valid_cr_model` existe déjà, exécute `save_model.R`.
Cela crée un dossier `credit_risk_pipeline_model/` (pas un simple fichier — une
arborescence avec du parquet et des métadonnées JSON).

Note bien la version de Spark affichée à la fin (`spark_version(spark_sc)`).

## Étape 2 — Transférer le modèle vers le projet Python

Copie/zippe tout le dossier `credit_risk_pipeline_model/` et place-le à la racine
de ce projet (`streamlit_credit_app/credit_risk_pipeline_model/`), au même niveau
que `streamlit_app.py`.

## Étape 3 — Compatibilité de version (déjà réglée, mais à savoir)

Ton R utilise `spark_connect(version = "3.0")`. J'ai donc réglé :

- `requirements.txt` → `pyspark==3.0.3` (dernier correctif de la branche 3.0)
- `packages.txt` → `openjdk-8-jdk` (Spark 3.0 est officiellement supporté sur
  Java 8 ; les JDK plus récents installés par défaut sur les images cloud
  peuvent causer des erreurs au démarrage de la JVM)
- `runtime.txt` → `python-3.8` (**important** : PySpark 3.0.x plante sur
  Python 3.10+ à cause d'un import de `collections.Iterable` qui a été retiré
  du module standard. Streamlit Cloud utilise `runtime.txt` pour choisir la
  version de Python — ne le supprime pas.)

Si tu changes un jour la version de Spark côté R (`spark_install(version = ...)`),
répercute le changement dans ces trois fichiers.

## Étape 4 — Vérifier les catégories des variables texte

Le pipeline utilise `ft_string_indexer` sur `person_home_ownership`, `loan_intent`
et `cb_person_default_on_file`. Ces indexeurs Spark lèvent une erreur si une
valeur inconnue leur est présentée (comportement par défaut `handleInvalid = "error"`).

Vérifie les catégories exactes dans ton notebook R :

```r
df %>% distinct(person_home_ownership) %>% collect()
df %>% distinct(loan_intent) %>% collect()
df %>% distinct(cb_person_default_on_file) %>% collect()
```

Ajuste les listes dans les `st.selectbox(...)` de `streamlit_app.py` si elles
diffèrent de ce que j'ai mis par défaut (valeurs standards du dataset Kaggle
"Credit Risk Dataset").

## Étape 5 — Tester en local

Utilise un environnement Python 3.8 (voir contrainte de compatibilité ci-dessus) :

```bash
python3.8 -m venv venv
source venv/bin/activate   # sous Windows : venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Si tu n'as pas Python 3.8 d'installé, `pyenv install 3.8.18` (Linux/Mac) ou
télécharge-le directement depuis python.org.

Le premier chargement sera lent (démarrage de la JVM Spark + lecture du modèle).
C'est normal.

## Étape 6 — Déployer

### Option A : Streamlit Community Cloud
- Pousse ce dossier (avec le modèle inclus) sur un repo GitHub.
- `packages.txt` installera Java automatiquement.
- ⚠️ Limite importante : le tier gratuit (~1 Go de RAM) est souvent **insuffisant**
  pour faire tourner une JVM Spark + charger un modèle. Teste, et si ça plante
  au chargement (OOM), passe à l'option B.

### Option B : Conteneur Docker sur une VM (recommandé pour PySpark)
Plus fiable pour ce cas d'usage. Exemple de `Dockerfile` minimal :

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y default-jdk && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Déployable ensuite sur Render, Railway, une VM EC2/GCE, etc., avec au moins
2 Go de RAM alloués au conteneur.

## Structure finale attendue

```
streamlit_credit_app/
├── streamlit_app.py
├── requirements.txt
├── packages.txt
├── runtime.txt
├── save_model.R
├── README.md
└── credit_risk_pipeline_model/   ← ajouté après l'étape 2
    ├── metadata/
    └── stages/
```