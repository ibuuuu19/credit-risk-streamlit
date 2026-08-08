# Déploiement du modèle de risque de crédit — CreditScore Pro

## Étape 1 — Sauvegarder le modèle depuis R

Dans ta session RStudio/sparklyr où `Valid_cr_model` existe déjà, exécute `save_model.R`.
Cela crée un dossier `credit_risk_pipeline_model/` (pas un simple fichier — une
arborescence avec du parquet et des métadonnées JSON, sous-dossiers `metadata/` et `stages/`).

Note bien la version de Spark affichée à la fin (`spark_version(spark_sc)`).
Référence actuelle : **Spark 3.0.3 / Hadoop 3.2**.

> ⚠️ **Sous Windows**, `ml_save()` peut échouer avec `UnsatisfiedLinkError` (NativeIO)
> si `winutils.exe` et `hadoop.dll` ne sont pas installés et ne correspondent pas
> exactement à la version Hadoop réelle (vérifiable via `sparklyr::spark_installed_versions()`).
> Voir l'historique du projet pour la procédure complète de résolution si ça se reproduit.

## Étape 2 — Transférer le modèle vers le projet

Copie tout le dossier `credit_risk_pipeline_model/` à la racine de ce projet
(`streamlit_credit_app/credit_risk_pipeline_model/`), au même niveau que `streamlit_app.py`.

## Étape 3 — Versions retenues (état actuel, validé en production)

Deux environnements distincts sont utilisés volontairement, et c'est normal — le format
de sauvegarde des pipelines Spark ML reste compatible d'une version à l'autre :

| | Local (développement, WSL/Ubuntu) | Cloud (Streamlit Community Cloud) |
|---|---|---|
| Python | 3.9.18 (via `pyenv`) | 3.14 (imposé par la plateforme, non configurable) |
| PySpark | 3.5.0 | **4.1.0** |
| Java | — (déjà en place localement) | OpenJDK **17** |

**Fichiers de configuration cloud (`requirements.txt` / `packages.txt`) :**

```
# requirements.txt
streamlit>=1.32
pyspark==4.1.0
plotly
pandas
```

```
# packages.txt
openjdk-17-jdk
```

> ℹ️ **`runtime.txt` a été supprimé du projet.** Streamlit Community Cloud ignore ce
> fichier (bug de plateforme confirmé, non résolu à ce jour) et impose sa propre version
> de Python — actuellement 3.14. Ne pas le recréer avec une ancienne version, ça n'aurait
> aucun effet et pourrait induire en erreur en cas de correction future de ce bug.
>
> C'est pour cette raison que **PySpark 4.1.0** est utilisé côté cloud : c'est la première
> version à supporter officiellement Python 3.14 (Spark Classic). Les versions antérieures
> (3.0.3, 3.5.0) provoquent des erreurs `RecursionError` / `ModuleNotFoundError: distutils`
> sur cette version de Python.
>
> Spark 4.x exige Java 17 minimum (Java 8/11 ne suffisent plus), d'où `openjdk-17-jdk`.

**Si tu changes la version de Spark côté R** (`spark_install(version = ...)`), il n'est
**pas nécessaire** de répercuter la même version côté cloud — seule la compatibilité du
*format de persistance* du modèle compte, et elle est stable sur toute la branche Spark 3.x/4.x.

## Étape 4 — Vérifier les catégories des variables texte

Le pipeline utilise `ft_string_indexer` sur `person_home_ownership`, `loan_intent`
et `cb_person_default_on_file`. Ces indexeurs Spark lèvent une erreur si une valeur
inconnue leur est présentée.

```r
df %>% distinct(person_home_ownership) %>% collect()
df %>% distinct(loan_intent) %>% collect()
df %>% distinct(cb_person_default_on_file) %>% collect()
```

✅ Déjà vérifié et confirmé identique aux listes codées dans `streamlit_app.py`
(RENT/MORTGAGE/OWN/OTHER, les 6 intents, Y/N).

## Étape 5 — Tester en local (WSL/Ubuntu recommandé)

Windows natif nécessite `winutils.exe`/`hadoop.dll` pour PySpark en plus de R — autant
éviter cette double gestion en travaillant sous WSL2/Ubuntu, où ces bibliothèques natives
ne sont pas nécessaires.

```bash
# Installation Python 3.9 via pyenv (une fois)
pyenv install 3.9.18
pyenv local 3.9.18

# Environnement virtuel du projet
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pyspark==3.5.0 streamlit plotly pandas

# Lancement
streamlit run streamlit_app.py
```

Le premier chargement est lent (démarrage JVM + lecture du modèle) — normal.

> Le script `fix_and_run.sh` automatise ces étapes pour un environnement local. Il n'est
> **pas destiné à être poussé sur GitHub** ni utilisé comme référence pour `requirements.txt`
> cloud — ses pins de version (Python 3.9, pyspark 3.5.0...) sont spécifiques au poste local.

## Étape 6 — Déployer sur Streamlit Community Cloud

1. Pousser le dépôt sur GitHub (dossier du modèle inclus).
2. Sur [share.streamlit.io](https://share.streamlit.io), New app → sélectionner le repo,
   branche `main`, fichier `streamlit_app.py` → Deploy.
3. `packages.txt` installe Java 17 automatiquement ; `requirements.txt` installe
   PySpark 4.1.0, Streamlit et Plotly.

✅ **Validé en production** — le tier gratuit (~1 Go de RAM) est suffisant avec cette
configuration de versions. Aucun souci mémoire observé.

En cas de nouvelle erreur après une future mise à jour de dépendance : toujours vérifier
la compatibilité avec **Python 3.14** avant de pousser (numpy/pandas/matplotlib anciens
n'ont souvent pas de wheel pour cette version — préférer ne pas épingler de version stricte
pour ces paquets, laisser `pip` résoudre automatiquement).

## Politique de confidentialité (requise pour Play Store)

Hébergée via GitHub Pages : **https://ibuuuu19.github.io/credit-risk-streamlit/privacy/**
(fichier source : `privacy/index.html`, publié depuis la racine du repo, branche `main`).
Un lien vers cette page est inclus dans le pied de page de l'application.

## Publication sur le Play Store

Voir la procédure complète (génération du package Android via PWABuilder, compte
développeur Google, soumission) — résumé :
1. Icône carrée 512×512px préparée.
2. Package `.aab` généré via [pwabuilder.com](https://www.pwabuilder.com) à partir de
   l'URL Streamlit Cloud.
3. Upload sur Google Play Console (compte développeur, 25 $ une fois) avec le lien de
   politique de confidentialité ci-dessus.

## Structure actuelle du projet

```
streamlit_credit_app/
├── streamlit_app.py
├── requirements.txt          # streamlit, pyspark==4.1.0, plotly, pandas
├── packages.txt               # openjdk-17-jdk
├── save_model.R
├── README.md
├── privacy/
│   └── index.html              # politique de confidentialité (GitHub Pages)
├── credit_risk_pipeline_model/ # modèle Spark (metadata/, stages/)
├── fix_and_run.sh              # LOCAL UNIQUEMENT — ne pas répercuter sur requirements.txt cloud
├── .gitignore                  # exclut venv/, __pycache__, .streamlit/secrets.toml
└── venv/                       # environnement local, jamais commité
```