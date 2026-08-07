# ===================================================================
# SAUVEGARDE DU MODELE POUR DEPLOIEMENT STREAMLIT
# A executer dans ta session R/sparklyr, APRES avoir entraine Valid_cr_model
# ===================================================================

# Le meilleur modele trouve par la validation croisee
best_model <- Valid_cr_model$best_model

# Si la ligne au-dessus renvoie une erreur selon ta version de sparklyr,
# essaie plutot :
# best_model <- ml_stage(Valid_cr_model, "bestModel")

# Chemin de sauvegarde local (dossier, pas un fichier unique)
chemin_modele <- "credit_risk_pipeline_model"

ml_save(
  best_model,
  chemin_modele,
  overwrite = TRUE
)

cat("Modele sauvegarde dans :", normalizePath(chemin_modele), "\n")
cat("Compresse ce dossier entier (zip) et transfere-le vers ton projet Python.\n")

# IMPORTANT : note la version de Spark utilisee ici, tu devras utiliser
# une version compatible (idealement identique) cote PySpark.
cat("Version Spark utilisee :", spark_version(spark_sc), "\n")

# Pour zipper directement depuis R :
# zip::zip("credit_risk_pipeline_model.zip", chemin_modele)