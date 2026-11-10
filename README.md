# Analyse de l'effet du Tolvaptan et prédiction du temps avant l'insuffisance rénale terminale chez les patients atteints de PKRAD

## Présentation

Ce dépôt regroupe les notebooks développés dans le cadre du projet de modélisation de la progression de la PKRAD au sein du service de Néphrologie du CHU de Brest.

Le projet poursuit deux objectifs complémentaires :

* **analyser l'impact du traitement par Tolvaptan** sur la progression de la polykystose rénale autosomique dominante (PKRAD) ;
* **développer et évaluer des modèles d'analyse de survie** afin de prédire le temps restant avant l'insuffisance rénale terminale (IRT) à partir de données cliniques, génétiques et longitudinales.

Les données utilisées proviennent de la cohorte **Genkyst** et ne sont pas distribuées.

---

# Pipeline du projet

```text
                           Cohorte Genkyst
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
 Analyse de l'effet du Tolvaptan                Prédiction de l'IRT
      (Jinarc Analysis)                                 │
                                                        │
                            ┌───────────────────────────┴───────────────────────────┐
                            │                                                       │
                            ▼                                                       ▼
                 Une seule mesure de DFGe                             Données longitudinales
                            │                                                       │
                            ▼                                                       ▼
              Prepare Static Dataset                           Build Longitudinal Cohort
                            │                                                       │
                            ▼                                                       ▼
            Parametric Survival Models                        Build Dynamic Datasets
                            │                                                       │
                            ▼                                                       ▼
            Baseline Survival ML Models     XGBoost Survival AFT ─ Survival Stacking
                                                                            │
                                                                            ▼
                                                              Neural ODE AFT ─ Neural CDE AFT
```

---

# Organisation du dépôt

## 1. Analyse de l'effet du Tolvaptan 

| Notebook                | Description                                                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `jinarc_analysis.ipynb` | Prépare les données longitudinales et analyse l'évolution du DFGe avant et après l'initiation du Tolvaptan à l'aide d'un modèle linéaire mixte segmenté. |

---

## 2. Analyse de la progression de la PKRAD

| Notebook            | Description                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ADPKD_OM.ipynb` | Reproduit l'approche de l'ADPKD Outcome Model sur la cohorte Genkyst afin d'analyser la progression du TKV et du DFG à l'aide de modèles linéaires mixtes. |

## 3. Prédiction de l'IRT à partir d'une mesure de référence du DFGe

| Notebook                            | Description                                                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `prepare_static_dataset.ipynb`      | Prépare le jeu de données statique et génère les jeux d'entraînement, de validation interne et de validation externe.                     |
| `parametric_survival_models.ipynb`  | Développe et compare les modèles Weibull AFT, Log-normal AFT et Log-logistic AFT.                                                         |
| `baseline_survival_ml_models.ipynb` | Développe les modèles Random Survival Forest et XGBoost Survival AFT avec optimisation des hyperparamètres et validation interne/externe. |

---

## 4. Prédiction de l'IRT à partir de données longitudinales

| Notebook                                     | Description                                                                                                     |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `build_longitudinal_cohort.ipynb`            | Construit la cohorte longitudinale et génère les cohortes de développement et de validation externe.            |
| `build_dynamic_datasets.ipynb`               | Construit les jeux de données dynamiques (snapshots longitudinaux) utilisés par les modèles de prédiction.      |
| `xgboost_survival_aft.ipynb`                 | Développe un modèle XGBoost Survival AFT, optimise les hyperparamètres avec Optuna et réalise une analyse SHAP. |
| `survival_stacking_prediction.ipynb` | Implémente une approche Survival Stacking basée sur un classifieur XGBoost.                                     |

---

## 5. Modèles neuronaux

| Notebook               | Description                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| `neural_ode_survival_final.ipynb` | Développe un modèle GRU et modèle Neural ODE adapté à l'analyse de survie pour exploiter les trajectoires longitudinales. |
| `neural_cde_survival.ipynb` | Développe un modèle Neural CDE avec une tête AFT log-normale pour prédire le temps restant avant l'IRT.     |

---

# Installation

Installer les dépendances des modèles classiques :

```bash
pip install -r requirements.txt
```

Installer les dépendances des modèles neuronaux :

```bash
pip install -r requirements-neural.txt
```

---

#

Auteur

Orlana Hashazinka
Master 2 Mathématiques Appliquées et Statistiques
Université Rennes 2

Encadrement scientifique :

* Dr Katleen BLANCHET

 
