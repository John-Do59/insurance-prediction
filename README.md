# Insurance Charges Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.0-blue)
![Status](https://img.shields.io/badge/Semaine-3%20Complétée-success)

## Présentation du Projet
Ce projet est réalisé dans le cadre d'un cabinet d'analytique pour un assureur souhaitant anticiper ses charges médicales. L'objectif est de construire un modèle de régression linéaire fiable pour estimer les coûts à partir de variables démographiques et de santé.

Le projet couvre l'ensemble du cycle de vie d'un modèle de machine learning, de l'exploration des données à la mise en production avec un pipeline industrialisé.

## Performance du Modèle Final (Semaine 3)
Modèle final optimisé avec Pipeline scikit-learn complet et GridSearchCV :

### Métriques sur le jeu de test
- **R² Score : 0.8857** (Le modèle explique ~88.6% de la variance)
- **MAE : $2,884.67** (Erreur moyenne absolue)
- **CV R² (5-fold) : 0.8250** (Score moyen en validation croisée, écart-type : 0.0209)

### Modèle sélectionné
- **Type : Ridge Regression**
- **Hyperparamètre optimal : alpha = 0.1**
- **Pipeline complet** : Feature Engineering → Preprocessing → Régression

## Structure du Dépôt
```
insurance-prediction/
├── data/
│   └── insurance.csv              # Dataset Kaggle
├── notebooks/
│   ├── 01_EDA.ipynb               # Analyse exploratoire complète
│   ├── 02_preprocessing_baseline_model.ipynb  # Préparation données + Baseline
│   └── 03_model_optimization_pipeline.ipynb   # Pipeline + Optimisation + MLflow
├── models/
│   ├── insurance_model_prod.joblib    # Modèle final sauvegardé
│   └── shap_explainer.joblib          # Explainer SHAP pour interprétabilité
├── src/                              # Code source de l'application Streamlit
│   ├── tabs/                         # Onglets du dashboard
│   ├── components/                   # Composants réutilisables
│   └── config.py                     # Configuration
├── app.py                            # Dashboard Streamlit principal
└── requirements.txt                  # Dépendances Python
```

## Fonctionnalités Principales

### Pipeline Scikit-Learn Industrialisé
- **ColumnTransformer** : Préprocessing automatique des variables numériques et catégorielles
- **Feature Engineering** : Interactions (`smoker * bmi`) et termes polynomiales (`age²`)
- **Pipeline complet** : Automatisation du flux de données de bout en bout

### Validation Robuste
- **Cross-Validation (5-fold)** : Évaluation fiable avec `cross_val_score`
- **GridSearchCV** : Optimisation des hyperparamètres pour Ridge, Lasso et ElasticNet
- **Split train/test** : Séparation rigoureuse (80/20) avec `random_state=42`

### Tracking MLflow
- Expérience localisée avec tracking file-based
- Logging des paramètres (modèle, alpha, nombre de folds)
- Logging des métriques (R² CV, R² test, MAE)
- Sauvegarde du pipeline complet comme artefact

### Interprétabilité Avancée
- **SHAP** : Explication locale et globale des prédictions
- **Permutation Importance** : Mesure de l'impact de chaque variable
- Visualisations intégrées dans l'application Streamlit

### Dashboard Streamlit V2
- **Onglet Prédiction** : Interface interactive pour estimer les charges
- **Onglet "Inside the Model"** : Visualisation des coefficients, SHAP et feature importance
- **9 onglets d'analyse** : EDA complète, corrélations, analyses expertes

## Insights & Méthodologie
1. **Feature Engineering** : L'ajout d'interactions (`smoker * bmi`) et de termes polynomiales (`age²`) améliore significativement la performance.
2. **Régularisation** : Utilisation de **Ridge (alpha=0.1)** pour stabiliser le modèle et éviter le sur-apprentissage.
3. **Pipeline Automatisé** : Industrialisation complète avec scikit-learn pour faciliter la maintenance et le déploiement.
4. **Reproductibilité** : Utilisation de `random_state=42` garantissant la stabilité des résultats.

## Installation & Utilisation
1. **Environnement virtuel** :
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

2. **Dépendances** :
```bash
pip install -r requirements.txt
```

3. **Lancement du Dashboard** :
```bash
streamlit run app.py
```

4. **Visualisation des Expériences MLflow** (optionnel) :
```bash
# Après avoir exécuté le notebook 03
mlflow ui --backend-store-uri file:./mlruns
# Puis ouvrir http://localhost:5000 dans votre navigateur
```

## Roadmap

###  Semaine 1 : Exploration des Données (EDA)
- Analyse exploratoire exhaustive
- Dashboard Streamlit interactif
- Identification des insights métier clés

###  Semaine 2 : Préparation & Modèle Baseline
- Feature Engineering (interactions, termes polynomiales)
- Modèles baseline : LinearRegression, Ridge, Lasso
- Performance : R² = 0.8856 (LinearRegression)

###  Semaine 3 : Pipeline & Optimisation (COMPLÉTÉE)
- Pipeline scikit-learn complet avec ColumnTransformer
- Validation croisée (5-fold)
-  GridSearchCV pour optimisation des hyperparamètres
-  Comparaison Ridge/Lasso/ElasticNet
-  Sélection du modèle optimal (Ridge alpha=0.1, R² = 0.8857)
-  MLflow pour tracking des expériences
-  Analyse des résidus et vérification des hypothèses
-  Interprétabilité SHAP et Permutation Importance
-  Dashboard Streamlit V2 (prédiction + explication)

---
*Projet réalisé par l'équipe **Dev Data IA**.*
