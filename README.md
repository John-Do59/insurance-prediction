# Insurance Charges Prediction - Semaine 2

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-orange)
![Status](https://img.shields.io/badge/Semaine-2%20Complétée-success)

## 🎯 Présentation du Projet
Ce projet est réalisé dans le cadre d'un cabinet d'analytique pour un assureur souhaitant anticiper ses charges médicales. L'objectif est de construire un modèle de régression linéaire fiable pour estimer les coûts à partir de variables démographiques et de santé.

## 🚀 Performance du Modèle (Fin Semaine 2)
Nous avons atteint un niveau de performance exceptionnel dépassant les objectifs fixés :
- **R² Score : 0.9390** (Objectif : 0.9324)
- **MAE : $1,989.76**
- **RMSE : $3,183.52**

## 📂 Structure du Dépôt
- **data/** : Contient le dataset Kaggle `insurance.csv`.
- **notebooks/** : 
    - `01_EDA.ipynb` : Analyse exhaustive et insights métier.
    - `02_preprocessing_baseline_model.ipynb` : Data prep + Baseline + **Modèle Optimal**.
- **app.py** : Dashboard interactif et outil de prédiction.
- **SEMAINE_2_RECAP.md** : Rapport détaillé de la méthodologie et des tests de la semaine 2.
- **requirements.txt** : Liste des dépendances.

## 💡 Insights & Méthodologie
1. **Feature Engineering** : L'ajout d'interactions complexes (ex: `bmi * smoker * age`) et de termes polynomiales a été la clé du succès.
2. **Régularisation** : Utilisation de la régression **Ridge (alpha=0.5)** pour stabiliser le modèle face au grand nombre de features (27).
3. **Reproductibilité** : Utilisation d'une seed fixe (1282) garantissant la stabilité des résultats.

## 🛠️ Installation & Utilisation
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

## 🗺️ Roadmap
- [x] **Semaine 1** : Analyse Exploratoire (EDA) & Dashboard Interactif.
- [x] **Semaine 2** : Data preparation & Modèle Optimal (R² : 0.9390).
- [ ] **Semaine 3** : Mise en place des Pipelines Sklearn & Industrialisation.

---
*Projet réalisé par l'équipe **Dev Data IA**.*
