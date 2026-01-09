# 🏥 Insurance Charges Prediction - Semaine 1

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B)
![Plotly](https://img.shields.io/badge/Plotly-6.5-brightgreen)
![Status](https://img.shields.io/badge/Semaine-1%20Complétée-success)

## 📋 Présentation du Projet
Ce projet est réalisé dans le cadre d'un cabinet d'analytique pour un assureur souhaitant anticiper ses charges médicales. L'objectif est de construire un modèle de régression linéaire fiable pour estimer les coûts à partir de variables démographiques et de santé.

## 🏗️ Structure du Dépôt
- 📁 **data/** : Contient le dataset Kaggle `insurance.csv` (1338 entrées).
- 📁 **notebooks/** : 
    - `01_EDA.ipynb` : Analyse exhaustive, dictionnaire des variables et rapport automatique.
    - `insurance_eda_report.html` : Rapport interactif généré par `ydata-profiling`.
- 📁 **presentations/** : Espace de stockage pour les présentations hebdomadaires.
- 📄 **app.py** : Dashboard interactif pour explorer les données en temps réel.
- 📄 **requirements.txt** : Liste des dépendances du projet.

## 💡 Insights Clés (Semaine 1)
L'Analyse Exploratoire des Données (EDA) a révélé plusieurs facteurs déterminants :
1.  **Tabagisme** : C'est le facteur le plus corrélé aux charges. Un fumeur coûte en moyenne 3 à 4 fois plus cher qu'un non-fumeur.
2.  **Synergie IMC x Fumeur** : Les charges explosent littéralement pour les fumeurs ayant un IMC > 30 (Obésité).
3.  **Âge** : On observe une progression linéaire des charges moyennes avec l'âge, répartie en trois "bandes" distinctes.
4.  **Distribution** : La variable cible `charges` est fortement asymétrique, ce qui justifiera une transformation logarithmique en Semaine 2.

## 🚀 Installation & Utilisation
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
- [ ] **Semaine 2** : Préparation des données (Encodage, Scaling) & Premier modèle baseline.
- [ ] **Semaine 3** : Optimisation des modèles & Mise en place d'un pipeline de production.

---
*Projet réalisé par l'équipe **Dev Data IA**.*
