# Insurance Prediction Project - Week 1

## Présentation du projet
Ce projet vise à prédire les charges d'assurance santé à partir de profils démographiques et de santé. Il s'inscrit dans une démarche de montée en compétences progressive sur la Data Science et l'IA (Semaine 1 : EDA).

## Variables du Dataset
Le dataset contient 1338 entrées avec les colonnes suivantes :
- **Age** : Âge du bénéficiaire principal.
- **Sex** : Genre de l'assuré (male/female).
- **BMI** (IMC) : Indice de masse corporelle, permettant d'évaluer la corpulence.
- **Children** : Nombre d'enfants ou de personnes à charge couverts par l'assurance.
- **Smoker** : Statut fumeur (yes/no).
- **Region** : Zone de résidence de l'assuré aux USA (northeast, northwest, southeast, southwest).
- **Charges** : Frais médicaux annuels facturés par l'assurance (**Variable cible**).

## Installation et Utilisation
1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Lancer le dashboard interactif :
```bash
streamlit run app.py
```

3. Explorer les analyses détaillées dans le notebook :
`notebooks/01_EDA.ipynb`
