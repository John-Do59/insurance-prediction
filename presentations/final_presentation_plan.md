# Plan de Présentation Finale : Prédiction des Charges d'Assurance
**Objectif :** Présenter les résultats de la Semaine 3 (Optimisation et Pipeline) et clore le projet.

---

## Slide 1 : Titre & Introduction
*   **Visuel :** Logo projet + Titre accrocheur ("Optimisation Prédictive : De la Baseline au Modèle Etat-de-l'Art").
*   **Contenu :** 
    *   Nom du projet.
    *   Auteur.
    *   Contexte : Automatisation de l'estimation des coûts de santé.

## Slide 2 : Le Défi Business
*   **Visuel :** Illustration simplifiée du "Gap" financier.
*   **Contenu :**
    *   Problème : Les erreurs d'estimation manuelles causent des pertes de marge.
    *   Objectif technique : Atteindre un R² > 0.9232 (Cible initiale).
    *   Objectif métier : Réduire l'erreur moyenne sous les $2037.

## Slide 3 : Exploration des Données (EDA Flash)
*   **Visuel :** Graphique de corrélation (heatmap) ou distribution des charges.
*   **Contenu :**
    *   Dataset : 1338 individus.
    *   Variables clés : Age, BMI, Smoker.
    *   Insight critique : Les coûts ne sont pas linéaires ; les interactions entre le tabagisme et l'obésité dominent le profil de risque.

## Slide 4 : Architecture du Pipeline ML
*   **Visuel :** Schéma de flux (Data -> Preprocessing -> Model -> Export).
*   **Contenu :**
    *   **Prétraitement** : Scaling numérique + One-Hot Encoding catégoriel via `ColumnTransformer`.
    *   **Pipeline** : Encapsulation robuste garantissant qu'aucune donnée d'entraînement ne "fuite" dans le test set.
    *   **Export** : Modèle sérialisé via `joblib` pour intégration immédiate.

## Slide 5 : Le "Game Changer" : Feature Engineering Stratégique
*   **Visuel :** Histogramme comparatif "Sans vs Avec" interactions.
*   **Contenu :**
    *   La découverte : L'impact du tabagisme n'est pas seulement additif, il est multiplicatif avec l'augmentation du BMI.
    *   Variables créées : `bmi * smoker`, `age²`, `is_obese_smoker`.
    *   Résultat : Cette étape seule a propulsé le modèle de 0.81 à 0.94.

## Slide 6 : Saut de Performance : Baseline vs Optimal
*   **Visuel :** Tableau comparatif ou bar chart.
*   **Vrais Chiffres :**
    *   **Baseline (Linéaire simple)** : R² = **0.8069** | MAE = **$4,177**.
    *   **Modèle Optimal (Strategic Interactions)** : R² = **0.9404** | MAE = **$1,980**.
    *   **Gain** : Amélioration massive de **+16.4%** de précision.

## Slide 7 : Synthèse Finale des Performances
*   **Visuel :** Tableau des métriques (OLS, Ridge, Lasso).
*   **Vrais Chiffres :**
    *   **Configuration Gagnante** : Ridge (Alpha=1.0) avec interactions stratégiques.
    *   **R² Final** : **0.9404** (Cible 0.92 dépassée).
    *   **MAE Final** : **$1,980** (Cible $2,037 atteinte).
    *   **RMSE Final** : **$3,183** (Cible $3,326 atteinte).

## Slide 8 : Fiabilité & Analyse des Résidus
*   **Visuel :** Graphique "Distribution des Résidus" (cloche de Gauss).
*   **Contenu :**
    *   Le modèle se trompe de manière aléatoire (bruit blanc), signe qu'il ne reste pas de structure non capturée.
    *   **Biais-Variance** : La stabilité entre Train et Test confirme l'absence de surapprentissage (Overfitting).

## Slide 9 : Atteinte du "Data Ceiling" (Plafond de Verre)
*   **Visuel :** Graphique de comparaison Bench (Random Forest) vs Linear.
*   **Contenu :**
    *   Performance RF : 0.90 (Incapacité des modèles complexes à faire mieux).
    *   Conclusion technique : Nous avons extrait **100% du signal utile** disponible dans les données actuelles.
    *   Les 6% restants sont du bruit incompressible (imprévisibilité individuelle).

## Slide 10 : Démo Produit - Simulateur Streamlit
*   **Visuel :** Screenshot de l'app ou GIF.
*   **Contenu :**
    *   Interface interactive pour les agents d'assurance.
    *   Saisie en temps réel : Age, BMI, Smoker.
    *   Affichage instantané du tarif "Fair Price" basé sur l'IA.

## Slide 11 : Interprétabilité Métier
*   **Visuel :** Graphique des coefficients (Feature Importance).
*   **Contenu :**
    *   Quel est le coût réel de fumer ?
    *   Comment l'âge impacte-t-il le risque ?
    *   L'algorithme n'est plus une "Boîte Noire", chaque décision est justifiable.

## Slide 12 : Conclusion & Prochaines Étapes
*   **Contenu :**
    *   **Acquis** : Un modèle robuste, conforme et prêt à l'emploi.
    *   **Roadmap** :
        1. Intégration API dans le SI.
        2. Collecte de nouvelles features (pression sanguine, cholestérol) pour briser le plafond des 0.94.
    *   **Mot de la fin** : "La précision mathématique au service de l'équité tarifaire."
