# Plan de Présentation : Projet Prediction Assurance (Semaines 1-3)

Ce document détaille le plan pour une présentation de 10 à 12 slides, destinée à une audience métier/technique.

## Structure Globale
*   **Slides 1-2** : Rappel du contexte et découvertes initiales (Semaines 1 & 2).
*   **Slides 3-10** : Cœur du sujet - Industrialisation et Optimisation (Semaine 3).
*   **Slides 11-12** : Résultats finaux, Démo et Conclusion.

---

## Détail des Slides

### Slide 1 : Titre et Contexte
*   **Titre** : Prédictions des Charges d'Assurance : De l'Analyse à l'Industrialisation.
*   **Sous-titre** : Synthèse des semaines 1 à 3.
*   **Contexte** : Un assureur souhaite prédire les frais médicaux pour ajuster sa tarification.
*   **Objectif du projet** : Créer un modèle prédictif fiable, automatisé et explicable.

### Slide 2 : Exploration des Facteurs de Coûts (Semaine 1 & 2)
*   **Message clé** : "Le tabac et l'obésité sont les facteurs de coûts majeurs."
*   **Chiffres Clés (1338 observations)** :
    *   **Charges Moyennes** : 13 270 $
    *   **Âge Médian** : 39 ans
    *   **IMC Moyen** : 30.7
    *   **% Fumeurs** : 20.5%
    *   (Données propres : pas de valeurs manquantes critiques).
*   **Insights Visuels (à inclure)** :
    *   Graphique : Scatter plot `BMI` vs `Charges` avec couleur `Smoker`. (Montre clairement deux groupes distincts).
    *   Constat : Les fumeurs obèses coûtent exponentiellement plus cher. L'âge augmente les coûts de façon non-linéaire.
*   **La réponse de la Semaine 2** : Création de variables expertes (`smoker_bmi`, `age_squared`) pour capturer ces phénomènes.

### Slide 3 : Semaine 3 - Le besoin de Robustesse
*   **Problème** : Le modèle de la semaine 2 (Régression Linéaire simple) était prometteur ($R^2 \approx 0.88$) mais fragile pour la mise en production (risque de sur-apprentissage, gestion manuelle des données).
*   **Objectif Semaine 3** : "Industrialiser" la démarche.
*   **Transition** : Passer du "bricolage" au "Pipeline Automatisé".

### Slide 4 : La Solution Technique - Le Pipeline
*   **Concept** : Un "tuyau" unique qui traite la donnée brute jusqu'à la prédiction.
*   **Composants du Pipeline** :
    1.  **Feature Engineering** : Calcul auto de l'interaction `Fumeur * BMI`.
    2.  **Nettoyage** : Imputation (remplir les trous) + Scaling (mettre à l'échelle).
    3.  **Encodage** : Transformer "Sud-Ouest" en chiffres.
    4.  **Modèle** : L'algorithme de prédiction.
*   **Diagramme suggéré** : Un schéma linéaire montrant : *Données Brutes -> [ Pipeline ] -> Prédiction*.

### Slide 5 : Stratégie de Validation - Éviter le hasard
*   **Méthode** : Validation Croisée à 5 plis (5-Fold Cross-Validation).
*   **Pourquoi ?** : Pour être sûr que le score du modèle n'est pas "un coup de chance" sur une partie facile des données.
*   **Visuel** : Schéma montrant le découpage du dataset en 5 parties tournantes.
*   **Résultat** : Score $R^2$ moyen de **0.82** (+/- 0.02). -> Le modèle est stable.

### Slide 6 : Optimisation des Hyperparamètres (Le "Tuning")
*   **Méthode** : GridSearch (Recherche sur grille). On a testé des centaines de configurations pour 3 algorithmes.
*   **Concurrents** :
    *   **Ridge** (Régularisation L2).
    *   **Lasso** (Régularisation L1).
    *   **ElasticNet** (Mixte).
*   **Pourquoi régulariser ?** : Pour empêcher le modèle d'apprendre "par cœur" le bruit des données, surtout avec nos nouvelles variables corrélées (`age` et `age_squared`).

### Slide 7 : Sélection du Champion
*   **Résultats du Match** :
    *   Ridge : $R^2 = 0.8250$
    *   Lasso : $R^2 = 0.8250$
    *   ElasticNet : $R^2 = 0.8250$
*   **Décision** : Choix du modèle **Ridge**.
*   **Justification Métier** : Performance identique, mais Ridge est mathématiquement plus stable pour gérer les variables corrélées que nous avons créées. Il garde toutes les variables actives, ce qui est utile pour l'explicabilité.

### Slide 8 : Performance Finale (Sur jeu de test)
*   **Verdict (Test Set)** :
    *   **$R^2$ : 0.8857** (Excellent pouvoir explicatif).
    *   **MAE : 2884 $** (Erreur moyenne par an).
*   **Interprétation** : Le modèle est très performant pour un modèle linéaire. L'optimisation Ridge n'a pas "boosté" le score brut par rapport à la baseline (0.8856), mais elle a apporté la **sécurité** nécessaire pour le déploiement.

### Slide 9 : Ouvrir la "Boîte Noire" - SHAP Global
*   **Outil** : SHAP (SHapley Additive exPlanations).
*   **Question** : "Quels sont les leviers principaux de la tarification ?"
*   **Graphique à inclure** : Bar plot des importances SHAP.
*   **Insights** :
    1.  `smoker` (statut fumeur) est le facteur #1 écrasant.
    2.  `age` et `bmi` arrivent ensuite.
    3.  `sex` et `region` ont un impact négligeable.

### Slide 10 : Explicabilité Locale - Cas Client
*   **Question** : "Pourquoi M. Dupont paye-t-il 1000$ de plus ?"
*   **Visuel** : Le graphique SHAP individuel (Waterfall ou Bar plot généré dans l'app).
*   **Exemple** :
    *   Base : 13 000 $
    *   +5 000 $ car Fumeur.
    *   -2 000 $ car Jeune.
    *   +500 $ car IMC élevé.
    *   = Prédiction Finale.
*   **Valeur** : Permet aux agents de justifier le tarif auprès du client.

### Slide 11 : Déploiement dans l'Application
*   **Réalisation** : Intégration du Pipeline dans une interface Streamlit.
*   **Fonctionnalités** :
    *   Saisie profil client.
    *   Prédiction instantanée.
    *   Explication visuelle (SHAP).
    *   Vision globale du portefeuille.
*   **Screenshot** : Une capture d'écran de l'onglet "Prédiction" de l'application.

### Slide 12 : Conclusion et Prochaines Étapes
*   **Bilan** : Objectif atteint. Modèle performant ($R^2 \approx 0.89$), robuste (Ridge + Pipeline) et explicable.
*   **Limites** : Le modèle linéaire a du mal avec les cas très complexes (ex: non-fumeurs avec charges très élevées "inexpliquées").
*   **Next Steps (Semaine 4+)** :
    *   Tester des modèles non-linéaires (Random Forest, Gradient Boosting) pour capturer les 12% de variance restants.
    *   Déployer l'API pour une utilisation temps réel.
    *   Mettre en place un monitoring de la performance.
