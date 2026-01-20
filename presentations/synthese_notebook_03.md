# Synthèse Détaillée : Semaine 3 - Industrialisation et Optimisation (Notebook 03)

Cette synthèse couvre les travaux réalisés dans le notebook `03_model_optimization_pipeline.ipynb`. L'objectif principal de cette troisième semaine était de passer d'un modèle expérimental (Semaine 2) à une solution **robuste**, **automatisée** et **explicable**, prête pour le déploiement.

## 1. Objectifs Métier et Techniques
*   **(Métier)** Garantir que le modèle est fiable sur de nouveaux profils (éviter l'overfitting) et qu'il est capable d'expliquer ses décisions (transparence pour la tarification).
*   **(Technique)** Créer un **Pipeline Scikit-learn** complet qui encapsule toutes les étapes de transformation, utiliser la **Validation Croisée** pour une évaluation réaliste, et optimiser les hyperparamètres via **GridSearch**.

## 2. Méthodologie et Étapes Clés

### A. Construction du Pipeline Automatisé ("Model as Code")
Au lieu de traiter les données manuellement étape par étape, nous avons créé un "tuyau" (Pipeline) unique qui prend les données brutes en entrée et sort une prédiction.
*   **Feature Engineering intégré** : Création d'une classe `FeatureEngineer` personnalisée qui génère automatiquement les variables clés découvertes en semaine 2 (`smoker_bmi` pour l'interaction tabac/obésité, et `age_squared` pour l'effet non-linéaire de l'âge).
*   **Pré-traitement robuste** :
    *   *Numérique* : Imputation des valeurs manquantes (médiane) + Standardisation (mise à l'échelle) pour que les modèles de régression fonctionnent correctement.
    *   *Catégoriel* : Encodage OneHot (transformation des catégories "oui/non", "région" en chiffres).
*   **Intêret** : Ce pipeline garantit que les mêmes transformations sont appliquées exactement de la même manière en entraînement et en production.

### B. Validation Robuste (Cross-Validation)
Plutôt que de se fier à une seule division "Train/Test" (qui peut être biaisée par le hasard), nous avons utilisé la **Validation Croisée à 5 plis (5-Fold CV)**.
*   **Le principe** : Le modèle est entraîné et testé 5 fois sur des sous-parties différentes des données.
*   **Résultat** : Score $R^2$ moyen de **0.8212** (+/- 0.02). Cela confirme que notre modèle est stable et généralise bien, quelle que soit la partie du dataset utilisée.

### C. Optimisation des Modèles (GridSearch)
Nous avons mis en compétition trois algorithmes de régression linéaire régularisée pour trouver le meilleur compromis biais/variance :
1.  **Ridge** (Pénalité L2) : Réduit l'impact des variables corrélées sans les supprimer.
2.  **Lasso** (Pénalité L1) : Peut supprimer certaines variables moins utiles.
3.  **ElasticNet** : Combinaison des deux.

**Résultats de la compétition :**
*   Les trois modèles ont obtenu des performances quasi-identiques ($R^2 \approx 0.8250$ en CV).
*   **Vainqueur choisi : Ridge**.
    *   *Raison* : Il offre une excellente stabilité mathématique, particulièrement utile ici car nous avons créé des variables corrélées (`age` et `age_squared`). Contrairement au Lasso, il conserve l'information de toutes les variables, ce qui est souvent préférable quand toutes les features ont du sens métier.

## 3. Performance Finale et Analyse
Après avoir sélectionné le modèle Ridge optimisé (`alpha=0.1`), nous l'avons évalué sur le jeu de test final (jamais vu durant l'optimisation).

*   **Score $R^2$** : **0.8857** (Le modèle explique ~88.6% de la variance des charges).
*   **MAE (Erreur Absolue Moyenne)** : **2884.67 $** (En moyenne, le modèle se trompe de ~2880$ sur le tarif annuel).
*   **Comparaison** : Ce résultat est très proche de la régression linéaire simple de la semaine 2 ($0.8856$), ce qui prouve que l'apport principal de cette semaine n'est pas tant la performance pure que la **robustesse** et la **sécurité** du modèle grâce à la régularisation Ridge et au Pipeline.

## 4. Interprétabilité (SHAP)
Pour répondre aux exigences de transparence, nous avons utilisé les valeurs SHAP (SHapley Additive exPlanations).
*   **Importance Globale** : Confirme que le statut de **Fumeur** (et son interaction avec le BMI) est de loin le facteur le plus déterminant, suivi par l'**Âge** et le **BMI**.
*   **Explication Locale** : Pour chaque assuré, nous pouvons désormais dire exactement : *"Votre prime est de 15 000$ car l'âge a ajouté +3000$, le tabac +10 000$, etc."*. C'est un atout majeur pour l'argumentaire commercial et la conformité réglementaire.

---
**En résumé** : La Semaine 3 a transformé un "prototype" (Semaine 2) en un "produit" (Pipeline Ridge scikit-learn), validé mathématiquement et prêt à être intégré dans l'application web.
