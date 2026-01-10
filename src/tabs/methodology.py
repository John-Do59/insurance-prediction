"""
Onglet Methodologie et Glossaire.
"""

import streamlit as st
import pandas as pd


def render_tab_methodology():
    """
    Affiche la methodologie et le glossaire des notions statistiques.
    """
    st.subheader("Methodologie et Glossaire Statistique")

    st.info(
        """
        Cet onglet presente les notions statistiques utilisees dans cette analyse,
        leurs formules, leur interpretation et leur utilite pour la modelisation.
        """
    )

    # Section 1: Analyse Univariee
    _render_univariate_methods()
    st.markdown("---")

    # Section 2: Analyse Bivariee
    _render_bivariate_methods()
    st.markdown("---")

    # Section 3: Mesures de qualite des donnees
    _render_quality_metrics()
    st.markdown("---")

    # Section 4: Preparation modelisation
    _render_modeling_concepts()


def _render_univariate_methods():
    """Affiche les methodes d'analyse univariee."""
    st.markdown("### Analyse Univariee")

    st.write(
        """
        L'analyse univariee etudie chaque variable individuellement.
        Elle permet de comprendre la distribution et les caracteristiques de chaque variable.
        """
    )

    # Mesures de tendance centrale
    st.markdown("#### Mesures de Tendance Centrale")

    measures_central = pd.DataFrame({
        "Mesure": ["Moyenne", "Mediane", "Mode"],
        "Formule": [
            "x_bar = (1/n) * sum(xi)",
            "Valeur centrale des donnees triees",
            "Valeur la plus frequente"
        ],
        "Utilite": [
            "Resume la valeur typique. Sensible aux valeurs extremes.",
            "Valeur centrale robuste. Insensible aux outliers.",
            "Valeur la plus commune. Utile pour les variables categorielles."
        ],
        "Quand l'utiliser": [
            "Distributions symetriques, pas d'outliers",
            "Distributions asymetriques, presence d'outliers",
            "Variables categorielles ou discretes"
        ]
    })
    st.dataframe(measures_central, use_container_width=True, hide_index=True)

    # Mesures de dispersion
    st.markdown("#### Mesures de Dispersion")

    measures_dispersion = pd.DataFrame({
        "Mesure": ["Ecart-type", "Variance", "IQR", "Etendue"],
        "Formule": [
            "s = sqrt((1/(n-1)) * sum((xi - x_bar)^2))",
            "s^2 = (1/(n-1)) * sum((xi - x_bar)^2)",
            "IQR = Q3 - Q1",
            "Etendue = Max - Min"
        ],
        "Utilite": [
            "Mesure la dispersion autour de la moyenne. Meme unite que les donnees.",
            "Carre de l'ecart-type. Utilisee dans les calculs statistiques.",
            "Dispersion des 50% centraux. Robuste aux outliers.",
            "Amplitude totale. Tres sensible aux valeurs extremes."
        ],
        "Quand l'utiliser": [
            "Distributions normales, comparaison de variabilite",
            "Calculs intermediaires (ANOVA, regression)",
            "Detection d'outliers, distributions asymetriques",
            "Apercu rapide, donnees sans outliers"
        ]
    })
    st.dataframe(measures_dispersion, use_container_width=True, hide_index=True)

    # Mesures de forme
    st.markdown("#### Mesures de Forme de la Distribution")

    measures_shape = pd.DataFrame({
        "Mesure": ["Skewness (Asymetrie)", "Kurtosis (Aplatissement)"],
        "Formule": [
            "g1 = (1/n) * sum(((xi - x_bar)/s)^3)",
            "g2 = (1/n) * sum(((xi - x_bar)/s)^4) - 3"
        ],
        "Interpretation": [
            "= 0: Symetrique | > 0: Queue a droite | < 0: Queue a gauche",
            "= 0: Normale | > 0: Pics, queues lourdes | < 0: Aplatie"
        ],
        "Impact Modelisation": [
            "Skewness > 1: Transformation log recommandee",
            "Kurtosis eleve: Attention aux outliers, modeles robustes"
        ]
    })
    st.dataframe(measures_shape, use_container_width=True, hide_index=True)

    # Formule du skewness expliquee
    with st.expander("Detail : Pourquoi le Skewness est important ?"):
        st.markdown(
            """
            **Le Skewness mesure l'asymetrie de la distribution.**

            Pour la regression lineaire, une variable cible tres asymetrique pose probleme :
            - Les residus ne seront pas normalement distribues
            - Le modele aura du mal a predire les valeurs extremes
            - Les metriques comme R2 peuvent etre trompeuses

            **Solution :** Appliquer une transformation logarithmique.

            ```
            y_transformed = log(y + 1)
            ```

            Cela "compresse" les grandes valeurs et "etire" les petites,
            rendant la distribution plus symetrique.
            """
        )


def _render_bivariate_methods():
    """Affiche les methodes d'analyse bivariee."""
    st.markdown("### Analyse Bivariee")

    st.write(
        """
        L'analyse bivariee etudie les relations entre deux variables.
        Le choix du test depend de la nature des variables.
        """
    )

    # Tableau de selection du test
    st.markdown("#### Choix du Test Statistique")

    test_selection = pd.DataFrame({
        "Variable 1": ["Numerique", "Categorielle", "Numerique"],
        "Variable 2": ["Numerique", "Categorielle", "Categorielle"],
        "Test": ["Correlation de Pearson", "Test du Chi²", "ANOVA / Test F"],
        "Mesure de Force": ["Coefficient r", "V de Cramer", "Eta² (Eta-squared)"]
    })
    st.dataframe(test_selection, use_container_width=True, hide_index=True)

    # Correlation de Pearson
    st.markdown("#### 1. Correlation de Pearson (Numerique x Numerique)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Formule :**
            ```
            r = sum((xi - x_bar)(yi - y_bar)) / sqrt(sum((xi - x_bar)^2) * sum((yi - y_bar)^2))
            ```

            **Interpretation de r :**
            - r = 1 : Correlation positive parfaite
            - r = 0 : Pas de correlation lineaire
            - r = -1 : Correlation negative parfaite
            """
        )

    with col2:
        st.markdown(
            """
            **Seuils d'interpretation :**

            | Valeur |r| | Force |
            |--------|-------|
            | < 0.3 | Faible |
            | 0.3 - 0.7 | Moderee |
            | > 0.7 | Forte |

            **Limites :**
            - Mesure uniquement les relations LINEAIRES
            - Sensible aux outliers
            """
        )

    # Test du Chi²
    st.markdown("#### 2. Test du Chi² (Categorielle x Categorielle)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Formule :**
            ```
            Chi² = sum((Observe - Attendu)^2 / Attendu)
            ```

            **Principe :**
            - Compare les effectifs observes aux effectifs theoriques
            - Si independance, Observe = Attendu
            - Plus Chi² est grand, plus les variables sont liees
            """
        )

    with col2:
        st.markdown(
            """
            **V de Cramer (force de la relation) :**
            ```
            V = sqrt(Chi² / (n * min(r-1, c-1)))
            ```

            | Valeur V | Force |
            |----------|-------|
            | < 0.1 | Negligeable |
            | 0.1 - 0.3 | Faible |
            | 0.3 - 0.5 | Moderee |
            | > 0.5 | Forte |
            """
        )

    with st.expander("Detail : Comment interpreter un test Chi² ?"):
        st.markdown(
            """
            **Etape 1 : Hypotheses**
            - H0 (nulle) : Les variables sont independantes
            - H1 (alternative) : Les variables sont dependantes

            **Etape 2 : Calcul du Chi²**
            - Construire la table de contingence (effectifs observes)
            - Calculer les effectifs attendus sous H0
            - Calculer Chi² = sum((O - E)² / E)

            **Etape 3 : Decision**
            - Si p-value < 0.05 : Rejeter H0, les variables sont liees
            - Si p-value >= 0.05 : Ne pas rejeter H0, pas de preuve de lien

            **Etape 4 : Force**
            - Calculer V de Cramer pour quantifier la force du lien
            """
        )

    # ANOVA
    st.markdown("#### 3. ANOVA / Test F (Categorielle x Numerique)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Principe :**
            - Compare les moyennes de la variable numerique entre les groupes
            - Decompose la variance totale en variance inter-groupe et intra-groupe

            **Formule F :**
            ```
            F = Variance inter-groupes / Variance intra-groupes
            ```

            Si F est grand, les moyennes different significativement.
            """
        )

    with col2:
        st.markdown(
            """
            **Eta² (force de la relation) :**
            ```
            Eta² = SS_between / SS_total
            ```

            | Valeur Eta² | Force |
            |-------------|-------|
            | < 0.01 | Negligeable |
            | 0.01 - 0.06 | Faible |
            | 0.06 - 0.14 | Moderee |
            | > 0.14 | Forte |

            **Interpretation :**
            Eta² = 0.14 signifie que 14% de la variance de Y
            est expliquee par X.
            """
        )

    with st.expander("Detail : Pourquoi utiliser ANOVA plutot que plusieurs t-tests ?"):
        st.markdown(
            """
            **Probleme des comparaisons multiples :**

            Si on compare 4 groupes avec des t-tests :
            - 4 groupes = 6 comparaisons (A-B, A-C, A-D, B-C, B-D, C-D)
            - Risque d'erreur Type I augmente : 1 - (0.95)^6 = 26%

            **Solution : ANOVA**
            - Un seul test global
            - Controle le risque d'erreur a 5%
            - Si significatif, on peut faire des tests post-hoc
            """
        )


def _render_quality_metrics():
    """Affiche les metriques de qualite des donnees."""
    st.markdown("### Metriques de Qualite des Donnees")

    quality_metrics = pd.DataFrame({
        "Metrique": ["Completude", "Unicite", "Validite", "Coherence"],
        "Definition": [
            "Pourcentage de valeurs non manquantes",
            "Pourcentage de lignes uniques (sans doublons)",
            "Pourcentage de valeurs dans les plages attendues",
            "Absence de contradictions entre variables"
        ],
        "Formule": [
            "(Total cellules - Manquantes) / Total cellules",
            "(Total lignes - Doublons) / Total lignes",
            "Valeurs valides / Total valeurs",
            "Verification manuelle des regles metier"
        ],
        "Seuil Acceptable": [
            "> 95%",
            "> 99%",
            "> 99%",
            "100%"
        ]
    })
    st.dataframe(quality_metrics, use_container_width=True, hide_index=True)


def _render_modeling_concepts():
    """Affiche les concepts de preparation a la modelisation."""
    st.markdown("### Concepts pour la Modelisation")

    # VIF
    st.markdown("#### Multicolinearite et VIF")

    st.markdown(
        """
        **Probleme :** Si deux variables explicatives sont tres correlees,
        le modele ne peut pas distinguer leurs effets individuels.

        **VIF (Variance Inflation Factor) :**
        ```
        VIF_j = 1 / (1 - R²_j)
        ```
        Ou R²_j est le R² de la regression de X_j sur toutes les autres variables X.

        | Valeur VIF | Interpretation |
        |------------|----------------|
        | 1 | Pas de colinearite |
        | 1 - 5 | Colinearite moderee, acceptable |
        | 5 - 10 | Colinearite elevee, attention |
        | > 10 | Colinearite severe, action requise |
        """
    )

    # Transformation Log
    st.markdown("#### Transformation Logarithmique")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Quand l'utiliser :**
            - Variable cible tres asymetrique (skewness > 1)
            - Relation non-lineaire avec les predicteurs
            - Variance qui augmente avec la moyenne

            **Formule :**
            ```
            y_log = log(y + 1)
            ```
            On ajoute 1 pour gerer les zeros.
            """
        )

    with col2:
        st.markdown(
            """
            **Avantages :**
            - Reduit l'asymetrie
            - Stabilise la variance
            - Rend les relations plus lineaires
            - Reduit l'impact des outliers

            **Attention :**
            - Les coefficients s'interpretent differemment
            - Predictions a retransformer : y = exp(y_log) - 1
            """
        )

    # Encodage
    st.markdown("#### Encodage des Variables Categorielles")

    encoding_methods = pd.DataFrame({
        "Methode": ["Label Encoding", "One-Hot Encoding", "Binary Encoding"],
        "Description": [
            "Conversion en entiers (0, 1, 2, ...)",
            "Creation d'une colonne par categorie (0/1)",
            "Conversion pour variables binaires (0/1)"
        ],
        "Quand l'utiliser": [
            "Variables ordinales (ex: taille S/M/L)",
            "Variables nominales avec peu de categories",
            "Variables binaires (ex: oui/non)"
        ],
        "Exemple": [
            "region: NE=0, NW=1, SE=2, SW=3",
            "region_NW=1, region_SE=0, region_SW=0",
            "smoker: yes=1, no=0"
        ]
    })
    st.dataframe(encoding_methods, use_container_width=True, hide_index=True)

    # Interactions
    st.markdown("#### Variables d'Interaction")

    st.markdown(
        """
        **Probleme :** Parfois l'effet d'une variable depend d'une autre.

        **Exemple dans notre dataset :**
        - L'effet du BMI sur les charges depend du statut fumeur
        - BMI eleve + Non-fumeur = Impact modere
        - BMI eleve + Fumeur = Impact TRES eleve

        **Solution : Creer une variable d'interaction**
        ```
        smoker_bmi = smoker * bmi
        smoker_bmi30 = smoker * (bmi >= 30)
        ```

        Ces variables capturent l'effet synergique.
        """
    )