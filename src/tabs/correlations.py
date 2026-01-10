"""
Onglet Correlations et Stats.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from src.utils import calculate_association_matrix, run_chi2_test
from src.components.metrics import display_chi2_result


def render_tab_correlations(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet des correlations et statistiques.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtre.
    """
    st.subheader("Analyse Bivariee Complete")

    st.info(
        """
        L'analyse bivariee etudie les relations entre 2 variables.
        Le choix du test depend de la nature des variables :
        - **Numerique x Numerique** : Correlation de Pearson
        - **Categorielle x Categorielle** : Test du Chi-carre
        - **Categorielle x Numerique** : Test ANOVA (Fisher)
        """
    )

    # Section 1: Matrice d'association globale
    _render_association_matrix(df_filtered)
    st.markdown("---")

    # Section 2: Correlation Pearson (Num x Num)
    _render_pearson_correlation(df_filtered)
    st.markdown("---")

    # Section 3: Test ANOVA (Cat x Num)
    _render_anova_tests(df_filtered)
    st.markdown("---")

    # Section 4: Test Chi-carre (Cat x Cat)
    _render_chi2_tests(df_filtered)
    st.markdown("---")

    # Section 5: Statistiques descriptives et outliers
    _render_descriptive_stats(df_filtered)
    st.markdown("---")

    # Section 6: Resume de l'analyse bivariee
    _render_bivariate_summary(df_filtered)


def _render_association_matrix(df_filtered: pd.DataFrame):
    """Affiche la matrice d'association."""
    st.markdown("#### Matrice d'Association Comprehensive")

    st.write(
        """
        Cette matrice combine differentes mesures d'association :
        - **Numerique x Numerique** : Correlation de Pearson
        - **Categorielle x Categorielle** : V de Cramer
        - **Numerique x Categorielle** : Eta-squared (ratio de correlation)
        """
    )

    main_cols = ["age", "bmi", "children", "charges", "sex", "smoker", "region"]
    df_for_corr = df_filtered[main_cols]

    numerical_cols = df_for_corr.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_for_corr.select_dtypes(exclude=np.number).columns.tolist()

    association_matrix = calculate_association_matrix(
        df_for_corr,
        numerical_cols,
        categorical_cols
    )

    fig_comp_heatmap = px.imshow(
        association_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Matrice d'Association Complete",
        labels={"color": "Force d'Association"},
        aspect="auto"
    )
    fig_comp_heatmap.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=500
    )
    st.plotly_chart(fig_comp_heatmap, use_container_width=True)

    # Legende
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            **Pearson (Num x Num) :**
            - |r| < 0.3 : Faible
            - 0.3 <= |r| < 0.7 : Moderee
            - |r| >= 0.7 : Forte
            """
        )

    with col2:
        st.markdown(
            """
            **V de Cramer (Cat x Cat) :**
            - V < 0.1 : Negligeable
            - 0.1 <= V < 0.3 : Faible
            - 0.3 <= V < 0.5 : Moderee
            - V >= 0.5 : Forte
            """
        )

    with col3:
        st.markdown(
            """
            **Eta-squared (Cat x Num) :**
            - Eta < 0.01 : Negligeable
            - 0.01 <= Eta < 0.06 : Faible
            - 0.06 <= Eta < 0.14 : Moderee
            - Eta >= 0.14 : Forte
            """
        )


def _render_pearson_correlation(df_filtered: pd.DataFrame):
    """Affiche la correlation de Pearson (Num x Num)."""
    st.markdown("#### Correlation de Pearson (Numerique x Numerique)")

    numerical_cols = ["age", "bmi", "children", "charges"]

    col1, col2 = st.columns([2, 1])

    with col1:
        corr_matrix = df_filtered[numerical_cols].corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Matrice de Correlation de Pearson",
            labels={"color": "Coefficient r"},
            aspect="auto"
        )
        fig_heatmap.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        st.markdown("##### Correlations avec Charges")

        results_pearson = []

        for var in ["age", "bmi", "children"]:
            r = corr_matrix.loc[var, "charges"]
            n = len(df_filtered)

            # Test de significativite
            if abs(r) < 1:
                t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            else:
                p_value = 0

            # Force
            if abs(r) < 0.3:
                force = "Faible"
            elif abs(r) < 0.7:
                force = "Moderee"
            else:
                force = "Forte"

            results_pearson.append({
                "Variable": var,
                "r": r,
                "p_value": p_value,
                "force": force
            })

            st.metric(var, f"r = {r:.3f}")

            if p_value < 0.001:
                st.caption(f"p < 0.001 - {force}")
            elif p_value < 0.05:
                st.caption(f"p = {p_value:.3f} - {force}")
            else:
                st.caption(f"p = {p_value:.3f} - Non significatif")

    # Tableau interpretation
    st.markdown("##### Interpretation des Correlations")

    interpretation_pearson = []
    for res in results_pearson:
        var = res["Variable"]
        r = res["r"]
        p = res["p_value"]
        force = res["force"]

        existe = "Oui" if p < 0.05 else "Non"

        if var == "age":
            traduction = "Les charges augmentent avec l'age"
        elif var == "bmi":
            traduction = "Effet limite seul, fort si combine avec smoker"
        else:
            traduction = "Pas d'impact significatif du nombre d'enfants"

        interpretation_pearson.append({
            "Variable": var,
            "Coefficient r": f"{r:.3f}",
            "p-value": f"{p:.2e}" if p < 0.001 else f"{p:.3f}",
            "1. Existe ?": existe,
            "2. Force": force,
            "3. Traduction Metier": traduction
        })

    st.dataframe(
        pd.DataFrame(interpretation_pearson),
        use_container_width=True,
        hide_index=True
    )


def _render_anova_tests(df_filtered: pd.DataFrame):
    """Affiche les tests ANOVA (Cat x Num)."""
    st.markdown("#### Test ANOVA / Fisher (Categorielle x Numerique)")

    st.write(
        """
        Le test ANOVA compare les moyennes de la variable numerique 
        entre les groupes definis par la variable categorielle.
        
        - **Hypothese nulle (H0)** : Les moyennes sont egales entre les groupes
        - **Si p < 0.05** : On rejette H0, les moyennes sont differentes
        """
    )

    # Liste des tests a effectuer
    anova_tests = [
        ("smoker", "charges", "Smoker vs Charges"),
        ("sex", "charges", "Sexe vs Charges"),
        ("region", "charges", "Region vs Charges"),
    ]

    results_anova = []

    for cat_var, num_var, title in anova_tests:
        st.markdown(f"##### {title}")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Boxplot
            fig = px.box(
                df_filtered,
                x=cat_var,
                y=num_var,
                color=cat_var,
                title=f"Distribution de {num_var} par {cat_var}"
            )
            fig.update_layout(
                height=350,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Calcul ANOVA
            groups = [
                df_filtered[df_filtered[cat_var] == cat][num_var].dropna()
                for cat in df_filtered[cat_var].unique()
            ]

            if all(len(g) > 1 for g in groups):
                f_stat, p_value = f_oneway(*groups)

                # Calcul Eta-squared
                grand_mean = df_filtered[num_var].mean()
                ss_between = sum(
                    len(g) * (g.mean() - grand_mean)**2 for g in groups
                )
                ss_total = sum((df_filtered[num_var] - grand_mean)**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                # Force
                if eta_squared < 0.01:
                    force = "Negligeable"
                elif eta_squared < 0.06:
                    force = "Faible"
                elif eta_squared < 0.14:
                    force = "Moderee"
                else:
                    force = "Forte"

                # Affichage metriques
                st.metric("F-statistique", f"{f_stat:.2f}")
                st.metric("p-value", f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}")
                st.metric("Eta-squared", f"{eta_squared:.3f}")

                # Interpretation
                if p_value < 0.001:
                    st.success(f"Relation TRES significative - Force : {force}")
                elif p_value < 0.05:
                    st.success(f"Relation significative - Force : {force}")
                else:
                    st.info(f"Relation NON significative - Force : {force}")

                # Moyennes par groupe
                st.markdown("**Moyennes par groupe :**")
                for cat in df_filtered[cat_var].unique():
                    mean_val = df_filtered[df_filtered[cat_var] == cat][num_var].mean()
                    st.write(f"- {cat} : {mean_val:,.0f} $")

                # Traduction metier
                if cat_var == "smoker":
                    traduction = "Le tabagisme est le facteur dominant des couts"
                elif cat_var == "sex":
                    traduction = "Pas de difference significative entre hommes et femmes"
                else:
                    traduction = "Differences regionales mineures"

                results_anova.append({
                    "Variables": f"{cat_var} vs {num_var}",
                    "F-stat": f"{f_stat:.2f}",
                    "p-value": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
                    "Eta-squared": f"{eta_squared:.3f}",
                    "1. Existe ?": "Oui" if p_value < 0.05 else "Non",
                    "2. Force": force,
                    "3. Traduction Metier": traduction
                })

        st.markdown("---")

    # Tableau recapitulatif ANOVA
    st.markdown("##### Tableau Recapitulatif ANOVA")
    st.dataframe(
        pd.DataFrame(results_anova),
        use_container_width=True,
        hide_index=True
    )


def _render_chi2_tests(df_filtered: pd.DataFrame):
    """Affiche les tests Chi-carre (Cat x Cat)."""
    st.markdown("#### Test du Chi-carre (Categorielle x Categorielle)")

    st.write(
        """
        Le test du Chi-carre verifie si deux variables categorielles sont independantes.
        
        - **Hypothese nulle (H0)** : Les variables sont independantes
        - **Si p < 0.05** : On rejette H0, les variables sont dependantes (liees)
        """
    )

    # Liste des tests a effectuer
    chi2_tests = [
        ("sex", "smoker", "Sexe vs Statut Fumeur"),
        ("region", "smoker", "Region vs Statut Fumeur"),
        ("sex", "region", "Sexe vs Region"),
    ]

    results_chi2 = []

    for cat1, cat2, title in chi2_tests:
        st.markdown(f"##### {title}")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Table de contingence
            contingency = pd.crosstab(df_filtered[cat1], df_filtered[cat2])
            st.markdown("**Table de contingence :**")
            st.dataframe(contingency, use_container_width=True)

        with col2:
            # Calcul Chi-carre
            chi2, p_value, dof, expected = chi2_contingency(contingency)

            # V de Cramer
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            if min_dim > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
            else:
                cramers_v = 0

            # Force
            if cramers_v < 0.1:
                force = "Negligeable"
            elif cramers_v < 0.3:
                force = "Faible"
            elif cramers_v < 0.5:
                force = "Moderee"
            else:
                force = "Forte"

            # Affichage
            st.metric("Chi-carre", f"{chi2:.2f}")
            st.metric("p-value", f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}")
            st.metric("V de Cramer", f"{cramers_v:.3f}")

            if p_value < 0.05:
                st.success(f"Variables DEPENDANTES - Force : {force}")
            else:
                st.info(f"Variables INDEPENDANTES - Force : {force}")

            # Traduction
            if cat1 == "sex" and cat2 == "smoker":
                traduction = "Le taux de fumeurs est similaire chez les hommes et les femmes"
            elif cat1 == "region" and cat2 == "smoker":
                traduction = "Le taux de fumeurs est similaire dans toutes les regions"
            else:
                traduction = "La repartition des sexes est similaire dans toutes les regions"

            results_chi2.append({
                "Variables": f"{cat1} vs {cat2}",
                "Chi-carre": f"{chi2:.2f}",
                "p-value": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}",
                "V de Cramer": f"{cramers_v:.3f}",
                "1. Existe ?": "Oui" if p_value < 0.05 else "Non",
                "2. Force": force,
                "3. Traduction Metier": traduction
            })

        st.markdown("---")

    # Tableau recapitulatif Chi-carre
    st.markdown("##### Tableau Recapitulatif Chi-carre")
    st.dataframe(
        pd.DataFrame(results_chi2),
        use_container_width=True,
        hide_index=True
    )


def _render_descriptive_stats(df_filtered: pd.DataFrame):
    """Affiche les statistiques descriptives et outliers."""
    st.markdown("#### Statistiques Descriptives et Outliers")

    numerical_cols = ["age", "bmi", "children", "charges"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Variables Numeriques")
        stats_df = df_filtered[numerical_cols].describe().T.round(2)
        st.dataframe(stats_df, use_container_width=True)

    with col2:
        st.markdown("##### Detection des Outliers (Charges)")

        q1 = df_filtered["charges"].quantile(0.25)
        q3 = df_filtered["charges"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df_filtered[
            (df_filtered["charges"] < lower_bound) |
            (df_filtered["charges"] > upper_bound)
        ]

        n_outliers = len(outliers)
        pct_outliers = (n_outliers / len(df_filtered)) * 100

        st.metric("Outliers detectes", n_outliers)
        st.metric("Pourcentage", f"{pct_outliers:.1f}%")
        st.metric("Seuil superieur", f"{upper_bound:,.0f} $")

        if n_outliers > 0:
            st.warning(
                f"{n_outliers} valeurs extremes detectees. "
                "Ces cas representent les assures a haut risque (fumeurs obeses)."
            )


def _render_bivariate_summary(df_filtered: pd.DataFrame):
    """Affiche le resume complet de l'analyse bivariee."""
    st.markdown("#### Resume de l'Analyse Bivariee")

    st.markdown(
        """
        ##### Grille d'Interpretation Finale
        
        Pour chaque relation, nous avons verifie :
        1. **Existence** : La relation est-elle statistiquement significative ? (p < 0.05)
        2. **Force** : Quelle est l'intensite de cette relation ?
        3. **Traduction** : Que signifie cette relation pour le metier ?
        """
    )

    # Tableau de synthese global
    summary_data = [
        {
            "Variables": "smoker vs charges",
            "Type Test": "ANOVA",
            "1. Existe ?": "Oui (p < 0.001)",
            "2. Force": "Tres Forte (Eta > 0.5)",
            "3. Traduction Metier": "Les fumeurs coutent 4x plus cher"
        },
        {
            "Variables": "age vs charges",
            "Type Test": "Pearson",
            "1. Existe ?": "Oui (p < 0.001)",
            "2. Force": "Moderee (r = 0.30)",
            "3. Traduction Metier": "Les charges augmentent avec l'age"
        },
        {
            "Variables": "bmi vs charges",
            "Type Test": "Pearson",
            "1. Existe ?": "Oui (p < 0.05)",
            "2. Force": "Faible (r = 0.20)",
            "3. Traduction Metier": "Effet limite seul, fort avec smoker"
        },
        {
            "Variables": "children vs charges",
            "Type Test": "Pearson",
            "1. Existe ?": "Non significatif",
            "2. Force": "Negligeable",
            "3. Traduction Metier": "Pas d'impact du nombre d'enfants"
        },
        {
            "Variables": "sex vs charges",
            "Type Test": "ANOVA",
            "1. Existe ?": "Non significatif",
            "2. Force": "Negligeable",
            "3. Traduction Metier": "Pas de difference homme/femme"
        },
        {
            "Variables": "region vs charges",
            "Type Test": "ANOVA",
            "1. Existe ?": "Faiblement",
            "2. Force": "Faible",
            "3. Traduction Metier": "Southeast legerement plus cher"
        },
        {
            "Variables": "sex vs smoker",
            "Type Test": "Chi-carre",
            "1. Existe ?": "Non significatif",
            "2. Force": "Negligeable",
            "3. Traduction Metier": "Taux de fumeurs similaire H/F"
        },
        {
            "Variables": "region vs smoker",
            "Type Test": "Chi-carre",
            "1. Existe ?": "Non significatif",
            "2. Force": "Negligeable",
            "3. Traduction Metier": "Taux de fumeurs similaire par region"
        },
    ]

    st.dataframe(
        pd.DataFrame(summary_data),
        use_container_width=True,
        hide_index=True
    )

    # Conclusions
    st.markdown("##### Conclusions pour la Modelisation")

    col1, col2 = st.columns(2)

    with col1:
        st.success(
            """
            **Variables Predictives Importantes :**
            1. `smoker` - Variable dominante
            2. `age` - Effet lineaire significatif
            3. `bmi` - Important en interaction avec smoker
            4. `smoker x bmi` - Interaction a creer
            """
        )

    with col2:
        st.info(
            """
            **Variables a Impact Limite :**
            1. `children` - Peut etre exclue
            2. `sex` - Faible pouvoir predictif
            3. `region` - Effet marginal
            """
        )