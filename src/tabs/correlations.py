"""
Onglet Corrélations & Stats.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from src.utils import calculate_association_matrix, run_chi2_test
from src.components.metrics import display_chi2_result


def render_tab_correlations(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet des corrélations et statistiques.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtré.
    """
    st.subheader("Analyse des Corrélations et Statistiques Descriptives")

    _render_association_matrix(df_filtered)
    st.markdown("---")
    _render_pearson_correlation(df_filtered)
    st.markdown("---")
    _render_descriptive_stats(df_filtered)
    st.markdown("---")
    _render_chi2_tests(df_filtered)


def _render_association_matrix(df_filtered: pd.DataFrame):
    """Affiche la matrice d'association."""
    st.markdown("#### Matrice d'Association Comprehensive")
    st.info(
        """
        Cette matrice combine différentes mesures d'association :
        - **Numérique-Numérique :** Corrélation de Pearson
        - **Catégorielle-Catégorielle :** Coefficient de Cramer (V)
        - **Numérique-Catégorielle :** Ratio de Corrélation (Eta-squared)
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
        title="Matrice d'Association Complète",
        labels={"color": "Force d'Association"},
        aspect="auto"
    )
    fig_comp_heatmap.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=500
    )
    st.plotly_chart(fig_comp_heatmap, use_container_width=True)


def _render_pearson_correlation(df_filtered: pd.DataFrame):
    """Affiche la corrélation de Pearson."""
    numerical_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    
    col_c1, col_c2 = st.columns([2, 1])

    with col_c1:
        st.markdown("#### Corrélation Linéaire (Pearson)")
        corr_matrix = df_filtered[numerical_cols].corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Corrélations entre variables numériques",
            labels={"color": "Coefficient"},
            aspect="auto"
        )
        fig_heatmap.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col_c2:
        st.info(
            """
            **Interprétation (Pearson) :**
            - **Forte** : |r| > 0.7
            - **Modérée** : 0.3 < |r| < 0.7
            - **Faible** : |r| < 0.3

            **Limites :**
            - Mesure uniquement les relations **linéaires**
            - Ne capture pas les interactions complexes
            """
        )

        st.markdown("#### Corrélations avec `charges`")
        if "charges" in corr_matrix.columns:
            corr_with_charges = (
                corr_matrix["charges"]
                .drop("charges")
                .sort_values(ascending=False)
            )
            for var, corr_val in corr_with_charges.items():
                st.metric(label=var, value=f"{corr_val:.3f}")


def _render_descriptive_stats(df_filtered: pd.DataFrame):
    """Affiche les statistiques descriptives."""
    numerical_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("#### Statistiques Descriptives")
        stats_df = df_filtered[numerical_cols].describe().T.round(2)
        st.dataframe(stats_df, use_container_width=True)

    with col_s2:
        st.markdown("#### Détection des Valeurs Extrêmes")
        _render_outliers_detection(df_filtered)


def _render_outliers_detection(df_filtered: pd.DataFrame):
    """Affiche la détection des outliers."""
    q1 = df_filtered["charges"].quantile(0.25)
    q3 = df_filtered["charges"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df_filtered[
        (df_filtered["charges"] < lower_bound)
        | (df_filtered["charges"] > upper_bound)
    ]

    col_o1, col_o2 = st.columns(2)
    col_o1.metric("Outliers détectés", len(outliers))
    col_o2.metric(
        "% du dataset",
        f"{len(outliers) / len(df_filtered) * 100:.1f}%"
    )

    col_o3, col_o4 = st.columns(2)
    col_o3.metric("Charge min", f"{df_filtered['charges'].min():,.0f} $")
    col_o4.metric("Charge max", f"{df_filtered['charges'].max():,.0f} $")

    st.markdown(
        f"""
        **Seuils IQR :**
        - Limite inférieure : {lower_bound:,.0f} $
        - Limite supérieure : {upper_bound:,.0f} $
        """
    )

    # Affichage des outliers
    if len(outliers) > 0:
        st.markdown("---")
        st.markdown("#### Liste des Cas Extrêmes (Top 10)")
        display_cols = [
            "age", "sex", "bmi", "children", "smoker", "region", "charges"
        ]
        outliers_display = (
            outliers[display_cols]
            .sort_values("charges", ascending=False)
            .head(10)
        )
        st.dataframe(outliers_display, use_container_width=True)


def _render_chi2_tests(df_filtered: pd.DataFrame):
    """Affiche les tests Chi-carré."""
    st.markdown("#### Tests d'Indépendance (Chi-carré)")

    st.info(
        """
        **Pourquoi le Chi-carré ?**
        - La corrélation de Pearson **ne fonctionne pas** avec
          les variables catégorielles
        - Le test du Chi-carré mesure l'**indépendance statistique**
        - Utile pour évaluer la significativité de l'association
        """
    )

    df_temp = df_filtered.copy()
    if "charges_cat" not in df_temp.columns:
        df_temp["charges_cat"] = pd.cut(
            df_temp["charges"],
            bins=3,
            labels=["Faible", "Moyen", "Élevé"]
        )

    col_chi1, col_chi2 = st.columns([1, 1])

    with col_chi1:
        st.markdown("##### Test 1 : Smoker x Charges")
        contingency_smoker = pd.crosstab(
            df_temp["smoker"],
            df_temp["charges_cat"]
        )
        st.dataframe(contingency_smoker, use_container_width=True)

        chi2, pval, valid = run_chi2_test(contingency_smoker)
        display_chi2_result(chi2, pval, valid, "Smoker x Charges")

    with col_chi2:
        st.markdown("##### Test 2 : Region x Smoker")
        contingency_region = pd.crosstab(
            df_temp["region"],
            df_temp["smoker"]
        )
        st.dataframe(contingency_region, use_container_width=True)

        chi2, pval, valid = run_chi2_test(contingency_region)
        display_chi2_result(chi2, pval, valid, "Region x Smoker")

    st.markdown("---")
    col_chi3, col_chi4 = st.columns([1, 1])

    with col_chi3:
        st.markdown("##### Test 3 : Sex x Charges")
        contingency_sex = pd.crosstab(
            df_temp["sex"],
            df_temp["charges_cat"]
        )
        st.dataframe(contingency_sex, use_container_width=True)

        chi2, pval, valid = run_chi2_test(contingency_sex)
        display_chi2_result(chi2, pval, valid, "Sex x Charges")

    with col_chi4:
        st.markdown("##### Interprétation du Chi-carré")
        st.markdown(
            """
            **Hypothèse nulle (H0)** : Les variables sont indépendantes

            **Règle de décision** :
            - Si **p < 0.05** : On rejette H0 -> Variables **dépendantes**
            - Si **p >= 0.05** : On ne rejette pas H0 -> Pas de preuve

            **Attendu** :
            - `smoker` x `charges` : **Forte dépendance**
            - `region` x `smoker` : Probablement **indépendant**
            - `sex` x `charges` : Probablement **indépendant**
            """
        )