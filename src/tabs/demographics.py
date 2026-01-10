"""
Onglet Démographie & Profils.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import COLOR_MAP_SMOKER


def render_tab_demographics(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet démographie.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtré.
    """
    st.subheader("Analyse Démographique Approfondie")

    _render_region_section(df_filtered)
    st.markdown("---")
    _render_sex_section(df_filtered)
    st.markdown("---")
    _render_children_section(df_filtered)


def _render_region_section(df_filtered: pd.DataFrame):
    """Section distribution par région."""
    st.markdown("#### Distribution des Charges par Région")
    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        fig_region = px.box(
            df_filtered,
            x="region",
            y="charges",
            color="smoker_label",
            title="Charges par région et statut fumeur",
            points="outliers",
            color_discrete_map=COLOR_MAP_SMOKER
        )
        fig_region.update_layout(
            xaxis_title="Région",
            yaxis_title="Charges ($)"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with col_r2:
        st.info(
            """
            **Lecture du Box Plot :**
            - **Boîte** : 50% des données (Q1 à Q3)
            - **Ligne médiane** : Valeur centrale
            - **Moustaches** : Étendue normale
            - **Points** : Valeurs extrêmes (outliers)

            **Insight :**
            Les fumeurs ont des charges plus élevées
            dans **toutes** les régions.
            """
        )

        region_stats = (
            df_filtered
            .groupby("region")["charges"]
            .agg(["mean", "median"])
            .round(0)
        )
        st.markdown("**Moyennes par région :**")
        for region, row in region_stats.iterrows():
            st.metric(
                label=region,
                value=f"{row['mean']:,.0f} $",
                delta=f"Médiane: {row['median']:,.0f} $"
            )


def _render_sex_section(df_filtered: pd.DataFrame):
    """Section comparaison homme/femme."""
    st.markdown("#### Comparaison Homme vs Femme")
    col_s1, col_s2 = st.columns([2, 1])

    with col_s1:
        avg_by_sex_smoker = (
            df_filtered
            .groupby(["sex", "smoker_label"])["charges"]
            .mean()
            .reset_index()
        )

        fig_sex = px.bar(
            avg_by_sex_smoker,
            x="sex",
            y="charges",
            color="smoker_label",
            barmode="group",
            title="Charges moyennes par sexe et statut fumeur",
            labels={
                "sex": "Sexe",
                "charges": "Charges moyennes ($)"
            },
            color_discrete_map=COLOR_MAP_SMOKER
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    with col_s2:
        st.success(
            """
            **Observation :**
            - Peu de différence entre hommes et femmes
            - Le **statut fumeur** est le facteur dominant
            - L'écart fumeur/non-fumeur est similaire pour les deux sexes
            """
        )

        _render_sex_ecart_metric(avg_by_sex_smoker)


def _render_sex_ecart_metric(avg_by_sex_smoker: pd.DataFrame):
    """Affiche la métrique d'écart homme fumeur/non-fumeur."""
    male_smoker_df = avg_by_sex_smoker[
        (avg_by_sex_smoker["sex"] == "male")
        & (avg_by_sex_smoker["smoker_label"] == "Fumeur")
    ]
    male_nonsmoker_df = avg_by_sex_smoker[
        (avg_by_sex_smoker["sex"] == "male")
        & (avg_by_sex_smoker["smoker_label"] == "Non-fumeur")
    ]

    if not male_smoker_df.empty and not male_nonsmoker_df.empty:
        male_smoker = male_smoker_df["charges"].values[0]
        male_nonsmoker = male_nonsmoker_df["charges"].values[0]
        ecart = male_smoker - male_nonsmoker
        pct_ecart = ((male_smoker / male_nonsmoker) - 1) * 100

        st.metric(
            "Écart Fumeur/Non-fumeur (Homme)",
            f"+{ecart:,.0f} $",
            delta=f"{pct_ecart:.0f}%"
        )


def _render_children_section(df_filtered: pd.DataFrame):
    """Section impact du nombre d'enfants."""
    st.markdown("#### Impact du Nombre d'Enfants")
    col_c1, col_c2 = st.columns([2, 1])

    with col_c1:
        fig_children = px.box(
            df_filtered,
            x="children",
            y="charges",
            color="smoker_label",
            title="Distribution des charges selon le nombre d'enfants",
            points="outliers",
            color_discrete_map=COLOR_MAP_SMOKER
        )
        fig_children.update_layout(
            xaxis_title="Nombre d'enfants",
            yaxis_title="Charges ($)"
        )
        st.plotly_chart(fig_children, use_container_width=True)

    with col_c2:
        st.warning(
            """
            **Tendance :**
            - Impact **modéré** du nombre d'enfants
            - Augmentation légère avec plus d'enfants
            - Toujours dominé par le statut fumeur

            **Note :** La plupart des assurés ont 0-2 enfants.
            """
        )

        children_dist = df_filtered["children"].value_counts().sort_index()
        st.markdown("**Répartition :**")
        for n_children, count in children_dist.items():
            pct = (count / len(df_filtered)) * 100
            st.write(f"{n_children} enfant(s): {count} ({pct:.1f}%)")