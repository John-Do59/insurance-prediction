"""
Onglet Distribution des Coûts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import COLOR_MAP_SMOKER


def render_tab_distribution(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet de distribution des coûts.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtré.
    """
    st.subheader("Analyse de la Variable Cible : Charges")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        _render_histogram(df_filtered)

    with col_right:
        _render_insights()
        _render_boxplot_smoker(df_filtered)
        _render_categorical_distribution(df_filtered)


def _render_histogram(df_filtered: pd.DataFrame):
    """Affiche l'histogramme des charges."""
    use_log = st.checkbox(
        "Appliquer l'échelle logarithmique (Log scale)",
        help="Aide à visualiser les distributions asymétriques"
    )

    fig_hist = px.histogram(
        df_filtered,
        x="charges",
        nbins=50,
        color_discrete_sequence=["#3b82f6"],
        marginal="box",
        log_x=use_log,
        title="Distribution des charges médicales"
    )
    fig_hist.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)


def _render_insights():
    """Affiche les insights métier."""
    st.info(
        """
        **Observation Métier :**
        - La distribution est fortement **asymétrique à droite**
          (skewness positive).
        - La majorité des dossiers sont sous les 15k$, mais une
          minorité génère des coûts très élevés (>40k$).
        - **Impact IA :** Une transformation Log sera probablement
          nécessaire pour améliorer la performance de la régression.
        """
    )


def _render_boxplot_smoker(df_filtered: pd.DataFrame):
    """Affiche le boxplot par statut fumeur."""
    fig_box_smoker = px.box(
        df_filtered,
        x="smoker_label",
        y="charges",
        color="smoker_label",
        points="all",
        title="Répartition des charges par statut fumeur",
        color_discrete_map=COLOR_MAP_SMOKER
    )
    st.plotly_chart(fig_box_smoker, use_container_width=True)


def _render_categorical_distribution(df_filtered: pd.DataFrame):
    """Affiche la répartition des variables catégorielles."""
    st.markdown("---")
    st.markdown("#### Répartition des Variables Catégorielles")

    col_cat1, col_cat2 = st.columns(2)

    with col_cat1:
        fig_sex_dist = px.pie(
            df_filtered,
            names="sex",
            title="Répartition par Sexe",
            hole=0.3
        )
        st.plotly_chart(fig_sex_dist, use_container_width=True)

    with col_cat2:
        fig_region_dist = px.pie(
            df_filtered,
            names="region",
            title="Répartition par Région",
            hole=0.3
        )
        st.plotly_chart(fig_region_dist, use_container_width=True)