"""
Onglet Facteurs de Santé.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import COLOR_MAP_SMOKER


def render_tab_health(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet des facteurs de santé.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtré.
    """
    st.subheader("L'influence de l'IMC et du Tabagisme")

    col_h1, col_h2 = st.columns([2, 1])

    with col_h1:
        _render_scatter_bmi(df_filtered)

    with col_h2:
        _render_health_insights()
        _render_bmi_distribution(df_filtered)


def _render_scatter_bmi(df_filtered: pd.DataFrame):
    """Affiche le scatter plot IMC vs Charges."""
    fig_scatter = px.scatter(
        df_filtered,
        x="bmi",
        y="charges",
        color="smoker_label",
        size="age",
        hover_data=["age", "children"],
        title="Impact combiné de l'IMC et du Tabac",
        labels={
            "bmi": "IMC (Indice de Masse Corporelle)",
            "charges": "Charges ($)"
        },
        color_discrete_map=COLOR_MAP_SMOKER
    )
    fig_scatter.add_vline(
        x=30,
        line_dash="dash",
        line_color="gray",
        annotation_text="Seuil Obésité (30)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def _render_health_insights():
    """Affiche les insights santé."""
    st.success(
        """
        **Insight Clé : L'Effet Synergie**
        - Chez les **non-fumeurs**, l'IMC a une influence
          linéaire modérée.
        - Chez les **fumeurs**, on observe une rupture brutale
          au-delà d'un **IMC de 30**.
        - Ce groupe (Fumeur + Obèse) représente le risque
          financier le plus élevé pour l'assureur.
        """
    )


def _render_bmi_distribution(df_filtered: pd.DataFrame):
    """Affiche la distribution des catégories d'IMC."""
    df_temp = df_filtered.copy()
    df_temp["bmi_cat"] = pd.cut(
        df_temp["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Insuffisant", "Normal", "Surpoids", "Obèse"]
    )
    fig_pie = px.pie(
        df_temp,
        names="bmi_cat",
        title="Répartition des catégories d'IMC",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)