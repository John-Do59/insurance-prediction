"""
Composant Sidebar avec filtres.
"""

import streamlit as st
import pandas as pd
from src.config import LOGO_URL


def format_smoker(x: str) -> str:
    """Formate le label du statut fumeur."""
    if x == "Tous":
        return "Tous"
    return "Fumeur" if x == "yes" else "Non-fumeur"


def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Affiche la sidebar avec les filtres et retourne le DataFrame filtré.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original.

    Returns
    -------
    pd.DataFrame
        DataFrame filtré.
    """
    st.sidebar.image(LOGO_URL, width=150)
    st.sidebar.title("Filtres de Données")
    st.sidebar.markdown("---")

    # Filtre Région
    st.sidebar.markdown("#### Région Géographique")
    region_list = ["Toutes"] + list(df["region"].unique())
    selected_region = st.sidebar.selectbox(
        "Sélectionner une région",
        region_list,
        label_visibility="collapsed"
    )

    # Filtre Age
    st.sidebar.markdown("#### Tranche d'Âge")
    age_min = int(df["age"].min())
    age_max = int(df["age"].max())
    age_range = st.sidebar.slider(
        "Age",
        age_min,
        age_max,
        (age_min, age_max),
        label_visibility="collapsed"
    )

    # Filtre Sexe
    st.sidebar.markdown("#### Sexe")
    sex_options = ["Tous", "male", "female"]
    selected_sex = st.sidebar.radio(
        "Sexe",
        sex_options,
        label_visibility="collapsed"
    )

    # Filtre Fumeur
    st.sidebar.markdown("#### Statut Fumeur")
    smoker_options = ["Tous", "yes", "no"]
    selected_smoker = st.sidebar.radio(
        "Fumeur",
        smoker_options,
        format_func=format_smoker,
        label_visibility="collapsed"
    )

    # Filtre IMC
    st.sidebar.markdown("#### Indice de Masse Corporelle (IMC)")
    bmi_min = float(df["bmi"].min())
    bmi_max = float(df["bmi"].max())
    bmi_range = st.sidebar.slider(
        "IMC",
        bmi_min,
        bmi_max,
        (bmi_min, bmi_max),
        step=0.5,
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Ajustez les filtres pour explorer les données")

    # Application des filtres
    df_filtered = df[
        (df["age"] >= age_range[0])
        & (df["age"] <= age_range[1])
        & (df["bmi"] >= bmi_range[0])
        & (df["bmi"] <= bmi_range[1])
    ].copy()

    if selected_region != "Toutes":
        df_filtered = df_filtered[df_filtered["region"] == selected_region]

    if selected_sex != "Tous":
        df_filtered = df_filtered[df_filtered["sex"] == selected_sex]

    if selected_smoker != "Tous":
        df_filtered = df_filtered[df_filtered["smoker"] == selected_smoker]

    # Statistiques de filtrage
    _render_filter_stats(df, df_filtered)

    return df_filtered


def _render_filter_stats(df: pd.DataFrame, df_filtered: pd.DataFrame):
    """Affiche les statistiques de filtrage."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Résultat du Filtrage")
    pct_filtered = (len(df_filtered) / len(df)) * 100
    st.sidebar.metric(
        "Observations sélectionnées",
        f"{len(df_filtered)} / {len(df)}",
        delta=f"{pct_filtered:.1f}%"
    )

    # Bouton de téléchargement
    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="Télécharger les données (CSV)",
        data=csv_data,
        file_name=f"insurance_filtered_{len(df_filtered)}_rows.csv",
        mime="text/csv",
        use_container_width=True
    )