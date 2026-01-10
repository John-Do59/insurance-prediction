"""
Composants d'affichage des métriques.
"""

import streamlit as st
import pandas as pd


def render_header_metrics(df_filtered: pd.DataFrame):
    """
    Affiche les métriques principales dans le header.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtré.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_charges = df_filtered["charges"].mean()
        st.metric("Charges Moyennes", f"{avg_charges:,.0f} $")

    with col2:
        median_age = df_filtered["age"].median()
        st.metric("Âge Médian", f"{median_age:.0f} ans")

    with col3:
        avg_bmi = df_filtered["bmi"].mean()
        st.metric("IMC Moyen", f"{avg_bmi:.1f}")

    with col4:
        smoker_ratio = (df_filtered["smoker"] == "yes").mean()
        st.metric("% Fumeurs", f"{smoker_ratio:.1%}")


def display_chi2_result(
    chi2_stat: float,
    p_value: float,
    is_valid: bool,
    test_name: str
):
    """
    Affiche les résultats d'un test Chi-carré.

    Parameters
    ----------
    chi2_stat : float
        Statistique du Chi-carré.
    p_value : float
        P-value du test.
    is_valid : bool
        Indique si le test est valide.
    test_name : str
        Nom du test pour l'affichage.
    """
    if not is_valid:
        st.warning(f"Données insuffisantes pour le test Chi2 ({test_name}).")
        return

    st.metric("Chi2 statistique", f"{chi2_stat:.2f}")
    st.metric("p-value", f"{p_value:.2e}")

    if p_value < 0.001:
        st.success("**Relation très significative** (p < 0.001)")
    elif p_value < 0.05:
        st.success("**Relation significative** (p < 0.05)")
    else:
        st.warning("Pas de relation significative (p >= 0.05)")