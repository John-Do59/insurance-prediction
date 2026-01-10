"""
Onglet Analyses Expert.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from src.config import COLORS, COLOR_MAP_SMOKER


def render_tab_expert(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet des analyses expert.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtré.
    """
    st.subheader("Analyses Avancées de Second Niveau")

    _render_pareto_analysis(df_filtered)
    st.markdown("---")
    _render_regional_health(df_filtered)
    st.markdown("---")
    _render_risk_volatility(df_filtered)
    st.markdown("---")
    _render_shock_effects(df_filtered)


def _render_pareto_analysis(df_filtered: pd.DataFrame):
    """Analyse de Pareto."""
    st.markdown("#### Concentration des Charges (Loi de Pareto)")
    col_p1, col_p2 = st.columns([2, 1])

    with col_p1:
        df_sorted = df_filtered.sort_values('charges', ascending=False).copy()
        df_sorted['cum_charges'] = df_sorted['charges'].cumsum() / df_sorted['charges'].sum()
        df_sorted['cum_population'] = np.arange(1, len(df_sorted) + 1) / len(df_sorted)

        fig_pareto = px.line(
            df_sorted, x='cum_population', y='cum_charges',
            title="Courbe de Pareto : Concentration des Charges",
            labels={'cum_population': '% Population', 'cum_charges': '% Charges Totales'},
            color_discrete_sequence=[COLORS["primary"]]
        )
        fig_pareto.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line=dict(dash="dash", color="grey")
        )
        st.plotly_chart(fig_pareto, use_container_width=True)

    with col_p2:
        top_20_percent = int(len(df_sorted) * 0.2)
        if top_20_percent > 0:
            top_20_share = df_sorted.iloc[top_20_percent - 1]['cum_charges']
        else:
            top_20_share = 0

        st.metric("Part du Top 20%", f"{top_20_share:.1%}")

        pareto_pie = pd.DataFrame({
            'Groupe': ['Top 20% des assurés', 'Les autres 80%'],
            'Coût total': [top_20_share, 1 - top_20_share]
        })
        fig_pie_pareto = px.pie(
            pareto_pie, values='Coût total', names='Groupe',
            hole=0.4, color_discrete_sequence=[COLORS["primary"], COLORS["neutral"]]
        )
        st.plotly_chart(fig_pie_pareto, use_container_width=True)

        st.info("**Insight :** Plus de 50% des coûts sont générés par seulement 20% des assurés.")


def _render_regional_health(df_filtered: pd.DataFrame):
    """Profil de santé régional."""
    st.markdown("#### Profil de Santé Régional (Taux d'Obésité)")
    col_ob1, col_ob2 = st.columns([2, 1])

    with col_ob1:
        df_temp = df_filtered.copy()
        df_temp['is_obese'] = df_temp['bmi'] >= 30
        reg_health = df_temp.groupby('region')['is_obese'].mean().reset_index()
        reg_health['is_obese'] *= 100

        fig_obesity = px.bar(
            reg_health, x='region', y='is_obese',
            title="Taux d'Obésité par Région (Fracture Sanitaire)",
            labels={'is_obese': '% Obésité', 'region': 'Région'},
            color='is_obese',
            color_continuous_scale="Purples"
        )
        st.plotly_chart(fig_obesity, use_container_width=True)

    with col_ob2:
        se_obesity = reg_health[reg_health['region'] == 'southeast']['is_obese'].values
        if len(se_obesity) > 0:
            st.metric("Taux Obésité Southeast", f"{se_obesity[0]:.1f}%")

        st.warning("**Alerte Fracture Sanitaire :** Le Southeast présente un taux d'obésité alarmant.")


def _render_risk_volatility(df_filtered: pd.DataFrame):
    """Volatilité du risque."""
    st.markdown("#### Volatilité du Risque (Imprévisibilité)")
    col_v1, col_v2 = st.columns([2, 1])

    with col_v1:
        cv_data = (
            df_filtered
            .groupby('smoker_label')['charges']
            .agg(lambda x: x.std() / x.mean())
            .reset_index()
        )
        cv_data.columns = ['Statut', 'Volatilité (CV)']

        fig_cv = px.bar(
            cv_data, x='Statut', y='Volatilité (CV)',
            title="Éclatement du Risque (Coefficient de Variation)",
            color='Statut',
            color_discrete_map=COLOR_MAP_SMOKER
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    with col_v2:
        st.info("""
        **Coefficient de Variation (CV) :**
        - Un CV élevé signifie que le risque est **plus imprévisible**.
        - Le risque chez les non-fumeurs est plus volatil.
        - Chez les fumeurs, le risque élevé est plus 'systématique'.
        """)


def _render_shock_effects(df_filtered: pd.DataFrame):
    """Visualisation des effets chocs."""
    st.markdown("#### ⚡ Visualisation des Écarts Majeurs (Effets Chocs)")
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        _render_age_vs_lifestyle(df_filtered)

    with col_ch2:
        _render_combo_effect(df_filtered)


def _render_age_vs_lifestyle(df_filtered: pd.DataFrame):
    """Duel Jeune Fumeur vs Senior Non-Fumeur."""
    df_temp = df_filtered.copy()
    df_temp['profile_group'] = 'Autres'
    df_temp.loc[
        (df_temp['age'] <= 35) & (df_temp['smoker'] == 'yes'),
        'profile_group'
    ] = 'Jeune Fumeur (<=35)'
    df_temp.loc[
        (df_temp['age'] >= 51) & (df_temp['smoker'] == 'no'),
        'profile_group'
    ] = 'Senior Non-Fumeur (>=51)'

    duel_df = (
        df_temp[df_temp['profile_group'] != 'Autres']
        .groupby('profile_group')['charges']
        .mean()
        .reset_index()
    )

    fig_duel = px.bar(
        duel_df, x='profile_group', y='charges',
        title="Le Duel : Style de Vie vs Vieillissement",
        labels={'charges': 'Charges Moyennes ($)', 'profile_group': 'Profil'},
        color='profile_group',
        color_discrete_map={
            'Jeune Fumeur (<=35)': COLORS["smoker"],
            'Senior Non-Fumeur (>=51)': COLORS["secondary"]
        }
    )
    st.plotly_chart(fig_duel, use_container_width=True)
    st.caption("Un jeune fumeur coûte ~2x plus cher qu'un senior qui ne fume pas.")


def _render_combo_effect(df_filtered: pd.DataFrame):
    """Effet combo IMC x Tabac."""
    df_temp = df_filtered.copy()
    df_temp['bmi_simple_label'] = (df_temp['bmi'] >= 30).map({
        True: 'IMC >= 30',
        False: 'IMC < 30'
    })

    matrix_df = (
        df_temp
        .groupby(['smoker_label', 'bmi_simple_label'])['charges']
        .mean()
        .reset_index()
    )
    matrix_df['Groupe'] = matrix_df['smoker_label'] + " (" + matrix_df['bmi_simple_label'] + ")"

    fig_matrix = px.bar(
        matrix_df, x='Groupe', y='charges',
        title="L'Effet Combo (Tabac + Obésité)",
        labels={'charges': 'Charges Moyennes ($)'},
        color='smoker_label',
        color_discrete_map=COLOR_MAP_SMOKER
    )
    st.plotly_chart(fig_matrix, use_container_width=True)
    st.caption("L'obésité multiplie par ~5 le coût pour un fumeur.")