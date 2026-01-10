"""
Dashboard d'analyse des données d'assurance.

Point d'entrée principal de l'application Streamlit.
"""

import streamlit as st

# Configuration et style
from src.config import setup_page_config, apply_custom_css

# Chargement des données
from src.data_loader import load_data_with_error_handling

# Composants
from src.components.sidebar import render_sidebar
from src.components.metrics import render_header_metrics

# Onglets
from src.tabs import (
    render_tab_distribution,
    render_tab_health,
    render_tab_demographics,
    render_tab_correlations,
    render_tab_expert,
)


def render_synthesis():
    """Affiche la synthèse et conclusions."""
    st.markdown("---")
    st.subheader("Synthèse & Hypothèses de Modélisation")

    expander = st.expander("Voir les conclusions de l'EDA", expanded=True)
    with expander:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                """
                **1. Le Tabagisme (Game Changer) :**
                - **Non-fumeur** : ~8 434 $
                - **Fumeur** : ~32 050 $ (**Impact x4**)
                
                **2. Synergie IMC x Tabac :**
                - Non-Fumeur + IMC >= 30 : **8 842 $**
                - Fumeur + IMC >= 30 : **41 557 $** (**Risque x4.7**)
                """
            )

        with c2:
            st.markdown(
                """
                **3. Évolution Âge & Tabac :**
                - Jeune (18-35) Non-fumeur : **~4 800 $**
                - Senior (51+) Non-fumeur : **~13 500 $**
                - Jeune fumeur (**~28 100 $**) coûte 2x plus qu'un senior non-fumeur.
                
                **4. Zoom Régions & Genre :**
                - **Southeast** : Plus chère (**14 735 $**) car plus de fumeurs.
                - **Genre** : Les hommes fumeurs sont les plus coûteux (~33k$).
                """
            )


def main():
    """Fonction principale de l'application."""
    # Configuration
    setup_page_config()
    apply_custom_css()

    # Chargement des données
    df = load_data_with_error_handling()

    # Sidebar avec filtres
    df_filtered = render_sidebar(df)

    # Header
    st.title("Analyse des Risques & Charges d'Assurance")
    st.markdown(
        f"**Semaine 1 : Exploration des Facteurs de Coûts** | "
        f"{df_filtered.shape[0]} observations sélectionnées"
    )

    # Métriques principales
    render_header_metrics(df_filtered)
    st.markdown("---")

    # Onglets principaux
    tab_dist, tab_health, tab_demo, tab_corr, tab_expert = st.tabs([
        "Distribution des Coûts",
        "Facteurs de Santé",
        "Démographie & Profils",
        "Corrélations & Stats",
        "Analyses Expert"
    ])

    with tab_dist:
        render_tab_distribution(df_filtered)

    with tab_health:
        render_tab_health(df_filtered)

    with tab_demo:
        render_tab_demographics(df_filtered)

    with tab_corr:
        render_tab_correlations(df_filtered)

    with tab_expert:
        render_tab_expert(df_filtered)

    # Synthèse
    render_synthesis()

    # Footer
    st.write("---")
    st.caption("Dashboard Analytics Insurance v2.1 - Équipe Dev Data IA")


if __name__ == "__main__":
    main()