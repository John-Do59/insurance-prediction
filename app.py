"""
Dashboard d'analyse des donnees d'assurance.

Point d'entree principal de l'application Streamlit.
"""

import streamlit as st

# Configuration et style
from src.config import setup_page_config, apply_custom_css

# Chargement des donnees
from src.data_loader import load_data_with_error_handling

# Composants
from src.components.sidebar import render_sidebar
from src.components.metrics import render_header_metrics

# Onglets
from src.tabs import (
    render_tab_overview,
    render_tab_distribution,
    render_tab_health,
    render_tab_demographics,
    render_tab_correlations,
    render_tab_expert,
    render_tab_modeling_prep,
    render_tab_prediction,
    render_tab_inside_model,
)

def render_synthesis(df_filtered):
    """
    Affiche la synthese et conclusions chiffrees.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtre pour calculer les stats dynamiques.
    """
    st.markdown("---")
    st.subheader("Synthese et Conclusions de l'EDA")

    # Calculs dynamiques
    total_obs = len(df_filtered)

    non_smoker_mean = df_filtered[
        df_filtered["smoker"] == "no"
    ]["charges"].mean()

    smoker_mean = df_filtered[
        df_filtered["smoker"] == "yes"
    ]["charges"].mean()

    smoker_obese = df_filtered[
        (df_filtered["smoker"] == "yes") & (df_filtered["bmi"] >= 30)
    ]["charges"].mean()

    non_smoker_obese = df_filtered[
        (df_filtered["smoker"] == "no") & (df_filtered["bmi"] >= 30)
    ]["charges"].mean()

    young_non_smoker = df_filtered[
        (df_filtered["age"] <= 35) & (df_filtered["smoker"] == "no")
    ]["charges"].mean()

    senior_non_smoker = df_filtered[
        (df_filtered["age"] >= 51) & (df_filtered["smoker"] == "no")
    ]["charges"].mean()

    young_smoker = df_filtered[
        (df_filtered["age"] <= 35) & (df_filtered["smoker"] == "yes")
    ]["charges"].mean()

    # Pourcentages
    pct_smokers = (df_filtered["smoker"] == "yes").mean() * 100
    pct_obese = (df_filtered["bmi"] >= 30).mean() * 100

    # Section 1: Chiffres cles
    st.markdown("### Chiffres Cles a Retenir")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Observations analysees",
            f"{total_obs:,}"
        )
    with col2:
        st.metric(
            "Charge moyenne",
            f"{df_filtered['charges'].mean():,.0f} $"
        )
    with col3:
        st.metric(
            "Taux de fumeurs",
            f"{pct_smokers:.1f}%"
        )
    with col4:
        st.metric(
            "Taux d'obesite",
            f"{pct_obese:.1f}%"
        )

    st.markdown("---")

    # Section 2: Conclusions chiffrees
    st.markdown("### Conclusions Chiffrees")

    expander = st.expander("Voir les conclusions detaillees de l'EDA", expanded=True)
    with expander:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Impact du Tabagisme")
            if smoker_mean > 0 and non_smoker_mean > 0:
                ratio_smoker = smoker_mean / non_smoker_mean
                diff_smoker = smoker_mean - non_smoker_mean

                st.write(f"- Non-fumeur moyen : **{non_smoker_mean:,.0f} $**")
                st.write(f"- Fumeur moyen : **{smoker_mean:,.0f} $**")
                st.write(f"- Difference : **+{diff_smoker:,.0f} $** (x{ratio_smoker:.1f})")
                st.success(
                    f"Un fumeur coute en moyenne **{ratio_smoker:.1f} fois plus** qu'un non-fumeur."
                )

            st.markdown("#### Synergie Obesite x Tabac")
            if smoker_obese > 0 and non_smoker_obese > 0:
                ratio_obese = smoker_obese / non_smoker_obese

                st.write(f"- Non-Fumeur obese (IMC >= 30) : **{non_smoker_obese:,.0f} $**")
                st.write(f"- Fumeur obese (IMC >= 30) : **{smoker_obese:,.0f} $**")
                st.write(f"- Ratio : **x{ratio_obese:.1f}**")
                st.warning(
                    f"L'obesite multiplie les couts par **{ratio_obese:.1f}** chez les fumeurs."
                )

        with c2:
            st.markdown("#### Impact de l'Age")
            if young_non_smoker > 0 and senior_non_smoker > 0:
                ratio_age = senior_non_smoker / young_non_smoker

                st.write(f"- Jeune non-fumeur (18-35) : **{young_non_smoker:,.0f} $**")
                st.write(f"- Senior non-fumeur (51+) : **{senior_non_smoker:,.0f} $**")
                st.write(f"- Ratio : **x{ratio_age:.1f}**")
                st.info(
                    f"Les charges augmentent de **{((ratio_age - 1) * 100):.0f}%** avec l'age."
                )

            st.markdown("#### Paradoxe Age vs Tabac")
            if young_smoker > 0 and senior_non_smoker > 0:
                ratio_paradox = young_smoker / senior_non_smoker

                st.write(f"- Jeune fumeur (18-35) : **{young_smoker:,.0f} $**")
                st.write(f"- Senior non-fumeur (51+) : **{senior_non_smoker:,.0f} $**")
                st.error(
                    f"Un jeune fumeur coute **{ratio_paradox:.1f}x plus** qu'un senior non-fumeur."
                )

    # Section 3: Resume pour la modelisation
    st.markdown("---")
    st.markdown("### Recommandations pour la Modelisation")

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.markdown(
            """
            **Variables Predictives Prioritaires :**
            1. `smoker` - Variable dominante (Eta carre > 0.6)
            2. `age` - Effet lineaire significatif (r = 0.30)
            3. `bmi` - Important en interaction avec smoker
            4. `smoker x bmi` - Interaction critique

            **Variables Secondaires :**
            - `children` - Faible pouvoir predictif
            - `sex` - Pas d'effet significatif
            - `region` - Effet marginal (Southeast legerement plus cher)
            """
        )

    with col_rec2:
        st.markdown(
            """
            **Transformations Recommandees :**
            1. **Interactions** : `bmi * smoker`, `is_obese_smoker` (Indispensable)
            2. **Terme quadratique** : `age^2`
            3. **Variable cible** : Pas de log necessaire avec les bons termes d'interaction
            
            **Metriques Obtenues (Semaine 2) :**
            - R2 avec interactions : **0.9324**
            - MAE cible : **< 2000 $**
            - RMSE cible : **< 3200 $**
            """
        )

    # Section 4: Qualite des donnees
    st.markdown("---")
    st.markdown("### Bilan Qualite des Donnees")

    missing = df_filtered.isnull().sum().sum()
    duplicates = df_filtered.duplicated().sum()
    skewness = df_filtered["charges"].skew()

    col_qual1, col_qual2, col_qual3 = st.columns(3)

    with col_qual1:
        if missing == 0:
            st.success("Valeurs manquantes : 0")
        else:
            st.warning(f"Valeurs manquantes : {missing}")

    with col_qual2:
        if duplicates == 0:
            st.success("Doublons : 0")
        else:
            st.warning(f"Doublons : {duplicates}")

    with col_qual3:
        if abs(skewness) > 1:
            st.warning(f"Skewness charges : {skewness:.2f} (transformation log recommandee)")
        else:
            st.success(f"Skewness charges : {skewness:.2f}")

    # Conclusion finale
    st.markdown("---")
    st.success(
        """
        **Conclusion Generale :**
        L'EDA revele que le **tabagisme** est le facteur dominant des couts d'assurance,
        avec un effet multiplie par l'obesite. Le dataset est de bonne qualite
        (pas de valeurs manquantes, pas de doublons). La transformation logarithmique
        de la variable cible et l'ajout de variables d'interaction seront essentiels
        pour obtenir un modele performant en Semaine 2.
        """
    )


def main():
    """Fonction principale de l'application."""
    # Configuration
    setup_page_config()
    apply_custom_css()

    # Chargement des donnees
    df = load_data_with_error_handling()

    # Sidebar avec filtres
    df_filtered = render_sidebar(df)

    # Header
    st.title("Analyse des Risques et Charges d'Assurance")
    st.markdown(
        f"**Exploration des Facteurs de Couts** | "
        f"{df_filtered.shape[0]} observations selectionnees"
    )

    # Metriques principales
    render_header_metrics(df_filtered)
    st.markdown("---")

    # Onglets principaux
    (
        tab_overview,
        tab_dist,
        tab_health,
        tab_demo,
        tab_corr,
        tab_expert,
        tab_modeling,
        tab_prediction,
        tab_inside
    ) = st.tabs([
        "Apercu des Donnees",
        "Distribution des Couts",
        "Facteurs de Sante",
        "Demographie et Profils",
        "Correlations et Stats",
        "Analyses Expert",
        "Preparation Modele",
        "Prediction",
        "Inside the Model"
    ])

    with tab_overview:
        render_tab_overview(df, df_filtered)

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

    with tab_modeling:
        render_tab_modeling_prep(df_filtered)

    with tab_prediction:
        render_tab_prediction()

    with tab_inside:
        render_tab_inside_model(df)

    # Synthese
    render_synthesis(df_filtered)

    # Footer
    st.write("---")
    st.caption("Dashboard Analytics Insurance v2.1 - Equipe Dev Data IA")


if __name__ == "__main__":
    main()