"""
Onglet Distribution des Couts.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import COLOR_MAP_SMOKER


def render_tab_distribution(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet de distribution des couts.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtre.
    """
    st.subheader("Analyse de la Variable Cible : Charges")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        _render_histogram(df_filtered)

    with col_right:
        _render_insights()
        _render_boxplot_smoker(df_filtered)

    st.markdown("---")

    # Nouvelles sections : distributions des variables numeriques
    _render_numerical_distributions(df_filtered)

    st.markdown("---")

    # Repartition des variables categorielles
    _render_categorical_distribution(df_filtered)


def _render_histogram(df_filtered: pd.DataFrame):
    """Affiche l'histogramme des charges."""
    use_log = st.checkbox(
        "Appliquer l'echelle logarithmique (Log scale)",
        help="Aide a visualiser les distributions asymetriques"
    )

    fig_hist = px.histogram(
        df_filtered,
        x="charges",
        nbins=50,
        color_discrete_sequence=["#3b82f6"],
        marginal="box",
        log_x=use_log,
        title="Distribution des charges medicales"
    )
    fig_hist.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)


def _render_insights():
    """Affiche les insights metier."""
    st.info(
        """
        **Observation Metier :**
        - La distribution est fortement **asymetrique a droite**
          (skewness positive).
        - La majorite des dossiers sont sous les 15k$, mais une
          minorite genere des couts tres eleves (>40k$).
        - **Impact IA :** Une transformation Log sera probablement
          necessaire pour ameliorer la performance de la regression.
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
        title="Repartition des charges par statut fumeur",
        color_discrete_map=COLOR_MAP_SMOKER
    )
    st.plotly_chart(fig_box_smoker, use_container_width=True)


def _render_numerical_distributions(df_filtered: pd.DataFrame):
    """Affiche les distributions des variables numeriques."""
    st.markdown("#### Distribution des Variables Numeriques")

    col1, col2 = st.columns(2)

    with col1:
        # Histogramme de l'age
        fig_age = px.histogram(
            df_filtered,
            x="age",
            nbins=30,
            color_discrete_sequence=["#3b82f6"],
            title="Distribution de l'Age",
            labels={"age": "Age", "count": "Nombre"}
        )
        fig_age.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_age, use_container_width=True)

        # Statistiques age
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        col_stat1.metric("Min", f"{df_filtered['age'].min()} ans")
        col_stat2.metric("Moyenne", f"{df_filtered['age'].mean():.1f} ans")
        col_stat3.metric("Max", f"{df_filtered['age'].max()} ans")

    with col2:
        # Histogramme du BMI
        fig_bmi = px.histogram(
            df_filtered,
            x="bmi",
            nbins=30,
            color_discrete_sequence=["#10b981"],
            title="Distribution de l'IMC",
            labels={"bmi": "IMC", "count": "Nombre"}
        )
        fig_bmi.add_vline(
            x=18.5,
            line_dash="dash",
            line_color="orange",
            annotation_text="Sous-poids"
        )
        fig_bmi.add_vline(
            x=25,
            line_dash="dash",
            line_color="orange",
            annotation_text="Surpoids"
        )
        fig_bmi.add_vline(
            x=30,
            line_dash="dash",
            line_color="red",
            annotation_text="Obesite"
        )
        fig_bmi.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_bmi, use_container_width=True)

        # Statistiques BMI
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        col_stat1.metric("Min", f"{df_filtered['bmi'].min():.1f}")
        col_stat2.metric("Moyenne", f"{df_filtered['bmi'].mean():.1f}")
        col_stat3.metric("Max", f"{df_filtered['bmi'].max():.1f}")

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        # Bar chart du nombre d'enfants
        children_counts = df_filtered["children"].value_counts().sort_index()
        fig_children = px.bar(
            x=children_counts.index,
            y=children_counts.values,
            title="Distribution du Nombre d'Enfants",
            labels={"x": "Nombre d'enfants", "y": "Nombre d'assures"},
            color_discrete_sequence=["#8b5cf6"]
        )
        fig_children.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_children, use_container_width=True)

        # Pourcentages enfants
        st.markdown("**Repartition :**")
        for n_children, count in children_counts.items():
            pct = count / len(df_filtered) * 100
            st.write(f"- {n_children} enfant(s) : {count} ({pct:.1f}%)")

    with col4:
        # Bar chart fumeur / non-fumeur
        smoker_counts = df_filtered["smoker"].value_counts()
        fig_smoker = px.bar(
            x=smoker_counts.index.map({"yes": "Fumeur", "no": "Non-fumeur"}),
            y=smoker_counts.values,
            title="Repartition Fumeur / Non-Fumeur",
            labels={"x": "Statut", "y": "Nombre d'assures"},
            color=smoker_counts.index,
            color_discrete_map={"yes": "#ef4444", "no": "#10b981"}
        )
        fig_smoker.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_smoker, use_container_width=True)

        # Pourcentages fumeur
        st.markdown("**Repartition :**")
        for status, count in smoker_counts.items():
            pct = count / len(df_filtered) * 100
            label = "Fumeur" if status == "yes" else "Non-fumeur"
            st.write(f"- {label} : {count} ({pct:.1f}%)")


def _render_categorical_distribution(df_filtered: pd.DataFrame):
    """Affiche la repartition des variables categorielles."""
    st.markdown("#### Repartition par Sexe et Region")

    col_cat1, col_cat2 = st.columns(2)

    with col_cat1:
        # Pie chart sexe
        fig_sex_dist = px.pie(
            df_filtered,
            names="sex",
            title="Repartition par Sexe",
            hole=0.3,
            color="sex",
            color_discrete_map={"male": "#3b82f6", "female": "#ec4899"}
        )
        fig_sex_dist.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_sex_dist, use_container_width=True)

        # Stats sexe
        sex_counts = df_filtered["sex"].value_counts()
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Hommes", f"{sex_counts.get('male', 0)}")
        col_s2.metric("Femmes", f"{sex_counts.get('female', 0)}")

    with col_cat2:
        # Pie chart region
        fig_region_dist = px.pie(
            df_filtered,
            names="region",
            title="Repartition par Region",
            hole=0.3
        )
        fig_region_dist.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_region_dist, use_container_width=True)

        # Stats region
        region_counts = df_filtered["region"].value_counts()
        cols = st.columns(4)
        for i, (region, count) in enumerate(region_counts.items()):
            cols[i].metric(region.capitalize(), f"{count}")

    st.markdown("---")

    # Bar chart comparatif des regions
    st.markdown("#### Comparaison des Regions")

    region_stats = df_filtered.groupby("region").agg({
        "charges": "mean",
        "age": "mean",
        "bmi": "mean"
    }).round(2).reset_index()

    fig_region_bar = px.bar(
        region_stats,
        x="region",
        y="charges",
        title="Charges Moyennes par Region",
        labels={"region": "Region", "charges": "Charges Moyennes ($)"},
        color="charges",
        color_continuous_scale="Blues"
    )
    fig_region_bar.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_region_bar, use_container_width=True)