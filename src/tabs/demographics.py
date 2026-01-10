"""
Onglet Demographie et Profils.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import COLOR_MAP_SMOKER


def render_tab_demographics(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet demographie.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtre.
    """
    st.subheader("Analyse Demographique Approfondie")

    # Section 1: Analyse par tranche d'age (NOUVEAU)
    _render_age_section(df_filtered)
    st.markdown("---")

    # Section 2: Distribution par region
    _render_region_section(df_filtered)
    st.markdown("---")

    # Section 3: Comparaison homme/femme
    _render_sex_section(df_filtered)
    st.markdown("---")

    # Section 4: Impact du nombre d'enfants
    _render_children_section(df_filtered)


def _render_age_section(df_filtered: pd.DataFrame):
    """Section analyse par tranche d'age."""
    st.markdown("#### Charges par Tranche d'Age")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Creer les tranches d'age
        df_temp = df_filtered.copy()
        df_temp["age_group"] = pd.cut(
            df_temp["age"],
            bins=[17, 25, 35, 45, 55, 65],
            labels=["18-25", "26-35", "36-45", "46-55", "56-64"]
        )

        # Charges moyennes par tranche d'age et statut fumeur
        age_stats = (
            df_temp
            .groupby(["age_group", "smoker_label"])["charges"]
            .mean()
            .reset_index()
        )

        fig_age = px.bar(
            age_stats,
            x="age_group",
            y="charges",
            color="smoker_label",
            barmode="group",
            title="Charges Moyennes par Tranche d'Age",
            labels={
                "age_group": "Tranche d'Age",
                "charges": "Charges Moyennes ($)",
                "smoker_label": "Statut"
            },
            color_discrete_map=COLOR_MAP_SMOKER
        )
        fig_age.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Tranche d'Age",
            yaxis_title="Charges Moyennes ($)"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col2:
        st.info(
            """
            **Observation :**
            - Les charges augmentent avec l'age
            - L'ecart fumeur/non-fumeur reste constant
            - Un fumeur de 25 ans coute plus qu'un non-fumeur de 55 ans
            """
        )

        # Tableau recapitulatif par tranche d'age
        age_summary = (
            df_temp
            .groupby("age_group")
            .agg({
                "charges": ["mean", "median", "count"]
            })
            .round(0)
        )
        age_summary.columns = ["Moyenne", "Mediane", "Effectif"]
        age_summary = age_summary.reset_index()
        age_summary.columns = ["Tranche", "Moyenne ($)", "Mediane ($)", "Effectif"]

        st.markdown("**Resume par tranche :**")
        st.dataframe(age_summary, use_container_width=True, hide_index=True)

    # Line chart evolution
    st.markdown("##### Evolution des Charges selon l'Age")

    col3, col4 = st.columns([2, 1])

    with col3:
        fig_line = px.line(
            age_stats,
            x="age_group",
            y="charges",
            color="smoker_label",
            markers=True,
            title="Evolution des Charges Moyennes avec l'Age",
            labels={
                "age_group": "Tranche d'Age",
                "charges": "Charges Moyennes ($)",
                "smoker_label": "Statut"
            },
            color_discrete_map=COLOR_MAP_SMOKER
        )
        fig_line.update_layout(
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col4:
        # Calcul de l'ecart jeune fumeur vs senior non-fumeur
        young_smoker = df_temp[
            (df_temp["age"] <= 35) & (df_temp["smoker"] == "yes")
        ]["charges"].mean()

        senior_nonsmoker = df_temp[
            (df_temp["age"] >= 51) & (df_temp["smoker"] == "no")
        ]["charges"].mean()

        if pd.notna(young_smoker) and pd.notna(senior_nonsmoker):
            st.markdown("**Comparaison cle :**")
            st.metric(
                "Jeune Fumeur (<=35 ans)",
                f"{young_smoker:,.0f} $"
            )
            st.metric(
                "Senior Non-Fumeur (>=51 ans)",
                f"{senior_nonsmoker:,.0f} $"
            )

            ratio = young_smoker / senior_nonsmoker
            st.warning(
                f"Un jeune fumeur coute **{ratio:.1f}x** plus "
                f"qu'un senior non-fumeur."
            )


def _render_region_section(df_filtered: pd.DataFrame):
    """Section distribution par region."""
    st.markdown("#### Distribution des Charges par Region")

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        fig_region = px.box(
            df_filtered,
            x="region",
            y="charges",
            color="smoker_label",
            title="Charges par region et statut fumeur",
            points="outliers",
            color_discrete_map=COLOR_MAP_SMOKER
        )
        fig_region.update_layout(
            xaxis_title="Region",
            yaxis_title="Charges ($)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with col_r2:
        st.info(
            """
            **Lecture du Box Plot :**
            - **Boite** : 50% des donnees (Q1 a Q3)
            - **Ligne mediane** : Valeur centrale
            - **Moustaches** : Etendue normale
            - **Points** : Valeurs extremes (outliers)

            **Insight :**
            Les fumeurs ont des charges plus elevees
            dans **toutes** les regions.
            """
        )

        region_stats = (
            df_filtered
            .groupby("region")["charges"]
            .agg(["mean", "median"])
            .round(0)
        )
        st.markdown("**Moyennes par region :**")
        for region, row in region_stats.iterrows():
            st.metric(
                label=region.capitalize(),
                value=f"{row['mean']:,.0f} $",
                delta=f"Mediane: {row['median']:,.0f} $"
            )

    # Ajout : Taux de fumeurs par region
    st.markdown("##### Profil des Regions")

    col_r3, col_r4 = st.columns(2)

    with col_r3:
        # Taux de fumeurs par region
        smoker_rate = (
            df_filtered
            .groupby("region")["smoker"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reset_index()
        )
        smoker_rate.columns = ["region", "taux_fumeur"]

        fig_smoker_region = px.bar(
            smoker_rate,
            x="region",
            y="taux_fumeur",
            title="Taux de Fumeurs par Region (%)",
            labels={"region": "Region", "taux_fumeur": "Taux de fumeurs (%)"},
            color="taux_fumeur",
            color_continuous_scale="Reds"
        )
        fig_smoker_region.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_smoker_region, use_container_width=True)

    with col_r4:
        # Age moyen et BMI moyen par region
        region_profile = (
            df_filtered
            .groupby("region")
            .agg({
                "age": "mean",
                "bmi": "mean",
                "charges": "mean"
            })
            .round(1)
            .reset_index()
        )
        region_profile.columns = ["Region", "Age Moyen", "IMC Moyen", "Charges Moy."]

        st.markdown("**Profil moyen par region :**")
        st.dataframe(region_profile, use_container_width=True, hide_index=True)

        # Region la plus chere
        max_region = region_profile.loc[
            region_profile["Charges Moy."].idxmax(), "Region"
        ]
        max_charges = region_profile["Charges Moy."].max()

        st.warning(
            f"La region **{max_region}** est la plus couteuse "
            f"avec une moyenne de **{max_charges:,.0f} $**."
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
        fig_sex.update_layout(
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    with col_s2:
        st.success(
            """
            **Observation :**
            - Peu de difference entre hommes et femmes
            - Le **statut fumeur** est le facteur dominant
            - L'ecart fumeur/non-fumeur est similaire pour les deux sexes
            """
        )

        _render_sex_ecart_metric(avg_by_sex_smoker)

    # Ajout : Boxplot comparatif homme/femme
    st.markdown("##### Distribution detaillee par Sexe")

    col_s3, col_s4 = st.columns(2)

    with col_s3:
        fig_box_sex = px.box(
            df_filtered,
            x="sex",
            y="charges",
            color="sex",
            title="Distribution des Charges par Sexe",
            points="outliers",
            color_discrete_map={"male": "#3b82f6", "female": "#ec4899"}
        )
        fig_box_sex.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Sexe",
            yaxis_title="Charges ($)"
        )
        st.plotly_chart(fig_box_sex, use_container_width=True)

    with col_s4:
        # Tableau comparatif
        sex_stats = (
            df_filtered
            .groupby("sex")
            .agg({
                "charges": ["mean", "median", "std"],
                "age": "mean",
                "bmi": "mean"
            })
            .round(1)
        )
        sex_stats.columns = [
            "Charges Moy.", "Charges Med.", "Ecart-type",
            "Age Moy.", "IMC Moy."
        ]
        sex_stats = sex_stats.reset_index()
        sex_stats["sex"] = sex_stats["sex"].map({
            "male": "Homme",
            "female": "Femme"
        })
        sex_stats.columns = ["Sexe"] + list(sex_stats.columns[1:])

        st.markdown("**Statistiques par sexe :**")
        st.dataframe(sex_stats, use_container_width=True, hide_index=True)

        # Difference en pourcentage
        male_mean = df_filtered[df_filtered["sex"] == "male"]["charges"].mean()
        female_mean = df_filtered[df_filtered["sex"] == "female"]["charges"].mean()
        diff_pct = ((male_mean - female_mean) / female_mean) * 100

        if diff_pct > 0:
            st.info(
                f"Les hommes coutent **{diff_pct:.1f}%** de plus que les femmes."
            )
        else:
            st.info(
                f"Les femmes coutent **{abs(diff_pct):.1f}%** de plus que les hommes."
            )


def _render_sex_ecart_metric(avg_by_sex_smoker: pd.DataFrame):
    """Affiche la metrique d'ecart homme fumeur/non-fumeur."""
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
            "Ecart Fumeur/Non-fumeur (Homme)",
            f"+{ecart:,.0f} $",
            delta=f"{pct_ecart:.0f}%"
        )

    # Ajout : meme calcul pour les femmes
    female_smoker_df = avg_by_sex_smoker[
        (avg_by_sex_smoker["sex"] == "female")
        & (avg_by_sex_smoker["smoker_label"] == "Fumeur")
    ]
    female_nonsmoker_df = avg_by_sex_smoker[
        (avg_by_sex_smoker["sex"] == "female")
        & (avg_by_sex_smoker["smoker_label"] == "Non-fumeur")
    ]

    if not female_smoker_df.empty and not female_nonsmoker_df.empty:
        female_smoker = female_smoker_df["charges"].values[0]
        female_nonsmoker = female_nonsmoker_df["charges"].values[0]
        ecart_f = female_smoker - female_nonsmoker
        pct_ecart_f = ((female_smoker / female_nonsmoker) - 1) * 100

        st.metric(
            "Ecart Fumeur/Non-fumeur (Femme)",
            f"+{ecart_f:,.0f} $",
            delta=f"{pct_ecart_f:.0f}%"
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
            yaxis_title="Charges ($)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_children, use_container_width=True)

    with col_c2:
        st.warning(
            """
            **Tendance :**
            - Impact **modere** du nombre d'enfants
            - Augmentation legere avec plus d'enfants
            - Toujours domine par le statut fumeur

            **Note :** La plupart des assures ont 0-2 enfants.
            """
        )

        children_dist = df_filtered["children"].value_counts().sort_index()
        st.markdown("**Repartition :**")
        for n_children, count in children_dist.items():
            pct = (count / len(df_filtered)) * 100
            st.write(f"- {n_children} enfant(s) : {count} ({pct:.1f}%)")

    # Ajout : Bar chart charges moyennes par nombre d'enfants
    st.markdown("##### Charges Moyennes par Nombre d'Enfants")

    col_c3, col_c4 = st.columns([2, 1])

    with col_c3:
        children_charges = (
            df_filtered
            .groupby(["children", "smoker_label"])["charges"]
            .mean()
            .reset_index()
        )

        fig_children_bar = px.bar(
            children_charges,
            x="children",
            y="charges",
            color="smoker_label",
            barmode="group",
            title="Charges Moyennes par Nombre d'Enfants",
            labels={
                "children": "Nombre d'enfants",
                "charges": "Charges Moyennes ($)",
                "smoker_label": "Statut"
            },
            color_discrete_map=COLOR_MAP_SMOKER
        )
        fig_children_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_children_bar, use_container_width=True)

    with col_c4:
        # Tableau recapitulatif
        children_summary = (
            df_filtered
            .groupby("children")
            .agg({
                "charges": ["mean", "median"],
                "age": "mean"
            })
            .round(0)
        )
        children_summary.columns = ["Charges Moy.", "Charges Med.", "Age Moy."]
        children_summary = children_summary.reset_index()

        st.markdown("**Resume par nombre d'enfants :**")
        st.dataframe(children_summary, use_container_width=True, hide_index=True)

        # Correlation enfants-charges
        corr_children = df_filtered["children"].corr(df_filtered["charges"])
        st.metric(
            "Correlation Enfants-Charges",
            f"{corr_children:.3f}"
        )
        st.caption("Correlation faible : le nombre d'enfants a peu d'impact direct.")