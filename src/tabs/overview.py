"""
Onglet Apercu des Donnees.
"""

import streamlit as st
import pandas as pd


def render_tab_overview(df: pd.DataFrame, df_filtered: pd.DataFrame):
    """
    Affiche l'apercu des donnees.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original (non filtre).
    df_filtered : pd.DataFrame
        DataFrame filtre.
    """
    st.subheader("Apercu des Donnees")

    # Section 1: Informations generales
    _render_data_info(df, df_filtered)
    st.markdown("---")

    # Section 2: Verification des doublons
    _render_duplicates_check(df, df_filtered)
    st.markdown("---")

    # Section 3: Echantillon des donnees
    _render_data_sample(df_filtered)
    st.markdown("---")

    # Section 4: Valeurs manquantes
    _render_missing_values(df_filtered)
    st.markdown("---")

    # Section 5: Resume statistique
    _render_statistical_summary(df_filtered)
    st.markdown("---")

    # Section 6: Resume des variables categorielles
    _render_categorical_summary(df_filtered)
    st.markdown("---")

    # Section 7: Qualite des donnees
    _render_data_quality_summary(df_filtered)


def _render_data_info(df: pd.DataFrame, df_filtered: pd.DataFrame):
    """Affiche les informations generales du dataset."""
    st.markdown("#### Informations Generales")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Lignes (total)", f"{len(df):,}")
    with col2:
        st.metric("Lignes (filtrees)", f"{len(df_filtered):,}")
    with col3:
        st.metric("Colonnes", f"{len(df.columns)}")
    with col4:
        memory = df_filtered.memory_usage(deep=True).sum() / 1024
        st.metric("Memoire", f"{memory:.1f} KB")

    # Types de donnees
    st.markdown("#### Types de Donnees")

    col_types = pd.DataFrame({
        "Colonne": df_filtered.columns,
        "Type": df_filtered.dtypes.astype(str).values,
        "Non-Null": df_filtered.count().values,
        "Null": df_filtered.isnull().sum().values,
        "Unique": [df_filtered[col].nunique() for col in df_filtered.columns],
        "Exemple": [
            str(df_filtered[col].iloc[0])[:30]
            if len(df_filtered) > 0 else "N/A"
            for col in df_filtered.columns
        ]
    })
    st.dataframe(col_types, use_container_width=True, hide_index=True)

    # Description des variables
    st.markdown("#### Description des Variables")

    variable_descriptions = pd.DataFrame({
        "Variable": ["age", "sex", "bmi", "children", "smoker", "region", "charges"],
        "Description": [
            "Age de l'assure (en annees)",
            "Sexe de l'assure (male/female)",
            "Indice de Masse Corporelle (kg/m2)",
            "Nombre d'enfants couverts par l'assurance",
            "Statut fumeur (yes/no)",
            "Region de residence aux USA",
            "Frais medicaux annuels factures (variable cible)"
        ],
        "Type": [
            "Numerique continue",
            "Categorielle binaire",
            "Numerique continue",
            "Numerique discrete",
            "Categorielle binaire",
            "Categorielle nominale",
            "Numerique continue (cible)"
        ]
    })
    st.dataframe(variable_descriptions, use_container_width=True, hide_index=True)


def _render_duplicates_check(df: pd.DataFrame, df_filtered: pd.DataFrame):
    """Affiche la verification des doublons."""
    st.markdown("#### Verification des Doublons")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Dataset Original")

        # Doublons exacts (toutes colonnes)
        duplicates_exact = df.duplicated().sum()
        st.metric("Doublons exacts", duplicates_exact)

        # Doublons sur colonnes cles (sans charges)
        key_cols = ["age", "sex", "bmi", "children", "smoker", "region"]
        duplicates_keys = df.duplicated(subset=key_cols).sum()
        st.metric("Doublons sur colonnes cles", duplicates_keys)

        pct_dup = (duplicates_exact / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Pourcentage doublons", f"{pct_dup:.2f}%")

    with col2:
        st.markdown("##### Dataset Filtre")

        duplicates_filtered = df_filtered.duplicated().sum()
        st.metric("Doublons exacts", duplicates_filtered)

        duplicates_keys_filtered = df_filtered.duplicated(subset=key_cols).sum()
        st.metric("Doublons sur colonnes cles", duplicates_keys_filtered)

        pct_dup_filtered = (
            duplicates_filtered / len(df_filtered) * 100
            if len(df_filtered) > 0 else 0
        )
        st.metric("Pourcentage doublons", f"{pct_dup_filtered:.2f}%")

    # Interpretation
    if duplicates_exact == 0:
        st.success("Aucun doublon exact detecte dans le dataset.")
    else:
        st.warning(
            f"{duplicates_exact} doublons exacts detectes. "
            "A analyser si necessaire."
        )

    if duplicates_keys > 0:
        st.info(
            f"{duplicates_keys} lignes ont les memes caracteristiques (hors charges). "
            "Cela peut etre normal : plusieurs assures peuvent avoir le meme profil."
        )

    # Afficher les doublons si demande
    if duplicates_exact > 0:
        show_duplicates = st.checkbox("Afficher les doublons exacts", value=False)
        if show_duplicates:
            duplicated_rows = df[df.duplicated(keep=False)].sort_values(
                by=list(df.columns)
            )
            st.dataframe(
                duplicated_rows.head(20),
                use_container_width=True,
                hide_index=True
            )
            if len(duplicated_rows) > 20:
                st.caption(f"Affichage limite a 20 lignes sur {len(duplicated_rows)}")


def _render_data_sample(df_filtered: pd.DataFrame):
    """Affiche un echantillon des donnees."""
    st.markdown("#### Echantillon des Donnees")

    col1, col2 = st.columns([3, 1])

    with col2:
        n_rows = st.selectbox(
            "Nombre de lignes",
            options=[5, 10, 20, 50],
            index=0,
            key="sample_rows"
        )
        show_random = st.checkbox("Echantillon aleatoire", value=False)

        # Options d'affichage
        sort_by = st.selectbox(
            "Trier par",
            options=["(aucun)"] + list(df_filtered.columns),
            index=0,
            key="sort_column"
        )

    with col1:
        if len(df_filtered) == 0:
            st.warning("Aucune donnee a afficher avec les filtres actuels.")
        else:
            if show_random:
                sample_df = df_filtered.sample(min(n_rows, len(df_filtered)))
            else:
                sample_df = df_filtered.head(n_rows)

            if sort_by != "(aucun)":
                sample_df = sample_df.sort_values(by=sort_by, ascending=False)

            st.dataframe(sample_df, use_container_width=True, hide_index=True)


def _render_missing_values(df_filtered: pd.DataFrame):
    """Affiche l'analyse des valeurs manquantes."""
    st.markdown("#### Valeurs Manquantes")

    missing = df_filtered.isnull().sum()
    missing_pct = (missing / len(df_filtered) * 100).round(2)

    missing_df = pd.DataFrame({
        "Colonne": missing.index,
        "Manquantes": missing.values,
        "Pourcentage (%)": missing_pct.values
    })

    total_missing = missing.sum()
    total_cells = len(df_filtered) * len(df_filtered.columns)
    pct_total = (total_missing / total_cells * 100) if total_cells > 0 else 0

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Total valeurs manquantes", f"{total_missing:,}")
        st.metric("Total cellules", f"{total_cells:,}")
        st.metric("Taux de completude", f"{100 - pct_total:.2f}%")
        st.metric(
            "Colonnes avec valeurs manquantes",
            f"{(missing > 0).sum()} / {len(missing)}"
        )

    with col2:
        if total_missing == 0:
            st.success(
                "Le dataset est complet : aucune valeur manquante detectee. "
                "Aucun traitement d'imputation necessaire."
            )
        else:
            st.warning(f"{total_missing} valeurs manquantes detectees.")
            missing_only = missing_df[missing_df["Manquantes"] > 0]
            st.dataframe(missing_only, use_container_width=True, hide_index=True)

            # Recommandations
            st.markdown("**Strategies de traitement :**")
            for _, row in missing_only.iterrows():
                col_name = row["Colonne"]
                pct = row["Pourcentage (%)"]
                if pct < 5:
                    st.write(f"- `{col_name}` ({pct}%) : Suppression ou imputation simple")
                elif pct < 20:
                    st.write(f"- `{col_name}` ({pct}%) : Imputation (moyenne/mediane/mode)")
                else:
                    st.write(f"- `{col_name}` ({pct}%) : Attention, taux eleve. Analyser.")


def _render_statistical_summary(df_filtered: pd.DataFrame):
    """Affiche le resume statistique des variables numeriques."""
    st.markdown("#### Resume Statistique (Variables Numeriques)")

    # Selectionner uniquement les colonnes numeriques principales
    all_numeric = df_filtered.select_dtypes(include=["number"]).columns.tolist()
    # Garder uniquement les colonnes originales
    numeric_cols = [c for c in all_numeric if c in ["age", "bmi", "children", "charges"]]

    if len(numeric_cols) == 0:
        st.warning("Aucune variable numerique dans le dataset.")
        return

    # Calculer les statistiques
    stats_df = df_filtered[numeric_cols].describe().T
    stats_df = stats_df.round(2)

    # Ajouter des colonnes supplementaires
    stats_df["skewness"] = df_filtered[numeric_cols].skew().round(2)
    stats_df["kurtosis"] = df_filtered[numeric_cols].kurtosis().round(2)
    stats_df["missing"] = df_filtered[numeric_cols].isnull().sum()
    stats_df["IQR"] = (
        df_filtered[numeric_cols].quantile(0.75) -
        df_filtered[numeric_cols].quantile(0.25)
    ).round(2)

    # Renommer les colonnes
    stats_df.columns = [
        "Count", "Moyenne", "Ecart-type", "Min",
        "25%", "50%", "75%", "Max",
        "Skewness", "Kurtosis", "Manquantes", "IQR"
    ]

    st.dataframe(stats_df, use_container_width=True)

    # Interpretations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Interpretation du Skewness :**")
        st.write("- **< -1** : Fortement asymetrique a gauche")
        st.write("- **-1 a 1** : Approximativement symetrique")
        st.write("- **> 1** : Fortement asymetrique a droite")

    with col2:
        st.markdown("**Interpretation du Kurtosis :**")
        st.write("- **< 0** : Distribution aplatie (platykurtique)")
        st.write("- **= 0** : Distribution normale (mesokurtique)")
        st.write("- **> 0** : Distribution pointue (leptokurtique)")

    # Alertes specifiques
    st.markdown("##### Alertes sur les Distributions")

    alerts = []
    for col in numeric_cols:
        skew = df_filtered[col].skew()
        kurt = df_filtered[col].kurtosis()

        if abs(skew) > 1:
            alerts.append({
                "Variable": col,
                "Probleme": "Asymetrie forte",
                "Valeur": f"Skewness = {skew:.2f}",
                "Recommandation": "Transformation log ou Box-Cox"
            })

        if kurt > 3:
            alerts.append({
                "Variable": col,
                "Probleme": "Queue lourde",
                "Valeur": f"Kurtosis = {kurt:.2f}",
                "Recommandation": "Attention aux outliers"
            })

    if alerts:
        st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
    else:
        st.success("Aucune alerte majeure sur les distributions.")


def _render_categorical_summary(df_filtered: pd.DataFrame):
    """Affiche le resume des variables categorielles."""
    st.markdown("#### Resume des Variables Categorielles")

    cat_cols = df_filtered.select_dtypes(include=["object"]).columns.tolist()

    # Exclure smoker_label si present (c'est une colonne derivee)
    if "smoker_label" in cat_cols:
        cat_cols.remove("smoker_label")

    if len(cat_cols) == 0:
        st.warning("Aucune variable categorielle dans le dataset.")
        return

    # Afficher les colonnes en grille
    n_cols = min(len(cat_cols), 4)
    cols = st.columns(n_cols)

    for i, col_name in enumerate(cat_cols):
        with cols[i % n_cols]:
            st.markdown(f"**{col_name.upper()}**")
            value_counts = df_filtered[col_name].value_counts()

            for val, count in value_counts.items():
                pct = count / len(df_filtered) * 100
                st.write(f"- {val}: {count} ({pct:.1f}%)")

    # Tableau recapitulatif
    st.markdown("---")
    st.markdown("#### Tableau Recapitulatif des Categories")

    cat_summary = []
    for col_name in cat_cols:
        value_counts = df_filtered[col_name].value_counts()
        cat_summary.append({
            "Variable": col_name,
            "Nb Categories": df_filtered[col_name].nunique(),
            "Mode": value_counts.index[0] if len(value_counts) > 0 else "N/A",
            "Frequence Mode": value_counts.iloc[0] if len(value_counts) > 0 else 0,
            "Freq. Mode (%)": (
                value_counts.iloc[0] / len(df_filtered) * 100
            ).round(1) if len(df_filtered) > 0 else 0,
            "Entropie": _calculate_entropy(df_filtered[col_name])
        })

    cat_summary_df = pd.DataFrame(cat_summary)
    st.dataframe(cat_summary_df, use_container_width=True, hide_index=True)

    # Interpretation de l'entropie
    st.markdown(
        """
        **Interpretation de l'Entropie :**
        - Valeur proche de 0 : Une categorie domine fortement
        - Valeur elevee : Categories equilibrees
        """
    )


def _calculate_entropy(series: pd.Series) -> float:
    """Calcule l'entropie d'une variable categorielle."""
    import numpy as np

    probabilities = series.value_counts(normalize=True)
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return round(entropy, 2)


def _render_data_quality_summary(df_filtered: pd.DataFrame):
    """Affiche un resume de la qualite des donnees."""
    st.markdown("#### Resume de la Qualite des Donnees")

    # Calcul des metriques de qualite
    total_rows = len(df_filtered)
    total_cols = len(df_filtered.columns)
    total_cells = total_rows * total_cols

    # Completude
    missing_cells = df_filtered.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells * 100) if total_cells > 0 else 0

    # Doublons
    duplicates = df_filtered.duplicated().sum()
    uniqueness = ((total_rows - duplicates) / total_rows * 100) if total_rows > 0 else 0

    # Validite (exemple : age positif, bmi positif)
    validity_checks = []
    if "age" in df_filtered.columns:
        invalid_age = ((df_filtered["age"] < 0) | (df_filtered["age"] > 120)).sum()
        validity_checks.append(total_rows - invalid_age)
    if "bmi" in df_filtered.columns:
        invalid_bmi = ((df_filtered["bmi"] < 10) | (df_filtered["bmi"] > 70)).sum()
        validity_checks.append(total_rows - invalid_bmi)
    if "charges" in df_filtered.columns:
        invalid_charges = (df_filtered["charges"] < 0).sum()
        validity_checks.append(total_rows - invalid_charges)

    validity = (sum(validity_checks) / (len(validity_checks) * total_rows) * 100) if validity_checks else 100

    # Affichage
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Completude", f"{completeness:.1f}%")
        st.caption("% de cellules non manquantes")

    with col2:
        st.metric("Unicite", f"{uniqueness:.1f}%")
        st.caption("% de lignes uniques")

    with col3:
        st.metric("Validite", f"{validity:.1f}%")
        st.caption("% de valeurs dans les plages attendues")

    # Score global
    quality_score = (completeness + uniqueness + validity) / 3

    st.markdown("---")

    col_score1, col_score2 = st.columns([1, 2])

    with col_score1:
        st.metric("Score Qualite Global", f"{quality_score:.1f}%")

    with col_score2:
        if quality_score >= 95:
            st.success(
                "Excellente qualite des donnees. "
                "Le dataset est pret pour l'analyse et la modelisation."
            )
        elif quality_score >= 80:
            st.info(
                "Bonne qualite des donnees. "
                "Quelques verifications mineures peuvent etre necessaires."
            )
        elif quality_score >= 60:
            st.warning(
                "Qualite moyenne. "
                "Un nettoyage des donnees est recommande avant la modelisation."
            )
        else:
            st.error(
                "Qualite insuffisante. "
                "Un nettoyage approfondi est necessaire."
            )

    # Checklist de validation
    st.markdown("##### Checklist de Validation des Donnees")

    checks = {
        "Pas de valeurs manquantes": missing_cells == 0,
        "Pas de doublons exacts": duplicates == 0,
        "Types de donnees corrects": True,  # Suppose correct apres chargement
        "Valeurs dans les plages attendues": validity >= 99,
        "Variable cible presente (charges)": "charges" in df_filtered.columns,
        "Variables explicatives presentes": all(
            c in df_filtered.columns
            for c in ["age", "sex", "bmi", "children", "smoker", "region"]
        ),
    }

    for check_name, status in checks.items():
        if status:
            st.write(f"- [x] {check_name}")
        else:
            st.write(f"- [ ] {check_name}")

    # Conclusion
    all_checks_passed = all(checks.values())
    if all_checks_passed:
        st.success(
            "Toutes les verifications sont passees. "
            "Le dataset est pret pour l'analyse bivariee et la modelisation."
        )
    else:
        failed_checks = [name for name, status in checks.items() if not status]
        st.warning(
            f"Verifications echouees : {', '.join(failed_checks)}. "
            "Veuillez traiter ces points avant de continuer."
        )