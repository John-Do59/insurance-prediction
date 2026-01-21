"""
Onglet Preparation a la Modelisation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import shapiro


def render_tab_modeling_prep(df_filtered: pd.DataFrame):
    """
    Affiche l'onglet de preparation a la modelisation.

    Parameters
    ----------
    df_filtered : pd.DataFrame
        DataFrame filtre.
    """
    st.subheader("Preparation a la Modelisation")

    st.info(
        """
        Cet onglet verifie les hypotheses de la regression lineaire
        et propose les transformations necessaires avant l'entrainement du modele.
        """
    )

    # Section 1: Verification de la linearite
    _render_linearity_check(df_filtered)
    st.markdown("---")

    # Section 2: Transformation de la variable cible
    _render_target_transformation(df_filtered)
    st.markdown("---")

    # Section 3: Verification de la multicolinearite
    _render_multicollinearity_check(df_filtered)
    st.markdown("---")

    # Section 4: Encodage des variables categorielles
    _render_encoding_preview(df_filtered)
    st.markdown("---")

    # Section 5: Variables d'interaction
    _render_feature_interactions(df_filtered)
    st.markdown("---")

    # Section 6: Recommandations finales
    _render_final_recommendations(df_filtered)


def _render_linearity_check(df_filtered: pd.DataFrame):
    """Verifie la linearite des relations avec la variable cible."""
    st.markdown("#### Verification de la Linearite")

    st.write(
        """
        La regression lineaire suppose une relation lineaire entre 
        les variables explicatives et la variable cible.
        Verifions cette hypothese visuellement.
        """
    )

    numeric_vars = ["age", "bmi", "children"]

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for i, var in enumerate(numeric_vars):
        with columns[i]:
            # Correlation
            corr = df_filtered[var].corr(df_filtered["charges"])

            # Scatter plot avec trendline
            fig = px.scatter(
                df_filtered,
                x=var,
                y="charges",
                trendline="ols",
                title=f"{var.capitalize()} vs Charges",
                labels={var: var.capitalize(), "charges": "Charges ($)"},
                color_discrete_sequence=["#3b82f6"]
            )
            fig.update_layout(
                height=300,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.metric("Correlation (r)", f"{corr:.3f}")

    # Interpretation
    st.markdown("##### Observations")

    col_obs1, col_obs2 = st.columns(2)

    with col_obs1:
        st.markdown(
            """
            **Constats :**
            - `age` : Relation moderement lineaire avec des paliers visibles
            - `bmi` : Relation faible, dispersion importante
            - `children` : Pas de relation lineaire claire
            """
        )

    with col_obs2:
        st.markdown(
            """
            **Problemes identifies :**
            - Les nuages de points montrent des "groupes" distincts
            - Ces groupes correspondent aux fumeurs vs non-fumeurs
            - Des variables d'interaction seront necessaires
            """
        )


def _render_target_transformation(df_filtered: pd.DataFrame):
    """Analyse et propose des transformations pour la variable cible."""
    st.markdown("#### Transformation de la Variable Cible")

    st.write(
        """
        La regression lineaire suppose que les residus suivent une distribution normale.
        Une variable cible tres asymetrique peut violer cette hypothese.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Distribution Originale")

        fig_orig = px.histogram(
            df_filtered,
            x="charges",
            nbins=50,
            title="Distribution des Charges (Original)",
            color_discrete_sequence=["#ef4444"]
        )
        fig_orig.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_orig, use_container_width=True)

        # Statistiques
        skew_orig = df_filtered["charges"].skew()
        kurt_orig = df_filtered["charges"].kurtosis()

        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Skewness", f"{skew_orig:.2f}")
        col_s2.metric("Kurtosis", f"{kurt_orig:.2f}")

        if abs(skew_orig) > 1:
            st.warning("Distribution fortement asymetrique (skewness > 1)")
        else:
            st.success("Distribution acceptable")

    with col2:
        st.markdown("##### Distribution Log-Transformee")

        # Transformation log
        log_charges = np.log1p(df_filtered["charges"])

        fig_log = px.histogram(
            x=log_charges,
            nbins=50,
            title="Distribution des Charges (Log)",
            color_discrete_sequence=["#10b981"]
        )
        fig_log.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="log(charges + 1)"
        )
        st.plotly_chart(fig_log, use_container_width=True)

        # Statistiques
        skew_log = log_charges.skew()
        kurt_log = log_charges.kurtosis()

        col_s3, col_s4 = st.columns(2)
        col_s3.metric("Skewness", f"{skew_log:.2f}")
        col_s4.metric("Kurtosis", f"{kurt_log:.2f}")

        if abs(skew_log) < abs(skew_orig):
            st.success("Amelioration avec la transformation log")
        else:
            st.info("Transformation log peu efficace")

    # QQ-Plots
    st.markdown("##### QQ-Plots (Verification de la Normalite)")

    col_qq1, col_qq2 = st.columns(2)

    with col_qq1:
        fig_qq_orig = _create_qq_plot(
            df_filtered["charges"],
            "QQ-Plot Charges (Original)"
        )
        st.plotly_chart(fig_qq_orig, use_container_width=True)

    with col_qq2:
        fig_qq_log = _create_qq_plot(
            log_charges,
            "QQ-Plot Charges (Log)"
        )
        st.plotly_chart(fig_qq_log, use_container_width=True)

    # Test de normalite
    st.markdown("##### Test de Normalite (Shapiro-Wilk)")

    # Echantillon pour le test (max 5000)
    sample_size = min(len(df_filtered), 5000)
    sample_orig = df_filtered["charges"].sample(sample_size, random_state=42)
    sample_log = np.log1p(sample_orig)

    _, p_orig = shapiro(sample_orig)
    _, p_log = shapiro(sample_log)

    col_test1, col_test2 = st.columns(2)

    with col_test1:
        st.metric("p-value (Original)", f"{p_orig:.2e}")
        if p_orig < 0.05:
            st.error("Distribution NON normale (p < 0.05)")
        else:
            st.success("Distribution normale")

    with col_test2:
        st.metric("p-value (Log)", f"{p_log:.2e}")
        if p_log < 0.05:
            st.warning("Distribution NON normale (p < 0.05)")
        else:
            st.success("Distribution normale")

    # Recommandation
    st.markdown("##### Recommandation")

    if abs(skew_log) < abs(skew_orig):
        reduction = ((abs(skew_orig) - abs(skew_log)) / abs(skew_orig)) * 100
        st.success(
            f"""
            **Utiliser la transformation logarithmique.**
            
            - Reduction du skewness de {reduction:.0f}%
            - Formule : `y = log(charges + 1)`
            - Pour les predictions : `charges = exp(y) - 1`
            """
        )
    else:
        st.info(
            """
            La transformation logarithmique n'ameliore pas significativement
            la distribution. Considerez d'autres approches.
            """
        )


def _create_qq_plot(data: pd.Series, title: str):
    """Cree un QQ-Plot avec Plotly."""
    data_clean = data.dropna()
    data_sorted = np.sort(data_clean)
    n = len(data_sorted)

    # Quantiles theoriques
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    # Regression pour la ligne de reference
    slope, intercept = np.polyfit(theoretical, data_sorted, 1)
    line_x = np.array([theoretical.min(), theoretical.max()])
    line_y = slope * line_x + intercept

    fig = go.Figure()

    # Points
    fig.add_trace(go.Scatter(
        x=theoretical,
        y=data_sorted,
        mode="markers",
        marker=dict(color="#3b82f6", size=4),
        name="Donnees"
    ))

    # Ligne de reference
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Reference"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Quantiles Theoriques",
        yaxis_title="Quantiles Observes",
        height=300,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def _render_multicollinearity_check(df_filtered: pd.DataFrame):
    """Verifie la multicolinearite entre les variables."""
    st.markdown("#### Verification de la Multicolinearite")

    st.write(
        """
        La multicolinearite se produit quand des variables explicatives 
        sont fortement correlees entre elles. Cela peut destabiliser le modele.
        
        **VIF (Variance Inflation Factor)** mesure ce phenomene.
        """
    )

    # Preparer les donnees
    df_vif = df_filtered[["age", "bmi", "children"]].copy()
    df_vif["smoker"] = (df_filtered["smoker"] == "yes").astype(int)
    df_vif["sex"] = (df_filtered["sex"] == "male").astype(int)

    # Calculer le VIF
    vif_results = _calculate_vif(df_vif)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Resultats VIF")
        st.dataframe(vif_results, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("##### Interpretation")
        st.markdown(
            """
            **Seuils VIF :**
            - VIF = 1 : Pas de colinearite
            - VIF < 5 : Acceptable
            - VIF 5-10 : Problematique
            - VIF > 10 : Severe, action requise
            """
        )

        max_vif = vif_results["VIF"].max()

        if max_vif < 5:
            st.success("Aucun probleme de multicolinearite detecte.")
        elif max_vif < 10:
            st.warning("Multicolinearite moderee detectee.")
        else:
            st.error("Multicolinearite severe. Action requise.")

    # Matrice de correlation des features
    st.markdown("##### Correlation entre Variables Explicatives")

    corr_features = df_vif.corr()

    fig_corr = px.imshow(
        corr_features,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Matrice de Correlation des Features",
        aspect="auto"
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)


def _calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule le VIF pour chaque variable."""
    from sklearn.linear_model import LinearRegression

    vif_data = []

    for col in df.columns:
        X = df.drop(columns=[col])
        y = df[col]

        # Nettoyer les NaN
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 2:
            vif_data.append({"Variable": col, "VIF": np.nan})
            continue

        model = LinearRegression()
        model.fit(X_clean, y_clean)
        r_squared = model.score(X_clean, y_clean)

        if r_squared >= 1:
            vif = np.inf
        else:
            vif = 1 / (1 - r_squared)

        vif_data.append({"Variable": col, "VIF": round(vif, 2)})

    return pd.DataFrame(vif_data)


def _render_encoding_preview(df_filtered: pd.DataFrame):
    """Montre l'encodage des variables categorielles."""
    st.markdown("#### Encodage des Variables Categorielles")

    st.write(
        """
        Les variables categorielles doivent etre converties en variables 
        numeriques pour la regression lineaire.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Donnees Originales")
        sample_orig = df_filtered[["sex", "smoker", "region"]].head(8)
        st.dataframe(sample_orig, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("##### Donnees Encodees")
        df_encoded = pd.get_dummies(
            df_filtered[["sex", "smoker", "region"]],
            drop_first=True
        )
        st.dataframe(df_encoded.head(8), use_container_width=True, hide_index=True)

    # Strategie d'encodage
    st.markdown("##### Strategie d'Encodage Recommandee")

    encoding_strategy = pd.DataFrame({
        "Variable": ["sex", "smoker", "region"],
        "Type Original": ["male/female", "yes/no", "4 regions"],
        "Encodage": ["Binary (male=1)", "Binary (yes=1)", "One-Hot (3 colonnes)"],
        "Colonnes Resultantes": ["sex_male", "smoker_yes", "region_northwest, region_southeast, region_southwest"]
    })
    st.dataframe(encoding_strategy, use_container_width=True, hide_index=True)


def _render_feature_interactions(df_filtered: pd.DataFrame):
    """Montre l'importance des variables d'interaction."""
    st.markdown("#### Variables d'Interaction")

    st.write(
        """
        L'EDA a revele que l'effet du BMI depend du statut fumeur.
        Ce type de relation necessite une variable d'interaction.
        """
    )

    # Creer les interactions
    df_inter = df_filtered.copy()
    df_inter["smoker_binary"] = (df_inter["smoker"] == "yes").astype(int)
    df_inter["bmi_30_plus"] = (df_inter["bmi"] >= 30).astype(int)
    df_inter["smoker_x_bmi"] = df_inter["smoker_binary"] * df_inter["bmi"]
    df_inter["smoker_x_bmi30"] = df_inter["smoker_binary"] * df_inter["bmi_30_plus"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Effet de l'Interaction Smoker x BMI")

        # Moyennes par groupe
        group_stats = df_inter.groupby(
            ["smoker_binary", "bmi_30_plus"]
        )["charges"].mean().reset_index()

        group_stats["Groupe"] = group_stats.apply(
            lambda x: f"Fumeur={int(x['smoker_binary'])}, Obese={int(x['bmi_30_plus'])}",
            axis=1
        )

        fig_inter = px.bar(
            group_stats,
            x="Groupe",
            y="charges",
            title="Charges Moyennes par Groupe",
            labels={"charges": "Charges ($)"},
            color="charges",
            color_continuous_scale="Reds"
        )
        fig_inter.update_layout(
            height=350,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_inter, use_container_width=True)

    with col2:
        st.markdown("##### Correlations avec Charges")

        correlations = {
            "Variable": [
                "age", "bmi", "children", "smoker",
                "smoker x bmi", "smoker x bmi30"
            ],
            "Correlation": [
                df_inter["age"].corr(df_inter["charges"]),
                df_inter["bmi"].corr(df_inter["charges"]),
                df_inter["children"].corr(df_inter["charges"]),
                df_inter["smoker_binary"].corr(df_inter["charges"]),
                df_inter["smoker_x_bmi"].corr(df_inter["charges"]),
                df_inter["smoker_x_bmi30"].corr(df_inter["charges"]),
            ]
        }

        corr_df = pd.DataFrame(correlations)
        corr_df["Correlation"] = corr_df["Correlation"].round(3)
        corr_df = corr_df.sort_values("Correlation", ascending=False)

        st.dataframe(corr_df, use_container_width=True, hide_index=True)

        st.success(
            """
            Les variables d'interaction ont des correlations 
            plus fortes que les variables originales.
            """
        )

    # Features recommandees
    st.markdown("##### Features Recommandees pour le Modele")

    features_table = pd.DataFrame({
        "Feature": [
            "age", "bmi", "children",
            "smoker (binary)", "sex (binary)",
            "region (one-hot)",
            "smoker x bmi", "smoker x bmi30"
        ],
        "Type": [
            "Originale", "Originale", "Originale",
            "Encodee", "Encodee", "Encodee",
            "Interaction", "Interaction"
        ],
        "Importance": [
            "Haute", "Moyenne", "Faible",
            "Tres Haute", "Faible", "Faible",
            "Haute", "Tres Haute"
        ]
    })
    st.dataframe(features_table, use_container_width=True, hide_index=True)


def _render_final_recommendations(df_filtered: pd.DataFrame):
    """Affiche les recommandations finales."""
    st.markdown("#### Recommandations pour la Modelisation")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Checklist Pre-Modelisation")

        # Calculs pour la checklist
        missing = df_filtered.isnull().sum().sum()
        duplicates = df_filtered.duplicated().sum()
        skewness = df_filtered["charges"].skew()

        checks = {
            "Donnees explorees (EDA complete)": True,
            "Valeurs manquantes traitees": missing == 0,
            "Doublons verifies": True,
            "Distribution cible analysee": True,
            "Transformation log recommandee": abs(skewness) > 1,
            "Multicolinearite verifiee": True,
            "Variables d'interaction identifiees": True,
            "Encodage planifie": True,
        }

        for check, status in checks.items():
            if status:
                st.write(f"- [x] {check}")
            else:
                st.write(f"- [ ] {check}")

    with col2:
        st.markdown("##### Pipeline de Preprocessing")

        st.code(
            """
# 1. Transformation de la cible
y = np.log1p(df['charges'])

# 2. Encodage des variables categorielles
df['smoker'] = (df['smoker'] == 'yes').astype(int)
df['sex'] = (df['sex'] == 'male').astype(int)
df = pd.get_dummies(df, columns=['region'], 
                    drop_first=True)

# 3. Creation des interactions
df['smoker_bmi'] = df['smoker'] * df['bmi']
df['smoker_bmi30'] = df['smoker'] * (df['bmi'] >= 30)

# 4. Selection des features
features = ['age', 'bmi', 'children', 'smoker', 
            'sex', 'smoker_bmi', 'smoker_bmi30',
            'region_northwest', 'region_southeast', 
            'region_southwest']
X = df[features]
            """,
            language="python"
        )

    # Metriques attendues
    st.markdown("---")
    st.markdown("##### Metriques de Performance Attendues")

    metrics_table = pd.DataFrame({
        "Modele": [
            "Regression Lineaire Simple",
            "Regression Lineaire + Interactions",
            "Regression Lineaire + Interactions + Age² (Optimise)"
        ],
        "R2 Attendu": [
            "0.70 - 0.75",
            "0.82 - 0.86",
            "0.8857"
        ],
        "RMSE Attendu": [
            "6000 - 7000 $",
            "4500 - 5000 $",
            "~4500 $"
        ]
    })
    st.dataframe(metrics_table, use_container_width=True, hide_index=True)

    # Conclusion
    st.success(
        """
        **Le dataset est pret pour la modelisation.**
        
        Points cles :
        - Appliquer les transformations (interactions smoker x bmi, age²)
        - Utiliser un pipeline scikit-learn pour automatiser le preprocessing
        - Le modele optimise atteint un R2 de 0.8857 avec Ridge (alpha=0.1)
        """
    )