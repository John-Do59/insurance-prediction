"""
Dashboard d'analyse des données d'assurance.

Ce module fournit une interface Streamlit pour l'exploration
et la visualisation des facteurs de coûts d'assurance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import chi2_contingency


# ==========================================
# Configuration & Style
# ==========================================
st.set_page_config(
    page_title="Insurance Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a premium look
st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    [data-testid="metric-container"] {
        background-color: #1e3a8a;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        border: none;
        color: white !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 400 !important;
        opacity: 0.9;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    h1, h2, h3 {
        color: #1e3a8a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# Fonctions Utilitaires
# ==========================================
def cramers_v(x, y):
    """
    Calcule le coefficient de Cramer (Cramer's V).

    Mesure l'association entre deux variables categorielles.

    Parameters
    ----------
    x : pd.Series
        Premiere variable categorielle.
    y : pd.Series
        Deuxieme variable categorielle.

    Returns
    -------
    float
        Coefficient de Cramer entre 0 et 1.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1) if r > 1 else r
    kcorr = k - ((k - 1) ** 2) / (n - 1) if k > 1 else k

    denominator = min((kcorr - 1), (rcorr - 1))
    if denominator == 0:
        return 0.0
    return np.sqrt(phi2corr / denominator)


def correlation_ratio(categories, values):
    """
    Calcule le ratio de correlation (Eta-squared).

    Mesure la proportion de variance d'une variable numerique
    expliquee par une variable categorielle.

    Parameters
    ----------
    categories : pd.Series
        Variable categorielle.
    values : pd.Series
        Variable numerique.

    Returns
    -------
    float
        Ratio de correlation entre 0 et 1.
    """
    df_temp = pd.DataFrame({
        "categories": categories,
        "values": values
    }).dropna()

    if df_temp.empty:
        return np.nan

    categories = df_temp["categories"]
    values = df_temp["values"]

    if len(categories.unique()) <= 1 or len(values) < 2:
        return 0.0

    groups = [values[categories == c] for c in categories.unique()]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return 0.0

    ss_total = np.sum((values - values.mean()) ** 2)

    if ss_total == 0:
        return 0.0

    ss_between = 0
    for group in groups:
        if len(group) > 0:
            group_mean = group.mean()
            global_mean = values.mean()
            ss_between += len(group) * (group_mean - global_mean) ** 2

    eta_squared = ss_between / ss_total
    return eta_squared


def calculate_association_matrix(df, numerical_cols, categorical_cols):
    """
    Calcule une matrice d'association comprehensive.

    Combine Pearson, Cramer's V et Eta-squared selon les types de variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    numerical_cols : list
        Liste des colonnes numeriques.
    categorical_cols : list
        Liste des colonnes categorielles.

    Returns
    -------
    pd.DataFrame
        Matrice d'association.
    """
    all_vars = numerical_cols + categorical_cols
    matrix = pd.DataFrame(np.nan, index=all_vars, columns=all_vars)

    for col1 in all_vars:
        for col2 in all_vars:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
                continue

            if pd.notna(matrix.loc[col2, col1]):
                matrix.loc[col1, col2] = matrix.loc[col2, col1]
                continue

            type1_is_num = col1 in numerical_cols
            type2_is_num = col2 in numerical_cols

            if type1_is_num and type2_is_num:
                val = df[col1].corr(df[col2])
            elif not type1_is_num and not type2_is_num:
                if (len(df[col1].unique()) > 1 and len(df[col2].unique()) > 1):
                    val = cramers_v(df[col1], df[col2])
                else:
                    val = 0.0
            else:
                num_col = col1 if type1_is_num else col2
                cat_col = col1 if not type1_is_num else col2
                val = correlation_ratio(df[cat_col], df[num_col])

            matrix.loc[col1, col2] = val
            matrix.loc[col2, col1] = val

    return matrix


def run_chi2_test(contingency_table):
    """
    Execute un test du Chi-carre et retourne les resultats.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Table de contingence.

    Returns
    -------
    tuple
        (chi2_stat, p_value, is_valid)
    """
    is_valid = (
        not contingency_table.empty
        and contingency_table.shape[0] > 1
        and contingency_table.shape[1] > 1
    )

    if not is_valid:
        return None, None, False

    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2_stat, p_value, True


def display_chi2_result(chi2_stat, p_value, is_valid, test_name):
    """
    Affiche les resultats d'un test Chi-carre.

    Parameters
    ----------
    chi2_stat : float
        Statistique du Chi-carre.
    p_value : float
        P-value du test.
    is_valid : bool
        Indique si le test est valide.
    test_name : str
        Nom du test pour l'affichage.
    """
    if not is_valid:
        st.warning(f"Donnees insuffisantes pour le test Chi2 ({test_name}).")
        return

    st.metric("Chi2 statistique", f"{chi2_stat:.2f}")
    st.metric("p-value", f"{p_value:.2e}")

    if p_value < 0.001:
        st.success("**Relation tres significative** (p < 0.001)")
    elif p_value < 0.05:
        st.success("**Relation significative** (p < 0.05)")
    else:
        st.warning("Pas de relation significative (p >= 0.05)")


# ==========================================
# Chargement des Donnees
# ==========================================
@st.cache_data
def load_data():
    """
    Charge et prepare les donnees d'assurance.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les donnees preparees.
    """
    data = pd.read_csv("data/insurance.csv")
    data["smoker_label"] = data["smoker"].map({
        "yes": "Fumeur",
        "no": "Non-fumeur"
    })
    return data


try:
    df = load_data()
except FileNotFoundError:
    st.error("Fichier 'data/insurance.csv' introuvable.")
    st.stop()
except Exception as e:
    st.error(f"Erreur de chargement des donnees : {e}")
    st.stop()


# ==========================================
# Sidebar - Filtres
# ==========================================
LOGO_URL = (
    "https://www.google.com/images/branding/"
    "googlelogo/2x/googlelogo_color_92x30dp.png"
)
st.sidebar.image(LOGO_URL, width=150)
st.sidebar.title("Filtres de Donnees")
st.sidebar.markdown("---")

# Filtre Region
st.sidebar.markdown("#### Region Geographique")
region_list = ["Toutes"] + list(df["region"].unique())
selected_region = st.sidebar.selectbox(
    "Selectionner une region",
    region_list,
    label_visibility="collapsed"
)

# Filtre Age
st.sidebar.markdown("#### Tranche d'Age")
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


def format_smoker(x):
    """Formate le label du statut fumeur."""
    if x == "Tous":
        return "Tous"
    return "Fumeur" if x == "yes" else "Non-fumeur"


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
st.sidebar.caption("Ajustez les filtres pour explorer les donnees")

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
st.sidebar.markdown("---")
st.sidebar.markdown("#### Resultat du Filtrage")
pct_filtered = (len(df_filtered) / len(df)) * 100
st.sidebar.metric(
    "Observations selectionnees",
    f"{len(df_filtered)} / {len(df)}",
    delta=f"{pct_filtered:.1f}%"
)

# Bouton de telechargement
csv_data = df_filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Telecharger les donnees (CSV)",
    data=csv_data,
    file_name=f"insurance_filtered_{len(df_filtered)}_rows.csv",
    mime="text/csv",
    use_container_width=True
)


# ==========================================
# Header & Metriques Principales
# ==========================================
st.title("Analyse des Risques & Charges d'Assurance")
st.markdown(
    f"**Semaine 1 : Exploration des Facteurs de Couts** | "
    f"{df_filtered.shape[0]} observations selectionnees"
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_charges = df_filtered["charges"].mean()
    st.metric("Charges Moyennes", f"{avg_charges:,.0f} $")

with col2:
    median_age = df_filtered["age"].median()
    st.metric("Age Median", f"{median_age:.0f} ans")

with col3:
    avg_bmi = df_filtered["bmi"].mean()
    st.metric("IMC Moyen", f"{avg_bmi:.1f}")

with col4:
    smoker_ratio = (df_filtered["smoker"] == "yes").mean()
    st.metric("% Fumeurs", f"{smoker_ratio:.1%}")

st.markdown("---")


# ==========================================
# Onglets Principaux
# ==========================================
tab_dist, tab_health, tab_demo, tab_corr, tab_expert = st.tabs([
    "Distribution des Couts",
    "Facteurs de Sante",
    "Demographie & Profils",
    "Correlations & Stats",
    "Analyses Expert"
])


# --- Tab 1: Distribution ---
with tab_dist:
    st.subheader("Analyse de la Variable Cible : Charges")

    col_left, col_right = st.columns([1, 1])

    with col_left:
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

    with col_right:
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

        color_map_smoker = {
            "Fumeur": "#ef4444",
            "Non-fumeur": "#10b981"
        }
        fig_box_smoker = px.box(
            df_filtered,
            x="smoker_label",
            y="charges",
            color="smoker_label",
            points="all",
            title="Repartition des charges par statut fumeur",
            color_discrete_map=color_map_smoker
        )
        st.plotly_chart(fig_box_smoker, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Repartition des Variables Categorielles")

        col_cat1, col_cat2 = st.columns(2)

        with col_cat1:
            fig_sex_dist = px.pie(
                df_filtered,
                names="sex",
                title="Repartition par Sexe",
                hole=0.3
            )
            st.plotly_chart(fig_sex_dist, use_container_width=True)

        with col_cat2:
            fig_region_dist = px.pie(
                df_filtered,
                names="region",
                title="Repartition par Region",
                hole=0.3
            )
            st.plotly_chart(fig_region_dist, use_container_width=True)


# --- Tab 2: Facteurs de Sante ---
with tab_health:
    st.subheader("L'influence de l'IMC et du Tabagisme")

    col_h1, col_h2 = st.columns([2, 1])

    with col_h1:
        color_map_health = {
            "Fumeur": "#ef4444",
            "Non-fumeur": "#10b981"
        }
        fig_scatter = px.scatter(
            df_filtered,
            x="bmi",
            y="charges",
            color="smoker_label",
            size="age",
            hover_data=["age", "children"],
            title="Impact combine de l'IMC et du Tabac",
            labels={
                "bmi": "IMC (Indice de Masse Corporelle)",
                "charges": "Charges ($)"
            },
            color_discrete_map=color_map_health
        )
        fig_scatter.add_vline(
            x=30,
            line_dash="dash",
            line_color="gray",
            annotation_text="Seuil Obesite (30)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_h2:
        st.success(
            """
            **Insight Cle : L'Effet Synergie**
            - Chez les **non-fumeurs**, l'IMC a une influence
              lineaire moderee.
            - Chez les **fumeurs**, on observe une rupture brutale
              au-dela d'un **IMC de 30**.
            - Ce groupe (Fumeur + Obese) represente le risque
              financier le plus eleve pour l'assureur.
            """
        )

        df_filtered["bmi_cat"] = pd.cut(
            df_filtered["bmi"],
            bins=[0, 18.5, 25, 30, 100],
            labels=["Insuffisant", "Normal", "Surpoids", "Obese"]
        )
        fig_pie = px.pie(
            df_filtered,
            names="bmi_cat",
            title="Repartition des categories d'IMC",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# --- Tab 3: Demographie ---
with tab_demo:
    st.subheader("Analyse Demographique Approfondie")

    # Section 1: Distribution par Region
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
            color_discrete_map={
                "Fumeur": "#ef4444",
                "Non-fumeur": "#10b981"
            }
        )
        fig_region.update_layout(
            xaxis_title="Region",
            yaxis_title="Charges ($)"
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
                label=region,
                value=f"{row['mean']:,.0f} $",
                delta=f"Mediane: {row['median']:,.0f} $"
            )

    st.markdown("---")

    # Section 2: Comparaison Homme/Femme
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
            color_discrete_map={
                "Fumeur": "#ef4444",
                "Non-fumeur": "#10b981"
            }
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

    st.markdown("---")

    # Section 3: Impact du nombre d'enfants
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
            color_discrete_map={
                "Fumeur": "#ef4444",
                "Non-fumeur": "#10b981"
            }
        )
        fig_children.update_layout(
            xaxis_title="Nombre d'enfants",
            yaxis_title="Charges ($)"
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
            st.write(f"{n_children} enfant(s): {count} ({pct:.1f}%)")


# --- Tab 4: Correlations & Stats ---
with tab_corr:
    st.subheader("Analyse des Correlations et Statistiques Descriptives")

    # Matrice d'Association Comprehensive
    st.markdown("#### Matrice d'Association Comprehensive")
    st.info(
        """
        Cette matrice combine differentes mesures d'association :
        - **Numerique-Numerique :** Correlation de Pearson
        - **Categorielle-Categorielle :** Coefficient de Cramer (V)
        - **Numerique-Categorielle :** Ratio de Correlation (Eta-squared)
        """
    )

    main_cols = ["age", "bmi", "children", "charges", "sex", "smoker", "region"]
    df_for_corr = df_filtered[main_cols]

    numerical_cols = df_for_corr.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_for_corr.select_dtypes(exclude=np.number).columns.tolist()

    association_matrix = calculate_association_matrix(
        df_for_corr,
        numerical_cols,
        categorical_cols
    )

    fig_comp_heatmap = px.imshow(
        association_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Matrice d'Association Complete",
        labels={"color": "Force d'Association"},
        aspect="auto"
    )
    fig_comp_heatmap.update_layout(
        xaxis_title="",
        yaxis_title="",
        height=500
    )
    st.plotly_chart(fig_comp_heatmap, use_container_width=True)

    st.markdown("---")

    # Correlation de Pearson (Numerique)
    col_c1, col_c2 = st.columns([2, 1])

    with col_c1:
        st.markdown("#### Correlation Lineaire (Pearson)")
        corr_matrix = df_filtered[numerical_cols].corr()

        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlations entre variables numeriques",
            labels={"color": "Coefficient"},
            aspect="auto"
        )
        fig_heatmap.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col_c2:
        st.info(
            """
            **Interpretation (Pearson) :**
            - **Forte** : |r| > 0.7
            - **Moderee** : 0.3 < |r| < 0.7
            - **Faible** : |r| < 0.3

            **Limites :**
            - Mesure uniquement les relations **lineaires**
            - Ne capture pas les interactions complexes
            """
        )

        st.markdown("#### Correlations avec `charges`")
        if "charges" in corr_matrix.columns:
            corr_with_charges = (
                corr_matrix["charges"]
                .drop("charges")
                .sort_values(ascending=False)
            )
            for var, corr_val in corr_with_charges.items():
                st.metric(label=var, value=f"{corr_val:.3f}")

    st.markdown("---")

    # Statistiques descriptives
    col_s1, col_s2 = st.columns([1, 1])

    with col_s1:
        st.markdown("#### Statistiques Descriptives")
        stats_df = df_filtered[numerical_cols].describe().T.round(2)
        st.dataframe(stats_df, use_container_width=True)

    with col_s2:
        st.markdown("#### Detection des Valeurs Extremes")

        q1 = df_filtered["charges"].quantile(0.25)
        q3 = df_filtered["charges"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = df_filtered[
            (df_filtered["charges"] < lower_bound)
            | (df_filtered["charges"] > upper_bound)
        ]

        col_o1, col_o2 = st.columns(2)
        col_o1.metric("Outliers detectes", len(outliers))
        col_o2.metric(
            "% du dataset",
            f"{len(outliers) / len(df_filtered) * 100:.1f}%"
        )

        col_o3, col_o4 = st.columns(2)
        col_o3.metric("Charge min", f"{df_filtered['charges'].min():,.0f} $")
        col_o4.metric("Charge max", f"{df_filtered['charges'].max():,.0f} $")

        st.markdown(
            f"""
            **Seuils IQR :**
            - Limite inferieure : {lower_bound:,.0f} $
            - Limite superieure : {upper_bound:,.0f} $
            """
        )

    # Affichage des outliers
    if len(outliers) > 0:
        st.markdown("---")
        st.markdown("#### Liste des Cas Extremes (Top 10)")
        display_cols = [
            "age", "sex", "bmi", "children", "smoker", "region", "charges"
        ]
        outliers_display = (
            outliers[display_cols]
            .sort_values("charges", ascending=False)
            .head(10)
        )
        st.dataframe(outliers_display, use_container_width=True)

    # Tests Chi-carre
    st.markdown("---")
    st.markdown("#### Tests d'Independance (Chi-carre)")

    st.info(
        """
        **Pourquoi le Chi-carre ?**
        - La correlation de Pearson **ne fonctionne pas** avec
          les variables categorielles
        - Le test du Chi-carre mesure l'**independance statistique**
        - Utile pour evaluer la significativite de l'association
        """
    )

    if "charges_cat" not in df_filtered.columns:
        df_filtered["charges_cat"] = pd.cut(
            df_filtered["charges"],
            bins=3,
            labels=["Faible", "Moyen", "Eleve"]
        )

    col_chi1, col_chi2 = st.columns([1, 1])

    with col_chi1:
        st.markdown("##### Test 1 : Smoker x Charges")
        contingency_smoker = pd.crosstab(
            df_filtered["smoker"],
            df_filtered["charges_cat"]
        )
        st.dataframe(contingency_smoker, use_container_width=True)

        chi2, pval, valid = run_chi2_test(contingency_smoker)
        display_chi2_result(chi2, pval, valid, "Smoker x Charges")

    with col_chi2:
        st.markdown("##### Test 2 : Region x Smoker")
        contingency_region = pd.crosstab(
            df_filtered["region"],
            df_filtered["smoker"]
        )
        st.dataframe(contingency_region, use_container_width=True)

        chi2, pval, valid = run_chi2_test(contingency_region)
        display_chi2_result(chi2, pval, valid, "Region x Smoker")

    st.markdown("---")
    col_chi3, col_chi4 = st.columns([1, 1])

    with col_chi3:
        st.markdown("##### Test 3 : Sex x Charges")
        contingency_sex = pd.crosstab(
            df_filtered["sex"],
            df_filtered["charges_cat"]
        )
        st.dataframe(contingency_sex, use_container_width=True)

        chi2, pval, valid = run_chi2_test(contingency_sex)
        display_chi2_result(chi2, pval, valid, "Sex x Charges")

    with col_chi4:
        st.markdown("##### Interpretation du Chi-carre")
        st.markdown(
            """
            **Hypothese nulle (H0)** : Les variables sont independantes

            **Regle de decision** :
            - Si **p < 0.05** : On rejette H0 -> Variables **dependantes**
            - Si **p >= 0.05** : On ne rejette pas H0 -> Pas de preuve

            **Attendu** :
            - `smoker` x `charges` : **Forte dependance**
            - `region` x `smoker` : Probablement **independant**
            - `sex` x `charges` : Probablement **independant**
            """
        )


# ==========================================
# Synthese & Conclusions
# ==========================================
st.markdown("---")
st.subheader("Synthese & Hypotheses de Modelisation")

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
            - L'obésité multiplie par 5 le coût pour un fumeur vs non-fumeur.
            """
        )

    with c2:
        st.markdown(
            """
            **3. Évolution Âge & Tabac :**
            - Jeune (18-35) Non-fumeur : **~4 800 $**
            - Senior (51+) Non-fumeur : **~13 500 $**
            - Un jeune fumeur (**~28 100 $**) coûte déjà 2x plus cher qu'un senior non-fumeur.
            
            **4. Zoom Régions & Genre :**
            - **Southeast** : Plus chère (**14 735 $**) car plus de fumeurs (25%).
            - **Genre** : Les hommes fumeurs sont les plus coûteux (~33k$).
            """
        )

# --- Tab 5: Analyses Expert ---
with tab_expert:
    st.subheader("Analyses Avancees de Second Niveau")
    
    # A. Pareto Analysis
    st.markdown("#### 🎯 Concentration des Charges (Loi de Pareto)")
    col_p1, col_p2 = st.columns([2, 1])
    
    with col_p1:
        df_sorted = df_filtered.sort_values('charges', ascending=False)
        df_sorted['cum_charges'] = df_sorted['charges'].cumsum() / df_sorted['charges'].sum()
        df_sorted['cum_population'] = np.arange(1, len(df_sorted) + 1) / len(df_sorted)
        
        fig_pareto = px.line(
            df_sorted, x='cum_population', y='cum_charges',
            title="Courbe de Pareto : Concentration des Charges",
            labels={'cum_population': '% Population', 'cum_charges': '% Charges Totales'},
            color_discrete_sequence=["#1e3a8a"]
        )
        fig_pareto.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        st.plotly_chart(fig_pareto, use_container_width=True)
        
    with col_p2:
        top_20_percent = int(len(df_sorted) * 0.2)
        if top_20_percent > 0:
            top_20_share = df_sorted.iloc[top_20_percent-1]['cum_charges']
        else:
            top_20_share = 0
            
        st.metric("Part du Top 20%", f"{top_20_share:.1%}")
        
        # Pie chart pour la loi de Pareto
        pareto_pie = pd.DataFrame({
            'Groupe': ['Top 20% des assurés', 'Les autres 80%'],
            'Coût total': [top_20_share, 1-top_20_share]
        })
        fig_pie_pareto = px.pie(
            pareto_pie, values='Coût total', names='Groupe',
            hole=0.4, color_discrete_sequence=["#1e3a8a", "#94a3b8"]
        )
        st.plotly_chart(fig_pie_pareto, use_container_width=True)
        
        st.info("**Insight :** Plus de 50% des coûts sont générés par seulement 20% des assurés (cas critiques).")

    st.markdown("---")

    # B. Regional Health Profile
    st.markdown("#### 🏥 Profil de Sante Regional (Taux d'Obesite)")
    col_ob1, col_ob2 = st.columns([2, 1])
    
    with col_ob1:
        df_filtered['is_obese'] = df_filtered['bmi'] >= 30
        reg_health = df_filtered.groupby('region')['is_obese'].mean().reset_index()
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
        
        st.warning("**Alerte Fracture Sanitaire :** Le Southeast présente un taux d'obésité alarmant (> 60%), expliquant ses charges plus élevées.")

    st.markdown("---")

    # C. Risk Volatility
    st.markdown("#### 📉 Volatilité du Risque (Imprévisibilité)")
    col_v1, col_v2 = st.columns([2, 1])
    
    with col_v1:
        cv_data = df_filtered.groupby('smoker_label')['charges'].agg(lambda x: x.std() / x.mean()).reset_index()
        cv_data.columns = ['Statut', 'Volatilité (CV)']
        
        fig_cv = px.bar(
            cv_data, x='Statut', y='Volatilité (CV)',
            title="Éclatement du Risque (Coefficient de Variation)",
            color='Statut',
            color_discrete_map={'Fumeur': '#ef4444', 'Non-fumeur': '#10b981'}
        )
        st.plotly_chart(fig_cv, use_container_width=True)
        
    with col_v2:
        st.info("""
        **Coefficient de Variation (CV) :**
        - Un CV élevé signifie que le risque est **plus imprévisible**.
        - On remarque que le risque chez les non-fumeurs est beaucoup plus volatil.
        - Chez les fumeurs, le risque de charges élevées est plus 'systématique'.
        """)

    st.markdown("---")

    # D. Visualisation des "Chocs"
    st.markdown("#### ⚡ Visualisation des Écarts Majeurs (Effets Chocs)")
    col_ch1, col_ch2 = st.columns(2)
    
    with col_ch1:
        # Duel Jeune Fumeur vs Senior Non-Fumeur
        df_filtered['profile_group'] = 'Autres'
        df_filtered.loc[(df_filtered['age'] <= 35) & (df_filtered['smoker'] == 'yes'), 'profile_group'] = 'Jeune Fumeur (<=35)'
        df_filtered.loc[(df_filtered['age'] >= 51) & (df_filtered['smoker'] == 'no'), 'profile_group'] = 'Senior Non-Fumeur (>=51)'
        
        duel_df = df_filtered[df_filtered['profile_group'] != 'Autres'].groupby('profile_group')['charges'].mean().reset_index()
        
        fig_duel = px.bar(
            duel_df, x='profile_group', y='charges',
            title="Le Duel : Style de Vie vs Vieillissement",
            labels={'charges': 'Charges Moyennes ($)', 'profile_group': 'Profil'},
            color='profile_group',
            color_discrete_map={'Jeune Fumeur (<=35)': '#ef4444', 'Senior Non-Fumeur (>=51)': '#3b82f6'}
        )
        st.plotly_chart(fig_duel, use_container_width=True)
        st.caption("Un jeune fumeur coûte ~2x plus cher qu'un senior qui ne fume pas.")

    with col_ch2:
        # Matrice 4-blocs IMC x Tabac
        df_filtered['imc_tabac_group'] = df_filtered['smoker_label'] + " + " + df_filtered['bmi_cat'].astype(str)
        # On ne veut que les 4 groupes principaux (basé sur le seuil 30)
        df_filtered['bmi_simple'] = df_filtered['bmi'] >= 30
        df_filtered['bmi_simple_label'] = df_filtered['bmi_simple'].map({True: 'IMC >= 30', False: 'IMC < 30'})
        
        matrix_df = df_filtered.groupby(['smoker_label', 'bmi_simple_label'])['charges'].mean().reset_index()
        matrix_df['Groupe'] = matrix_df['smoker_label'] + " (" + matrix_df['bmi_simple_label'] + ")"
        
        fig_matrix = px.bar(
            matrix_df, x='Groupe', y='charges',
            title="L'Effet Combo (Tabac + Obésité)",
            labels={'charges': 'Charges Moyennes ($)'},
            color='smoker_label',
            color_discrete_map={'Fumeur': '#ef4444', 'Non-fumeur': '#10b981'}
        )
        st.plotly_chart(fig_matrix, use_container_width=True)
        st.caption("L'obésité multiplie par ~5 le coût pour un fumeur (Risque de 41k$).")

st.write("---")
st.caption("Dashboard Analytics Insurance v2.1 - Equipe Dev Data IA")