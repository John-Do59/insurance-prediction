import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance EDA", layout="wide")
sns.set_style("whitegrid")

PALETTE_SMOKER = {"yes": "red", "no": "blue"}
PALETTE_SEX = {"male": "steelblue", "female": "salmon"}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/insurance.csv")
    # Log transformation for charges
    df["log_charges"] = np.log1p(df["charges"])
    # Encode smoker
    df["smoker_flag"] = df["smoker"].map({"yes": 1, "no": 0})
    # Interaction features
    df["bmi_smoker"] = df["bmi"] * df["smoker_flag"]
    df["age_smoker"] = df["age"] * df["smoker_flag"]
    df["children_smoker"] = df["children"] * df["smoker_flag"]
    return df


df = load_data()

st.title("Insurance Dataset EDA")
st.markdown(
    "Exploration interactive des charges d'assurance et facteurs associés."
)

# Sidebar filters
st.sidebar.header("Filtres")
smoker_filter = st.sidebar.multiselect(
    "Fumeur", options=df["smoker"].unique(), default=df["smoker"].unique()
)

region_filter = st.sidebar.multiselect(
    "Région", options=df["region"].unique(), default=df["region"].unique()
)

df_filtered = df[
    (df["smoker"].isin(smoker_filter)) & (df["region"].isin(region_filter))
]
st.sidebar.markdown(f"Nombre d'observations : {df_filtered.shape[0]}")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Distributions", "BMI & Age", "Smoker Analysis", "Children & Region", "Correlations"]
)

# Tab 1: Distributions
with tab1:
    st.header("Distribution des charges")
    col1, col2 = st.columns(2)

    with col1:
        # Histogramme charges
        fig, ax = plt.subplots()
        sns.histplot(df_filtered["charges"], bins=50, kde=True, ax=ax)
        ax.set_title("Distribution des charges")
        st.pyplot(fig)

        # Histogramme log-charges
        fig, ax = plt.subplots()
        sns.histplot(df_filtered["log_charges"], bins=50, kde=True, ax=ax, color="green")
        ax.set_title("Distribution log-transformée des charges")
        st.pyplot(fig)

        # Boxplot charges
        fig, ax = plt.subplots()
        sns.boxplot(x=df_filtered["charges"], ax=ax)
        ax.set_title("Boxplot des charges")
        st.pyplot(fig)

    with col2:
        # Histogramme BMI
        fig, ax = plt.subplots()
        sns.histplot(df_filtered["bmi"], bins=30, kde=True, ax=ax)
        ax.set_title("Distribution du BMI (valeurs absolues)")
        st.pyplot(fig)

        # Histogramme BMI %
        fig, ax = plt.subplots()
        sns.histplot(df_filtered["bmi"], bins=30, stat="percent", kde=True, ax=ax)
        ax.set_title("Distribution du BMI (%)")
        st.pyplot(fig)

        # Boxplot BMI
        fig, ax = plt.subplots()
        sns.boxplot(x=df_filtered["bmi"], ax=ax)
        ax.set_title("Boxplot du BMI")
        st.pyplot(fig)

        # Boxplot Age
        fig, ax = plt.subplots()
        sns.boxplot(x=df_filtered["age"], ax=ax)
        ax.set_title("Boxplot de l'âge")
        st.pyplot(fig)

# Tab 2: BMI & Age Scatterplots
with tab2:
    st.header("Relation Charges vs BMI et Age")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.scatterplot(x="bmi", y="charges", hue="smoker",
                        palette=PALETTE_SMOKER, data=df_filtered, ax=ax)
        ax.set_title("Charges vs BMI par fumeur")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(x="age", y="charges", hue="smoker",
                        palette=PALETTE_SMOKER, data=df_filtered, ax=ax)
        ax.set_title("Charges vs Age par fumeur")
        st.pyplot(fig)

# --- Tab 3: Smoker Analysis ---
with tab3:
    st.header("Analyse par fumeur")
    smoker_stats = (
        df_filtered.groupby("smoker")["charges"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    st.dataframe(smoker_stats)

    fig, ax = plt.subplots()
    sns.boxplot(x="smoker", y="charges", data=df_filtered,
                palette=PALETTE_SMOKER, ax=ax)
    ax.set_title("Charges par fumeur")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.violinplot(x="smoker", y="charges", data=df_filtered,
                   palette=PALETTE_SMOKER, ax=ax)
    ax.set_title("Distribution des charges par fumeur")
    st.pyplot(fig)

# Tab 4: Children & Region
with tab4:
    st.header("Analyse enfants et région")

    fig, ax = plt.subplots()
    sns.boxplot(x="children", y="charges", hue="smoker",
                data=df_filtered, palette=PALETTE_SMOKER, ax=ax)
    ax.set_title("Charges par nombre d'enfants et fumeur")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.violinplot(x="children", y="charges", hue="smoker",
                   data=df_filtered, palette=PALETTE_SMOKER, split=True, ax=ax)
    ax.set_title("Distribution des charges par enfants et fumeur")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.barplot(x="region", y="charges", hue="smoker",
                data=df_filtered, ci=None, ax=ax)
    ax.set_title("Charges moyennes par région et fumeur")
    st.pyplot(fig)

# Tab 5: Correlations
with tab5:
    st.header("Analyse des corrélations")
    numeric_cols = [
        "age", "bmi", "children", "charges", "log_charges",
        "bmi_smoker", "age_smoker", "children_smoker"
    ]
    df_pair = df_filtered[numeric_cols].copy()
    df_pair["smoker"] = df_filtered["smoker"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_filtered[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Matrice de corrélation")
    st.pyplot(fig)

    fig = sns.pairplot(df_pair, hue="smoker", palette=PALETTE_SMOKER)
    st.pyplot(fig)

#  Summary
st.header("Synthèse finale")
st.markdown(
    "- Les fumeurs ont des charges nettement plus élevées que les non-fumeurs.\n"
    "- Le BMI amplifie les charges pour les fumeurs.\n"
    "- L'âge a un effet positif surtout pour les fumeurs.\n"
    "- Le nombre d’enfants a un effet faible, visible surtout chez les fumeurs.\n"
    "- La région a un effet modéré sur les charges.\n"
    "- La transformation logarithmique des charges peut aider pour la modélisation.\n"
    "- Les interactions BMI*smoker, Age*smoker, Children*smoker peuvent améliorer un modèle prédictif."
)
