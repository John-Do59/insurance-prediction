import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Configuration générale
st.set_page_config(page_title="EDA Insurance", layout="wide")
sns.set_style("whitegrid")

PALETTE_SMOKER = {"yes": "red", "no": "blue"}
PALETTE_SEX = {"male": "steelblue", "female": "salmon"}


# =========================
# Chargement des données
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/insurance.csv")


df = load_data()


# Titre
st.title(" Mini-dashboard EDA — Insurance Dataset")

st.markdown(
    """
Exploration interactive du dataset **Insurance Charges**  
Objectif : identifier les facteurs influençant les charges d’assurance.
"""
)


# Filtres

st.sidebar.header("Filtres")

smoker_filter = st.sidebar.multiselect(
    "Tabagisme",
    options=df["smoker"].unique(),
    default=df["smoker"].unique(),
)

region_filter = st.sidebar.multiselect(
    "Région",
    options=df["region"].unique(),
    default=df["region"].unique(),
)

df_filtered = df[
    (df["smoker"].isin(smoker_filter))
    & (df["region"].isin(region_filter))
]

st.sidebar.markdown(f"**Observations : {df_filtered.shape[0]}**")

# Charges & tabagisme
st.header(" Charges selon le statut fumeur")

smoker_stats = (
    df_filtered
    .groupby("smoker")["charges"]
    .agg(["count", "mean", "median"])
    .reset_index()
)

st.dataframe(smoker_stats)

fig, ax = plt.subplots()
sns.boxplot(
    data=df_filtered,
    x="smoker",
    y="charges",
    palette=PALETTE_SMOKER,
    ax=ax,
)
ax.set_title("Charges selon le statut fumeur")
st.pyplot(fig)


# Charges vs BMI × fumeur
st.header("Charges vs IMC (BMI) selon le tabagisme")

fig, ax = plt.subplots()
sns.scatterplot(
    data=df_filtered,
    x="bmi",
    y="charges",
    hue="smoker",
    palette=PALETTE_SMOKER,
    ax=ax,
)
ax.set_title("Charges vs BMI selon le statut fumeur")
st.pyplot(fig)


# Charges vs enfants × fumeur
st.header(" Charges selon le nombre d’enfants et le tabagisme")

fig, ax = plt.subplots()
sns.boxplot(
    data=df_filtered,
    x="children",
    y="charges",
    hue="smoker",
    palette=PALETTE_SMOKER,
    ax=ax,
)
ax.set_title("Charges vs enfants selon le statut fumeur")
st.pyplot(fig)


children_smoker_mean = (
    df_filtered
    .groupby(["children", "smoker"])["charges"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots()
sns.lineplot(
    data=children_smoker_mean,
    x="children",
    y="charges",
    hue="smoker",
    palette=PALETTE_SMOKER,
    marker="o",
    ax=ax,
)
ax.set_title("Charges moyennes vs enfants selon le statut fumeur")
st.pyplot(fig)


# Analyse par région

st.header("Analyse par région")

region_counts = (
    df_filtered
    .groupby("region")
    .size()
    .reset_index(name="count")
)

region_charge_sum = (
    df_filtered
    .groupby("region")["charges"]
    .sum()
    .reset_index(name="total_charges")
)

region_charge_mean = (
    df_filtered
    .groupby("region")["charges"]
    .mean()
    .reset_index(name="mean_charges")
)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Nombre d'assurés**")
    st.dataframe(region_counts)

with col2:
    st.markdown("**Charges totales**")
    st.dataframe(region_charge_sum)

with col3:
    st.markdown("**Charges moyennes**")
    st.dataframe(region_charge_mean)


fig, ax = plt.subplots()
sns.barplot(
    data=region_counts,
    x="region",
    y="count",
    ax=ax,
)
ax.set_title("Nombre d'assurés par région")
st.pyplot(fig)


# Corrélations
st.header(" Corrélations numériques")

fig, ax = plt.subplots()
sns.heatmap(
    df_filtered[["age", "bmi", "children", "charges"]].corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax,
)
ax.set_title("Matrice de corrélation")
st.pyplot(fig)


# Conclusion métier

st.header(" Synthèse")

st.markdown(
    """
- Le **tabagisme** est le facteur le plus impactant sur les charges.
- L’**IMC élevé chez les fumeurs** amplifie fortement les coûts.
- Le nombre d’**enfants** a un effet secondaire.
- Les **régions** sont globalement équilibrées.
- Une transformation **log(charges)** est recommandée avant modélisation ML.
"""
)
