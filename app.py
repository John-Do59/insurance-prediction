import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement du dataset
df = pd.read_csv("data/insurance.csv")

st.title("Mini-dashboard EDA - Insurance Dataset")

# Filtres interactifs
smoker_filter = st.multiselect(
    "Sélectionner fumeur(s) :", 
    options=df['smoker'].unique(), 
    default=df['smoker'].unique()
)
region_filter = st.multiselect(
    "Sélectionner région(s) :", 
    options=df['region'].unique(), 
    default=df['region'].unique()
)

# Appliquer les filtres
df_filtered = df[(df['smoker'].isin(smoker_filter)) & (df['region'].isin(region_filter))]

# Histogramme des charges
st.subheader("Distribution des charges")
fig, ax = plt.subplots()
sns.histplot(df_filtered['charges'], bins=50, kde=True, ax=ax)
st.pyplot(fig)

# Scatterplot charges vs BMI
st.subheader("Charges vs BMI selon tabagisme")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df_filtered, ax=ax2)
st.pyplot(fig2)

# GroupBy sex: total et moyenne des charges
st.subheader("Charges totales et moyennes par sexe")
charges_by_sex = df_filtered.groupby('sex')['charges'].agg(['sum', 'mean']).reset_index()
st.dataframe(charges_by_sex)

# Barplot charges moyennes par sexe
st.subheader("Charges moyennes par sexe")
fig3, ax3 = plt.subplots()
sns.barplot(x='sex', y='mean', data=charges_by_sex, ax=ax3)
ax3.set_ylabel("Charges moyennes")
st.pyplot(fig3)
