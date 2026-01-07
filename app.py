import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement du dataset
df = pd.read_csv("data/insurance.csv")


st.title("Mini-dashboard EDA - Insurance Dataset")

# Filtres interactifs
smoker_filter = st.multiselect("Sélectionner fumeur(s) :", options=df['smoker'].unique(), default=df['smoker'].unique())
region_filter = st.multiselect("Sélectionner région(s) :", options=df['region'].unique(), default=df['region'].unique())

# Appliquer les filtres
df_filtered = df[(df['smoker'].isin(smoker_filter)) & (df['region'].isin(region_filter))]


st.subheader("Distribution des charges")
fig, ax = plt.subplots()
sns.histplot(df_filtered['charges'], bins=50, kde=True, ax=ax)
st.pyplot(fig)

st.subheader("Charges vs BMI")
fig2, ax2 = plt.subplots()
sns.scatterplot(x='bmi', y='charges', hue='smoker', data=df_filtered, ax=ax2)
st.pyplot(fig2)
