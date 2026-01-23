import streamlit as st
import joblib
import pandas as pd
import os

MODEL_DIR = "notebooks/models"
MODELS = {
    "Linear": "linear_model.joblib",
    "Ridge": "ridge_model.joblib",
    "Lasso": "lasso_model.joblib"
}

# ======================
# Titre et description
# ======================
st.set_page_config(page_title="Prédiction Assurance Santé", layout="centered")
st.title("🩺 Prédiction des frais d'assurance santé")
st.markdown("""
Entrez les caractéristiques d'un client pour estimer ses frais annuels.
Les modèles ont été entraînés sur le jeu de données *insurance.csv*.
""")

# ======================
# Sélection du modèle
# ======================
st.sidebar.header("⚙️ Paramètres")
selected_model_name = st.sidebar.selectbox(
    "Choisir un modèle",
    options=list(MODELS.keys()),
    index=2  # Lasso par défaut
)

model_path = os.path.join(MODEL_DIR, MODELS[selected_model_name])

if not os.path.exists(model_path):
    st.error(f"❌ Modèle non trouvé : `{model_path}`\n\nAssurez-vous d'avoir exécuté le script d'entraînement.")
    st.stop()

try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle :\n`{e}`")
    st.stop()

st.sidebar.success(f"✅ Modèle **{selected_model_name}** chargé")

# ======================
# Inputs utilisateur
# ======================
st.subheader("📋 Informations du client")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Âge", min_value=18, max_value=65, value=30)
    bmi = st.slider("IMC (BMI)", min_value=15.0, max_value=50.0, value=25.0)
    children = st.slider("Nombre d'enfants", min_value=0, max_value=5, value=1)

with col2:
    sex = st.selectbox("Sexe", options=["female", "male"])
    smoker = st.selectbox("Fumeur ?", options=["no", "yes"])
    region = st.selectbox(
        "Région",
        options=["southeast", "southwest", "northeast", "northwest"]
    )

# ======================
# Prédiction
# ======================
if st.button("🔍 Prédire les frais"):
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'sex': [sex],
        'smoker': [smoker],
        'region': [region]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 **Frais prédits : {prediction:,.0f} $**")
        if smoker == "yes":
            st.info("💡 Le tabac augmente fortement les frais (≈ +23 000 $).")
    except Exception as e:
        st.error(f"Erreur de prédiction :\n`{e}`")

# ======================
# Pied de page
# ======================
st.markdown("---")
st.caption("Projet Semaine 3 — Pipeline + Optimisation des Modèles")