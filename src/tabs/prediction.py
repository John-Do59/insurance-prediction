"""
Onglet Prediction des Charges.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

def render_tab_prediction():
    """Affiche l'onglet de prediction interactive."""
    st.subheader("🔮 Predicteur de Charges d'Assurance")
    
    st.markdown(
        """
        Cette interface utilise notre **modele lineaire optimise (R² = 0.9324)** 
        pour estimer vos frais medicaux annuels en fonction de votre profil.
        """
    )

    # Chargement du modele
    model_path = "models/model.joblib"
    if not os.path.exists(model_path):
        st.error("Le modele n'est pas encore genere. Veuillez executer le script d'entrainement.")
        return

    model = joblib.load(model_path)

    # Formulaire
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 100, 30)
            bmi = st.number_input("Indice de Masse Corporelle (BMI)", 15.0, 60.0, 25.0, step=0.1)
            children = st.selectbox("Nombre d'enfants", [0, 1, 2, 3, 4, 5])
            
        with col2:
            smoker = st.radio("Etes-vous fumeur ?", ["Non", "Oui"])
            sex = st.radio("Genre", ["female", "male"])
            region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])
            
        submit = st.form_submit_button("Calculer l'estimation")

    if submit:
        # Preparation des donnees (le format doit correspondre exactement a X lors du fit)
        # On cree un DataFrame avec les memes noms de colonnes originales
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": ["yes" if smoker == "Oui" else "no"],
            "region": [region]
        })

        # Calcul des features ingenierees (necessaire avant de passer au pipeline)
        # Note: Le pipeline contient le preprocesseur, mais pas les transformations de colonnes 
        # effectuees avant le fit (age2, bmi_smoker, is_obese_smoker)
        # On doit les ajouter ici car le modele a ete entraine AVEC ces colonnes supplementaires dans X.
        
        input_data['is_obese_smoker'] = ((input_data['bmi'] >= 30) & (input_data['smoker'] == 'yes')).astype(int)
        input_data['bmi_smoker'] = input_data['bmi'] * input_data['smoker'].map({'yes': 1, 'no': 0})
        input_data['age2'] = input_data['age'] ** 2

        # Prediction
        prediction = model.predict(input_data)[0]

        # Affichage du resultat
        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.markdown(f"### Estimation :\n# {prediction:,.2f} $")
        
        with c2:
            # Message contextuel
            if smoker == "Oui" and bmi >= 30:
                st.warning("⚠️ Profil a haut risque (Obese + Fumeur). Les charges sont fortement augmentees par l'interaction de ces deux facteurs.")
            elif age > 50:
                st.info("ℹ️ L'age avance contribue de maniere polynomiale a l'augmentation des frais.")
            else:
                st.success("✅ Votre profil presente des charges estimees moderees.")

        # Explication visuelle simple
        st.progress(min(prediction / 64000, 1.0))
        st.caption("Positionnement de l'estimation par rapport au maximum du dataset (~64k $)")
