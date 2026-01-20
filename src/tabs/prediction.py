"""
Onglet Prediction des Charges.
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import shap
import matplotlib.pyplot as plt

def render_tab_prediction():
    """Affiche l'onglet de prediction interactive."""
    st.subheader(" Predicteur de Charges d'Assurance")
    
    st.markdown(
        """
        Cette interface utilise notre **modele lineaire optimise (Ridge)** 
        pour estimer vos frais medicaux annuels en fonction de votre profil.
        """
    )

    # Chargement du modele
    model_path = "models/insurance_model_prod.joblib"
    explainer_path = "models/shap_explainer.joblib"
    
    if not os.path.exists(model_path):
        st.error("Le modele n'est pas encore genere. Veuillez executer le notebook 03.")
        return

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modele : {e}")
        return

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
        # Preparation des donnees
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": ["yes" if smoker == "Oui" else "no"],
            "region": [region]
        })

        # Feature Engineering AUTOMATIQUE via le Pipeline
        # On n'a plus besoin de calculer manuellement les interactions ici !
        # Le pipeline s'en charge.

        try:
            # Prediction
            prediction = model.predict(input_data)[0]

            # Affichage du resultat
            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown(f"### Estimation :\n# {prediction:,.2f} $")
            
            with c2:
                if smoker == "Oui" and bmi >= 30:
                    st.warning(" Profil a haut risque (Obese + Fumeur).")
                elif age > 50:
                    st.info(" L'age contribue a l'augmentation des frais.")
                else:
                    st.success(" Estimation moderee.")

            st.progress(min(max(prediction, 0) / 64000, 1.0))

            # Explication SHAP
            if os.path.exists(explainer_path):
                st.markdown("---")
                st.subheader(" Comprendre cette prediction (SHAP)")
                
                with st.spinner("Calcul des facteurs d'influence..."):
                    try:
                        explainer = joblib.load(explainer_path)
                        
                        # Il faut transformer les donnees comme lors de l'entrainement pour SHAP
                        # 1. Feature Engineering
                        eng_step = model.named_steps['engineer']
                        prep_step = model.named_steps['preprocessor']
                        
                        X_eng = eng_step.transform(input_data)
                        X_trans = prep_step.transform(X_eng)
                        
                        # Calcul SHAP val
                        shap_values = explainer(X_trans)
                        
                        # Assign feature names to correct "feature 1, feature 2..." issue
                        feature_names = prep_step.get_feature_names_out()
                        shap_values.feature_names = list(feature_names)
                        
                        # Bar plot (simpler than waterfall)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        shap.plots.bar(shap_values[0], show=False, max_display=10)
                        st.pyplot(fig)
                        
                        st.caption("Ce graphique montre comment chaque caracteristique contribue a augmenter (rouge) ou diminuer (bleu) la prediction par rapport a la moyenne.")
                        
                    except Exception as e:
                        st.warning(f"Impossible d'afficher l'explication detaillee : {e}")
            
        except Exception as e:
            st.error(f"Erreur lors de la prediction : {e}")
