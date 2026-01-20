"""
Onglet "Inside the Model" pour l'explicabilité globale.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

def render_tab_inside_model(df):
    """
    Affiche l'analyse interne du modèle (Coefficients, SHAP).
    
    Parameters
    
    df : pd.DataFrame
        Le dataframe complet (utilisé pour un échantillon SHAP).
    """
    st.subheader(" Inside the Model : Comprendre la « Boîte Noire »")
    
    model_path = "models/insurance_model_prod.joblib"
    explainer_path = "models/shap_explainer.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(explainer_path):
        st.warning("Modèle ou Explainer introuvable. Veuillez exécuter le Notebook 03.")
        return

    try:
        model = joblib.load(model_path)
        explainer = joblib.load(explainer_path)
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return

    # --- 1. Coefficients du Modèle (Feature Importance Linéaire) ---
    st.markdown("### 1. Importance Globale des Features (Coefficients)")
    st.markdown(
        "Puisque nous utilisons un modèle linéaire (Ridge/Lasso), "
        "la valeur absolue des coefficients indique l'importance de chaque variable."
    )
    
    try:
        # Extraction des noms de features
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']
        
        # Noms numériques (issus du FeatureEngineer - attention à l'ordre)
        # On sait que FeatureEngineer ajoute 'smoker_bmi' et 'age_squared'
        # et que numeric_features = ["age", "bmi", "children"]
        # L'ordre dans ColumnTransformer : 'num' puis 'cat'
        
        # On essaie de récupérer dynamiquement si possible, sinon hardcode intelligent
        num_cols = ["age", "bmi", "children", "age_squared", "smoker_bmi"]
        cat_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(["sex", "smoker", "region"])
        
        feature_names = np.concatenate([num_cols, cat_cols])
        coefs = regressor.coef_
        
        # Création DataFrame
        feat_imp = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefs,
            "Abs_Coef": np.abs(coefs)
        }).sort_values(by="Abs_Coef", ascending=False)
        
        st.dataframe(
            feat_imp[["Feature", "Coefficient"]].style.background_gradient(cmap="coolwarm", subset=["Coefficient"]),
            use_container_width=True
        )
        
        st.caption("Note : Les données sont standardisées, donc les coefficients sont comparables.")
        
    except Exception as e:
        st.error(f"Impossible d'extraire les coefficients simplement : {e}")

    st.markdown("---")

    # --- 2. SHAP Summary Plot ---
    st.markdown("### 2. Analyse SHAP Globale")
    st.markdown(
        "Le graphique SHAP résume l'impact de chaque feature sur l'ensemble du dataset."
    )
    
    if st.button("Générer le graphique SHAP (peut prendre quelques secondes)"):
        with st.spinner("Calcul des valeurs SHAP sur un échantillon..."):
            try:
                # On prend un échantillon pour la rapidité
                sample = df.sample(min(200, len(df)), random_state=42).drop("charges", axis=1)
                
                # Transformation
                eng_step = model.named_steps['engineer']
                prep_step = model.named_steps['preprocessor']
                
                X_eng = eng_step.transform(sample)
                X_trans = prep_step.transform(X_eng)
                
                # Calcul SHAP
                shap_values = explainer(X_trans)
                
                # Assign names (cast to list to avoid numpy errors)
                shap_values.feature_names = list(feature_names)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.title("Importance Globale des Features (Impact absolu moyen)")
                # Pass list of feature names explicitly and use bar plot
                shap.summary_plot(shap_values, X_trans, feature_names=list(feature_names), plot_type="bar", show=False)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Erreur lors du calcul SHAP : {e}")
