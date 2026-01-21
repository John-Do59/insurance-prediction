
import pandas as pd
import numpy as np
from joblib import dump
import shap


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from src.feature_engineering import FeatureEngineer

# Chargement des données
path = "data/insurance.csv"
df = pd.read_csv(path).drop_duplicates()

# Définition des colonnes
numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]
engineered_features = ["age_squared", "smoker_bmi"]

# Preprocessing numérique : Imputation + Scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing catégoriel : Imputation + OneHot
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Assembleur global
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features + engineered_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Pipeline Maître
# On recrée le pipeline avec les meilleurs params trouvés précédemment (Ridge alpha=0.1)
# D'après le notebook : Best Params : {'regressor__alpha': 0.1}
best_ridge = Ridge(alpha=0.1)

master_pipeline = Pipeline(steps=[
    ('engineer', FeatureEngineer()),
    ('preprocessor', preprocessor),
    ('regressor', best_ridge)
])

X = df.drop("charges", axis=1)
y = df["charges"]

# Split initial pour garder un jeu de test final intact
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrainement final
master_pipeline.fit(X_train, y_train)

# Évaluation finale sur le jeu de test
y_pred_final = master_pipeline.predict(X_test)
final_r2 = r2_score(y_test, y_pred_final)
final_mae = mean_absolute_error(y_test, y_pred_final)

print(f"Performance Finale sur Test Set :")
print(f"R²  : {final_r2:.4f}")
print(f"MAE : ${final_mae:.2f}")

# Sauvegarde
model_path = 'models/insurance_model_prod.joblib'
dump(master_pipeline, model_path)
print(f"Modèle sauvegardé sous : {model_path}")

# --- SHAP EXPLATION GENERATION ---
print("Génération de l'explainer SHAP...")

# Récupération des étapes du pipeline entrainé
engineer = master_pipeline.named_steps['engineer']
preprocessor = master_pipeline.named_steps['preprocessor']
regressor = master_pipeline.named_steps['regressor']

# Transformation des données de test pour le background dataset
# On utilise X_test pour calibrer l'explainer (comme dans le notebook)
X_test_transformed = preprocessor.transform(engineer.transform(X_test))

# Création de l'explainer
# LinearExplainer est optimisé pour les modèles linéaires (Ridge, Lasso...)
explainer = shap.LinearExplainer(regressor, X_test_transformed)

# Sauvegarde de l'explainer
explainer_path = 'models/shap_explainer.joblib'
dump(explainer, explainer_path)
print(f"Explainer SHAP sauvegardé sous : {explainer_path}")

