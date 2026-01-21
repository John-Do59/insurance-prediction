from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_ = X.copy()
        # Interactions et termes polynomiaux
        # Note: mapping is done internally, this assumes input is raw or partially processed?
        # In the notebook, it appeared right before the ColumnTransformer.
        # Let's ensure these columns exist or fail gracefully if needed.
        # Based on notebook logic:
        if 'smoker' in X_.columns and 'bmi' in X_.columns:
            # We need to handle potential categorical string values if they haven't been encoded yet.
            # The notebook code used: X_['smoker'].map({'yes': 1, 'no': 0}) * X_['bmi']
            # This implies 'smoker' column has 'yes'/'no' values at this stage.
            X_['smoker_bmi'] = X_['smoker'].map({'yes': 1, 'no': 0}) * X_['bmi']
        
        if 'age' in X_.columns:
            X_['age_squared'] = X_['age'] ** 2
            
        return X_
