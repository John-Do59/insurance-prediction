"""
Module de chargement et préparation des données.
"""

import streamlit as st
import pandas as pd


@st.cache_data
def load_data(filepath: str = "data/insurance.csv") -> pd.DataFrame:
    """
    Charge et prépare les données d'assurance.

    Parameters
    ----------
    filepath : str
        Chemin vers le fichier CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les données préparées.
    """
    data = pd.read_csv(filepath)
    data["smoker_label"] = data["smoker"].map({
        "yes": "Fumeur",
        "no": "Non-fumeur"
    })
    return data


def load_data_with_error_handling(filepath: str = "data/insurance.csv"):
    """
    Charge les données avec gestion d'erreurs.

    Returns
    -------
    pd.DataFrame or None
        DataFrame si succès, None sinon.
    """
    try:
        return load_data(filepath)
    except FileNotFoundError:
        st.error(f"Fichier '{filepath}' introuvable.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur de chargement des données : {e}")
        st.stop()