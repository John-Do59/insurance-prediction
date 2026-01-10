"""
Configuration et styles de l'application.
"""

import streamlit as st

# URLs et constantes
LOGO_URL = (
    "https://www.google.com/images/branding/"
    "googlelogo/2x/googlelogo_color_92x30dp.png"
)

# Couleurs
COLORS = {
    "primary": "#1e3a8a",
    "smoker": "#ef4444",
    "non_smoker": "#10b981",
    "secondary": "#3b82f6",
    "neutral": "#94a3b8",
}

COLOR_MAP_SMOKER = {
    "Fumeur": COLORS["smoker"],
    "Non-fumeur": COLORS["non_smoker"],
}


def setup_page_config():
    """Configure la page Streamlit."""
    st.set_page_config(
        page_title="Insurance Analytics Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_css():
    """Applique le CSS personnalisé."""
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8f9fa;
        }
        [data-testid="metric-container"] {
            background-color: #1e3a8a;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            border: none;
            color: white !important;
        }
        [data-testid="stMetricLabel"] {
            color: #ffffff !important;
            font-size: 1rem !important;
            font-weight: 400 !important;
            opacity: 0.9;
        }
        [data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 2rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricDelta"] {
            color: #ffffff !important;
        }
        h1, h2, h3 {
            color: #1e3a8a !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )