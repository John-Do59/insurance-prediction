"""Dashboard d'analyse des données d'assurance.

Ce module fournit une interface Streamlit pour l'exploration
et la visualisation des facteurs de coûts d'assurance.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# ==========================================
# Configuration & Style
# ==========================================
st.set_page_config(
    page_title="Insurance Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# Data Loading
# ==========================================
@st.cache_data
def load_data():
    """Charge et prépare les données d'assurance."""
    data = pd.read_csv("data/insurance.csv")
    # Basic cleaning
    data['smoker_label'] = data['smoker'].map(
        {'yes': 'Fumeur', 'no': 'Non-fumeur'}
    )
    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"Erreur de chargement des données : {e}")
    st.stop()


# ==========================================
# Sidebar Filters
# ==========================================
# Placeholder for branding
LOGO_URL = (
    "https://www.google.com/images/branding/"
    "googlelogo/2x/googlelogo_color_92x30dp.png"
)
st.sidebar.image(LOGO_URL, width=150)
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

region_list = ["Toutes"] + list(df['region'].unique())
selected_region = st.sidebar.selectbox("Région Géographique", region_list)

age_range = st.sidebar.slider(
    "Tranche d'âge",
    int(df['age'].min()),
    int(df['age'].max()),
    (int(df['age'].min()), int(df['age'].max()))
)

# Filtering logic
df_filtered = df[
    (df['age'] >= age_range[0]) & (df['age'] <= age_range[1])
]
if selected_region != "Toutes":
    df_filtered = df_filtered[df_filtered['region'] == selected_region]


# ==========================================
# Header & Key Metrics
# ==========================================
st.title("Analyse des Risques & Charges d'Assurance")
st.markdown(
    f"**Semaine 1 : Exploration des Facteurs de Coûts** | "
    f"{df_filtered.shape[0]} observations sélectionnées"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "Charges Moyennes",
    f"{df_filtered['charges'].mean():,.0f} $",
    delta=None
)
col2.metric("Âge Médian", f"{df_filtered['age'].median():.0f} ans")
col3.metric("IMC Moyen", f"{df_filtered['bmi'].mean():.1f}")
smoker_ratio = (df_filtered['smoker'] == 'yes').mean()
col4.metric("% Fumeurs", f"{smoker_ratio:.1%}")

st.markdown("---")


# ==========================================
# Main Dashboard Tabs
# ==========================================
tab_dist, tab_health, tab_demo = st.tabs([
    "Distribution des Coûts",
    "Facteurs de Santé",
    "Démographie & Profils"
])

# --- Tab 1: Distribution ---
with tab_dist:
    st.subheader("Analyse de la Variable Cible : Charges")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        use_log = st.checkbox(
            "Appliquer l'échelle logarithmique (Log scale)",
            help="Aide à visualiser les distributions asymétriques"
        )

        fig_hist = px.histogram(
            df_filtered,
            x="charges",
            nbins=50,
            color_discrete_sequence=['#3b82f6'],
            marginal="box",
            log_x=use_log,
            title="Distribution des charges médicales"
        )
        fig_hist.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.info("""
        **Observation Métier :**
        - La distribution est fortement **asymétrique à droite**
          (skewness positive).
        - La majorité des dossiers sont sous les 15k$, mais une
          minorité génère des coûts très élevés (>40k$).
        - **Impact IA :** Une transformation Log sera probablement
          nécessaire pour améliorer la performance de la régression.
        """)

        # Charges vs Smoker Insight directly below info
        color_map_smoker = {
            'Fumeur': '#ef4444',
            'Non-fumeur': '#10b981'
        }
        fig_box_smoker = px.box(
            df_filtered,
            x="smoker_label",
            y="charges",
            color="smoker_label",
            points="all",
            title="Répartition des charges par statut fumeur",
            color_discrete_map=color_map_smoker
        )
        st.plotly_chart(fig_box_smoker, use_container_width=True)

# --- Tab 2: Health Factors ---
with tab_health:
    st.subheader("L'influence de l'IMC et du Tabagisme")

    col_h1, col_h2 = st.columns([2, 1])

    with col_h1:
        color_map_health = {
            'Fumeur': '#ef4444',
            'Non-fumeur': '#10b981'
        }
        fig_scatter = px.scatter(
            df_filtered,
            x="bmi",
            y="charges",
            color="smoker_label",
            size="age",
            hover_data=['age', 'children'],
            title="Impact combiné de l'IMC et du Tabac",
            labels={
                "bmi": "IMC (Indice de Masse Corporelle)",
                "charges": "Charges ($)"
            },
            color_discrete_map=color_map_health
        )
        # Adding a reference line at BMI 30
        fig_scatter.add_vline(
            x=30,
            line_dash="dash",
            line_color="gray",
            annotation_text="Seuil Obésité (30)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_h2:
        st.success("""
        **Insight Clé : L'Effet Synergie**
        - Chez les **non-fumeurs**, l'IMC a une influence
          linéaire modérée.
        - Chez les **fumeurs**, on observe une rupture brutale
          au-delà d'un **IMC de 30**.
        - Ce groupe (Fumeur + Obèse) représente le risque
          financier le plus élevé pour l'assureur.
        """)

        # Pie chart for BMI categories
        df_filtered['bmi_cat'] = pd.cut(
            df_filtered['bmi'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Insuffisant', 'Normal', 'Surpoids', 'Obèse']
        )
        fig_pie = px.pie(
            df_filtered,
            names='bmi_cat',
            title="Répartition des catégories d'IMC",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --- Tab 3: Demographics ---
with tab_demo:
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.subheader("Analyse par Âge & Dépendants")
        fig_age = px.scatter(
            df_filtered,
            x="age",
            y="charges",
            color="smoker_label",
            trendline="ols",
            title="Évolution des charges avec l'âge"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with col_d2:
        st.subheader("Analyse Géographique & Genre")

        demo_option = st.selectbox(
            "Comparer les charges moyennes par :",
            ["Région", "Sexe", "Nombre d'enfants"]
        )

        if demo_option == "Région":
            target_col = "region"
        elif demo_option == "Sexe":
            target_col = "sex"
        else:
            target_col = "children"

        avg_charges = (
            df_filtered
            .groupby(target_col)['charges']
            .mean()
            .reset_index()
            .sort_values('charges', ascending=False)
        )

        fig_bar = px.bar(
            avg_charges,
            x=target_col,
            y="charges",
            color="charges",
            title=f"Charges moyennes par {demo_option}",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# Synthesis & Next Steps
# ==========================================
st.markdown("---")
st.subheader("Synthèse & Hypothèses de Modélisation")

expander = st.expander("Voir les conclusions de l'EDA", expanded=True)
with expander:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Facteurs Dominants :**
        1. **Statut Fumeur** : Premier prédicteur de coût.
        2. **IMC (BMI)** : Prédicteur multiplicateur chez les fumeurs.
        3. **Âge** : Augmentation régulière et stratifiée des coûts.
        """)
    with c2:
        st.markdown("""
        **Risques identifiés pour la S2 :**
        - **Déséquilibre** : Moins de fumeurs (20%)
          que de non-fumeurs.
        - **Outliers** : Cas complexes (Maladies graves ?)
          générant des charges > 45k$.
        - **Non-linéarité** : Interaction forte IMC x Smoker.
        """)

st.write("---")
st.caption("Dashboard Analytics Insurance v2.0 - Equipe Dev Data IA")