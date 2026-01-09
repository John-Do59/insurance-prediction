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
    /* Style pour les cartes de métriques - Fond sombre, texte blanc */
    [data-testid="metric-container"] {
        background-color: #1e3a8a; /* Bleu foncé business */
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        border: none;
        color: white !important;
    }
    /* Forcer le texte en blanc pour la lisibilité sur fond sombre */
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
    /* Delta (variation) en blanc aussi si présent */
    [data-testid="stMetricDelta"] {
        color: #ffffff !important;
    }
    /* Amélioration des titres */
    h1, h2, h3 {
        color: #1e3a8a !important;
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
st.sidebar.title("Filtres de Données")
st.sidebar.markdown("---")

# Filtre 1: Région
st.sidebar.markdown("#### Région Géographique")
region_list = ["Toutes"] + list(df['region'].unique())
selected_region = st.sidebar.selectbox(
    "Sélectionner une région",
    region_list,
    label_visibility="collapsed"
)

# Filtre 2: Tranche d'âge
st.sidebar.markdown("#### Tranche d'Âge")
age_range = st.sidebar.slider(
    "Âge",
    int(df['age'].min()),
    int(df['age'].max()),
    (int(df['age'].min()), int(df['age'].max())),
    label_visibility="collapsed"
)

# Filtre 3: Sexe (NOUVEAU)
st.sidebar.markdown("#### Sexe")
sex_options = ["Tous", "male", "female"]
selected_sex = st.sidebar.radio(
    "Sexe",
    sex_options,
    label_visibility="collapsed"
)

# Filtre 4: Statut Fumeur (NOUVEAU)
st.sidebar.markdown("#### Statut Fumeur")
smoker_options = ["Tous", "yes", "no"]
selected_smoker = st.sidebar.radio(
    "Fumeur",
    smoker_options,
    format_func=lambda x: "Tous" if x == "Tous" else ("Fumeur" if x == "yes" else "Non-fumeur"),
    label_visibility="collapsed"
)

# Filtre 5: Plage IMC (NOUVEAU)
st.sidebar.markdown("#### Indice de Masse Corporelle (IMC)")
bmi_range = st.sidebar.slider(
    "IMC",
    float(df['bmi'].min()),
    float(df['bmi'].max()),
    (float(df['bmi'].min()), float(df['bmi'].max())),
    step=0.5,
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("Ajustez les filtres pour explorer les données")

# Filtering logic
df_filtered = df[
    (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
    (df['bmi'] >= bmi_range[0]) & (df['bmi'] <= bmi_range[1])
]

if selected_region != "Toutes":
    df_filtered = df_filtered[df_filtered['region'] == selected_region]

if selected_sex != "Tous":
    df_filtered = df_filtered[df_filtered['sex'] == selected_sex]

if selected_smoker != "Tous":
    df_filtered = df_filtered[df_filtered['smoker'] == selected_smoker]

# Statistiques de filtrage
st.sidebar.markdown("---")
st.sidebar.markdown("#### Résultat du Filtrage")
pct_filtered = (len(df_filtered) / len(df)) * 100
st.sidebar.metric(
    "Observations sélectionnées",
    f"{len(df_filtered)} / {len(df)}",
    delta=f"{pct_filtered:.1f}%"
)

# Bouton de téléchargement
csv_data = df_filtered.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Télécharger les données (CSV)",
    data=csv_data,
    file_name=f"insurance_filtered_{len(df_filtered)}_rows.csv",
    mime="text/csv",
    use_container_width=True
)

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
tab_dist, tab_health, tab_demo, tab_corr = st.tabs([
    "Distribution des Coûts",
    "Facteurs de Santé",
    "Démographie & Profils",
    "Corrélations & Stats"
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
    st.subheader("Analyse Démographique Approfondie")
    
    # Section 1: Distribution par Région
    st.markdown("#### Distribution des Charges par Région")
    col_r1, col_r2 = st.columns([2, 1])
    
    with col_r1:
        fig_region = px.box(
            df_filtered,
            x="region",
            y="charges",
            color="smoker_label",
            title="Charges par région et statut fumeur",
            points="outliers",  # Afficher uniquement les outliers
            color_discrete_map={
                'Fumeur': '#ef4444',
                'Non-fumeur': '#10b981'
            }
        )
        fig_region.update_layout(
            xaxis_title="Région",
            yaxis_title="Charges ($)"
        )
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col_r2:
        st.info("""
        **Lecture du Box Plot :**
        - **Boîte** : 50% des données (Q1 à Q3)
        - **Ligne médiane** : Valeur centrale
        - **Moustaches** : Étendue normale
        - **Points** : Valeurs extrêmes (outliers)
        
        **Insight :**
        Les fumeurs ont des charges plus élevées dans **toutes** les régions.
        """)
        
        # Statistiques par région
        region_stats = df_filtered.groupby('region')['charges'].agg(['mean', 'median']).round(0)
        st.markdown("**Moyennes par région :**")
        for region, row in region_stats.iterrows():
            st.metric(
                label=region,
                value=f"{row['mean']:,.0f} $",
                delta=f"Médiane: {row['median']:,.0f} $"
            )
    
    st.markdown("---")
    
    # Section 2: Comparaison Homme/Femme
    st.markdown("#### Comparaison Homme vs Femme")
    col_s1, col_s2 = st.columns([2, 1])
    
    with col_s1:
        # Grouped bar chart
        avg_by_sex_smoker = (
            df_filtered
            .groupby(['sex', 'smoker_label'])['charges']
            .mean()
            .reset_index()
        )
        
        fig_sex = px.bar(
            avg_by_sex_smoker,
            x="sex",
            y="charges",
            color="smoker_label",
            barmode="group",
            title="Charges moyennes par sexe et statut fumeur",
            labels={"sex": "Sexe", "charges": "Charges moyennes ($)"},
            color_discrete_map={
                'Fumeur': '#ef4444',
                'Non-fumeur': '#10b981'
            }
        )
        st.plotly_chart(fig_sex, use_container_width=True)
    
    with col_s2:
        st.success("""
        **Observation :**
        - Peu de différence entre hommes et femmes
        - Le **statut fumeur** est le facteur dominant
        - L'écart fumeur/non-fumeur est similaire pour les deux sexes
        """)
        
        # Calcul de l'écart
        male_smoker = avg_by_sex_smoker[
            (avg_by_sex_smoker['sex'] == 'male') & 
            (avg_by_sex_smoker['smoker_label'] == 'Fumeur')
        ]['charges'].values[0]
        
        male_nonsmoker = avg_by_sex_smoker[
            (avg_by_sex_smoker['sex'] == 'male') & 
            (avg_by_sex_smoker['smoker_label'] == 'Non-fumeur')
        ]['charges'].values[0]
        
        st.metric(
            "Écart Fumeur/Non-fumeur (Homme)",
            f"+{(male_smoker - male_nonsmoker):,.0f} $",
            delta=f"{((male_smoker/male_nonsmoker - 1) * 100):.0f}%"
        )
    
    st.markdown("---")
    
    # Section 3: Impact du nombre d'enfants
    st.markdown("#### Impact du Nombre d'Enfants")
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        fig_children = px.box(
            df_filtered,
            x="children",
            y="charges",
            color="smoker_label",
            title="Distribution des charges selon le nombre d'enfants",
            points="outliers",
            color_discrete_map={
                'Fumeur': '#ef4444',
                'Non-fumeur': '#10b981'
            }
        )
        fig_children.update_layout(
            xaxis_title="Nombre d'enfants",
            yaxis_title="Charges ($)"
        )
        st.plotly_chart(fig_children, use_container_width=True)
    
    with col_c2:
        st.warning("""
        **Tendance :**
        - Impact **modéré** du nombre d'enfants
        - Augmentation légère avec plus d'enfants
        - Toujours dominé par le statut fumeur
        
        **Note :** La plupart des assurés ont 0-2 enfants.
        """)
        
        # Distribution du nombre d'enfants
        children_dist = df_filtered['children'].value_counts().sort_index()
        st.markdown("**Répartition :**")
        for n_children, count in children_dist.items():
            pct = (count / len(df_filtered)) * 100
            st.write(f"{n_children} enfant(s): {count} ({pct:.1f}%)")


# --- Tab 4: Correlations & Stats ---
with tab_corr:
    st.subheader("Analyse des Corrélations et Statistiques Descriptives")
    
    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        # Heatmap de corrélation
        st.markdown("#### Matrice de Corrélation (Pearson)")
        corr_matrix = df_filtered[['age', 'bmi', 'children', 'charges']].corr()
        
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title="Corrélations entre variables numériques",
            labels=dict(color="Coefficient"),
            aspect="auto"
        )
        fig_heatmap.update_layout(
            xaxis_title="",
            yaxis_title="",
            height=400
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col_c2:
        st.info("""
        **Interprétation :**
        - **Forte** : |r| > 0.7
        - **Modérée** : 0.3 < |r| < 0.7
        - **Faible** : |r| < 0.3
        
        **Limites :**
        - Mesure uniquement les relations **linéaires**
        - Ne capture pas les interactions complexes
        - Variables catégorielles (`smoker`, `region`) exclues
        """)
        
        # Afficher les corrélations avec charges
        st.markdown("#### Corrélations avec `charges`")
        corr_with_charges = corr_matrix['charges'].drop('charges').sort_values(ascending=False)
        for var, corr_val in corr_with_charges.items():
            st.metric(
                label=var,
                value=f"{corr_val:.3f}",
                delta=None
            )
    
    st.markdown("---")
    
    # Statistiques descriptives
    col_s1, col_s2 = st.columns([1, 1])
    
    with col_s1:
        st.markdown("#### Statistiques Descriptives")
        stats_df = df_filtered[['age', 'bmi', 'children', 'charges']].describe().T
        stats_df = stats_df.round(2)
        st.dataframe(stats_df, use_container_width=True)
    
    with col_s2:
        st.markdown("#### Détection des Valeurs Extrêmes")
        
        # Calcul des outliers (méthode IQR)
        Q1 = df_filtered['charges'].quantile(0.25)
        Q3 = df_filtered['charges'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_filtered[
            (df_filtered['charges'] < lower_bound) | 
            (df_filtered['charges'] > upper_bound)
        ]
        
        col_o1, col_o2 = st.columns(2)
        col_o1.metric("Outliers détectés", len(outliers))
        col_o2.metric("% du dataset", f"{len(outliers)/len(df_filtered)*100:.1f}%")
        
        col_o3, col_o4 = st.columns(2)
        col_o3.metric("Charge min", f"{df_filtered['charges'].min():,.0f} $")
        col_o4.metric("Charge max", f"{df_filtered['charges'].max():,.0f} $")
        
        st.markdown(f"""
        **Seuils IQR :**
        - Limite inférieure : {lower_bound:,.0f} $
        - Limite supérieure : {upper_bound:,.0f} $
        """)
    
    # Affichage des outliers si présents
    if len(outliers) > 0:
        st.markdown("---")
        st.markdown("#### Liste des Cas Extrêmes (Top 10)")
        outliers_display = outliers[
            ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
        ].sort_values('charges', ascending=False).head(10)
        st.dataframe(outliers_display, use_container_width=True)
    
    # Section Chi² pour variables catégorielles
    st.markdown("---")
    st.markdown("#### Test du Chi² (Variables Catégorielles)")
    
    st.info("""
    **Pourquoi le Chi² ?**
    - La corrélation de Pearson **ne fonctionne pas** avec les variables catégorielles
    - Le test du Chi² mesure l'**indépendance** entre variables catégorielles
    - Permet d'analyser `smoker`, `region`, et `sex` (exclus de la heatmap)
    """)
    
    # Import scipy pour le test du Chi²
    from scipy.stats import chi2_contingency
    import numpy as np
    
    # Section Matrice de Corrélation Catégorielle (V de Cramér)
    st.markdown("#### Matrice de Corrélation Catégorielle (V de Cramér)")
    
    # Fonction pour calculer le V de Cramér
    def calculate_cramers_v(x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

    # Préparer les données pour la matrice
    cat_vars = ['sex', 'smoker', 'region']
    # On ajoute la version catégorisée des charges
    df_filtered['charges_cat'] = pd.cut(
        df_filtered['charges'],
        bins=3,
        labels=['Faible', 'Moyen', 'Elevé']
    )
    cols_to_compare = cat_vars + ['charges_cat']
    
    # Calculer la matrice
    cramers_matrix = pd.DataFrame(
        np.zeros((len(cols_to_compare), len(cols_to_compare))),
        columns=cols_to_compare,
        index=cols_to_compare
    )
    
    for col1 in cols_to_compare:
        for col2 in cols_to_compare:
            if col1 == col2:
                cramers_matrix.loc[col1, col2] = 1.0
            else:
                cramers_matrix.loc[col1, col2] = calculate_cramers_v(df_filtered[col1], df_filtered[col2])
    
    col_cm1, col_cm2 = st.columns([2, 1])
    
    with col_cm1:
        fig_cramers = px.imshow(
            cramers_matrix,
            text_auto='.2f',
            color_continuous_scale='Purples',
            title="Intensité de l'association (V de Cramér)",
            labels=dict(color="Association"),
            aspect="auto"
        )
        fig_cramers.update_layout(height=450)
        st.plotly_chart(fig_cramers, use_container_width=True)
    
    with col_cm2:
        st.info("""
        **V de Cramér :**
        - **0** : Indépendance totale
        - **1** : Association parfaite
        - Contrairement à Pearson, il n'y a pas de direction (+/-).
        
        **Observations :**
        - Le **smoker** a la plus forte association avec les **charges**.
        - La **region** et le **sex** ont une influence beaucoup plus faible.
        """)

    st.markdown("---")
    st.markdown("#### Tests de Dépendance Individuels")
    
    col_chi1, col_chi2 = st.columns([1, 1])
    
    with col_chi1:
        st.markdown("##### Test 1 : Smoker × Charges")
        
        # Table de contingence
        contingency_smoker = pd.crosstab(
            df_filtered['smoker'],
            df_filtered['charges_cat']
        )
        
        st.dataframe(contingency_smoker, use_container_width=True)
        
        # Test du Chi²
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_smoker)
        
        st.metric("Chi² statistique", f"{chi2_stat:.2f}")
        st.metric("p-value", f"{p_value:.2e}")
        
        if p_value < 0.001:
            st.success("**Relation très significative** (p < 0.001)")
        elif p_value < 0.05:
            st.success("**Relation significative** (p < 0.05)")
        else:
            st.warning("Pas de relation significative (p ≥ 0.05)")
    
    with col_chi2:
        st.markdown("##### Test 2 : Region × Smoker")
        
        # Table de contingence
        contingency_region = pd.crosstab(
            df_filtered['region'],
            df_filtered['smoker']
        )
        
        st.dataframe(contingency_region, use_container_width=True)
        
        # Test du Chi²
        chi2_stat2, p_value2, dof2, expected2 = chi2_contingency(contingency_region)
        
        st.metric("Chi² statistique", f"{chi2_stat2:.2f}")
        st.metric("p-value", f"{p_value2:.2e}")
        
        if p_value2 < 0.001:
            st.success("**Relation très significative** (p < 0.001)")
        elif p_value2 < 0.05:
            st.success("**Relation significative** (p < 0.05)")
        else:
            st.warning("Pas de relation significative (p ≥ 0.05)")
    
    # Test supplémentaire : Sex × Charges
    st.markdown("---")
    col_chi3, col_chi4 = st.columns([1, 1])
    
    with col_chi3:
        st.markdown("##### Test 3 : Sex × Charges (catégorisées)")
        
        contingency_sex = pd.crosstab(
            df_filtered['sex'],
            df_filtered['charges_cat']
        )
        
        st.dataframe(contingency_sex, use_container_width=True)
        
        chi2_stat3, p_value3, dof3, expected3 = chi2_contingency(contingency_sex)
        
        st.metric("Chi² statistique", f"{chi2_stat3:.2f}")
        st.metric("p-value", f"{p_value3:.2e}")
        
        if p_value3 < 0.001:
            st.success("**Relation très significative** (p < 0.001)")
        elif p_value3 < 0.05:
            st.success("**Relation significative** (p < 0.05)")
        else:
            st.warning("Pas de relation significative (p ≥ 0.05)")
    
    with col_chi4:
        st.markdown("##### Interprétation du Chi²")
        st.markdown("""
        **Hypothèse nulle (H₀)** : Les variables sont indépendantes
        
        **Règle de décision** :
        - Si **p < 0.05** : On rejette H₀ → Les variables sont **dépendantes**
        - Si **p ≥ 0.05** : On ne rejette pas H₀ → Pas de preuve de dépendance
        
        **Attendu** :
        - `smoker` × `charges` : **Forte dépendance** (facteur dominant)
        - `region` × `smoker` : Probablement **indépendant**
        - `sex` × `charges` : Probablement **indépendant**
        """)



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