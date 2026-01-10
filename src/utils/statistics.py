"""
Fonctions statistiques pour l'analyse des données.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Calcule le coefficient de Cramer (Cramer's V).

    Mesure l'association entre deux variables catégorielles.

    Parameters
    ----------
    x : pd.Series
        Première variable catégorielle.
    y : pd.Series
        Deuxième variable catégorielle.

    Returns
    -------
    float
        Coefficient de Cramer entre 0 et 1.
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape

    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1) if r > 1 else r
    kcorr = k - ((k - 1) ** 2) / (n - 1) if k > 1 else k

    denominator = min((kcorr - 1), (rcorr - 1))
    if denominator == 0:
        return 0.0
    return np.sqrt(phi2corr / denominator)


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """
    Calcule le ratio de corrélation (Eta-squared).

    Mesure la proportion de variance d'une variable numérique
    expliquée par une variable catégorielle.

    Parameters
    ----------
    categories : pd.Series
        Variable catégorielle.
    values : pd.Series
        Variable numérique.

    Returns
    -------
    float
        Ratio de corrélation entre 0 et 1.
    """
    df_temp = pd.DataFrame({
        "categories": categories,
        "values": values
    }).dropna()

    if df_temp.empty:
        return np.nan

    categories = df_temp["categories"]
    values = df_temp["values"]

    if len(categories.unique()) <= 1 or len(values) < 2:
        return 0.0

    groups = [values[categories == c] for c in categories.unique()]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return 0.0

    ss_total = np.sum((values - values.mean()) ** 2)

    if ss_total == 0:
        return 0.0

    ss_between = 0
    for group in groups:
        if len(group) > 0:
            group_mean = group.mean()
            global_mean = values.mean()
            ss_between += len(group) * (group_mean - global_mean) ** 2

    eta_squared = ss_between / ss_total
    return eta_squared


def calculate_association_matrix(
    df: pd.DataFrame,
    numerical_cols: list,
    categorical_cols: list
) -> pd.DataFrame:
    """
    Calcule une matrice d'association comprehensive.

    Combine Pearson, Cramer's V et Eta-squared selon les types de variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame source.
    numerical_cols : list
        Liste des colonnes numériques.
    categorical_cols : list
        Liste des colonnes catégorielles.

    Returns
    -------
    pd.DataFrame
        Matrice d'association.
    """
    all_vars = numerical_cols + categorical_cols
    matrix = pd.DataFrame(np.nan, index=all_vars, columns=all_vars)

    for col1 in all_vars:
        for col2 in all_vars:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
                continue

            if pd.notna(matrix.loc[col2, col1]):
                matrix.loc[col1, col2] = matrix.loc[col2, col1]
                continue

            type1_is_num = col1 in numerical_cols
            type2_is_num = col2 in numerical_cols

            if type1_is_num and type2_is_num:
                val = df[col1].corr(df[col2])
            elif not type1_is_num and not type2_is_num:
                if (len(df[col1].unique()) > 1 and len(df[col2].unique()) > 1):
                    val = cramers_v(df[col1], df[col2])
                else:
                    val = 0.0
            else:
                num_col = col1 if type1_is_num else col2
                cat_col = col1 if not type1_is_num else col2
                val = correlation_ratio(df[cat_col], df[num_col])

            matrix.loc[col1, col2] = val
            matrix.loc[col2, col1] = val

    return matrix


def run_chi2_test(contingency_table: pd.DataFrame) -> tuple:
    """
    Exécute un test du Chi-carré et retourne les résultats.

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Table de contingence.

    Returns
    -------
    tuple
        (chi2_stat, p_value, is_valid)
    """
    is_valid = (
        not contingency_table.empty
        and contingency_table.shape[0] > 1
        and contingency_table.shape[1] > 1
    )

    if not is_valid:
        return None, None, False

    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2_stat, p_value, True