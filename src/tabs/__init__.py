"""Package des onglets de l'application."""

from .overview import render_tab_overview
from .distribution import render_tab_distribution
from .health import render_tab_health
from .demographics import render_tab_demographics
from .correlations import render_tab_correlations
from .expert import render_tab_expert
from .modeling_prep import render_tab_modeling_prep
from .prediction import render_tab_prediction
from .inside_model import render_tab_inside_model

__all__ = [
    "render_tab_overview",
    "render_tab_distribution",
    "render_tab_health",
    "render_tab_demographics",
    "render_tab_correlations",
    "render_tab_expert",
    "render_tab_modeling_prep",
    "render_tab_prediction",
    "render_tab_inside_model",
]