"""Package des onglets de l'application."""

from .distribution import render_tab_distribution
from .health import render_tab_health
from .demographics import render_tab_demographics
from .correlations import render_tab_correlations
from .expert import render_tab_expert

__all__ = [
    "render_tab_distribution",
    "render_tab_health",
    "render_tab_demographics",
    "render_tab_correlations",
    "render_tab_expert",
]