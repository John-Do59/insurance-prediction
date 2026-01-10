"""Package utilitaires."""

from .statistics import (
    cramers_v,
    correlation_ratio,
    calculate_association_matrix,
    run_chi2_test,
)

__all__ = [
    "cramers_v",
    "correlation_ratio", 
    "calculate_association_matrix",
    "run_chi2_test",
]