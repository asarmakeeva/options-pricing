"""Variance reduction techniques for Monte Carlo simulation."""

from .antithetic import AntitheticVariates
from .control_variates import ControlVariates
from .importance_sampling import ImportanceSampling

__all__ = ['AntitheticVariates', 'ControlVariates', 'ImportanceSampling']
