"""Monte Carlo Options Pricing Engine.

A comprehensive library for pricing European and Asian options using
Monte Carlo simulation with variance reduction techniques.
"""

__version__ = "0.1.0"

# Core option classes
from .core import (
    BaseOption,
    EuropeanOption,
    AsianOption,
    BlackScholes
)

# Monte Carlo simulation
from .monte_carlo import (
    MonteCarloSimulator,
    PathGenerator,
    ConvergenceAnalyzer
)

# Variance reduction
from .variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling
)

# Greeks
from .greeks import (
    FiniteDifferenceGreeks,
    PathwiseGreeks
)

# Calibration
from .calibration import (
    SmileCalibration,
    VolatilitySurface
)

# Utilities
from .utils import (
    Visualizer,
    PerformanceMetrics,
    DataLoader
)

__all__ = [
    # Core
    'BaseOption',
    'EuropeanOption',
    'AsianOption',
    'BlackScholes',
    # Monte Carlo
    'MonteCarloSimulator',
    'PathGenerator',
    'ConvergenceAnalyzer',
    # Variance Reduction
    'AntitheticVariates',
    'ControlVariates',
    'ImportanceSampling',
    # Greeks
    'FiniteDifferenceGreeks',
    'PathwiseGreeks',
    # Calibration
    'SmileCalibration',
    'VolatilitySurface',
    # Utilities
    'Visualizer',
    'PerformanceMetrics',
    'DataLoader',
]
