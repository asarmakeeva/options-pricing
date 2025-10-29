# API Reference

## Core Classes

### BaseOption

Abstract base class for all option types.

```python
class BaseOption(ABC):
    def __init__(
        self,
        option_type: str,      # 'call' or 'put'
        strike: float,          # Strike price
        maturity: float,        # Time to maturity in years
        spot: float,            # Current spot price
        rate: float,            # Risk-free rate
        volatility: float,      # Volatility (sigma)
        dividend: float = 0.0   # Dividend yield
    )

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate payoff for given price paths."""
```

### EuropeanOption

European-style option (exercise only at maturity).

```python
class EuropeanOption(BaseOption):
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate European option payoff.

        Args:
            paths: Terminal prices or full paths

        Returns:
            Array of payoffs
        """
```

**Example:**
```python
option = EuropeanOption(
    option_type='call',
    strike=100.0,
    maturity=1.0,
    spot=100.0,
    rate=0.05,
    volatility=0.2
)
```

### AsianOption

Asian option with payoff based on average price.

```python
class AsianOption(BaseOption):
    def __init__(
        self,
        ...,  # BaseOption parameters
        averaging_type: str = 'arithmetic'  # 'arithmetic' or 'geometric'
    )
```

**Example:**
```python
asian = AsianOption(
    option_type='call',
    strike=100.0,
    maturity=1.0,
    spot=100.0,
    rate=0.05,
    volatility=0.2,
    averaging_type='arithmetic'
)
```

### BlackScholes

Black-Scholes analytical pricing.

```python
class BlackScholes:
    @staticmethod
    def price(
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0
    ) -> float:
        """Calculate European option price."""

    @staticmethod
    def greeks(
        option_type: str,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        volatility: float,
        dividend: float = 0.0
    ) -> Dict[str, float]:
        """Calculate all Greeks."""

    @staticmethod
    def implied_volatility(
        option_type: str,
        market_price: float,
        spot: float,
        strike: float,
        maturity: float,
        rate: float,
        dividend: float = 0.0,
        initial_guess: float = 0.2,
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> Optional[float]:
        """Calculate implied volatility."""
```

## Monte Carlo Simulation

### MonteCarloSimulator

Main Monte Carlo pricing engine.

```python
class MonteCarloSimulator:
    def __init__(
        self,
        n_simulations: int = 100000,
        n_steps: int = 252,
        seed: Optional[int] = None
    )

    def price(
        self,
        option: BaseOption,
        variance_reduction: Optional[Any] = None,
        return_paths: bool = False
    ) -> Tuple[float, float]:
        """
        Price option using Monte Carlo.

        Args:
            option: Option to price
            variance_reduction: Variance reduction technique
            return_paths: Whether to return price paths

        Returns:
            (price, standard_error) or (price, std_error, paths)
        """

    def price_with_confidence_interval(
        self,
        option: BaseOption,
        confidence: float = 0.95,
        variance_reduction: Optional[Any] = None
    ) -> Tuple[float, float, float]:
        """
        Price with confidence interval.

        Returns:
            (price, lower_bound, upper_bound)
        """
```

**Example:**
```python
simulator = MonteCarloSimulator(n_simulations=100000, seed=42)
price, std_error = simulator.price(option)
```

### PathGenerator

Generate price paths using GBM.

```python
class PathGenerator:
    def __init__(self, seed: Optional[int] = None)

    def generate_paths(
        self,
        spot: float,
        maturity: float,
        rate: float,
        volatility: float,
        n_paths: int,
        n_steps: int,
        dividend: float = 0.0,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Generate price paths.

        Returns:
            Array of shape (n_paths, n_steps + 1)
        """

    def generate_terminal_prices(
        self,
        spot: float,
        maturity: float,
        rate: float,
        volatility: float,
        n_paths: int,
        dividend: float = 0.0,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Generate only terminal prices (more efficient).

        Returns:
            Array of shape (n_paths,)
        """
```

## Variance Reduction

### AntitheticVariates

```python
class AntitheticVariates:
    def apply(
        self,
        option: BaseOption,
        payoffs: np.ndarray,
        paths: Optional[np.ndarray],
        path_generator: PathGenerator
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply antithetic variates technique."""
```

**Example:**
```python
from mc_pricing.variance_reduction import AntitheticVariates

price, error = simulator.price(option, variance_reduction=AntitheticVariates())
```

### ControlVariates

```python
class ControlVariates:
    def __init__(self, control_option: Optional[EuropeanOption] = None)

    def apply(...) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply control variates technique."""
```

### ImportanceSampling

```python
class ImportanceSampling:
    def __init__(self, theta: float = 0.5)

    def apply(...) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply importance sampling."""
```

## Greeks

### FiniteDifferenceGreeks

```python
class FiniteDifferenceGreeks:
    def __init__(
        self,
        simulator: MonteCarloSimulator,
        spot_bump: float = 0.01,
        vol_bump: float = 0.01,
        rate_bump: float = 0.0001,
        time_bump: float = 1.0 / 365.0
    )

    def delta(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> float:
        """Calculate Delta."""

    def gamma(self, option: BaseOption, ...) -> float:
        """Calculate Gamma."""

    def vega(self, option: BaseOption, ...) -> float:
        """Calculate Vega."""

    def theta(self, option: BaseOption, ...) -> float:
        """Calculate Theta."""

    def rho(self, option: BaseOption, ...) -> float:
        """Calculate Rho."""

    def all_greeks(
        self,
        option: BaseOption,
        variance_reduction: Optional[object] = None
    ) -> Dict[str, float]:
        """Calculate all Greeks at once."""
```

### PathwiseGreeks

```python
class PathwiseGreeks:
    def __init__(
        self,
        n_simulations: int = 100000,
        seed: Optional[int] = None
    )

    def delta(self, option: BaseOption) -> float:
        """Calculate Delta using pathwise method."""

    def vega(self, option: BaseOption) -> float:
        """Calculate Vega using pathwise method."""
```

## Calibration

### SmileCalibration

```python
class SmileCalibration:
    def __init__(self, model: str = 'quadratic')

    def fit(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        spot: float,
        maturity: float,
        rate: float,
        option_type: str = 'call',
        dividend: float = 0.0
    ) -> dict:
        """
        Fit smile model to market data.

        Returns:
            Dictionary with params and fit statistics
        """

    def volatility(self, strikes: np.ndarray) -> np.ndarray:
        """Get implied volatility for strikes."""
```

**Models:**
- `'quadratic'`: σ(K) = a + b*(K-K₀) + c*(K-K₀)²
- `'svi'`: SVI parameterization
- `'polynomial'`: Higher-order polynomial

### VolatilitySurface

```python
class VolatilitySurface:
    def fit(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        volatilities: np.ndarray,
        spot: float
    )

    def volatility(self, strike: float, maturity: float) -> float:
        """Get volatility for given strike and maturity."""

    def slice_by_maturity(
        self,
        maturity: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get smile for specific maturity."""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
```

## Utilities

### Visualizer

```python
class Visualizer:
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid')

    def plot_convergence(
        self,
        n_simulations: np.ndarray,
        prices: np.ndarray,
        std_errors: np.ndarray,
        true_price: Optional[float] = None,
        title: str = "Monte Carlo Convergence",
        save_path: Optional[str] = None
    )

    def plot_volatility_smile(...)
    def plot_volatility_surface(...)
    def plot_greeks_surface(...)
    def plot_price_paths(...)
    def compare_variance_reduction(...)
```

### PerformanceMetrics

```python
class PerformanceMetrics:
    @staticmethod
    def calculate_accuracy(
        estimated_price: float,
        true_price: float
    ) -> Dict[str, float]

    @staticmethod
    def variance_reduction_efficiency(
        std_error_baseline: float,
        std_error_reduced: float,
        time_baseline: float,
        time_reduced: float
    ) -> Dict[str, float]

    @staticmethod
    def compare_methods(
        option: BaseOption,
        methods: Dict[str, Any],
        true_price: Optional[float] = None,
        n_simulations: int = 100000,
        n_runs: int = 10
    ) -> Dict[str, Dict]
```

### DataLoader

```python
class DataLoader:
    @staticmethod
    def load_volatility_surface(
        filepath: str,
        delimiter: str = ','
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]

    @staticmethod
    def generate_sample_volatility_surface(
        spot: float = 100.0,
        strikes_pct: np.ndarray = np.linspace(0.8, 1.2, 11),
        maturities: np.ndarray = np.array([0.25, 0.5, 1.0, 2.0]),
        atm_vol: float = 0.2,
        smile_amplitude: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

## Type Hints

```python
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import pandas as pd
```
