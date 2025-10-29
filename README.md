# Monte Carlo Options Pricing Engine

A comprehensive Python library for pricing European and Asian options using Monte Carlo simulation with advanced variance reduction techniques.

## Features

- **Multiple Option Types**: European and Asian options (arithmetic and geometric averages)
- **Monte Carlo Simulation**: Flexible path generation with Geometric Brownian Motion
- **Variance Reduction Techniques**:
  - Antithetic Variates
  - Control Variates
  - Importance Sampling
- **Greeks Calculation**:
  - Finite Difference Method (Delta, Gamma, Vega, Theta, Rho)
  - Pathwise Derivative Method
- **Black-Scholes Comparison**: Closed-form solutions for validation
- **Volatility Smile Calibration**: Fit implied volatility surfaces to market data
- **Comprehensive Analysis**: Jupyter notebooks with detailed examples

## Installation

### From Source

```bash
git clone https://github.com/yourusername/monte-carlo-options-pricer.git
cd monte-carlo-options-pricer
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
```

## Quick Start

### Price a European Call Option

```python
from mc_pricing.core import EuropeanOption
from mc_pricing.monte_carlo import MonteCarloSimulator

# Define option parameters
option = EuropeanOption(
    option_type='call',
    strike=100.0,
    maturity=1.0,
    spot=100.0,
    rate=0.05,
    volatility=0.2
)

# Create simulator
simulator = MonteCarloSimulator(n_simulations=100000, seed=42)

# Price the option
price, std_error = simulator.price(option)
print(f"Option Price: ${price:.4f} ± ${std_error:.4f}")
```

### Apply Variance Reduction

```python
from mc_pricing.variance_reduction import AntitheticVariates

# Use antithetic variates
price_av, std_error_av = simulator.price(
    option,
    variance_reduction=AntitheticVariates()
)
print(f"Price with Antithetic Variates: ${price_av:.4f} ± ${std_error_av:.4f}")
```

### Calculate Greeks

```python
from mc_pricing.greeks import FiniteDifferenceGreeks

greeks_calc = FiniteDifferenceGreeks(simulator)
delta = greeks_calc.delta(option)
gamma = greeks_calc.gamma(option)
vega = greeks_calc.vega(option)

print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Vega: {vega:.4f}")
```

## Project Structure

```
monte-carlo-options-pricer/
├── src/mc_pricing/          # Main source code
│   ├── core/                # Option classes and Black-Scholes
│   ├── monte_carlo/         # MC simulation engines
│   ├── variance_reduction/  # Variance reduction techniques
│   ├── greeks/              # Greeks calculation
│   ├── calibration/         # Volatility surface calibration
│   └── utils/               # Utility functions
├── tests/                   # Unit tests
├── notebooks/               # Jupyter notebooks with examples
├── examples/                # Standalone example scripts
├── docs/                    # Documentation
└── data/                    # Sample data
```

## Documentation

- [Theory and Mathematical Background](docs/theory.md)
- [API Reference](docs/api_reference.md)
- [Algorithm Descriptions](docs/algorithms.md)
- [Performance Benchmarks](docs/benchmarks.md)

## Jupyter Notebooks

The `notebooks/` directory contains detailed analysis examples:

1. **01_basic_pricing.ipynb**: Introduction to European and Asian options
2. **02_variance_reduction.ipynb**: Comparison of variance reduction techniques
3. **03_greeks_analysis.ipynb**: Greeks surfaces and hedging strategies
4. **04_convergence_study.ipynb**: Monte Carlo convergence analysis
5. **05_smile_calibration.ipynb**: Volatility smile fitting
6. **06_full_comparison.ipynb**: Monte Carlo vs Black-Scholes comparison

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src/mc_pricing --cov-report=html

# Run specific test file
pytest tests/test_core/test_european_option.py
```

## Benchmarks

Run the benchmark suite to compare different techniques:

```bash
python scripts/run_benchmarks.py
```

## Performance

- Efficient NumPy-based implementations
- Optional Numba JIT compilation for critical paths
- Parallel processing support for large-scale simulations
- Typical pricing time: ~100ms for 100,000 paths

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. Pearson.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
- Shreve, S. E. (2004). *Stochastic Calculus for Finance II*. Springer.

## Authors

Options Pricing Team

## Acknowledgments

Special thanks to the quantitative finance community for their valuable insights and contributions to options pricing theory.
