"""Example: Benchmark and compare variance reduction techniques."""

from mc_pricing import EuropeanOption, MonteCarloSimulator, BlackScholes
from mc_pricing.variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling
)
from mc_pricing.utils import PerformanceMetrics
import time


def main():
    """Benchmark different variance reduction techniques."""

    print("=" * 70)
    print("Variance Reduction Techniques Benchmark")
    print("=" * 70)

    # Define option parameters
    spot = 100.0
    strike = 110.0  # OTM call
    maturity = 1.0
    rate = 0.05
    volatility = 0.25

    # Create European call option
    option = EuropeanOption(
        option_type='call',
        strike=strike,
        maturity=maturity,
        spot=spot,
        rate=rate,
        volatility=volatility
    )

    print(f"\nOption Parameters:")
    print(f"  Type: Call (OTM)")
    print(f"  Spot: ${spot:.2f}")
    print(f"  Strike: ${strike:.2f} (Moneyness: {spot/strike:.2f})")
    print(f"  Maturity: {maturity} years")
    print(f"  Rate: {rate * 100:.1f}%")
    print(f"  Volatility: {volatility * 100:.1f}%")

    # Black-Scholes price
    bs_price = BlackScholes.price(
        option_type='call',
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility
    )

    print(f"\nBlack-Scholes Price: ${bs_price:.4f}")

    # Define variance reduction methods
    methods = {
        'Standard MC': None,
        'Antithetic Variates': AntitheticVariates(),
        'Control Variates': ControlVariates(),
        'Importance Sampling': ImportanceSampling()
    }

    # Benchmark parameters
    n_simulations = 50000
    n_runs = 10

    print(f"\n{'-' * 70}")
    print(f"Benchmark Settings:")
    print(f"  Simulations per run: {n_simulations:,}")
    print(f"  Number of runs: {n_runs}")
    print(f"{'-' * 70}\n")

    # Run benchmark
    results = {}

    for name, variance_reduction in methods.items():
        print(f"Running {name}...")

        prices = []
        std_errors = []
        times = []

        for run in range(n_runs):
            simulator = MonteCarloSimulator(n_simulations=n_simulations, seed=42 + run)

            start = time.time()
            price, std_error = simulator.price(option, variance_reduction=variance_reduction)
            end = time.time()

            prices.append(price)
            std_errors.append(std_error)
            times.append(end - start)

        results[name] = {
            'mean_price': sum(prices) / len(prices),
            'mean_error': sum(std_errors) / len(std_errors),
            'mean_time': sum(times) / len(times),
            'accuracy': abs(sum(prices) / len(prices) - bs_price)
        }

    # Display results
    print(f"\n{'-' * 70}")
    print("Results:")
    print(f"{'-' * 70}")

    print(f"\n{'Method':<25} {'Price':<12} {'Std Error':<12} {'Time (s)':<10} {'Error':<10}")
    print(f"{'-' * 70}")

    for name, res in results.items():
        print(f"{name:<25} "
              f"${res['mean_price']:<11.4f} "
              f"${res['mean_error']:<11.4f} "
              f"{res['mean_time']:<10.3f} "
              f"${res['accuracy']:<10.4f}")

    # Calculate variance reduction percentages
    baseline_var = results['Standard MC']['mean_error'] ** 2
    baseline_time = results['Standard MC']['mean_time']

    print(f"\n{'-' * 70}")
    print("Variance Reduction Analysis:")
    print(f"{'-' * 70}\n")

    for name in results:
        if name == 'Standard MC':
            continue

        var = results[name]['mean_error'] ** 2
        var_reduction = (1 - var / baseline_var) * 100
        time_ratio = results[name]['mean_time'] / baseline_time

        # Efficiency: variance reduction per unit time
        efficiency = var_reduction / (time_ratio * 100)

        print(f"{name}:")
        print(f"  Variance Reduction: {var_reduction:>6.1f}%")
        print(f"  Time Ratio: {time_ratio:>6.2f}x")
        print(f"  Efficiency Score: {efficiency:>6.2f}")
        print()

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
