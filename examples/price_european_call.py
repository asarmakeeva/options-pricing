"""Example: Price a European call option using Monte Carlo simulation."""

from mc_pricing import EuropeanOption, MonteCarloSimulator, BlackScholes
from mc_pricing.variance_reduction import AntitheticVariates


def main():
    """Price a European call option and compare with Black-Scholes."""

    print("=" * 60)
    print("European Call Option Pricing Example")
    print("=" * 60)

    # Define option parameters
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    rate = 0.05
    volatility = 0.2

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
    print(f"  Spot Price: ${spot:.2f}")
    print(f"  Strike Price: ${strike:.2f}")
    print(f"  Time to Maturity: {maturity} years")
    print(f"  Risk-free Rate: {rate * 100:.1f}%")
    print(f"  Volatility: {volatility * 100:.1f}%")

    # Black-Scholes closed-form solution
    bs_price = BlackScholes.price(
        option_type='call',
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility
    )

    print(f"\n{'-' * 60}")
    print("Black-Scholes Price (Analytical):")
    print(f"  Price: ${bs_price:.4f}")

    # Monte Carlo pricing without variance reduction
    print(f"\n{'-' * 60}")
    print("Monte Carlo Pricing (Standard):")

    simulator = MonteCarloSimulator(n_simulations=100000, seed=42)
    mc_price, mc_error = simulator.price(option)

    print(f"  Simulations: {simulator.n_simulations:,}")
    print(f"  Price: ${mc_price:.4f} ± ${mc_error:.4f}")
    print(f"  Error vs BS: ${abs(mc_price - bs_price):.4f}")
    print(f"  Relative Error: {abs(mc_price - bs_price) / bs_price * 100:.2f}%")

    # Monte Carlo with antithetic variates
    print(f"\n{'-' * 60}")
    print("Monte Carlo with Antithetic Variates:")

    av_price, av_error = simulator.price(option, variance_reduction=AntitheticVariates())

    print(f"  Simulations: {simulator.n_simulations:,}")
    print(f"  Price: ${av_price:.4f} ± ${av_error:.4f}")
    print(f"  Error vs BS: ${abs(av_price - bs_price):.4f}")
    print(f"  Relative Error: {abs(av_price - bs_price) / bs_price * 100:.2f}%")

    # Variance reduction effectiveness
    variance_reduction_pct = (1 - (av_error / mc_error) ** 2) * 100
    print(f"\n  Variance Reduction: {variance_reduction_pct:.1f}%")

    # Confidence interval
    print(f"\n{'-' * 60}")
    print("95% Confidence Interval:")

    price_ci, lower, upper = simulator.price_with_confidence_interval(
        option, confidence=0.95, variance_reduction=AntitheticVariates()
    )

    print(f"  Price: ${price_ci:.4f}")
    print(f"  95% CI: [${lower:.4f}, ${upper:.4f}]")
    print(f"  Width: ${upper - lower:.4f}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
