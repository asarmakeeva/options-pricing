"""Example: Calculate option Greeks using Monte Carlo and Black-Scholes."""

from mc_pricing import EuropeanOption, MonteCarloSimulator, BlackScholes
from mc_pricing.greeks import FiniteDifferenceGreeks, PathwiseGreeks


def main():
    """Calculate and compare Greeks from different methods."""

    print("=" * 70)
    print("Option Greeks Calculation Example")
    print("=" * 70)

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
    print(f"  Type: Call")
    print(f"  Spot Price: ${spot:.2f}")
    print(f"  Strike Price: ${strike:.2f}")
    print(f"  Time to Maturity: {maturity} years")
    print(f"  Risk-free Rate: {rate * 100:.1f}%")
    print(f"  Volatility: {volatility * 100:.1f}%")

    # Black-Scholes Greeks (analytical)
    print(f"\n{'-' * 70}")
    print("Black-Scholes Greeks (Analytical):")
    print(f"{'-' * 70}")

    bs_greeks = BlackScholes.greeks(
        option_type='call',
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        volatility=volatility
    )

    print(f"  Delta:  {bs_greeks['delta']:>8.4f}  (∂V/∂S)")
    print(f"  Gamma:  {bs_greeks['gamma']:>8.4f}  (∂²V/∂S²)")
    print(f"  Vega:   {bs_greeks['vega']:>8.4f}  (∂V/∂σ per 1%)")
    print(f"  Theta:  {bs_greeks['theta']:>8.4f}  (∂V/∂t per day)")
    print(f"  Rho:    {bs_greeks['rho']:>8.4f}  (∂V/∂r per 1%)")

    # Monte Carlo Greeks using finite differences
    print(f"\n{'-' * 70}")
    print("Monte Carlo Greeks (Finite Difference Method):")
    print(f"{'-' * 70}")

    simulator = MonteCarloSimulator(n_simulations=50000, seed=42)
    fd_greeks_calc = FiniteDifferenceGreeks(simulator)

    print(f"  Computing with {simulator.n_simulations:,} simulations...")
    print()

    mc_delta = fd_greeks_calc.delta(option)
    mc_gamma = fd_greeks_calc.gamma(option)
    mc_vega = fd_greeks_calc.vega(option)
    mc_theta = fd_greeks_calc.theta(option)
    mc_rho = fd_greeks_calc.rho(option)

    print(f"  Delta:  {mc_delta:>8.4f}  (Error: {abs(mc_delta - bs_greeks['delta']):>7.4f})")
    print(f"  Gamma:  {mc_gamma:>8.4f}  (Error: {abs(mc_gamma - bs_greeks['gamma']):>7.4f})")
    print(f"  Vega:   {mc_vega:>8.4f}  (Error: {abs(mc_vega - bs_greeks['vega']):>7.4f})")
    print(f"  Theta:  {mc_theta:>8.4f}  (Error: {abs(mc_theta - bs_greeks['theta']):>7.4f})")
    print(f"  Rho:    {mc_rho:>8.4f}  (Error: {abs(mc_rho - bs_greeks['rho']):>7.4f})")

    # Pathwise derivative method
    print(f"\n{'-' * 70}")
    print("Monte Carlo Greeks (Pathwise Derivative Method):")
    print(f"{'-' * 70}")

    pw_greeks_calc = PathwiseGreeks(n_simulations=100000, seed=42)

    print(f"  Computing with {pw_greeks_calc.n_simulations:,} simulations...")
    print()

    pw_delta = pw_greeks_calc.delta(option)
    pw_vega = pw_greeks_calc.vega(option)

    print(f"  Delta:  {pw_delta:>8.4f}  (Error: {abs(pw_delta - bs_greeks['delta']):>7.4f})")
    print(f"  Vega:   {pw_vega:>8.4f}  (Error: {abs(pw_vega - bs_greeks['vega']):>7.4f})")

    # Greeks interpretation
    print(f"\n{'-' * 70}")
    print("Greeks Interpretation:")
    print(f"{'-' * 70}")

    print(f"\nDelta = {bs_greeks['delta']:.4f}")
    print(f"  A $1 increase in spot would increase the option price by ~${bs_greeks['delta']:.2f}")

    print(f"\nGamma = {bs_greeks['gamma']:.4f}")
    print(f"  A $1 increase in spot would change Delta by {bs_greeks['gamma']:.4f}")

    print(f"\nVega = {bs_greeks['vega']:.4f}")
    print(f"  A 1% increase in volatility would increase the option price by ~${bs_greeks['vega']:.2f}")

    print(f"\nTheta = {bs_greeks['theta']:.4f}")
    print(f"  One day of time decay decreases the option price by ~${-bs_greeks['theta']:.2f}")

    print(f"\nRho = {bs_greeks['rho']:.4f}")
    print(f"  A 1% increase in interest rate would increase the option price by ~${bs_greeks['rho']:.2f}")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
