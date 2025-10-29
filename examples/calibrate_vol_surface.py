"""Example: Calibrate volatility smile to market data."""

import numpy as np
from mc_pricing import BlackScholes
from mc_pricing.calibration import SmileCalibration
from mc_pricing.utils import DataLoader


def main():
    """Calibrate volatility smile to simulated market data."""

    print("=" * 70)
    print("Volatility Smile Calibration Example")
    print("=" * 70)

    # Market parameters
    spot = 100.0
    maturity = 1.0
    rate = 0.05

    # Generate sample market data (simulating a volatility smile)
    print(f"\nGenerating sample market data...")
    print(f"  Spot: ${spot:.2f}")
    print(f"  Maturity: {maturity} years")
    print(f"  Rate: {rate * 100:.1f}%\n")

    # Strikes around ATM
    strikes = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120])

    # True volatilities with smile (higher for OTM options)
    true_vols = []
    for strike in strikes:
        moneyness = strike / spot
        # Quadratic smile
        smile = 0.2 + 0.15 * (moneyness - 1.0) ** 2
        true_vols.append(smile)

    true_vols = np.array(true_vols)

    # Generate market prices from these volatilities
    market_prices = []
    for strike, vol in zip(strikes, true_vols):
        price = BlackScholes.price(
            option_type='call',
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            volatility=vol
        )
        # Add small noise
        price += np.random.normal(0, 0.05)
        market_prices.append(max(price, 0.01))

    market_prices = np.array(market_prices)

    # Display market data
    print("Market Data:")
    print(f"{'Strike':<10} {'Price':<12} {'Impl Vol':<12} {'Moneyness':<12}")
    print("-" * 50)

    for strike, price, vol in zip(strikes, market_prices, true_vols):
        moneyness = strike / spot
        print(f"{strike:<10.2f} ${price:<11.2f} {vol*100:<11.1f}% {moneyness:<12.2f}")

    # Calibrate using different models
    models = ['quadratic', 'svi', 'polynomial']

    for model_name in models:
        print(f"\n{'-' * 70}")
        print(f"Calibrating {model_name.upper()} model...")
        print(f"{'-' * 70}")

        # Create calibrator
        calibrator = SmileCalibration(model=model_name)

        # Fit to market data
        fit_result = calibrator.fit(
            strikes=strikes,
            market_prices=market_prices,
            spot=spot,
            maturity=maturity,
            rate=rate,
            option_type='call'
        )

        # Display calibration results
        print(f"\nCalibration Results:")
        print(f"  RMSE: {fit_result['rmse']:.6f}")
        print(f"  Max Error: {fit_result['max_error']:.6f}")
        print(f"  Points: {fit_result['n_points']}")

        print(f"\nParameters:")
        for key, value in fit_result['params'].items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:.6f}")

        # Generate fitted curve
        strike_fine = np.linspace(strikes.min(), strikes.max(), 100)
        fitted_vols = calibrator.volatility(strike_fine)

        # Calculate prices using fitted vols
        print(f"\n{'Strike':<10} {'Market':<12} {'Fitted':<12} {'Difference':<12}")
        print("-" * 50)

        for strike, market_vol in zip(strikes, true_vols):
            fitted_vol = calibrator.volatility(np.array([strike]))[0]
            diff = abs(fitted_vol - market_vol)

            print(f"{strike:<10.2f} "
                  f"{market_vol*100:<11.1f}% "
                  f"{fitted_vol*100:<11.1f}% "
                  f"{diff*100:<11.2f}%")

    # Compare models
    print(f"\n{'-' * 70}")
    print("Model Comparison Summary:")
    print(f"{'-' * 70}\n")

    summary = []
    for model_name in models:
        calibrator = SmileCalibration(model=model_name)
        fit_result = calibrator.fit(
            strikes=strikes,
            market_prices=market_prices,
            spot=spot,
            maturity=maturity,
            rate=rate,
            option_type='call'
        )
        summary.append({
            'model': model_name,
            'rmse': fit_result['rmse'],
            'max_error': fit_result['max_error']
        })

    print(f"{'Model':<15} {'RMSE':<12} {'Max Error':<12}")
    print("-" * 40)
    for s in summary:
        print(f"{s['model']:<15} {s['rmse']:<12.6f} {s['max_error']:<12.6f}")

    best_model = min(summary, key=lambda x: x['rmse'])
    print(f"\nBest Model: {best_model['model'].upper()} (RMSE: {best_model['rmse']:.6f})")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    np.random.seed(42)
    main()
