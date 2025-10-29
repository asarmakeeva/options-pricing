"""Performance metrics for option pricing."""

import numpy as np
import time
from typing import Dict, Callable, Optional, Any
from ..core.base_option import BaseOption
from ..monte_carlo.simulator import MonteCarloSimulator


class PerformanceMetrics:
    """Calculate and track performance metrics for option pricing.

    Metrics include:
    - Accuracy (vs Black-Scholes or other benchmark)
    - Computational time
    - Convergence rate
    - Variance reduction effectiveness
    """

    @staticmethod
    def calculate_accuracy(
        estimated_price: float,
        true_price: float
    ) -> Dict[str, float]:
        """Calculate accuracy metrics.

        Args:
            estimated_price: Monte Carlo estimate
            true_price: True or benchmark price

        Returns:
            Dictionary with error metrics
        """
        absolute_error = abs(estimated_price - true_price)
        relative_error = absolute_error / abs(true_price) if true_price != 0 else np.inf
        percentage_error = relative_error * 100

        return {
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'percentage_error': percentage_error
        }

    @staticmethod
    def benchmark_pricing_speed(
        pricing_function: Callable,
        option: BaseOption,
        n_runs: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """Benchmark pricing speed.

        Args:
            pricing_function: Function that prices the option
            option: Option to price
            n_runs: Number of runs for timing
            **kwargs: Additional arguments for pricing function

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for _ in range(n_runs):
            start = time.time()
            pricing_function(option, **kwargs)
            end = time.time()
            times.append(end - start)

        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }

    @staticmethod
    def variance_reduction_efficiency(
        std_error_baseline: float,
        std_error_reduced: float,
        time_baseline: float,
        time_reduced: float
    ) -> Dict[str, float]:
        """Calculate variance reduction efficiency metrics.

        Args:
            std_error_baseline: Standard error without VR
            std_error_reduced: Standard error with VR
            time_baseline: Computation time without VR
            time_reduced: Computation time with VR

        Returns:
            Dictionary with efficiency metrics
        """
        # Variance reduction factor
        variance_ratio = (std_error_reduced / std_error_baseline) ** 2
        variance_reduction = 1 - variance_ratio
        variance_reduction_pct = variance_reduction * 100

        # Effective sample size increase
        # If variance is reduced by factor R, it's equivalent to R times more samples
        effective_samples_ratio = 1 / variance_ratio

        # Computational efficiency
        # How many times faster to achieve same accuracy?
        time_ratio = time_reduced / time_baseline
        efficiency = effective_samples_ratio / time_ratio

        return {
            'variance_reduction_pct': variance_reduction_pct,
            'variance_ratio': variance_ratio,
            'effective_samples_ratio': effective_samples_ratio,
            'time_ratio': time_ratio,
            'efficiency_ratio': efficiency
        }

    @staticmethod
    def confidence_interval_coverage(
        prices: np.ndarray,
        std_errors: np.ndarray,
        true_price: float,
        confidence: float = 0.95
    ) -> float:
        """Calculate empirical coverage of confidence intervals.

        Args:
            prices: Array of price estimates
            std_errors: Array of standard errors
            true_price: True price
            confidence: Confidence level

        Returns:
            Empirical coverage probability
        """
        from scipy.stats import norm

        z_score = norm.ppf((1 + confidence) / 2)

        # Calculate confidence intervals
        lower_bounds = prices - z_score * std_errors
        upper_bounds = prices + z_score * std_errors

        # Check coverage
        covered = (lower_bounds <= true_price) & (true_price <= upper_bounds)
        coverage = np.mean(covered)

        return coverage

    @staticmethod
    def convergence_rate(
        n_simulations: np.ndarray,
        std_errors: np.ndarray
    ) -> Dict[str, float]:
        """Estimate convergence rate.

        Theoretical rate for MC is O(n^(-0.5)).

        Args:
            n_simulations: Array of simulation counts
            std_errors: Array of standard errors

        Returns:
            Dictionary with convergence statistics
        """
        # Fit power law: SE = a * n^b
        # log(SE) = log(a) + b * log(n)

        log_n = np.log(n_simulations)
        log_se = np.log(std_errors)

        # Linear regression
        coeffs = np.polyfit(log_n, log_se, deg=1)
        exponent = coeffs[0]
        log_a = coeffs[1]

        # Theoretical is -0.5
        return {
            'estimated_exponent': exponent,
            'theoretical_exponent': -0.5,
            'difference': abs(exponent - (-0.5))
        }

    @staticmethod
    def compare_methods(
        option: BaseOption,
        methods: Dict[str, Any],
        true_price: Optional[float] = None,
        n_simulations: int = 100000,
        n_runs: int = 10
    ) -> Dict[str, Dict]:
        """Comprehensive comparison of different pricing methods.

        Args:
            option: Option to price
            methods: Dictionary of {name: variance_reduction_object}
            true_price: True price for accuracy comparison
            n_simulations: Number of simulations per run
            n_runs: Number of independent runs

        Returns:
            Dictionary with detailed comparison metrics
        """
        results = {}

        for name, var_reduction in methods.items():
            prices = []
            errors = []
            times = []

            simulator = MonteCarloSimulator(n_simulations=n_simulations)

            for _ in range(n_runs):
                start = time.time()
                price, std_error = simulator.price(option, variance_reduction=var_reduction)
                end = time.time()

                prices.append(price)
                errors.append(std_error)
                times.append(end - start)

            results[name] = {
                'mean_price': np.mean(prices),
                'std_price': np.std(prices),
                'mean_error': np.mean(errors),
                'mean_time': np.mean(times),
                'std_time': np.std(times)
            }

            if true_price is not None:
                accuracy = PerformanceMetrics.calculate_accuracy(
                    np.mean(prices), true_price
                )
                results[name].update(accuracy)

        # Calculate relative efficiency
        if 'Standard MC' in results:
            baseline = results['Standard MC']
            for name in results:
                if name != 'Standard MC':
                    eff = PerformanceMetrics.variance_reduction_efficiency(
                        baseline['mean_error'],
                        results[name]['mean_error'],
                        baseline['mean_time'],
                        results[name]['mean_time']
                    )
                    results[name]['efficiency'] = eff

        return results
