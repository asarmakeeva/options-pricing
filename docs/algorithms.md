# Algorithm Descriptions

## Monte Carlo Option Pricing Algorithm

### Basic Monte Carlo Pricing

```
Algorithm: Monte Carlo Option Pricing
Input: Option parameters (S, K, T, r, σ, q), number of simulations n
Output: Option price and standard error

1. Initialize: sum_payoff = 0, sum_payoff_sq = 0

2. For i = 1 to n:
   a. Generate Z ~ N(0,1)
   b. Calculate S_T = S * exp((r - q - σ²/2)T + σ√T * Z)
   c. Calculate payoff_i = max(S_T - K, 0)  [for call]
   d. sum_payoff += payoff_i
   e. sum_payoff_sq += payoff_i²

3. mean_payoff = sum_payoff / n
4. variance = (sum_payoff_sq - n * mean_payoff²) / (n - 1)
5. price = exp(-rT) * mean_payoff
6. std_error = exp(-rT) * sqrt(variance / n)

Return (price, std_error)

Time Complexity: O(n)
Space Complexity: O(1)
```

### Path-Dependent Options (Asian)

```
Algorithm: Asian Option Pricing
Input: Option parameters, number of paths n, steps m
Output: Option price and standard error

1. For i = 1 to n:
   a. Generate path: S_0, S_1, ..., S_m
      - For j = 1 to m:
        * Z_j ~ N(0,1)
        * S_j = S_(j-1) * exp((r - q - σ²/2)Δt + σ√Δt * Z_j)

   b. Calculate average: A = (1/m) Σ S_j
   c. Calculate payoff: payoff_i = max(A - K, 0)

2. Discount and average payoffs

Time Complexity: O(n * m)
Space Complexity: O(m)  [store one path at a time]
```

## Variance Reduction Algorithms

### Antithetic Variates

```
Algorithm: Antithetic Variates
Input: Option parameters, n/2 simulation pairs
Output: Price estimate with reduced variance

1. For i = 1 to n/2:
   a. Generate Z ~ N(0,1)

   b. Calculate positive path:
      S_T^+ = S * exp((r - q - σ²/2)T + σ√T * Z)
      payoff^+ = max(S_T^+ - K, 0)

   c. Calculate antithetic path:
      S_T^- = S * exp((r - q - σ²/2)T + σ√T * (-Z))
      payoff^- = max(S_T^- - K, 0)

   d. paired_payoff = (payoff^+ + payoff^-) / 2

2. Average paired payoffs and discount

Variance Reduction: Up to 50% for monotonic payoffs
```

### Control Variates

```
Algorithm: Control Variates
Input: Target option, control option with known price
Output: Adjusted price estimate

1. Simulate both target and control on same paths:
   For i = 1 to n:
      - Generate path
      - Calculate Y_i = target_payoff
      - Calculate X_i = control_payoff

2. Calculate control expectation E[X] from Black-Scholes

3. Compute covariance and variance:
   cov = Cov(Y, X)
   var_x = Var(X)

4. Optimal coefficient: c* = -cov / var_x

5. Adjusted estimate:
   Ŷ_adjusted = Ŷ + c*(X̂ - E[X])

Variance Reduction: Depends on correlation, can be >90%
```

### Importance Sampling

```
Algorithm: Importance Sampling for Options
Input: Option parameters, shift parameter θ
Output: Weighted price estimate

1. Choose shift parameter θ (positive for OTM calls)

2. For i = 1 to n:
   a. Generate Z ~ N(0,1)

   b. Shifted path:
      S_T = S * exp((r - q - σ²/2)T + σ√T * (Z + θ√T))

   c. Calculate payoff: f = max(S_T - K, 0)

   d. Likelihood ratio:
      L = exp(-θ*Z*√T - θ²T/2)

   e. Weighted payoff: f * L

3. Average weighted payoffs and discount

Optimal θ: Depends on moneyness and option type
```

## Greeks Algorithms

### Finite Difference Delta

```
Algorithm: Finite Difference Delta
Input: Option, bump size h (e.g., 1% of spot)
Output: Delta estimate

1. Price option at S + h → V_up
2. Price option at S - h → V_down
3. Delta = (V_up - V_down) / (2h)

Accuracy: O(h²) for central difference
Computational Cost: 2 option pricings
```

### Pathwise Delta

```
Algorithm: Pathwise Delta (European)
Input: Option parameters, n simulations
Output: Delta estimate

1. For i = 1 to n:
   a. Generate Z ~ N(0,1)
   b. Calculate S_T and ∂S_T/∂S_0 = S_T / S_0
   c. If S_T > K:  # In the money
      contribution_i = S_T / S_0
   Else:
      contribution_i = 0

2. Delta = exp(-rT) * mean(contributions)

Advantage: Single simulation, lower variance
```

## Volatility Smile Calibration

### SVI Calibration

```
Algorithm: SVI Parameter Calibration
Input: Market strikes K_i, implied vols σ_i
Output: SVI parameters (a, b, ρ, m, σ)

1. Convert to total variance:
   w_i = σ_i² * T

2. Convert to log-moneyness:
   k_i = ln(K_i / F)

3. Initialize parameters:
   a = mean(w)
   b = 0.1
   ρ = 0.0
   m = 0.0
   σ = 0.1

4. Optimization (L-BFGS-B):
   Minimize: Σ (w_i - w_SVI(k_i))²

   where: w_SVI(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

   Subject to:
   - a ≥ 0
   - b ≥ 0
   - -1 < ρ < 1
   - σ > 0

5. Return fitted parameters

Convergence: Typically <100 iterations
```

## Path Generation

### Exact Discretization

```
Algorithm: Exact GBM Path Generation
Input: S_0, r, q, σ, T, n_steps
Output: Price path [S_0, S_1, ..., S_n]

1. Δt = T / n_steps
2. S[0] = S_0

3. For i = 1 to n_steps:
   a. Z ~ N(0,1)
   b. drift = (r - q - σ²/2) * Δt
   c. diffusion = σ * √Δt * Z
   d. S[i] = S[i-1] * exp(drift + diffusion)

Return S

Advantages:
- Exact at discrete times
- No discretization error
- Positive prices guaranteed
```

### Milstein Scheme (Alternative)

```
Algorithm: Milstein Discretization
Input: Same as above
Output: Price path

1. Δt = T / n_steps
2. S[0] = S_0

3. For i = 1 to n_steps:
   a. Z ~ N(0,1)
   b. S[i] = S[i-1] + (r-q)*S[i-1]*Δt + σ*S[i-1]*√Δt*Z
           + 0.5*σ²*S[i-1]*Δt*(Z² - 1)

Return S

Order of Convergence: O(Δt)
Use when: Need higher-order scheme for SDEs
```

## Convergence Testing

```
Algorithm: Convergence Analysis
Input: Option, n_trials array (e.g., [1K, 10K, 100K, 1M])
Output: Prices and errors for each n

1. For each n in n_trials:
   a. Run multiple independent simulations
   b. Record price and std_error
   c. Calculate empirical variance

2. Fit power law to errors:
   log(error) = a + b*log(n)

3. Compare b to theoretical -0.5

4. Plot convergence curves

Expected: error ∝ 1/√n
```

## Optimization Tips

### Memory Efficiency

```python
# Bad: Store all paths
paths = np.zeros((n_simulations, n_steps))  # Large memory

# Good: Process one path at a time
for i in range(n_simulations):
    path = generate_single_path()
    payoff = calculate_payoff(path)
    sum_payoff += payoff
```

### Vectorization

```python
# Bad: Loop over simulations
for i in range(n):
    Z = np.random.randn()
    S_T[i] = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)

# Good: Vectorized
Z = np.random.randn(n)
S_T = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
```

### Parallel Processing

```python
from multiprocessing import Pool

def price_batch(seed):
    simulator = MonteCarloSimulator(seed=seed)
    return simulator.price(option)

# Run in parallel
with Pool(n_cpus) as pool:
    results = pool.map(price_batch, range(n_batches))
```

## References

1. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*
2. Jäckel, P. (2002). *Monte Carlo Methods in Finance*
3. Higham, D. J. (2004). "An Introduction to Financial Option Valuation"
