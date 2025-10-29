# Mathematical Theory and Background

## Table of Contents
1. [Option Pricing Fundamentals](#option-pricing-fundamentals)
2. [Monte Carlo Simulation](#monte-carlo-simulation)
3. [Variance Reduction Techniques](#variance-reduction-techniques)
4. [Greeks Calculation](#greeks-calculation)
5. [Volatility Smile](#volatility-smile)

## Option Pricing Fundamentals

### Risk-Neutral Pricing

Under the risk-neutral measure, the price of a derivative is the expected discounted payoff:

```
V(0) = E^Q[e^(-rT) * Payoff(S_T)]
```

where:
- `V(0)` is the current option price
- `E^Q` is expectation under risk-neutral measure Q
- `r` is the risk-free rate
- `T` is time to maturity
- `S_T` is the asset price at maturity

### Geometric Brownian Motion

The underlying asset follows:

```
dS_t = μ S_t dt + σ S_t dW_t
```

Under risk-neutral measure:

```
dS_t = (r - q) S_t dt + σ S_t dW_t
```

where:
- `μ` is the drift
- `σ` is volatility
- `q` is dividend yield
- `W_t` is a Wiener process

### Black-Scholes Formula

For European options, the Black-Scholes formula gives closed-form prices:

**Call Option:**
```
C = S_0 e^(-qT) N(d_1) - K e^(-rT) N(d_2)
```

**Put Option:**
```
P = K e^(-rT) N(-d_2) - S_0 e^(-qT) N(-d_1)
```

where:
```
d_1 = [ln(S_0/K) + (r - q + σ²/2)T] / (σ√T)
d_2 = d_1 - σ√T
```

## Monte Carlo Simulation

### Basic Algorithm

1. Generate random paths for the underlying asset
2. Calculate payoff for each path
3. Average and discount to present value

```python
# Pseudocode
for i in range(n_simulations):
    S_T = simulate_path(S_0, r, σ, T)
    payoff[i] = max(S_T - K, 0)  # Call option

price = exp(-r*T) * mean(payoff)
```

### Convergence

Monte Carlo estimators converge at rate O(1/√n):

```
Standard Error = σ̂ / √n
```

where σ̂ is the sample standard deviation.

### Path Generation

Asset price at time T:

```
S_T = S_0 * exp((r - q - σ²/2)T + σ√T * Z)
```

where Z ~ N(0,1) is a standard normal random variable.

## Variance Reduction Techniques

### Antithetic Variates

For each random variable Z, also use -Z:

```
V̂ = (V(Z) + V(-Z)) / 2
```

This induces negative correlation, reducing variance.

**Variance Reduction:**
```
Var(V̂_antithetic) ≤ Var(V̂_standard) / 2
```

### Control Variates

Use a correlated variable X with known expectation E[X]:

```
V̂_adjusted = V̂ + c(X̂ - E[X])
```

Optimal coefficient:
```
c* = -Cov(V, X) / Var(X)
```

For options, use European option as control for Asian option.

### Importance Sampling

Sample from modified distribution to emphasize important regions:

```
E[f(X)] = E_Q[f(X) * L(X)]
```

where L(X) is the likelihood ratio.

For options, shift the drift to increase probability of being in-the-money.

## Greeks Calculation

### Finite Difference Method

Approximate derivatives by finite differences:

**Delta:**
```
Δ ≈ [V(S + h) - V(S - h)] / (2h)
```

**Gamma:**
```
Γ ≈ [V(S + h) - 2V(S) + V(S - h)] / h²
```

**Vega:**
```
ν ≈ [V(σ + h) - V(σ - h)] / (2h)
```

### Pathwise Derivative Method

Differentiate the payoff function directly:

**Delta (European Call):**
```
Δ = E[1_{S_T > K} * (S_T / S_0) * e^(-rT)]
```

where 1_{S_T > K} is the indicator function.

## Volatility Smile

### Implied Volatility

The volatility that makes Black-Scholes price equal to market price:

```
V_market = V_BS(σ_implied)
```

Solved using Newton-Raphson:
```
σ_(n+1) = σ_n + (V_market - V_BS(σ_n)) / vega(σ_n)
```

### Smile Models

**Quadratic:**
```
σ(K) = a + b(K - K_0) + c(K - K_0)²
```

**SVI (Stochastic Volatility Inspired):**
```
w(k) = a + b[ρ(k - m) + √((k - m)² + σ²)]
```

where:
- w = σ²T (total variance)
- k = ln(K/F) (log-moneyness)
- a, b, ρ, m, σ are parameters

## Asian Options

### Arithmetic Average

Payoff based on arithmetic mean:

```
Payoff = max(A - K, 0)
where A = (1/n) Σ S_i
```

### Geometric Average

Payoff based on geometric mean:

```
Payoff = max(G - K, 0)
where G = (Π S_i)^(1/n)
```

Geometric Asian options have semi-closed-form solutions, but arithmetic do not (hence Monte Carlo is valuable).

## References

1. Hull, J. C. (2018). *Options, Futures, and Other Derivatives*. Pearson.
2. Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*. Springer.
3. Shreve, S. E. (2004). *Stochastic Calculus for Finance II*. Springer.
4. Gatheral, J. (2006). *The Volatility Surface*. Wiley.
