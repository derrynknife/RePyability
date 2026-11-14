# Stress-strength interference

A part fails when the stress put on it reaches its strength. Neither is a
single number: strengths vary from part to part, and stresses from use to use.
`stress_strength` answers the question that follows: **what is the chance that
a part's strength is above the stress it meets?**

It takes the two models as they are (fitted in surpyval, or anything with the
methods named in the API reference) and fits nothing. It is pure probability:
there is no fatigue, wear or crack-growth law in it, and where a strength comes
from is up to you.

!!! warning "One unit"
    The strength and the stress must be in the same unit (both in MPa, both
    in newtons, ...). RePyability does not check units, and a mismatch gives a
    wrong answer without a warning.

## The idea in plain words

Draw a part at random and a use at random. The part survives the use if its
strength is above the stress. `R` is how often that happens. A big gap between
the average strength and the average stress helps, but so does little spread:
two overlapping bell curves put some parts below some stresses even when the
averages are far apart. That overlap is the *interference*.

```python
import numpy as np
import surpyval as surv
from repyability import stress_strength

strength = surv.Normal.from_params([500, 40])
stress = surv.Normal.from_params([400, 50])

result = stress_strength(strength, stress)
result.reliability                  # -> 0.9408
result.failure_probability          # -> 0.0592
result.method                       # -> 'exact'
```

The average strength is 500 and the average stress 400, a margin of 100. The
spread of that margin is `sqrt(40² + 50²) = 64.0`, so the margin is 1.56
spreads above zero: `R = Φ(1.56) = 0.9408`. `reliability_index` is that 1.56
(the normal quantile of `R`, often called the safety index).

## Repeated loads

A part usually meets the stress many times. Each use draws a new, independent
stress while the part's strength stays what it is, so the part survives `n`
uses when its strength is above the **largest** of the `n` stresses:

```python
stress = surv.Weibull.from_params([380, 6])    # a different kind of model

for n in (1, 10, 100):
    print(n, round(stress_strength(strength, stress, n_loads=n).reliability, 4))
# 1 0.9769
# 10 0.8424
# 100 0.5203
```

The more uses, the more likely one of them is high: reliability falls with
`n_loads` even though no single use got worse.

## Strength that changes with age

RePyability has no law for how a strength changes: you give the model the part
has at each age. `stress_strength_over_time` takes a function from an age to a
model and returns `R(t)`, the chance a part of age `t` survives one use (or
`n_loads` uses in a row):

```python
from repyability import stress_strength_over_time

curve = stress_strength_over_time(
    lambda t: surv.Normal.from_params([500 - 4 * t, 40]),
    surv.Normal.from_params([400, 50]),
    times=[0, 10, 20],
)
curve.reliability.round(4).tolist()     # -> [0.9408, 0.8256, 0.6226]
```

This is not the chance of surviving from new to age `t`. It is the chance of
surviving a use at age `t`.

## How it is worked out

| Models | `n_loads` | Result |
| --- | --- | --- |
| Both Normal | 1 | closed form |
| Both LogNormal | 1 | closed form |
| Both Exponential | any | closed form |
| Both Weibull, same shape | any | closed form |
| Anything else (including any model with an offset, a limited failure population or zero inflation) | any | quadrature |

`result.method` says which. `method="numerical"` forces quadrature even where a
closed form exists, and `method="simulate"` gives the Monte-Carlo estimate with
its `standard_error` (use `mc_samples` and `seed`).

The formulas, for strength `Y` and stress `X`:

- `R = P(Y > X) = ∫ f_X(x) S_Y(x) dx`, where `f_X` is the stress density and
  `S_Y` the strength survival function.
- For `n` uses, `R_n = P(Y > max(X₁..Xₙ)) = ∫ n f_X(x) F_X(x)^(n-1) S_Y(x) dx`.
- Both Normal: `R = Φ((μ_Y − μ_X) / √(σ_X² + σ_Y²))`. Both LogNormal: the same
  with the parameters of the logs.
- Both Exponential with failure rates `λ_X` and `λ_Y`: `R = λ_X / (λ_X + λ_Y)`,
  and with a common Weibull shape `R = α_Y^β / (α_X^β + α_Y^β)`. For `n` uses
  both are `n · B(n, 1 + r)` with `r = λ_Y / λ_X` (`r = (α_X / α_Y)^β` for
  Weibulls), where `B` is the beta function.
- Quadrature substitutes `u = F_X(x)`, so the integral is over the stress's
  probability from 0 to 1 of `n uⁿ⁻¹ S_Y(Q_X(u))`, which stays between 0 and
  1. It is split at the probabilities where the strength's quantiles fall and
  refined until the Gauss-Kronrod (7, 15) rule agrees with its halves, as in
  the mean lifetime integral.

A model that has a share that never occurs (a limited failure population) is
refused as the stress, as are fixed probabilities such as a Bernoulli event.

## How sure are we? Parameter uncertainty

Fitted models are estimates. `uncertainty="fit"` draws both models' parameters
from the normal approximation of their fits (surpyval's `covariance()`), the
two independently since they come from separate data, works out `R` for each
draw and reports the central `confidence` share of them:

```python
rng = np.random.default_rng(0)
strength = surv.Normal.fit(rng.normal(500, 40, 30))
stress = surv.Normal.fit(rng.normal(400, 50, 30))
result = stress_strength(strength, stress, uncertainty="fit", seed=1)
(result.lower < result.reliability < result.upper)   # -> True
```

A model without a parameter covariance (a mixture, a non-parametric fit, a model
made with `from_params`) is refused with a message that says so.
