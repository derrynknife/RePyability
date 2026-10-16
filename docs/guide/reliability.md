# Reliability of a system

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson 1](../learn/lifetimes.md)
    (lifetimes, hazard and MTTF) and [Lesson 2](../learn/systems.md)
    (systems of components).

A [`NonRepairableRBD`][repyability.NonRepairableRBD] answers questions about a
system that is not repaired: the probability it survives to a time, its
lifetime distribution, its mean time to failure, and the time by which it
reaches a given reliability. The examples on this page use one system:

```python
import numpy as np
import surpyval as surv
from repyability import NonRepairableRBD

#   s -> (pump1 | pump2) -> valve -> t
rbd = NonRepairableRBD(
    [("s", "pump1"), ("s", "pump2"),
     ("pump1", "valve"), ("pump2", "valve"),
     ("valve", "t")],
    {
        "pump1": surv.Weibull.from_params([100, 2]),
        "pump2": surv.Weibull.from_params([100, 2]),
        "valve": surv.Weibull.from_params([200, 1.5]),
    },
)
```

## Reliability and unreliability

```python
rbd.sf(50)                 # -> 0.8393   R(50), the system reliability
rbd.ff(50)                 # -> 0.1607   F(50) = 1 − R(50)
rbd.sf([25, 50, 100])      # array([0.9533, 0.8393, 0.4216])
rbd.sf(50, method="c")     # -> 0.8393   the cut-set route gives the same
```

`reliability` and `unreliability` are aliases of `sf` and `ff`. The system
reliability is computed **exactly** from the node reliabilities (see
[Concepts](../concepts.md#how-the-system-quantity-is-computed)); nothing is
simulated.

When every node is a fixed probability the system does not depend on time,
and the time can be left out:

```python
from surpyval import FixedEventProbability

one_of_two = NonRepairableRBD(
    [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    {
        "a": FixedEventProbability.from_params(0.1),   # fails with p = 0.1
        "b": FixedEventProbability.from_params(0.1),
    },
)
one_of_two.sf()   # -> 0.99
```

Leaving the time out of a time-varying RBD raises `ValueError`.

## Density, hazard and cumulative hazard

```python
rbd.df(50)   # -> 0.006188   failure density f(t) = −dR/dt
rbd.hf(50)   # -> 0.007373   hazard rate h(t) = f(t) / R(t)
rbd.Hf(50)   # -> 0.1752     cumulative hazard H(t) = −ln R(t)
```

`Hf` is exact. `df` (and so `hf`) differentiates the exact reliability by a
central finite difference with relative step `dx` (default `1e-6`); the step
never crosses below zero. `hf` and `Hf` are infinite where the reliability has
reached zero.

## Conditional survival

`cs(x, X)` is the probability of surviving a further `x`, given the system has
survived to `X`: `R(x | X) = R(X + x) / R(X)`.

```python
rbd.cs(50, 50)   # -> 0.5023   survive to 100, given alive at 50
```

This conditions the *whole system* on one age. To condition each component on
its own age, see [Condition-based evaluation](condition-based.md).

## Per-node values

```python
rbd.node_sf(50)       # {'pump1': 0.7788, 'pump2': 0.7788, 'valve': 0.8825, 's': 1.0, 't': 1.0}
rbd.node_ff(50)       # the complements
rbd.node_mttf()       # {'pump1': 88.62, 'pump2': 88.62, 'valve': 180.55}
```

`node_sf` and `node_ff` include the input and output nodes (always 1 and 0).
`node_mttf` excludes them. It takes each model's own mean, without
simulating: a nested RBD's or repeated node's is the area under its
reliability, and a standby or load-sharing node's is exact where it has a
closed form or convolution, and otherwise the mean of the lifetimes its
reliability was fitted to. A fixed-probability node, which has no lifetime,
gets 0.

## Forcing nodes working or failed

`working_nodes` and `broken_nodes` pin the named nodes to perfectly reliable
or failed. They are accepted by `sf`, `ff`, `df`, `hf`, `Hf`, `cs`,
`time_to_reliability`, `bx_life` and the importance measures.

```python
rbd.sf(50, working_nodes=["pump1"])   # -> 0.8825   = R_valve: pump1 never fails
rbd.sf(50, broken_nodes=["pump1"])    # -> 0.6873   pump2 must carry it alone
rbd.sf(50, broken_nodes=["valve"])    # -> 0.0      the valve is a single point of failure
```

For any node `A` the system reliability is the blend of the two forced cases,
weighted by `A`'s own reliability (the *pivotal decomposition*):

```python
R_A = rbd.node_sf(50)["pump1"]
R_A * rbd.sf(50, working_nodes=["pump1"]) + (1 - R_A) * rbd.sf(50, broken_nodes=["pump1"])
# -> 0.8393   the unforced value
```

A node in both sets, an unknown node, the input or output node, or a
[repeated component](building.md#one-component-in-several-places) raises
`ValueError`. On an RBD with common-cause groups, forcing a group member
raises `NotImplementedError`.

## Lifetimes and mean time to failure

`random(size, seed=None)` draws system lifetimes by simulating each node's
lifetime; the system fails when its last working path is broken.

```python
rbd.random(5, seed=1)   # array([ 0.47, 42.19, 65.11, 87.97, 18.34])
```

The mean time to failure is exact: the area under the system reliability,
`MTTF = ∫ R(t) dt`, which is integrated numerically to about `1e-10`.
`mean` and `mean_time_to_failure` are the same. `method="simulate"`
estimates it instead, as the mean of `mc_samples` such lifetimes (default
100 000), and `mean_time_to_failure_interval` gives that estimate with its
sampling uncertainty, as a
[`ConfidenceInterval`][repyability.ConfidenceInterval]:

```python
rbd.mean_time_to_failure()   # -> 93.82
rbd.mean_time_to_failure(method="simulate", seed=0)   # -> 93.76
ci = rbd.mean_time_to_failure_interval(mc_samples=100_000, confidence=0.95, seed=0)
ci.estimate          # -> 93.76
ci.lower, ci.upper   # (93.49, 94.03)
ci.standard_error    # -> 0.1373   sample std / √mc_samples
```

The interval narrows like `1/√mc_samples`. A common-cause group that
splits the failure rate (`basis="rate"`) is in both the exact MTTF and the
simulation; one that splits a probability (the default) the exact MTTF
refuses and the simulation leaves out (see
[Common-cause failures](common-cause.md#what-honours-a-ccf-group)). A system
that can outlast its failing nodes has an infinite MTTF: one that needs only
a node some of whose units never fail (a surpyval model with `p < 1`).

To simulate until the MTTF is known to a given precision, pass a
`tolerance`: `mean_time_to_failure_interval(tolerance=0.5)` keeps adding
`mc_samples` lifetimes until the interval is at most 0.5 either side. The
same methods draw antithetic pairs (`antithetic=True`) and run over several
processes (`n_jobs`), and `compare(other)` estimates how much longer one
design's MTTF is than another's, with common random numbers: see
[Simulation precision and speed](simulation.md).

## From a target reliability to a time

`time_to_reliability(target)` solves `R(t) = target`; `bx_life(x)` is the time
by which `x` percent have failed, `time_to_reliability(1 − x/100)`.

```python
rbd.time_to_reliability(0.9)                           # -> 38.84
rbd.bx_life(10)                                        # -> 38.84   the B10 life
rbd.time_to_reliability(0.9, working_nodes=["pump1"])  # -> 44.62
```

Both accept the `sf` arguments (`working_nodes`, `broken_nodes`, `method`) and
`upper_bound` to bound the search (found automatically otherwise). They raise
`ValueError` if the target is not in (0, 1), if it is above the reliability at
time zero, or if the RBD is fixed-probability (time plays no part).

## Uncertainty in the component models

A component's model is estimated from data, so its parameters are uncertain.
This is *epistemic* uncertainty, about what the model is, as opposed to the
*aleatory* variability the model itself describes (whether a given unit
survives). `sf_uncertainty(x, uncertainty, n_draws=1000, seed=None)` carries
it to the system: each draw gives every uncertain node a plausible model, the
system reliability is computed exactly for that draw (all draws at once, by
the vectorised exact engine), and the result says how well the system
reliability is known. RePyability does not fit models; the draws use what the
fit, made in surpyval, provides:

```python
data = surv.Weibull.from_params([100, 2]).qf(np.linspace(0.025, 0.975, 20))
pump_fit = surv.Weibull.fit(data)              # 20 failure times, fitted by surpyval
fitted = NonRepairableRBD(
    [("s", "pump1"), ("s", "pump2"),
     ("pump1", "valve"), ("pump2", "valve"),
     ("valve", "t")],
    {"pump1": pump_fit, "pump2": pump_fit,
     "valve": surv.Weibull.from_params([200, 1.5])},
)
result = fitted.sf_uncertainty(50, {("pump1", "pump2"): "fit"}, n_draws=10_000, seed=0)
result.nominal                # -> 0.8426   with the fitted models
lower, upper = result.interval(0.9)
lower, upper                  # (0.7785, 0.8728)   the 5th to 95th percentile
```

A node's uncertainty is one of:

| Given | Draws |
|---|---|
| `"fit"` | Every parameter the fit estimated, from the normal approximation of the model's maximum-likelihood fit (surpyval's `covariance()`), on the log scale for a positive parameter and the logit scale for one in (0, 1), so every draw is valid. That includes a limited failure population's share that ever fails (`lfp_p`) and a zero-inflated fit's share dead on arrival (`f0`). An offset keeps its fitted value, with a warning, since surpyval's covariance leaves it out (SurPyval#830). |
| `{"alpha": distribution, ...}` | Each named parameter from its distribution (anything with `qf` or `ppf`: surpyval or `scipy.stats`); the others keep their values. |
| A list of models | One of them, with replacement: for example refits to bootstrap resamples, or posterior draws, made in surpyval. |

```python
import scipy.stats as st

known = {"alpha": st.uniform(80, 40)}    # the scale lies between 80 and 120 h
rbd.sf_uncertainty(50, {("pump1", "pump2"): known}, n_draws=10_000, seed=0).interval(0.9)
# (0.7974, 0.8587)

rng = np.random.default_rng(1)
refits = [surv.Weibull.fit(rng.choice(data, len(data))) for _ in range(100)]   # bootstrap, in surpyval
fitted.sf_uncertainty(50, {("pump1", "pump2"): refits}, n_draws=10_000, seed=0).interval(0.9)
# (0.7836, 0.8752)
```

**Nodes of one population share their uncertainty.** Two pumps of one type,
fitted to the same data, have the same unknown parameters: give them together
as a tuple of node names, and each draw gives both the same model. Giving
them separately draws their parameters independently, which averages part of
the uncertainty away: here the interval would narrow to (0.811, 0.855).

The result is an [`UncertaintyResult`][repyability.UncertaintyResult]:
`samples` (one value per draw, or one row per draw for an array of times),
`nominal` (with the nodes' own models), `mean`, `median`, `std`,
`percentile(q)` and `interval(level)`. For an array of times the summaries
are per time, and the draws are the same at every time:

```python
hours = np.array([25.0, 50.0, 100.0])
curve = fitted.sf_uncertainty(hours, {("pump1", "pump2"): "fit"}, n_draws=10_000, seed=0)
lower, upper = curve.interval(0.9)
np.round(lower, 4)            # array([0.9414, 0.7785, 0.291 ])
np.round(upper, 4)            # array([0.9565, 0.8728, 0.5419])
```

The same draws give the uncertainty of the MTTF, a B*X* life and the time to
a reliability: `mean_uncertainty(uncertainty)`,
`bx_life_uncertainty(x, uncertainty)` and
`time_to_reliability_uncertainty(target, uncertainty)` work each draw's value
out exactly, as `mean` and `time_to_reliability` do, and return an
`UncertaintyResult` like `sf_uncertainty`'s. The MTTF's interval from the
fit is far wider than its simulation error, which is all
`mean_time_to_failure_interval` reports:

```python
pumps = {("pump1", "pump2"): "fit"}
mttf = fitted.mean_uncertainty(pumps, n_draws=2000, seed=0)
mttf.nominal                  # -> 93.42   the exact MTTF with the fitted models
lower, upper = mttf.interval(0.9)
lower, upper                  # (81.78, 106.75)   from the data behind the fits
ci = fitted.mean_time_to_failure_interval(seed=0)
ci.lower, ci.upper            # (93.10, 93.63)   simulation error only
b10 = fitted.bx_life_uncertainty(10, pumps, n_draws=2000, seed=0)
b10.interval(0.9)             # (33.36, 43.26)   around a B10 of 39.26
```

The interval's width is a property of what is known about the models, and
does not shrink with more draws, which only make its ends more precise. More
failure data (a refit in surpyval) is what narrows it. `sampling="sobol"`
makes them precise sooner: it takes the draws from a scrambled Sobol
sequence rather than from random numbers, points that cover the
parameters' range evenly, so the mean, the percentiles and the Sobol
indices settle with fewer draws (a power of 2 of them suits the sequence
best). Every method here that draws takes it. With common-cause
groups each draw is worked out with them: a group's members are given
together, in one tuple, and the group's own model can be uncertain too (see
[Common-cause
failures](common-cause.md#importance-sensitivity-uncertainty-and-allocation)).
A node whose model is not a parametric distribution (a standby
arrangement, a nested diagram) can only be given a list of models.

### Whose uncertainty widens the interval

`uncertainty_importance(x, uncertainty)` says which input's uncertainty
makes the answer uncertain: each one's share of the quantity's variance,
and so where more failure data would narrow the interval most. Here the
valve's scale is known only to lie between 150 and 250 h:

```python
both = {("pump1", "pump2"): "fit", "valve": {"alpha": st.uniform(150, 100)}}
parts = fitted.uncertainty_importance(50, both)
parts.first_order[("pump1", "pump2")]   # -> 0.6004
parts.first_order["valve"]              # -> 0.3996
fitted.uncertainty_importance(None, both, of="mean").first_order["valve"]   # -> 0.1877
```

The pumps' fit accounts for 60% of the reliability's variance at 50 h, but
81% of the MTTF's. `of` picks the quantity: `"sf"` (at the times `x`),
`"mean"`, `"bx_life"` (`x` the percentage) or `"time_to_reliability"` (`x`
the reliability). By default it is worked out by the delta method: the
quantity's derivative in each input's parameters (central differences of the
exact value) with their covariance (the fit's `covariance()`, or the variances
of the distributions given). The inputs are independent, so each input's
part is its own term, and the shares add up to 1.

`method="sobol"` draws the inputs as `sf_uncertainty` does and estimates the
variance-based (Sobol) indices instead. The first-order index is the share
of the variance that knowing the input exactly would remove. The total index
counts the input's interactions with the others too. Both take the
quantity's nonlinearity in, at the cost of sampling error and of
`n_draws × (inputs + 2)` evaluations:

```python
sobol = fitted.uncertainty_importance(50, both, method="sobol", n_draws=5000, seed=0)
sobol.first_order["valve"]   # -> 0.395
sobol.total["valve"]         # -> 0.409
```

Close to the delta method's 0.40: at this spread the reliability is nearly
linear in the parameters. A list of models has no parameters for the delta
method to move, so it needs `method="sobol"`.
