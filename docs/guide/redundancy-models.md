# Redundancy models

Parallel branches in a diagram model *active* redundancy: every unit runs,
and each fails independently. Real redundancy is often not like that. Spares
wait in standby, identical parts are replicated, and units that share a load
wear each other out. The models on this page capture each of these as a
single node. Theory: [Concepts](../concepts.md#standby-and-repeated-nodes)
and [load sharing](../concepts.md#dependent-failures-load-sharing).

```python
import numpy as np
import surpyval as surv
from repyability import (
    LoadSharingModel,
    NonRepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
)

pump = surv.Weibull.from_params([100, 2])
```

## Standby: cold, warm and hot

A [`StandbyModel`][repyability.StandbyModel] holds `n` units of which `k`
operate at a time. When an operating unit fails, the next spare in list order
is switched in; the arrangement fails when fewer than `k` units survive.
`dormancy_factor` sets how fast a *dormant* spare ages, relative to an
operating one:

```python
cold = StandbyModel([pump, pump])                         # dormancy_factor=0
warm = StandbyModel([pump, pump], dormancy_factor=0.3, seed=0)
hot = StandbyModel([pump, pump], dormancy_factor=1.0, seed=0)

cold.sf(150)      # -> 0.6342   the spare only starts ageing when switched in
warm.sf(150)[0]   # -> 0.4735   the spare ages at 30% of the rate while dormant
hot.sf(150)[0]    # -> 0.2006   the same as two units in parallel (0.1997)
```

- **Cold** (`0`, the default): a dormant spare does not age. With `k = 1`,
  the lifetime is the sum of the units' lifetimes.
- **Warm** (between 0 and 1): a dormant spare ages at that fraction of the
  operating rate, so it can fail *latent*, dead before it is ever needed
  (it is skipped when its turn comes).
- **Hot** (`1`): spares age as fast as operating units, which is exactly
  `k`-out-of-`n` active parallel.

`k` operating units at once is supported for every dormancy:

```python
two_of_three = StandbyModel([pump, pump, pump], k=2, n_sims=20_000, seed=0)
two_of_three.sf(100)[0]   # -> 0.512
```

### Imperfect switching

`switching_probability` is the probability that each switch-over succeeds;
a failed switch ends the arrangement. It is supported for cold standby with
`k = 1`:

```python
unreliable_switch = StandbyModel([pump, pump], switching_probability=0.9)
unreliable_switch.sf(150)   # -> 0.5813   against 0.6342 with a perfect switch
```

Other combinations raise `NotImplementedError`.

### How the survival function is obtained

| Arrangement | Method |
|---|---|
| Identical Exponential units, cold, perfect switching (any `k`) | Exact: Erlang. |
| Identical Exponential units, warm or hot | Exact: hypoexponential. |
| Cold, `k = 1` (any units, including imperfect switching) | Numerical convolution of the units' lifetimes: deterministic. |
| Everything else (warm or hot non-Exponential units, cold `k ≥ 2` non-Exponential units) | Simulation: a Kaplan–Meier fit to `n_sims` simulated lifetimes (default 10 000), seeded by `seed`, with `lower` passed as the fit's lower limit. |

The simulated cases carry Monte-Carlo error, and their `sf` returns
one-element arrays even for a scalar time. For hot standby with non-Exponential
units, drawing the units as ordinary parallel nodes gives the exact answer.

`mean()` and `random(size, seed=None)` give the arrangement's mean lifetime
and draw lifetimes; `cs(x, X)` is its conditional survival. A standby node
cannot take a [condition-based state](condition-based.md).

## Repeated nodes: n identical copies

[`RepeatedNode`][repyability.RepeatedNode]`(model, repeats, kind)` is
`repeats` independent copies of `model` in `"series"` or `"parallel"`, as one
node. It saves drawing them out one by one:

```python
three_in_series = RepeatedNode(pump, 3, "series")
three_in_parallel = RepeatedNode(pump, 3, "parallel")
three_in_series.sf(50)     # -> 0.4724   = pump.sf(50) ** 3
three_in_parallel.sf(50)   # -> 0.9892   = 1 − pump.ff(50) ** 3
```

Its reliability is exact. `mean(N=1_000_000, seed=None)` is a Monte-Carlo
estimate from `N` draws. A `RepeatedNode` is *n* distinct copies; for the
*same* component in several places, see
[One component in several places](building.md#one-component-in-several-places).

## Repeated standby

[`RepeatedStandbyNode`][repyability.RepeatedStandbyNode]`(model, repeats,
switching_probability=1.0)` is `repeats` identical units in cold standby,
one operating at a time: the same as a `StandbyModel` of `repeats` copies of
one model with `k = 1`, and computed by the same numerical convolution.

```python
RepeatedStandbyNode(pump, 2).sf(150)                            # -> 0.6342
RepeatedStandbyNode(pump, 2, switching_probability=0.9).sf(150) # -> 0.5813
```

Its `N` and `lower` arguments are accepted for backwards compatibility and
unused.

## Load sharing

Units that share a load fail dependently: when one fails, the survivors take
over its share, run harder, and wear faster. A
[`LoadSharingModel`][repyability.LoadSharingModel] makes such a group one
node. Each unit is an **accelerated-failure-time** model fitted in surpyval
with the load as its covariate; with `s` survivors sharing a total load `L`,
each carries `L / s`, and the group works while at least `k` survive.

```python
from repyability import RegressionNode

# Field data: life at various loads (higher load, shorter life).
rng = np.random.default_rng(0)
load = rng.uniform(0.5, 2.0, size=800)
life = rng.exponential(scale=100.0 / np.exp(0.6 * (load - 1.0)), size=800)
unit = surv.ExponentialAFT.fit(life + 1e-3, Z=load.reshape(-1, 1))

# Three units share a total load of 3.0; the group needs two.
group = LoadSharingModel([unit, unit, unit], load=3.0, k=2)
group.sf(50)          # -> 0.5842
group.is_simulated    # False: identical Exponential baselines have a closed form
group.mean()          # -> 70.36
```

Treating the three units as independent, each pinned at its initial share
(load 1.0), would overstate the group:

```python
p = RegressionNode(unit, covariates=[1.0]).sf(50)[0]
3 * p**2 - 2 * p**3   # -> 0.6608   two-out-of-three, ignoring the load transfer
```

- Identical units with an **Exponential** baseline have an exact
  (hypoexponential) group lifetime and `is_simulated` is `False`. Otherwise
  the survival function is a Kaplan–Meier fit to `n_sims` simulated lifetimes
  (seeded by `seed`), and `is_simulated` is `True`.
- With no load effect the units neither share stress nor accelerate, and the
  group reduces exactly to `k`-out-of-`n` parallel.
- The units must be AFT models (they need the time-scaling `phi(load)`);
  anything else raises `ValueError`.
- Warm standby uses the same cumulative-exposure (virtual-age) machinery.

## Using them in an RBD

Each of these is an ordinary node:

```python
line = NonRepairableRBD(
    [("s", "pumps"), ("pumps", "filters"), ("filters", "t")],
    {"pumps": cold, "filters": three_in_parallel},
)
line.sf(50)   # -> 0.9798
line.is_analytically_solvable()   # False: a standby node is simulation-backed
```

`is_analytically_solvable()` counts every standby, repeated-standby and
load-sharing node as simulation-backed, even when (as here) its reliability
is exact; see
[Is the system time-dependent, and is it exact?](building.md#is-the-system-time-dependent-and-is-it-exact).
All of these models save with the RBD.
