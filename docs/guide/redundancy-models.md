# Redundancy models

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson
    5](../learn/dependence.md) (when redundancy disappoints).

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
warm = StandbyModel([pump, pump], dormancy_factor=0.3)
hot = StandbyModel([pump, pump], dormancy_factor=1.0)

cold.sf(150)      # -> 0.6342   the spare only starts ageing when switched in
warm.sf(150)      # -> 0.4815   the spare ages at 30% of the rate while dormant
hot.sf(150)       # -> 0.1997   exactly two units in parallel
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
two_of_three = StandbyModel([pump, pump, pump], k=2)
two_of_three.sf(100)   # -> 0.5172
```

### Spares that fail to switch in or to start

A spare can let the arrangement down on demand in two ways, each a
probability given once for all spares or one per spare (in list order):

- `switching_probability`: the switch onto it works. A failed switch ends
  the arrangement then.
- `start_probability`: it starts once switched in. A spare that fails to
  start is lost, and the next spare is switched in at once.

Both apply to cold, warm and hot standby, with any `k`. A warm spare that
has died while it waited is passed over without a switch.

```python
unreliable_switch = StandbyModel([pump, pump], switching_probability=0.9)
unreliable_switch.sf(150)   # -> 0.5813   against 0.6342 with a perfect switch
StandbyModel([pump, pump, pump], k=2, switching_probability=0.9).sf(100)
# -> 0.479   against 0.5172
may_not_start = StandbyModel([pump] * 3, start_probability=0.8)
may_not_start.sf(200)   # -> 0.6133   against 0.786 if both spares always start
StandbyModel([pump, pump], dormancy_factor=0.3, switching_probability=0.9).sf(150)
# -> 0.4439   warm, against 0.4815
```

With two exponential units of rate `λ` and one warm spare, for example, the
spare is still alive when it is needed with probability `exp(-κλu)`, so the
reliability is `exp(-λt) (1 + sp (1 - exp(-κλt)) / κ)`, which falls to
`exp(-λt) (1 + sp λt)` cold (`κ` the `dormancy_factor`, `s` and `p` the
two probabilities).

### How the survival function is obtained

| Arrangement | Method |
|---|---|
| Identical Exponential units, cold, perfect switching and starts (any `k`) | Exact: Erlang. |
| Identical Exponential units, warm or hot, perfect switching and starts | Exact: hypoexponential. |
| Hot, any units (any `k`), perfect switching and starts | Exact: `k`-out-of-`n` of the units, each working independently. |
| Any other arrangement of Exponential units: different rates, any `k`, any dormancy, spares that may fail to switch in or start | Exact: a Markov chain of which units are left, operating or waiting, followed by uniformization (each probability, of working or of having failed, to full precision however small). A chain of more than 200 states (many different units, warm) is simulated, saying so. |
| Cold, `k = 1` (any units) | Numerical convolution of the units' lifetimes: deterministic, and accurate to about 1e-6 (1e-5 for the steepest early-life densities, a gamma with shape 0.5 or less). A spare that may fail to start is one of life 0 with that probability. |
| Cold, `k ≥ 2`, identical units | Numerical: each operating position runs a renewal process of the units' lives, and the arrangement fails at the `n − k + 1`-th failure in all; the counts come from the same convolution, weighted by the chance that that many spares switch in and start. |
| Cold, `k = 2`, different units | Numerical: a recursion over the switch-ins on the time and the other operating unit's start (the newcomer starts new), accurate to about 1e-4 (1e-3 for lives with a steep start, such as a Weibull of shape below 1). |
| Warm or hot, `k = 1` (any units) | Numerical: a recursion over the spares' switch-ins on a time grid (a spare switched in at `τ` has aged `dormancy_factor · τ`), accurate to about 1e-5. |
| Everything else: warm with `k ≥ 2`, hot with `k ≥ 2` and spares that may fail to switch in or start, cold with `k ≥ 3` different units, each of units not all Exponential | None: the arrangement draws lifetimes for the system's simulations, and `is_simulated` is `True`. Which spare goes where, and how much each operating unit and spare has aged, branch with the order of the failures; the refusal says so. |

An arrangement with no exact or numerical survival function is simulated
where it is used. Its `sf`, `ff`, `cs` and `mean()` raise
`NotImplementedError`, as do the analyses of a diagram that need its
reliability, naming the simulations that take it: `random`,
`mean(method="simulate")` and `unreliability_interval` of a
`NonRepairableRBD`, or `availability` and `cost` of a `RepairableRBD`.
Until 0.12 such an arrangement was given a Kaplan–Meier fit to simulated
lifetimes, which carried their Monte-Carlo error into every exact analysis
of the system; the `mc_samples`, `lower` and `seed` that set the fit went
in 0.13.

`mean()` and `random(size, seed=None)` give the arrangement's mean lifetime
and draw lifetimes; `cs(x, X)` is its conditional survival. A standby node
cannot take a [condition-based state](condition-based.md).
`mean(method="simulate", mc_samples=..., seed=...)` estimates its mean
life from that many new draws: the way to a simulated arrangement's mean,
and a check on an exact one.

A `StandbyModel` is one lifetime: in a repairable RBD the whole arrangement
is replaced as one unit when it fails. For a duty unit and a standby that
are repaired one at a time, the failed unit returning as the new spare, make
the node a [standby group](repairable.md#standby-groups) instead.

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

Its reliability is exact, and so is its `mean()`, the area under it;
`mean(method="simulate", mc_samples=1_000_000)` estimates it from that many
draws instead. A `RepeatedNode` is *n* distinct copies; for the
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
p = RegressionNode(unit, covariates=[1.0]).sf(50)
3 * p**2 - 2 * p**3   # -> 0.6608   two-out-of-three, ignoring the load transfer
```

- Identical units with an **Exponential** baseline have an exact
  (hypoexponential) group lifetime. Other identical units (one baseline and
  one load response) have a numerical one: they all age alike, so they fail
  in the order of their exposures to failure, and a recursion over the
  failures on a grid of exposure and time gives the lifetime's
  distribution, to about 1e-4. Either way `is_simulated` is `False`.
- Different units have no exact or numerical survival function, and
  `is_simulated` is `True`: as for [standby](#how-the-survival-function-is-obtained),
  the group draws lifetimes for the system's simulations, its `sf`, `ff`,
  `cs` and `mean()` raise `NotImplementedError`, and
  `mean(method="simulate", mc_samples=..., seed=...)` estimates its mean
  life.
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
line.is_analytically_solvable()   # True: no node's reliability is simulated
```

`is_analytically_solvable()` is `False` only when some node has no exact or
numerical reliability, and `get_non_analytic_nodes()` names those nodes; `analysis_routes()` says how each analysis is computed (see
[Is the system time-dependent, and is it exact?](building.md#is-the-system-time-dependent-and-is-it-exact)).
All of these models save with the RBD.
