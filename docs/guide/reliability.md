# Reliability of a system

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
rbd.node_mttf(seed=0) # {'pump1': 88.62, 'pump2': 88.62, 'valve': 180.55}
```

`node_sf` and `node_ff` include the input and output nodes (always 1 and 0).
`node_mttf` excludes them; it takes each model's own mean, uses `mc_samples`
Monte-Carlo draws for simulation-backed models (standby, load sharing,
repeated and nested nodes), and reports 0 for a fixed-probability node, which
has no lifetime.

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

The mean time to failure is the mean of `mc_samples` such lifetimes
(default 100 000); `mean` and `mean_time_to_failure` are the same.
`mean_time_to_failure_interval` adds its sampling uncertainty as a
[`ConfidenceInterval`][repyability.ConfidenceInterval]:

```python
rbd.mean_time_to_failure(seed=0)   # -> 93.76
ci = rbd.mean_time_to_failure_interval(mc_samples=100_000, confidence=0.95, seed=0)
ci.estimate          # -> 93.76
ci.lower, ci.upper   # (93.49, 94.03)
ci.standard_error    # -> 0.1373   sample std / √mc_samples
```

The interval narrows like `1/√mc_samples`. Common-cause groups are not
included in these simulated quantities (see
[Common-cause failures](common-cause.md#what-honours-a-ccf-group)).

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
