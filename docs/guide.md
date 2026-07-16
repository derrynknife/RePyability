# User guide

## Building an RBD

An RBD is defined by its **edges** (a directed graph from a single input node
to a single output node) and a model for each intermediate node. The input and
output nodes are inferred as the unique source and sink of the graph.

```python
import surpyval as surv
from repyability import NonRepairableRBD

edges = [("s", 1), (1, 2), (2, "t")]          # series: s -> 1 -> 2 -> t
reliabilities = {
    1: surv.Weibull.from_params([20, 2]),
    2: surv.Weibull.from_params([100, 3]),
}
rbd = NonRepairableRBD(edges, reliabilities)
```

Any model exposing `sf(t)` (and `ff(t)`) can be a node: surpyval parametric and
non-parametric distributions, `FixedEventProbability`, standby arrangements
([`StandbyModel`][repyability.StandbyModel]), replicated nodes
([`RepeatedNode`][repyability.RepeatedNode]), and even nested
`NonRepairableRBD`s.

### k-out-of-n nodes

A node can require `k` of its `n` incoming paths to be working:

```python
rbd = NonRepairableRBD(edges, reliabilities, k={"valve": 2})
```

## Reliability, unreliability, and related functions

```python
rbd.sf(t)              # survival / reliability R(t)
rbd.ff(t)              # unreliability F(t) = 1 - R(t)
rbd.df(t)              # failure density f(t)
rbd.hf(t)              # hazard rate h(t)
rbd.Hf(t)              # cumulative hazard H(t) = -ln R(t)
rbd.cs(t, T)           # conditional survival R(t | already survived to T)
```

The return contract is numpy-style: a **scalar `t` returns a float**, an
**array `t` returns a numpy array** (the per-node and importance methods
likewise return dicts of floats or arrays). Both the exact **path-set**
(`method="p"`, default) and **cut-set** (`method="c"`) evaluations give the
same result.

### Mean time to failure

MTTF is estimated by Monte-Carlo simulation. Pass a `seed` for reproducible
results:

```python
rbd.mean_time_to_failure(seed=0)
rbd.node_mttf(seed=0)     # per-node MTTF
```

## Forcing nodes working or failed

`sf`/`ff`/`reliability`/`unreliability`/`cs` accept `working_nodes` and
`broken_nodes` to force nodes to perfect reliability or perfect failure — useful
for conditional analyses:

```python
rbd.sf(t, working_nodes=[1])   # R(t) given node 1 never fails
rbd.sf(t, broken_nodes=[2])    # R(t) given node 2 has failed
```

Invalid input raises rather than being silently ignored: an unknown node name,
the input/output node, or the same node in both sets.

!!! note "Pivotal decomposition"
    For any node, `sf == R_A * sf(A working) + (1 - R_A) * sf(A broken)`. This
    Birnbaum identity underpins the importance measures below.

## Importance measures

```python
rbd.birnbaum_importance(t)
rbd.improvement_potential(t)
rbd.risk_achievement_worth(t)
rbd.risk_reduction_worth(t)
rbd.criticality_importance(t)
rbd.fussell_vesely(t, fv_type="c")   # "c" cut-set (default) or "p" path-set
```

Each returns a dict mapping node name to its importance at time(s) `t`. All
importance measures also accept `working_nodes`/`broken_nodes` to condition
the analysis, e.g. `rbd.birnbaum_importance(t, broken_nodes=["pump2"])`.

The same measures exist on `RepairableRBD`, evaluated at the nodes' long-run
availabilities (no time argument): `rbd.birnbaum_importance()`, etc.

## Repairable systems and availability

A [`RepairableRBD`][repyability.RepairableRBD] takes components with both a
reliability and a repairability distribution. Long-run availability is
closed-form; the time-resolved availability (and a rich set of criticality
measures) comes from discrete-event simulation.

```python
import surpyval as surv
from repyability import RepairableRBD

components = {
    "A": {
        "reliability": surv.Exponential.from_params([0.1]),
        "repairability": surv.Exponential.from_params([1.0]),
    },
    "B": {
        "reliability": surv.Exponential.from_params([0.2]),
        "repairability": surv.Exponential.from_params([1.0]),
    },
}
rbd = RepairableRBD([("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")], components)

rbd.mean_availability()                        # steady-state, closed form
result = rbd.availability(t_simulation=100.0, N=10_000, seed=0)   # simulated
```

### Steady-state metrics (exact, no simulation)

The classic repairable-system metrics are available in closed form via the
Birnbaum/Vesely frequency formula (exact for independent nodes):

```python
rbd.system_failure_frequency()      # system failures per unit time
rbd.mean_up_time()                  # MUT: mean uninterrupted working period
rbd.mean_down_time()                # MDT: mean outage duration
rbd.mean_time_between_failures()    # MTBF = MUT + MDT = 1 / frequency
```

All of them (and `mean_availability`) accept `working_nodes`/`broken_nodes`
for conditional analyses. The simulated `AvailabilityResult` carries matching
*estimates* (`result.mean_up_time`, `result.mean_down_time`,
`result.failure_frequency`) you can cross-check against the exact values.

### The availability result

`availability()` returns a typed
[`AvailabilityResult`][repyability.AvailabilityResult]. It supports **both**
documented attribute access and the previous dict-style access, so existing
code keeps working:

```python
result.availability                       # numpy array of mean availability
result["availability"]                    # same thing (dict-style still works)

result.criticalities.iou.up               # attribute access
result["criticalities"]["iou"]["up"]      # equivalent dict access

result.criticalities.failure_criticality_index.per_system_failure
```

See [`Criticalities`][repyability.Criticalities] for what each measure means.

## Reproducibility

Every Monte-Carlo entry point (`random`, `mean`, `mean_time_to_failure`,
`node_mttf`, `availability`, and the component models' `random`) accepts a
`seed`. Because surpyval draws from NumPy's global RNG, a seed is applied to it
for the duration of the call and the caller's RNG state is restored afterwards —
so seeding is reproducible without disturbing the surrounding program.
