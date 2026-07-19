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

### Inverse reliability (BX life, design life)

To go the other way — from a target reliability to a time — invert the
survival function:

```python
rbd.time_to_reliability(0.9)   # time at which R(t) drops to 0.9
rbd.bx_life(10)                # B10 life: time by which 10% have failed
```

`bx_life(x)` is `time_to_reliability(1 - x/100)`. Both accept the same
`working_nodes`/`broken_nodes`/`method` arguments as `sf`, and raise if the
target reliability is never reached or the RBD is fixed-probability (constant
in time).

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

### Structural importance (design-time, model-free)

```python
rbd.structural_importance()
```

`structural_importance` is the Birnbaum importance with every node reliability
set to 1/2 — the fraction of the states of the *other* nodes in which a node is
pivotal. It depends only on the diagram, not on any failure model, so you can
rank where redundancy matters most **before** any life data exists. Being a
structural property, it is identical for a `NonRepairableRBD` and a
`RepairableRBD` built on the same diagram. For example, a node in series with a
parallel pair scores `0.75` against the pair's `0.25` each. It accepts the same
`working_nodes`/`broken_nodes` conditioning.

### Parameter sensitivity (which fitted parameter matters most)

```python
rbd.parameter_sensitivity(t)
# {node: {"alpha": dR_sys/d_alpha, "beta": ...}, ...}
```

`parameter_sensitivity` (on `NonRepairableRBD`) reports how much the system
reliability moves per unit change in each node's distribution parameters:
`dR_sys/d_theta = birnbaum_importance(node) * d sf_node/d_theta`. The parameter
derivative is taken numerically (rebuilding the distribution with a perturbed
parameter), so it works for any surpyval parametric model without
per-distribution formulae. Use it to target data collection or to gauge the
impact of estimation uncertainty. Composite nodes (a nested RBD, a standby
arrangement, a repeated node) and fitted non-parametric models have no
parameters to perturb and are omitted; a node forced via
`working_nodes`/`broken_nodes` is pinned regardless of its parameters and so
reports zero. As elsewhere, a scalar `t` returns floats and an array returns
numpy arrays.

## Condition-based reliability (a "digital twin")

The methods above assume every component is brand new. In a condition-based
setting each component instead has a *current state* — how much life it has
already accumulated (from sensors/telemetry), and whether it is still working —
and each conditions on its own state:

```
R_i(x | X_i)     = R_i(X_i + x) / R_i(X_i)
R_sys(x | {X_i}) = system reliability of the conditioned components
```

You pass the state as a dict of [`NodeState`][repyability.NodeState] (a node
omitted from the dict is treated as new):

```python
from repyability import NodeState

state = {
    "pump1": NodeState(age=1200),           # 1200 hours into life
    "pump2": NodeState(age=800),
    "valve": NodeState(alive=False),        # already failed
}

rbd.sf_given_state(x, state)          # reliability a further x from now
rbd.remaining_life(0.9, state)        # remaining useful life (RUL) to R=0.9
rbd.importances_given_state(x, state) # {"birnbaum": {...}, "criticality": {...}}
```

- `sf_given_state(x, state)` is the conditional generalisation of `sf` — with an
  empty state it is exactly `sf(x)`, and at `x=0` it is 1 (nothing has failed
  yet).
- `remaining_life(target, state)` is the state-conditioned inverse: the further
  time until system reliability falls to `target`. As redundancy is worn down,
  RUL collapses toward the weakest surviving path.
- `importances_given_state(x, state)` re-evaluates the Birnbaum and criticality
  importances at the conditioned reliabilities, so the ranking reflects the
  current wear rather than the as-new design.

Only lifetime (time-varying) distributions age; a fixed-probability component
conditioned on being alive contributes reliability 1 going forward. This
release supports ordinary distribution components — dynamic nodes (standby) and
composite nodes raise if given a state. The structure is static (persist it via
serialisation); state is transient input you supply per evaluation.

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

### Simulation uncertainty

Monte-Carlo results are estimates, and the result objects quantify their
sampling error:

```python
result = rbd.availability(t_simulation=100.0, N=10_000, seed=0)

result.availability_se                    # pointwise standard error
lower, upper = result.availability_interval(confidence=0.95)   # Wilson band
```

`lower`/`upper` align with `result.timeline`, so a consumer can render the
availability curve with its confidence band. For MTTF:

```python
interval = nonrepairable_rbd.mean_time_to_failure_interval(
    mc_samples=100_000, confidence=0.95, seed=0
)
interval.estimate, interval.lower, interval.upper, interval.standard_error
```

The simulator itself is validated against exact Markov solutions: the
transient availability of exponential systems (single component, series,
parallel) is held to the closed-form curve within sampling error in the test
suite.

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

## Maintenance policies: replacement vs overhaul

Two component-level classes price preventive maintenance. They sit at the
two ends of the repair-effectiveness spectrum:

- [`NonRepairable`][repyability.NonRepairable] — the unit is **replaced** on
  failure ("as good as new"): every cycle is a statistical renewal. This is
  also the component model `RepairableRBD` uses internally, because RBD
  repairs are assumed to restore a component to as-new.
- [`Repairable`][repyability.Repairable] — the unit is patched by **repair**
  that rejuvenates it anywhere from not at all (**minimal repair**, "as bad as
  old") to partially (**imperfect repair** / generalized renewal), then
  periodically **overhauled or replaced** to as-new. It prices the same policy
  from the model's expected number of failures `E[N(t)]`: analytic `cif` for
  minimal repair (e.g. surpyval's Crow-AMSAA) or seeded `mcf` for imperfect
  repair (a fitted `GeneralizedRenewal`, Kijima I/II). Because a repaired unit
  does not renew, `Repairable` is a standalone tool — it is **not** a valid RBD
  node model.

### Age replacement (`NonRepairable`)

Replace preventively at age `t` (planned, cost `cp`) or on failure
(unplanned, cost `cu > cp`); minimise the long-run cost rate:

```python
import surpyval as surv
from repyability import NonRepairable

comp = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
comp.set_costs_planned_and_unplanned(1, 5)

comp.find_optimal_replacement()       # optimal replacement age (float)
policy = comp.optimal_replacement_policy()
policy.interval, policy.cost_rate     # typed MaintenancePolicy result
```

If the lifetime shows no aging (an exponential, or a Weibull with shape
&le; 1), preventive replacement never pays: the interval is `inf` and the
policy cost rate is the run-to-failure rate `cu / MTTF`.

### Overhaul under minimal repair (`Repairable`)

Minimal repairs cost `cr` each; an overhaul costs `co > cr` and renews the
unit. The optimal overhaul interval minimises
`(cr * Lambda(t) + co) / t`, where `Lambda` is the cumulative intensity —
the classic Barlow-Hunter policy:

```python
from surpyval.recurrent import CrowAMSAA
from repyability import Repairable

unit = Repairable(CrowAMSAA.from_params([1500, 1.5]))
unit.set_repair_and_overhaul_costs(100, 10_000)

unit.find_optimal_overhaul_interval()   # inf if the unit never wears out
policy = unit.optimal_overhaul_policy()
policy.interval, policy.cost_rate
```

A finite optimum requires wear-out (Crow-AMSAA `beta > 1`). For HPP-like
behaviour (`beta <= 1`) repairs never become more frequent, overhauls never
pay, and the interval is `inf` with the cost rate of repairs alone.

### Imperfect repair (`Repairable`)

Real repairs sit between the extremes: each one rejuvenates the unit
*partially* (a restoration factor `0 < q < 1` — the generalized-renewal /
Kijima virtual-age model), so failures still accelerate, but more slowly than
under minimal repair. Give `Repairable` a fitted surpyval `GeneralizedRenewal`
and the *same* overhaul/replacement policy applies — only now `E[N(t)]` has no
closed form and is estimated by simulation, so pass a `seed` for a reproducible
result:

```python
from surpyval import Weibull
from surpyval.recurrent import GeneralizedRenewal
from repyability import Repairable

model = GeneralizedRenewal.fit_from_parameters(
    [1500, 2.0], q=0.4, kijima="i", dist=Weibull
)
unit = Repairable(model)                 # unit.is_simulated is True
unit.set_repair_and_overhaul_costs(100, 10_000)

policy = unit.optimal_overhaul_policy(seed=0)   # seeded → reproducible
policy.interval, policy.cost_rate
```

Minimal repair is the `q = 1` edge of this model and perfect repair (`q = 0`,
every repair a renewal) the `NonRepairable` boundary. `n_simulations` sets the
Monte-Carlo sample size and `max_interval` the search horizon — raise it if the
optimum (which grows as repairs get more effective) sits beyond the default.

Instead of renewing at a fixed *age*, you can renew at a fixed **failure
count** — repair on each failure and replace at the N-th:

```python
unit.optimal_failure_limit_policy(seed=0)   # -> FailureLimitPolicy(failure_count, cost_rate)
unit.expected_time_to_nth_failure(5, seed=0)  # E[T_5] by simulation
```

For the minimal-repair (`q = 1`, power-law) limit, `E[T_n]` is exact and needs
no simulation:

```python
from repyability import minimal_repair_time_to_nth_failure
minimal_repair_time_to_nth_failure(alpha=1500, beta=2.0, n=5)
```

## Saving and loading

An RBD round-trips to a plain, JSON-friendly structure, so it can be saved,
loaded, shared, and version controlled:

```python
data = rbd.to_dict()          # JSON-friendly dict
text = rbd.to_json(indent=2)  # JSON string (kwargs pass to json.dumps)

clone = NonRepairableRBD.from_dict(data)
clone = NonRepairableRBD.from_json(text)
```

The reconstructed RBD is equivalent — same structure (edges, k-out-of-n,
repeated nodes, nested RBDs) and same reliability/availability. Node models
are serialised structurally: surpyval parametric distributions as their name
and parameters, and the standby/repeated/`NonRepairable` wrappers recursively.
Integer *and* string node names survive JSON (the per-node collections are
encoded as lists of entries, not string-keyed objects).

`RBD.from_dict`/`from_json` dispatch on the document's type, so
`RBD.from_dict(data)` returns the right subclass; calling the wrong subclass
(`RepairableRBD.from_dict(a_non_repairable_document)`) raises.

!!! note
    Fitted non-parametric models (e.g. a Kaplan-Meier estimate) cannot be
    serialised — surpyval has no public reconstruction API for them — and
    raise a clear `NotImplementedError`. Fit in surpyval and pass parametric
    models in if you need to persist the diagram.

## Reproducibility

Every Monte-Carlo entry point (`random`, `mean`, `mean_time_to_failure`,
`node_mttf`, `availability`, and the component models' `random`) accepts a
`seed`. Because surpyval draws from NumPy's global RNG, a seed is applied to it
for the duration of the call and the caller's RNG state is restored afterwards —
so seeding is reproducible without disturbing the surrounding program.
