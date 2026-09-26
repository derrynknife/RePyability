# Saving, reproducibility and performance

## Saving and loading

An RBD round-trips to a plain, JSON-friendly structure, so it can be saved,
shared, version-controlled, and loaded wherever the analysis runs:

```python
import surpyval as surv
from surpyval import FixedEventProbability
from repyability import RBD, NonRepairableRBD

rbd = NonRepairableRBD(
    [("s", 1), (1, 2), (2, "t"), ("s", 3), (3, "t")],
    {1: surv.Weibull.from_params([100, 2]),
     2: FixedEventProbability.from_params(0.01),
     3: surv.Weibull.from_params([80, 2])},
)
data = rbd.to_dict()            # a JSON-friendly dict
text = rbd.to_json(indent=2)    # a JSON string (keyword arguments go to json.dumps)

clone = NonRepairableRBD.from_json(text)
clone.sf(30) == rbd.sf(30)      # True
type(RBD.from_dict(data)).__name__   # 'NonRepairableRBD': the base class dispatches on type
data["type"], data["repyability_version"]   # ('NonRepairableRBD', '0.8.0')
```

What is saved:

- the structure: edges, `k`, input and output nodes, `on_infeasible_rbd`,
  repeated components and nested RBDs (of either kind);
- common-cause groups, and for a `RepairableRBD` every cost, including cost
  distributions, `"instant"` repairs and `NonRepairable` components;
- the node models: surpyval parametric distributions and
  `FixedEventProbability` by name and parameters, `PerfectReliability` and
  `PerfectUnreliability`, and the standby, repeated, repeated-standby,
  load-sharing, regression and `NonRepairable` wrappers recursively (a
  regression model through `surpyval.from_dict`).

Integer and string node names both survive JSON; tuple node names survive
`to_dict`/`from_dict` but not JSON (JSON turns them into lists, and loading
fails with `TypeError`). Loading with the wrong class
(`NonRepairableRBD.from_dict` on a `RepairableRBD` document) raises
`ValueError`; `RBD.from_dict` and `RBD.from_json` always pick the right one.

!!! note "Two limits"
    - **Non-parametric fits** (Kaplan–Meier and friends) cannot be saved:
      surpyval has no public way to rebuild them, so `to_dict` raises
      `NotImplementedError`. Fit a parametric model in surpyval if you need to
      save the diagram.
    - **Simulation-backed standby and load-sharing nodes** are saved by their
      inputs (`n_sims`, `dormancy_factor`, ...) but not their `seed`, and are
      re-simulated when loaded. A reloaded simulated node's reliability can
      therefore differ from the original within Monte-Carlo error. Nodes with
      an exact reliability (cold `k = 1` standby, identical Exponential
      units) reload exactly.

Condition-based state (`NodeState`) is not part of the RBD and is not saved.

## Reproducibility

Every Monte-Carlo method takes a `seed`:

| Where | Methods |
|---|---|
| `NonRepairableRBD` | `random`, `mean`, `mean_time_to_failure`, `mean_time_to_failure_interval`, `node_mttf` |
| `RepairableRBD` | `availability`, `cost` |
| Node models | `StandbyModel(seed=...)`, `LoadSharingModel(seed=...)`, `RepeatedNode.random`/`mean`, `RepeatedStandbyNode.random`, `StandbyModel.random`, `LoadSharingModel.random` |
| `Repairable` | every simulation-backed method |

surpyval samples from numpy's **global** random number generator, so a seed
is applied to it for the duration of the call and the caller's generator
state is restored afterwards. Seeded calls are reproducible without
disturbing the surrounding program; unseeded calls draw from wherever the
global generator is.

```python
rbd.mean(1_000, seed=0) == rbd.mean(1_000, seed=0)   # True
```

**The exception: non-parametric nodes.** surpyval draws Kaplan–Meier (and
other non-parametric) samples from a fresh, unseeded generator, ignoring the
global one, so a simulation involving such a node is not reproducible even
with a seed. This is tracked as
[surpyval issue #361](https://github.com/derrynknife/SurPyval/issues/361).
The exact quantities (`sf`, `ff`, the importance measures, ...) are
unaffected.

The simulations draw the same random numbers in the same order however they
are computed internally (in blocks for speed, or one at a time), so seeded
results do not depend on which internal path a model takes.

## What is exact and what is simulated

| Quantity | How it is computed |
|---|---|
| `sf`, `ff`, `Hf`, `cs`, the importance measures, `sf_given_state`, `structural_importance` | Exact, from the node reliabilities. |
| `df`, `hf` | The exact reliability, differentiated numerically. |
| `time_to_reliability`, `bx_life`, `remaining_life` | Exact reliability, inverted by root-finding. |
| `parameter_sensitivity` | Exact Birnbaum importance times a numerical parameter derivative. |
| `random`, `mean`, `mean_time_to_failure(_interval)`, `node_mttf` (composite nodes) | Monte-Carlo. |
| `mean_availability`, `system_failure_frequency`, MUT/MDT/MTBF, `expected_cost_rate`, the repairable importance measures | Exact, from the long-run node availabilities. |
| `availability`, `cost`, the simulated criticality measures | Discrete-event simulation. |
| Standby and load-sharing node reliability | Exact or numerical where a closed form or convolution applies, otherwise simulated (see [Redundancy models](redundancy-models.md#how-the-survival-function-is-obtained)). |
| `allocate_redundancy` | Exact scoring; `method="exact"` is a proven optimum, `"greedy"` a heuristic. |
| `Repairable` policies | Analytic for a power-law process, simulated for imperfect repair. |

## Performance

- **The exact engine is fast and cached.** It decomposes the system once per
  RBD and replays the decomposition for every evaluation, so importance
  measures, allocation searches and array-valued times are cheap after the
  first call. Minimal cut sets are also derived once per RBD.
- **Structure size.** The number of minimal path and cut sets can grow very
  fast with the size and meshing of a diagram; systems of a few hundred
  nodes in series or parallel are routine, and wide ones of over a thousand
  work, while densely meshed diagrams get expensive sooner. A single chain of
  about a thousand nodes in series exceeds Python's default recursion limit
  while its path sets are found, when the RBD is built.
  `get_all_path_sets()` enumerates every simple path and is the first thing
  to avoid on a large diagram.
- **Simulations** are vectorised where the models allow it: `mean()` of a
  system of parametric components draws 100 000 lifetimes in well under a
  second. Availability simulations step through events, so their cost grows
  with `N` times the number of failures and repairs in the window.
- **Monte-Carlo error** shrinks like `1/√N`: use the confidence intervals
  (`mean_time_to_failure_interval`, `availability_interval`,
  `CostResult.mean_interval`) to choose `N`.
