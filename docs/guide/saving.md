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
data["type"], data["repyability_version"]   # ('NonRepairableRBD', '0.10.1')
```

What is saved:

- the structure: edges, `k`, input and output nodes, `on_infeasible_rbd`,
  repeated components and nested RBDs (of either kind);
- common-cause groups, and for a `RepairableRBD` every cost, including cost
  distributions and acquisition costs, preventive and inspection schedules,
  `"instant"` repairs and `NonRepairable` components;
- the node models: surpyval models, parametric (including
  `FixedEventProbability`) and non-parametric (Kaplan–Meier and friends), in
  surpyval's own format (`model.to_dict()`, loaded with `surpyval.from_dict`),
  so everything surpyval keeps round-trips: an offset, a
  limited-failure-population or zero-inflation parameter, and a fit's
  covariance, so parameter uncertainty can still be propagated after
  loading; `PerfectReliability` and `PerfectUnreliability`; and the standby,
  repeated, repeated-standby, load-sharing, regression and `NonRepairable`
  wrappers recursively. Files saved before 0.10.0, which stored a
  parametric model by name and parameters, still load.

String, integer and tuple node names all survive JSON (JSON turns a tuple
into a list, and loading turns it back). Loading with the wrong class
(`NonRepairableRBD.from_dict` on a `RepairableRBD` document) raises
`ValueError`; `RBD.from_dict` and `RBD.from_json` always pick the right one.

!!! note "Two limits"
    - **Models surpyval does not know** (your own class with an `sf`) cannot
      be saved: `to_dict` raises `NotImplementedError`.
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
| `NonRepairableRBD` | `random`, `mean`, `mean_time_to_failure`, `mean_time_to_failure_interval`, `compare`, `node_mttf` |
| `RepairableRBD` | `availability`, `cost`, `compare` |
| Node models | `StandbyModel(seed=...)`, `LoadSharingModel(seed=...)`, `RepeatedNode.random`/`mean`, `RepeatedStandbyNode.random`, `StandbyModel.random`, `LoadSharingModel.random` |
| `Repairable` | every simulation-backed method |

surpyval samples from numpy's **global** random number generator, so a seed
is applied to it for the duration of the call and the caller's generator
state is restored afterwards. Seeded calls are reproducible without
disturbing the surrounding program; unseeded calls draw from wherever the
global generator is. A `RepairableRBD`'s simulations draw from random
streams of their own instead, one for each component and quantity, seeded
from `seed` (see [Random streams](simulation.md#random-streams)); an
unseeded run takes its seed from the global generator.

```python
rbd.mean(1_000, seed=0) == rbd.mean(1_000, seed=0)   # True
```

A seed reproduces a result on the same platform. numpy's mathematical
functions can differ in the last bit between operating systems and
processors. So on another machine, the same seed can give times that differ
in their last digits, and rarely a different count, where two events
nearly coincide.

Simulations involving non-parametric nodes (Kaplan–Meier and the other
surpyval non-parametric fits) are reproducible too: surpyval seeds their
draws from the global generator
([surpyval issue #361](https://github.com/derrynknife/SurPyval/issues/361)).

The simulations draw the same random numbers in the same order however they
are computed internally (in blocks for speed, or one at a time), so seeded
results do not depend on which internal path a model takes. Non-parametric
nodes are the exception: surpyval takes one seed from the global generator
for each call rather than one random number for each draw, so a block of
their draws differs from the same draws made one at a time. Their results
are still reproducible. A parallel run of a `NonRepairableRBD` (`n_jobs`)
seeds each block of simulations in turn from `seed`, so its results do not
depend on the number of processes; they differ from a run without
`n_jobs`. A `RepairableRBD`'s results are the same with `n_jobs` or
without, and with either engine (see [Parallel
runs](simulation.md#parallel-runs)).

## What is exact and what is simulated

| Quantity | How it is computed |
|---|---|
| `sf`, `ff`, `Hf`, `cs`, the importance measures, `sf_given_state`, `structural_importance` | Exact, from the node reliabilities. |
| `df`, `hf` | The exact reliability, differentiated numerically. |
| `time_to_reliability`, `bx_life`, `remaining_life` | Exact reliability, inverted by root-finding. |
| `parameter_sensitivity` | Exact Birnbaum importance times a numerical parameter derivative. |
| `random`, `mean`, `mean_time_to_failure(_interval)`, `node_mttf` (composite nodes) | Monte-Carlo. |
| `mean_availability`, `system_failure_frequency`, MUT/MDT/MTBF, `expected_cost_rate`, `total_cost`, the repairable importance measures | Exact, from the long-run node availabilities. |
| `capacity_distribution`, `system_capacity` | Exact, from the node reliabilities (at a time) or long-run availabilities. |
| `point_availability`, `mission_availability` | Exact: each component's renewal equation, solved numerically (to about `1e-7`), and the system at its components' availabilities at each time. |
| `availability` (with the capacity over time and the delivered fraction), `cost`, the simulated criticality measures | Discrete-event simulation. |
| Standby and load-sharing node reliability | Exact or numerical where a closed form or convolution applies, otherwise simulated (see [Redundancy models](redundancy-models.md#how-the-survival-function-is-obtained)). |
| `allocate_redundancy` (both kinds of RBD) | Exact scoring, except cold standby (`strategy="cold"` or `"choose"`) that needs two or more units working, of units that are not identical Exponentials: its reliability is simulated from 10 000 lifetimes, seeded, so the scores are reproducible but carry Monte-Carlo error. `method="exact"` is a proven optimum of the scores, `"greedy"` a heuristic. |
| `Repairable` policies | Analytic for a power-law process, simulated for imperfect repair. |

## Performance

- **The exact engine is fast and cached.** On construction it reduces the
  diagram's series, parallel and *k*-out-of-*n* parts to modules with closed
  forms, and decomposes whatever is left once; every evaluation replays the
  result, so importance measures, allocation searches and array-valued times
  are cheap. Minimal path and cut sets are derived once per RBD, on first
  use.
- **Structure size.** Series-parallel diagrams stay fast at any size: they
  reduce to a single module, and their path sets are never listed. Thirty
  duplicated stages in series have `2**30` minimal path sets, yet `sf` over
  400 times takes milliseconds. The cost is in the part that does not reduce
  (bridges, cross-ties, shared nodes): it grows with that part's number of
  minimal path sets, which multiplies with meshing, so a long chain of
  bridges gets expensive. Only `get_min_path_sets()`, `path_set_probabilities`
  and `fussell_vesely(fv_type="p")` list every path set, and
  `get_all_path_sets()` every simple path: avoid them on large redundant
  diagrams.
- **Simulations** are vectorised where the models allow it: `mean()` of a
  system of parametric components draws 100 000 lifetimes in well under a
  second. Availability simulations step through events, so their cost grows
  with `N` times the number of failures and repairs in the window; with
  numba installed (`pip install "repyability[fast]"`) they run compiled,
  about ten times as fast (see [The compiled
  engine](simulation.md#the-compiled-engine)).
- **Monte-Carlo error** shrinks like `1/√N`: use the confidence intervals
  (`mean_time_to_failure_interval`, `mean_availability_interval`,
  `availability_interval`, `CostResult.mean_interval`) to judge `N`, or pass
  a `tolerance` to simulate until they are narrow enough. Antithetic pairs
  and common random numbers (`compare`) get more precision from each
  simulation, and `n_jobs` spreads the simulations over several CPUs: see
  [Simulation precision and speed](simulation.md).

!!! note "Known limit: long paths through a mesh"
    The minimal path sets of the part of a diagram that does not reduce are
    found recursively, one level per node along its longest path, so a
    non-series-parallel part with a path of about a thousand or more nodes
    exceeds Python's default recursion limit of 1,000 and raises
    `RecursionError`. Raise the limit before building such an RBD:
    `sys.setrecursionlimit(10_000)` handles paths of several thousand nodes.
    Series chains and parallel groups are reduced first, so long chains and
    wide systems are not affected.
