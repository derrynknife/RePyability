# RePyability

Reliability engineering tools for Python.

RePyability is the **computational reliability engine** for building and
analysing systems as [Reliability Block Diagrams
(RBDs)](https://en.wikipedia.org/wiki/Reliability_block_diagram). It consumes
already-fitted lifetime models (from
[surpyval](https://github.com/derrynknife/SurPyval) or any equivalent that
exposes `sf`/`ff`) as node inputs and computes system reliability,
availability, importance measures, costs, and more.

Scope notes:

- **Fitting lives in surpyval, not here.** RePyability consumes fitted models;
  fit your failure data in surpyval and pass the models in.
- **Visualisation lives in the Reliafy app, not here.** RePyability returns
  numbers and typed result objects; plotting and dashboards are a separate
  layer.

## Install

```bash
pip install repyability
```

## Quickstart

### System reliability over time

```python
import surpyval as surv
from repyability import NonRepairableRBD

# Two pumps in parallel feeding a valve in series:
#   start -> (pump1 | pump2) -> valve -> end
edges = [
    ("start", "pump1"), ("start", "pump2"),
    ("pump1", "valve"), ("pump2", "valve"),
    ("valve", "end"),
]
reliabilities = {
    "pump1": surv.Weibull.from_params([100, 2]),
    "pump2": surv.Weibull.from_params([100, 2]),
    "valve": surv.Weibull.from_params([200, 1.5]),
}
rbd = NonRepairableRBD(edges, reliabilities)

rbd.sf(50)                          # -> 0.8393   system reliability at t = 50
rbd.ff([50, 100])                   # system unreliability at t = 50, 100 (an array)
rbd.mean_time_to_failure(seed=0)    # -> 93.76    MTTF by Monte-Carlo (seeded)
rbd.birnbaum_importance(50)         # per-node Birnbaum importance at t = 50
```

Minimal path and cut sets are available too:

```python
rbd.get_min_path_sets(include_in_out_nodes=False)
rbd.get_min_cut_sets()
```

You can force nodes working or failed to explore conditional behaviour:

```python
rbd.sf(50, working_nodes=["pump1"])   # -> 0.8825   given pump1 is perfect
rbd.sf(50, broken_nodes=["valve"])    # -> 0.0      given the valve has failed
```

### Repairable systems: availability

For repairable systems, give each component a reliability **and** a
repairability (time-to-repair) distribution:

```python
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
plant = RepairableRBD([("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")], components)

plant.mean_availability()    # -> 0.9848   long-run availability, exact

result = plant.availability(t_simulation=100.0, N=2_000, seed=0)
result.availability          # mean availability at each event time
result.timeline              # the event times
result.criticalities.iou.up  # intersection-over-union importance (system up)
```

`availability()` returns a typed
[`AvailabilityResult`][repyability.AvailabilityResult]; see
[Repairable systems](guide/repairable.md).

## What it does

| Area | Capabilities | Guide |
|---|---|---|
| Structure | Series, parallel, *k*-out-of-*n*, shared components, nested subsystems, path and cut sets, validation | [Building an RBD](guide/building.md) |
| Reliability | Exact `sf`/`ff`, density, hazard, conditional survival, MTTF with confidence intervals, B*X* life | [Reliability of a system](guide/reliability.md) |
| Importance | Birnbaum, improvement potential, RAW, RRW, criticality, Fussell–Vesely, structural importance, parameter sensitivity | [Importance measures](guide/importance.md) |
| Live state | Reliability, remaining life and importance given each component's age; covariate-dependent components and load schedules | [Condition-based evaluation](guide/condition-based.md) |
| Redundancy | Cold, warm and hot standby, imperfect switching, repeated nodes, load sharing | [Redundancy models](guide/redundancy-models.md) |
| Dependence | Beta-factor and Multiple Greek Letter common-cause groups | [Common-cause failures](guide/common-cause.md) |
| Availability | Long-run availability, failure frequency, MUT/MDT/MTBF; simulated availability over time with criticality measures | [Repairable systems](guide/repairable.md) |
| Cost | Exact long-run cost rate; simulated cost distributions with percentiles; costs drawn from distributions | [Costs](guide/costs.md) |
| Design | Optimal redundancy allocation within a budget or to a target; reliability allocation | [Design and allocation](guide/design.md) |
| Maintenance | Age replacement; overhaul under minimal or imperfect repair; replace at the *N*-th failure | [Maintenance policies](guide/maintenance.md) |
| Persistence | JSON round-trips, seeded reproducibility | [Saving, reproducibility and performance](guide/saving.md) |

## Where to next

- **[Tutorial](tutorial.md)**: a start-to-finish worked example. Model a
  system, find its weak link, price extra redundancy, read its remaining life
  from live state, and cost it.
- **[User guide](guide/index.md)**: every capability, argument and return
  contract, with runnable examples.
- **[Concepts](concepts.md)**: the theory: path and cut sets, the exact engine,
  choosing an importance measure, conditioning, standby and load sharing,
  common cause, availability, cost, allocation and maintenance models.
- **[API reference](api.md)**: every public class and method.
