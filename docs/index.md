# RePyability

Reliability engineering tools for Python.

RePyability is the **computational reliability engine** for building and
analysing systems as [Reliability Block Diagrams
(RBDs)](https://en.wikipedia.org/wiki/Reliability_block_diagram). It consumes
already-fitted lifetime models (from
[surpyval](https://github.com/derrynknife/SurPyval) or any equivalent that
exposes `sf`/`ff`) as node inputs and computes system reliability, availability,
importance measures, and more.

Scope notes:

- **Fitting lives in surpyval, not here.** RePyability consumes fitted models;
  fit your failure data in surpyval and pass the models in.
- **Visualisation lives in the Reliafy app, not here.** RePyability returns
  numbers and typed result objects; plotting/dashboards are a separate layer.

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

rbd.sf(50)                       # system reliability at t=50   -> 0.839...
rbd.ff([50, 100])                # system unreliability at t=50, 100 (array)
rbd.mean_time_to_failure(seed=0) # MTTF via Monte-Carlo (seed for reproducibility)
rbd.birnbaum_importance(50)      # per-node Birnbaum importance at t=50
```

Minimal path and cut sets are available too:

```python
rbd.get_min_path_sets(include_in_out_nodes=False)
rbd.get_min_cut_sets()
```

You can force nodes working or failed to explore conditional behaviour:

```python
rbd.sf(50, working_nodes=["pump1"])   # reliability given pump1 is perfect
rbd.sf(50, broken_nodes=["valve"])    # reliability given valve has failed (-> 0)
```

### Repairable systems: availability

For repairable systems, give each component a reliability **and** a
repairability (time-to-repair) distribution and simulate availability:

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

result = rbd.availability(t_simulation=100.0, N=10_000, seed=0)

result.availability          # mean availability at each event time
result.timeline              # the event times
result.criticalities.iou.up  # intersection-over-union importance (system up)
```

`availability()` returns a typed [`AvailabilityResult`][repyability.AvailabilityResult]
— see the [user guide](guide.md) and [API reference](api.md).

Long-run (steady-state) availability has a closed form and needs no simulation:

```python
rbd.mean_availability()
```

## Where to next

- **[User guide](guide.md)** — building RBDs, reliability, importance measures,
  forcing nodes, and availability results.
- **[API reference](api.md)** — every public class and method.
