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
rbd.mean_time_to_failure()          # -> 93.82    mean time to failure, exactly
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

result = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
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
| Fault trees | OR, AND and VOTE gates with repeated events; exact top event probability, minimal cut sets ranked by probability, importance measures; conversion to and from block diagrams | [Fault trees](guide/fault-trees.md) |
| Reliability | Exact `sf`/`ff`, density, hazard, conditional survival, the exact MTTF (or simulated, with confidence intervals), B*X* life; uncertainty intervals from uncertain component models | [Reliability of a system](guide/reliability.md) |
| Phased missions | Missions through phases with their own durations and diagrams over the same components: the exact mission reliability and the chance of failing in each phase, or simulated | [Phased missions](guide/phased-missions.md) |
| Networks | Undirected networks with failing links and nodes: the exact two-terminal reliability, minimal paths and cuts, link importance and mean time to disconnection, or simulated | [Networks](guide/networks.md) |
| Importance | Birnbaum, improvement potential, RAW, RRW, criticality, Fussell–Vesely, structural importance, parameter sensitivity | [Importance measures](guide/importance.md) |
| Live state | Reliability, remaining life and importance given each component's age; covariate-dependent components and load schedules | [Condition-based evaluation](guide/condition-based.md) |
| Redundancy | Cold, warm and hot standby, imperfect switching, repeated nodes, load sharing | [Redundancy models](guide/redundancy-models.md) |
| Dependence | Beta-factor and Multiple Greek Letter common-cause groups | [Common-cause failures](guide/common-cause.md) |
| Availability | Long-run availability, failure frequency, MUT/MDT/MTBF; exact availability over time and over a mission; simulated histories with criticality measures; shared repair crews, exact for exponential components in the long run and numerical over time; repairable standby groups | [Repairable systems](guide/repairable.md) |
| Capacity | The exact distribution of how much a system can deliver, from its components' capacities (with several levels, or degrading through stages over time), at a time or in the long run; the probability of meeting a demand, the expected capacity and the production availability | [System capacity](guide/capacity.md) |
| Cost | Exact long-run cost rate; simulated cost distributions with percentiles; costs drawn from distributions; scheduled preventive maintenance (age or block replacement) priced exactly at system level, and its intervals chosen for a cost or availability target; replacement on condition at periodic inspections, numerical in the long run and over time; hidden failures found by periodic inspection (PFDavg); total cost of ownership, and the redundancy that minimises it | [Costs](guide/costs.md) |
| Spares | The distribution of each component's replacements over a horizon, for a system or a fleet, and the stock that meets a fill rate or a stock-out target for a replenishment lead time | [Spares](guide/spares.md) |
| Design | Optimal redundancy allocation within a budget (of one or several resources) or to a target, with a choice of component types, k-out-of-n nodes and cold standby spares, and the whole cost-reliability trade-off; reliability-redundancy allocation; reliability allocation by equal and ARINC-style apportionment, minimum effort (Albert) and cost-based (Mettas) methods; availability allocation, to MTTF and MTTR targets or the cheapest mix of the two | [Design and allocation](guide/design.md) |
| Maintenance | Age replacement; overhaul under minimal or imperfect repair; replace at the *N*-th failure | [Maintenance policies](guide/maintenance.md) |
| Testing | Demonstration test plans: the units, or the test time, that demonstrate a reliability or an MTBF at a confidence level, including Weibayes; what a test demonstrated; the chance a design passes | [Demonstration testing](guide/demonstration.md) |
| Simulation | Simulating to a tolerance, antithetic pairs, parallel runs, and comparing designs with common random numbers | [Simulation precision and speed](guide/simulation.md) |
| Persistence | JSON round-trips, seeded reproducibility | [Saving, reproducibility and performance](guide/saving.md) |

## Where to next

- **[Learn](learn/index.md)**: a short course in system reliability
  engineering, from a single part's lifetime to designing and maintaining
  whole systems. Each lesson works the ideas out by hand and then with
  RePyability, with exercises and worked answers. Start here if the theory is
  new to you.
- **[Tutorial](tutorial.md)**: a start-to-finish worked example. Model a
  system, find its weak link, price extra redundancy, read its remaining life
  from live state, and cost it.
- **[User guide](guide/index.md)**: every capability, argument and return
  contract, with runnable examples.
- **[Concepts](concepts.md)**: the theory: path and cut sets, the exact engine,
  choosing an importance measure, conditioning, standby and load sharing,
  common cause, availability, cost, allocation and maintenance models.
- **[API reference](api.md)**: every public class and method.
