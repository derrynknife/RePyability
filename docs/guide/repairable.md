# Repairable systems

A [`RepairableRBD`][repyability.RepairableRBD] models a system whose
components are repaired when they fail. The question changes from "has it
failed yet?" to "is it up?": **availability**. Long-run quantities have exact
closed forms; the availability over time, and a family of criticality
measures, come from a discrete-event simulation. Theory:
[Concepts](../concepts.md#availability).

## Components

Each component needs a reliability (time-to-failure) model and a
repairability (time-to-repair) model:

```python
import numpy as np
import surpyval as surv
from repyability import NonRepairable, RepairableRBD

def unit(failure_rate, repair_rate):
    return {
        "reliability": surv.Exponential.from_params([failure_rate]),
        "repairability": surv.Exponential.from_params([repair_rate]),
    }

#   s -> (A | B) -> C -> t        MTTF 10 / MTTR 1 pumps, MTTF 50 / MTTR 2 valve
edges = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]
plant = RepairableRBD(edges, {"A": unit(0.1, 1.0), "B": unit(0.1, 1.0), "C": unit(0.02, 0.5)})
```

A component can be given as:

- a dict with `"reliability"` and `"repairability"`, plus optional costs
  (`"repair_cost"`, `"replace_cost"`, `"downtime_cost"`; see [Costs](costs.md)).
  Any other key raises `ValueError`, so a mistyped cost key is never silently
  priced at zero;
- `"repairability": "instant"` for a component repaired in zero time (see
  [below](#instantly-repaired-components));
- a [`NonRepairable`][repyability.NonRepairable]`(reliability,
  time_to_replace)` object. The RBD keeps its own copy for each node, so one
  object can stand for several identical parts;
- another `RepairableRBD`, nested as a subsystem (see
  [below](#nested-repairable-rbds)).

The constructor also takes `k`, `input_node`, `output_node` and
`on_infeasible_rbd` exactly as for a
[`NonRepairableRBD`](building.md), and `downtime_cost_rate` (see
[Costs](costs.md)).

Every repair restores a component to as good as new, components fail and are
repaired independently of each other, and a component keeps its own
failure/repair cycle whether or not the system is up.

## Long-run availability and frequencies (exact)

```python
plant.mean_availability()           # -> 0.9536   long-run fraction of time up
plant.mean_unavailability()         # -> 0.04641
plant.node_availability()           # {'A': 0.9091, 'B': 0.9091, 'C': 0.9615, 's': 1.0, 't': 1.0}
plant.system_failure_frequency()    # -> 0.03497  system failures per unit time
plant.mean_up_time()                # -> 27.27    MUT: mean length of an up period
plant.mean_down_time()              # -> 1.327    MDT: mean length of an outage
plant.mean_time_between_failures()  # -> 28.6     MTBF = MUT + MDT = 1 / frequency
```

A component's availability is `MTTF / (MTTF + MTTR)`; the system's is the
exact system computation at those availabilities. The failure frequency is
the Birnbaum/Vesely formula: each node's Birnbaum importance times its own
failure frequency `1 / (MTTF + MTTR)`, summed. All of these accept
`working_nodes`/`broken_nodes`, and `mean_availability` accepts
`method="p"/"c"`:

```python
plant.mean_availability(working_nodes=["A"])        # -> 0.9615   only C can fail now
plant.system_failure_frequency(working_nodes=["A"]) # -> 0.01923
```

A forced node never changes state, so it contributes no failures. The
importance measures on a repairable system are covered on
[Importance measures](importance.md#on-a-repairable-system).

## Availability over time (simulated)

`availability(t_simulation, ...)` runs `N` independent simulations of the
system from time 0 (everything new, except any `broken_nodes`) to
`t_simulation`, and averages them:

```python
result = plant.availability(t_simulation=100.0, N=2_000, seed=0)
result.timeline[:3]       # array([0.    , 0.0037, 0.0289])  times the mean availability changes
result.availability[:3]   # array([1.    , 0.9995, 0.999 ])  mean availability at those times
result.availability[-1]   # -> 0.957   at t = 100
np.interp(50, result.timeline, result.availability)   # -> 0.9531   at t = 50
```

| Argument | Meaning |
|---|---|
| `t_simulation` | The length of each simulated history. |
| `N` | The number of histories (default 10 000). Error shrinks like `1/√N`. |
| `seed` | Seeds the run for reproducibility; the caller's RNG state is restored. |
| `working_nodes`, `broken_nodes` | Components that never fail, or that are down throughout. |
| `method` | `"p"` or `"c"`, for deciding whether the system is up; same result. |
| `verbose` | Show a progress bar. |

The curve starts at 1 and settles towards the long-run availability
(`0.9536` here). Its sampling error is available pointwise:

```python
result.availability_se[-1]                            # -> 0.004536   standard error at t = 100
lower, upper = result.availability_interval(confidence=0.95)   # Wilson band
lower[-1], upper[-1]                                  # (0.9472, 0.965)
```

`lower`/`upper` align with `result.timeline`, ready to draw as a band.

### What the result holds

`availability()` returns an [`AvailabilityResult`][repyability.AvailabilityResult]:

| Attribute | Meaning |
|---|---|
| `timeline`, `availability` | The mean availability curve. |
| `availability_se`, `availability_interval(confidence)` | Its sampling error. |
| `system_uptime`, `system_downtime` | Total up and down time over all histories. |
| `system_failures`, `system_restorations` | Counts over all histories. |
| `node_uptime`, `node_downtime` | Per-node totals. |
| `mean_up_time`, `mean_down_time`, `failure_frequency` | Simulation estimates of the exact MUT, MDT and frequency above. |
| `n_simulations`, `time_simulated_to` | `N` and `t_simulation`. |
| `criticalities` | The criticality measures (below). |
| `cost` | The simulated costs, or `None` when nothing is priced (see [Costs](costs.md#the-simulated-cost-distribution)). |

```python
result.mean_up_time      # -> 27.29    against the exact 27.27
result.failure_frequency # -> 0.03496  against the exact 0.03497
```

The result also behaves as a read-only mapping (`result["availability"]`,
`result.keys()`, `dict(result)`), so dict-style code keeps working.

### Criticality measures

`result.criticalities` holds four measures computed from the simulated
histories. Each has an "up" and a "down" (or a "system" and a "component")
view:

```python
c = result.criticalities
c.operational_criticality_index.down   # {'A': 0.238, 'B': 0.2337, 'C': 0.8384}
c.failure_criticality_index.per_system_failure   # {'A': 0.2217, 'B': 0.2191, 'C': 0.5592}
c.failure_criticality_index.per_system_failure["C"]   # -> 0.5592
```

| Measure | `up` / `by_system` / `per_system_failure` | `down` / `by_component` / `per_component_failure` |
|---|---|---|
| `operational_criticality_index` | Time node and system were both up, over system uptime. | Time node and system were both down, over system downtime. |
| `iou` | Intersection over union of the node's and the system's up time. | Same for down time. |
| `failure_criticality_index` | Fraction of system failures this node's failure caused. | Fraction of this node's failures that failed the system. |
| `restoration_criticality_index` | Fraction of system restorations this node's repair caused. | Fraction of this node's repairs that restored the system. |

Here the valve caused 56% of system failures although it fails a fifth as
often as a pump, and 99% of its failures took the system down: the pumps are
redundant and it is not. A node's failure "causes" a system failure when it
is the event that takes the system from up to down.

## Instantly repaired components

When repairs are much faster than the time scale of interest, or there is no
repair-time data, give `"repairability": "instant"`. The component still
fails (and any repair or replacement cost is charged), but every outage has
zero length, so its availability is exactly 1:

```python
fuse = RepairableRBD(
    [("s", "f"), ("f", "t")],
    {"f": {"reliability": surv.Exponential.from_params([0.1]),
           "repairability": "instant"}},
)
fuse.mean_availability()                                     # -> 1.0
fuse.availability(50.0, N=100, seed=0).availability.min()    # -> 1.0
```

Invisible to availability, visible to cost: see [Costs](costs.md).

## Nested repairable RBDs

A `RepairableRBD` can be a component of another. The nested RBD runs its own
simulation on the same clock, and the outer system sees it change state when
its own system state changes. The exact quantities recurse:

```python
pair = RepairableRBD(
    [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
    {"A": unit(0.1, 1.0), "B": unit(0.1, 1.0)},
)
nested = RepairableRBD(
    [("s", "pumps"), ("pumps", "C"), ("C", "t")],
    {"pumps": pair, "C": unit(0.02, 0.5)},
)
nested.mean_availability()         # -> 0.9536   the same system as `plant`
nested.system_failure_frequency()  # -> 0.03497
nested.availability(t_simulation=100.0, N=2_000, seed=0).availability[-1]   # -> 0.9535
```

Use one `RepairableRBD` object per place it appears: the same object used for
two nodes shares one simulation state and raises `ValueError` during
`availability()`. (One `NonRepairable` object *can* be reused; each node gets
its own copy.)

```python
pump = NonRepairable(surv.Weibull.from_params([30, 1.5]), surv.Exponential.from_params([0.5]))
twin_pumps = RepairableRBD(
    [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")], {"a": pump, "b": pump}
)
twin_pumps.components["a"] is twin_pumps.components["b"]   # False: separate copies
twin_pumps.mean_availability()                             # -> 0.9953
```

## Stepping a simulation by hand

`initialize_event_queue(t_simulation)` sets up one history, and each call to
`next_event()` advances it to the next change in the *system* state,
returning `(time, system_is_up)`. At the end of the window it returns
`(t_simulation, state)`; calling it again raises `ValueError` until the queue
is initialised again.

```python
np.random.seed(3)
plant.initialize_event_queue(100.0)
events = [plant.next_event()]
while events[-1][0] < 100.0:
    events.append(plant.next_event())
events[0]   # (15.06..., False): the plant first went down at t = 15.06
```

This is the interface a nested RBD presents to its parent; `availability()`
drives the same machinery. It uses numpy's global RNG as is (there is no
`seed` argument).
