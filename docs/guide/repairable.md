# Repairable systems

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson
    6](../learn/availability.md) (repair and availability).

A [`RepairableRBD`][repyability.RepairableRBD] models a system whose
components are repaired when they fail. The question changes from "has it
failed yet?" to "is it up?": **availability**. Long-run quantities have exact
closed forms, and the availability over time from new is exact too; the
histories behind it (failure counts, downtime and cost over a window) and a
family of criticality measures come from a discrete-event simulation.
Theory: [Concepts](../concepts.md#availability).

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
  (`"repair_cost"`, `"replace_cost"`, `"downtime_cost"`, and the one-off
  `"acquisition_cost"`; see [Costs](costs.md)),
  scheduled preventive replacement (`"preventive"`; see
  [Costs](costs.md#preventive-maintenance)), for a component whose
  failures are hidden until a proof test finds them, periodic inspection
  (`"inspection"`; see [Costs](costs.md#hidden-failures-and-inspection)),
  and its place in the queue for a repair crew (`"priority"`; see
  [below](#repair-crews)). Any other key raises `ValueError`, so a
  mistyped cost key is never silently priced at zero;
- `"repairability": "instant"` for a component repaired in zero time (see
  [below](#instantly-repaired-components));
- a [`NonRepairable`][repyability.NonRepairable]`(reliability,
  time_to_replace)` object. The RBD keeps its own copy for each node, so one
  object can stand for several identical parts;
- another `RepairableRBD`, nested as a subsystem (see
  [below](#nested-repairable-rbds)).

The constructor also takes `k`, `input_node`, `output_node` and
`on_infeasible_rbd` exactly as for a
[`NonRepairableRBD`](building.md), `downtime_cost_rate` (see
[Costs](costs.md)) and `repair_crews` (see [below](#repair-crews)).

Every repair restores a component to as good as new, components fail and are
repaired independently of each other (unless they wait for a repair crew),
and a component keeps its own failure/repair cycle whether or not the system
is up.

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
exact system computation at those availabilities. A component some of whose
units never fail (a surpyval model with `p < 1`) sooner or later gets one of
them, and is then up for good: its long-run availability is 1 and its
failure frequency 0 (simulate `availability()` for the years before). The failure frequency is
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

## Availability over time (exact)

With every component new at time 0, the probability that the system is up
at a time `t`, its **point availability** `A(t)`, is exact, and so is its
mean over a mission `[0, t]`:

```python
plant.point_availability([0.0, 1.0, 5.0, 50.0])   # array([1.    , 0.9808, 0.9565, 0.9536])
plant.point_availability(1.0)                      # -> 0.9808
plant.mission_availability(100.0)                  # -> 0.9544   mean over [0, 100]
plant.mission_availability([10.0, 100.0, 1000.0]) # array([0.962 , 0.9544, 0.9537])
```

Each component's `A(t)` follows from the distributions of its up and down
times by the renewal equation, solved numerically to about `1e-7`; the
components fail and are repaired independently, so the system's is the
exact system computation at theirs, at each time (see
[Concepts](../concepts.md#availability)). The curve starts at 1 (less any
units dead on arrival) and settles at `mean_availability()`, and a mission
average differs from the long-run value by about `b / t`, for a constant `b`
of the components' up and down times. `b` is positive unless the lives vary
more than exponential ones do, and largest for components that wear out,
which fail less early on. A mission of decades costs no more than one of
hours: past the time the components have settled, the integral is extended
exactly.

Both take `working_nodes`, `broken_nodes` and `method` as
`mean_availability` does, and cover what it covers: age and block
replacement, nested RBDs, and hidden failures with a constant failure rate
and instant tests and repair. A component with any other hidden failures
raises `NotImplementedError`; simulate it. Each component's curve is
computed on a grid of 2,000 steps over its typical up time: within one step
of a time at which its units start or stop on a schedule (at 0, and at its
scheduled replacements), what happens faster than a step, such as a short
repair, is smoothed over it, so a point value there can be off by up to
about the probability that the component is under repair; mission averages
are not affected.

## Availability over time (simulated)

`availability(t_simulation, ...)` runs `mc_samples` independent simulations of the
system from time 0 (everything new, except any `broken_nodes`) to
`t_simulation`, and averages them: the same curve as `point_availability`,
with the histories behind it, which also give the failure counts, downtime,
costs and criticality measures over the window:

```python
result = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
result.timeline[:3]       # array([0.    , 0.0259, 0.027 ])  times the mean availability changes
result.availability[:3]   # array([1.    , 0.9995, 0.999 ])  mean availability at those times
result.availability[-1]   # -> 0.9495   at t = 100
np.interp(50, result.timeline, result.availability)   # -> 0.9607   at t = 50
```

| Argument | Meaning |
|---|---|
| `t_simulation` | The length of each simulated history. |
| `mc_samples` | The number of histories `N` (default 10 000). Error shrinks like `1/√N`. |
| `seed` | Seeds the run for reproducibility: each component draws from random streams of its own (see [Random streams](simulation.md#random-streams)). numpy's global RNG is left as it was. |
| `working_nodes`, `broken_nodes` | Components that never fail, or that are down throughout. |
| `method` | `"p"` or `"c"`, for deciding whether the system is up; same result. |
| `verbose` | Show a progress bar. |
| `tolerance`, `confidence`, `max_samples` | Simulate until the mean availability over the window is known to within `tolerance` (see [Simulation precision and speed](simulation.md#simulating-to-a-tolerance)). |
| `antithetic` | Simulate in antithetic pairs, for a more precise mean from the same `mc_samples` (see [Antithetic pairs](simulation.md#antithetic-pairs)). |
| `n_jobs` | Run the simulations on several CPUs, with the same result as on one (see [Parallel runs](simulation.md#parallel-runs)). |
| `engine` | `"python"`, `"numba"` (compiled) or `"auto"`, the default: the same results, faster compiled (see [The compiled engine](simulation.md#the-compiled-engine)). |
| `demand` | With node capacities, the demand the delivered fraction is measured against (see [System capacity](capacity.md#over-a-window-simulated)). |

The curve starts at 1 and settles towards the long-run availability
(`0.9536` here). Its sampling error is available pointwise:

```python
result.availability_se[-1]                            # -> 0.004896   standard error at t = 100
lower, upper = result.availability_interval(confidence=0.95)   # Wilson band
lower[-1], upper[-1]                                  # (0.939, 0.9583)
```

`lower`/`upper` align with `result.timeline`, ready to draw as a band. The
mean availability over the whole window, the fraction of it the system was
up, has an interval of its own:

```python
window = result.mean_availability_interval(confidence=0.95)
window.estimate                   # -> 0.9542   plant.mission_availability(100.0) is 0.9544
window.lower, window.upper        # (0.9526, 0.9558)
```

To compare two designs, simulate them with common random numbers:
`faster.compare(plant, t_simulation)` estimates how much more of the window
one is up than the other far more precisely than two separate runs (see
[Comparing two designs](simulation.md#comparing-two-designs)).

### What the result holds

`availability()` returns an [`AvailabilityResult`][repyability.AvailabilityResult]:

| Attribute | Meaning |
|---|---|
| `timeline`, `availability` | The mean availability curve. |
| `availability_se`, `availability_interval(confidence)` | Its sampling error. |
| `uptimes`, `mean_availability_interval(confidence)` | Each history's up time, and the interval of the mean availability over the window. |
| `antithetic` | Whether the histories ran in antithetic pairs. |
| `system_uptime`, `system_downtime` | Total up and down time over all histories. |
| `system_failures`, `system_restorations` | Counts over all histories. |
| `system_planned_outages` | The times preventive maintenance or a test took the system down (not failures). |
| `node_uptime`, `node_downtime` | Per-node totals. |
| `mean_up_time`, `mean_down_time`, `failure_frequency` | Simulation estimates of the exact MUT, MDT and frequency above. |
| `n_simulations`, `time_simulated_to` | `mc_samples` and `t_simulation`. |
| `criticalities` | The criticality measures (below). |
| `cost` | The simulated costs, or `None` when nothing is priced (see [Costs](costs.md#the-simulated-cost-distribution)). |
| `capacity_timeline`, `capacity`, `capacity_time`, `mean_capacity`, `demand`, `delivered`, `delivered_fraction`, `delivered_fraction_interval(confidence)` | With node capacities: the mean capacity over time, the time at each capacity, and the fraction of the demand delivered (see [System capacity](capacity.md#over-a-window-simulated)). `None` without capacities. |

```python
result.mean_up_time      # -> 27.20    against the exact 27.27
result.failure_frequency # -> 0.035075 against the exact 0.03497
```

The result also behaves as a read-only mapping (`result["availability"]`,
`result.keys()`, `dict(result)`), so dict-style code keeps working.

### Criticality measures

`result.criticalities` holds four measures computed from the simulated
histories. Each has an "up" and a "down" (or a "system" and a "component")
view:

```python
c = result.criticalities
c.operational_criticality_index.down   # {'A': 0.246, 'B': 0.2491, 'C': 0.8252}
c.failure_criticality_index.per_system_failure   # {'A': 0.222, 'B': 0.2294, 'C': 0.5487}
c.failure_criticality_index.per_system_failure["C"]   # -> 0.5487
```

| Measure | `up` / `by_system` / `per_system_failure` | `down` / `by_component` / `per_component_failure` |
|---|---|---|
| `operational_criticality_index` | Time node and system were both up, over system uptime. | Time node and system were both down, over system downtime. |
| `iou` | Intersection over union of the node's and the system's up time. | Same for down time. |
| `failure_criticality_index` | Fraction of system failures this node's failure caused. | Fraction of this node's failures that failed the system. |
| `restoration_criticality_index` | Fraction of system restorations this node's repair caused. | Fraction of this node's repairs that restored the system. |

Here the valve caused 55% of system failures although it fails a fifth as
often as a pump, and 99% of its failures took the system down: the pumps are
redundant and it is not. A node's failure "causes" a system failure when it
is the event that takes the system from up to down.

## Repair crews

By default every failed component is repaired at once, as though each had a
crew of its own. `repair_crews` limits how many repairs can proceed at once.
A plant with one technician repairs one pump while the next waits, so it is
down more often, and for longer:

```python
pump = {"reliability": surv.Exponential.from_params([0.1]),     # MTTF 10 h
        "repairability": surv.Exponential.from_params([0.5])}   # MTTR 2 h
three = [("s", p) for p in "xyz"] + [(p, "t") for p in "xyz"]
one_crew = RepairableRBD(three, {p: dict(pump) for p in "xyz"}, repair_crews=1)
one_crew.mean_availability()    # -> 0.9746
result = one_crew.availability(20_000.0, mc_samples=40, seed=1)
result.mean_availability_interval().estimate    # -> 0.9747   simulated
RepairableRBD(three, {p: dict(pump) for p in "xyz"}).mean_availability()   # -> 0.9954   a crew each
```

(With one crew the number of pumps down is the machine-repair model's
birth-death chain, and the system is down when all three are.)

- **What needs a crew.** Every job that brings a component back up: a repair
  or replacement, preventive maintenance that takes time, and a test that
  takes time (with the repair of any failure it finds). The component is
  down from when the job falls due until it is done. Maintenance or a test
  in no time needs no crew.
- **The queue.** A job that finds every crew busy waits. The next crew to
  finish takes the waiting job of the highest `"priority"` (a component
  spec key, by default 0), and of those the one that fell due first, and
  stays with it until it is done. A test that waits keeps the component
  off-line, and it does not age. A job with `"instant"` repair takes a crew
  for no time: it waits only when every crew is busy.
- **Nested RBDs** have crews of their own: a nested `RepairableRBD`'s
  components are repaired by its `repair_crews`, not by its parent's.
- **Exact long-run values.** With fewer crews than components, components
  wait for each other, so they no longer fail and recover independently.
  When the components the crews work on all have exponential lives and
  exponential (or instant) repairs, with no scheduled maintenance or
  inspection, the system is a Markov chain: its state is which components
  are under repair and which are waiting, in the order the crews will take
  them. `mean_availability`, `node_availability`,
  `system_failure_frequency`, `mean_up_time`, `mean_down_time`,
  `mean_time_between_failures`, `expected_cost_rate`, `total_cost` and
  `capacity_distribution` solve it exactly, for up to 15,000 states. A
  component held working or broken (`working_nodes`, `broken_nodes`) needs
  no crew, and the others share them.
- **What is simulated.** Other lives or repair times, scheduled maintenance
  and inspections, and larger chains make the exact values refuse, with the
  reason. The importance measures, the availability over time
  (`point_availability`, `mission_availability`) and the allocations assume
  independent components, so they refuse whenever a job can wait. The
  simulations (`availability`, `cost`, `compare`) follow the queue whatever
  the components, in Python. With at least as many crews as components,
  nothing waits, and every result is as without crews.

```python
routes = one_crew.analysis_routes()
routes["mean_availability"].route     # 'exact'
routes["birnbaum_importance"].route   # 'refused'
```

The chain's size is set by the queue. First come, first served, every order
in which the waiting components can have failed is a state of its own: seven
components with one crew make 13,700 states, and eight with two crews make
54,805, too many. Priorities fix much of the order, so a priority each lets
eight components with two crews through in 1,801 states. The chain is solved
in well under a second for most diagrams, and in a few seconds near the
limit.

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
fuse.availability(50.0, mc_samples=100, seed=0).availability.min()    # -> 1.0
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
nested.availability(t_simulation=100.0, mc_samples=2_000, seed=0).availability[-1]   # -> 0.951
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
drives the same machinery, with each component drawing from its own random
streams. Stepped by hand, the components draw from numpy's global RNG as it
is (there is no `seed` argument), so a history stepped by hand is not one of
`availability()`'s.
