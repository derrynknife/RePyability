# User guide

The user guide is the how-to reference: every capability, the arguments it
takes, what it returns, and its limits. Each page is self-contained and its
examples run as written. To learn the ideas step by step, with worked
examples and exercises, take the [Learn](../learn/index.md) course; for a
compact summary of the theory, see [Concepts](../concepts.md); for every
signature and docstring, see the [API reference](../api/index.md).

| Page | What it covers |
|---|---|
| [Building an RBD](building.md) | Edges and node models, k-out-of-n nodes, junctions (a vote point that never fails), a component that appears in several places, nested RBDs, validation, path and cut sets, and the structural checks. |
| [Fault trees](fault-trees.md) | Static fault trees (OR, AND and VOTE gates, repeated events): the exact top event probability, minimal cut sets ranked by probability, importance measures, common-cause groups, the mean time to the top event, B-lives, hazard, parameter sensitivities and the uncertainty from fitted event models, and conversion to and from block diagrams; NOT gates (prime implicants), dynamic gates (PAND, SPARE, SEQ, FDEP) over time, and Open-PSA export. |
| [Reliability of a system](reliability.md) | `sf`/`ff`, density and hazard, conditional survival, per-node values, forcing nodes working or failed, lifetimes and the exact MTTF, inverting reliability to a time (B*X* life), and the uncertainty that uncertain component models give the system reliability. |
| [Phased missions](phased-missions.md) | Missions through phases (take-off, cruise, landing), each with its own duration and diagram over the same components: the exact mission reliability and the chance of failing in each phase, and their simulation; each component's importance over the mission, and the mission's cut sets. |
| [Networks](networks.md) | Undirected networks whose links (and nodes) fail: the exact reliability of the connection between two terminals, its minimal paths and cuts, the importance measures, B-lives, parameter uncertainty and saving of a diagram, the mean time to disconnection, and their simulation. |
| [Importance measures](importance.md) | Birnbaum, improvement potential, risk achievement and reduction worth, criticality, Fussell–Vesely, structural importance, and parameter sensitivity; on a repairable system in the long run or over time; differential and joint importance, the rate of change split by component, and Barlow–Proschan. |
| [Sensitivities: the Greeks](greeks.md) | The sensitivity measures as one family, named after the option Greeks (delta, the levers' deltas, the differential importance, gamma, theta, vega), what they share, and one pumping station run through every one. |
| [Condition-based evaluation](condition-based.md) | Reliability, remaining life and importance given each component's current age, covariate-dependent components (fixed operating conditions or a load schedule), and a unit of a joint model of readings and failures, given its readings (surpyval `Measurements` too). |
| [Redundancy models](redundancy-models.md) | Cold, warm and hot standby, repeated nodes, repeated standby, and load-sharing groups. |
| [Common-cause failures](common-cause.md) | Beta-factor and Multiple Greek Letter groups, splitting a probability or a failure rate, in block diagrams, repairable systems (with staggered tests) and fault trees, and where they apply. |
| [Repairable systems](repairable.md) | Long-run availability, failure frequency, MUT/MDT/MTBF; components operated part of the time, with tasks counted in running hours; the exact availability over time from new and over a mission, and the unavailability to its own precision; junctions; the simulated histories and their criticality measures; shared repair crews, with exact long-run values and importance and numerical values over time for exponential components; standby groups repaired one unit at a time; imperfect repair, and models fitted to a unit's repair history (Poisson processes, Kijima, G1, ARA and ARI); nested repairable RBDs, and stepping a simulation by hand. |
| [System capacity](capacity.md) | How much a system can deliver: node capacities, components with several levels (derated, or degrading through stages over time), the exact distribution of the system's capacity at a time or in the long run, the probability of meeting a demand, the expected capacity, and the production availability. |
| [Costs](costs.md) | The long-run cost rate in closed form, the simulated cost distribution, costs drawn from distributions, and the total cost of ownership, discounted to a present value, with the redundancy that minimises it, whole trains among it. |
| [Maintaining a system](system-maintenance.md) | Scheduled preventive maintenance in a diagram and the intervals that meet a system target, replacement on condition at periodic inspections, opportunistic maintenance, planned maintenance that restores partly, hidden failures found by periodic inspection (PFDavg for safety functions) with their intervals chosen, shutdowns, and planned against unplanned downtime. |
| [Spares](spares.md) | How many spares each component uses over a horizon, for a system or a fleet, and the stock that meets a fill rate or a stock-out target when each spare used is reordered and arrives a lead time later; interchangeable components' spares pooled on one shelf. |
| [Design and allocation](design.md) | How many redundant copies to fit (redundancy allocation), apportioning a reliability target among components (reliability allocation), and an availability target, as MTTF and MTTR targets (availability allocation). |
| [Maintenance policies](maintenance.md) | Age replacement, replacement over a calendar of shuts, overhaul under minimal or imperfect repair, failure-count replacement, the expected time to the *n*-th failure, repair again or replace for a fleet whose units differ (`RepairOrReplace`), the options from a part's state, and the four ways of managing a failure mode. |
| [Stress-strength interference](stress-strength.md) | The chance a strength exceeds a stress, for two fitted models: exact where a closed form exists, repeated loads, a strength that changes with age, and parameter uncertainty. |
| [Demonstration testing](demonstration.md) | How many units, or how long a test, demonstrates a reliability or an MTBF at a confidence level, plans that keep both the producer's and the consumer's risk, what a finished test demonstrated, and the chance a design passes. Deprecated in 0.17 for `surpyval.test_planning`; 0.18 removes these functions. |
| [Timelines](timelines.md) | Up/down histories: a unit's from an outage log or its durations, their measures (time up, failures and planned outages, first failure, outages), merges as a diagram's structure with the component that caused each change, a system's from its components' through its diagram, and a repairable system's simulated histories, kept whole. |
| [Simulation precision and speed](simulation.md) | Simulating until an estimate is precise enough, antithetic pairs, parallel runs, and comparing two designs with common random numbers. |
| [Saving, reproducibility and performance](saving.md) | JSON round-trips, results as plain data (`to_dict()`), seeding, what is exact and what is simulated, and how the engine scales. |

## Conventions used throughout

**Node models.** RePyability consumes *fitted* models; fitting data is
surpyval's job. A node model is anything that exposes `sf(t)` and `ff(t)`:
surpyval parametric distributions,
`FixedEventProbability`, and the composite models in this package. Methods
that simulate lifetimes also need the model's `random(size)`.

**Scalars and arrays.** A scalar time returns a float; an array of times
returns a numpy array. Per-node methods return a dict keyed by node name
whose values follow the same rule.

```python
import surpyval as surv
from repyability import NonRepairableRBD

rbd = NonRepairableRBD(
    [("s", "a"), ("a", "t")], {"a": surv.Weibull.from_params([100, 2])}
)
rbd.sf(50)          # -> 0.7788   a float
rbd.sf([25, 50])    # a numpy array: array([0.9394, 0.7788])
```

**Forcing nodes.** Most analysis methods accept `working_nodes` and
`broken_nodes`: the named nodes are treated as perfectly reliable or as
failed. An unknown node, the input or output node, or a node named in both
raises `ValueError` rather than being ignored.

**Path sets or cut sets.** Exact system quantities accept
`method="p"` (the default) or `method="c"`: whether the engine computes the
probability that the system works, or that it fails and returns the
complement. Both are exact and give the same answer. (In `is_system_working`
they choose whether the part of the diagram that is not series-parallel is
checked through its minimal path sets or cut sets.)

**Seeds.** Every Monte-Carlo method takes a `seed`. surpyval samples from
numpy's global random number generator, so a seed is applied to it for the
duration of the call and the caller's generator state is restored afterwards.
A `RepairableRBD`'s simulations draw from random streams of their own (see
[Saving, reproducibility and performance](saving.md#reproducibility)).

**Units.** Times are in whatever unit your models use; RePyability never
converts them.
