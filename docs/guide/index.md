# User guide

The user guide is the how-to reference: every capability, the arguments it
takes, what it returns, and its limits. Each page is self-contained and its
examples run as written. To learn the ideas step by step, with worked
examples and exercises, take the [Learn](../learn/index.md) course; for a
compact summary of the theory, see [Concepts](../concepts.md); for every
signature and docstring, see the [API reference](../api.md).

| Page | What it covers |
|---|---|
| [Building an RBD](building.md) | Edges and node models, k-out-of-n nodes, a component that appears in several places, nested RBDs, validation, path and cut sets, and the structural checks. |
| [Fault trees](fault-trees.md) | Static fault trees (OR, AND and VOTE gates, repeated events): the exact top event probability, minimal cut sets ranked by probability, importance measures, and conversion to and from block diagrams. |
| [Reliability of a system](reliability.md) | `sf`/`ff`, density and hazard, conditional survival, per-node values, forcing nodes working or failed, lifetimes and the exact MTTF, inverting reliability to a time (B*X* life), and the uncertainty that uncertain component models give the system reliability. |
| [Importance measures](importance.md) | Birnbaum, improvement potential, risk achievement and reduction worth, criticality, Fussell–Vesely, structural importance, and parameter sensitivity. |
| [Condition-based evaluation](condition-based.md) | Reliability, remaining life and importance given each component's current age, and covariate-dependent components (fixed operating conditions or a load schedule). |
| [Redundancy models](redundancy-models.md) | Cold, warm and hot standby, repeated nodes, repeated standby, and load-sharing groups. |
| [Common-cause failures](common-cause.md) | Beta-factor and Multiple Greek Letter groups, and where they apply. |
| [Repairable systems](repairable.md) | Long-run availability, failure frequency, MUT/MDT/MTBF; the exact availability over time from new and over a mission; the simulated histories and their criticality measures; shared repair crews, and their exact long-run values for exponential components; standby groups repaired one unit at a time; nested repairable RBDs, and stepping a simulation by hand. |
| [System capacity](capacity.md) | How much a system can deliver: node capacities, components with several levels (derated, or degrading through stages over time), the exact distribution of the system's capacity at a time or in the long run, the probability of meeting a demand, the expected capacity, and the production availability. |
| [Costs](costs.md) | The long-run cost rate in closed form, the simulated cost distribution, costs drawn from distributions, scheduled preventive maintenance and the intervals that meet a system target, hidden failures found by periodic inspection (PFDavg for safety functions), and the total cost of ownership with the redundancy that minimises it. |
| [Design and allocation](design.md) | How many redundant copies to fit (redundancy allocation), apportioning a reliability target among components (reliability allocation), and an availability target, as MTTF and MTTR targets (availability allocation). |
| [Maintenance policies](maintenance.md) | Age replacement, overhaul under minimal or imperfect repair, failure-count replacement, and the expected time to the *n*-th failure. |
| [Demonstration testing](demonstration.md) | How many units, or how long a test, demonstrates a reliability or an MTBF at a confidence level, what a finished test demonstrated, and the chance a design passes. |
| [Simulation precision and speed](simulation.md) | Simulating until an estimate is precise enough, antithetic pairs, parallel runs, and comparing two designs with common random numbers. |
| [Saving, reproducibility and performance](saving.md) | JSON round-trips, seeding, what is exact and what is simulated, and how the engine scales. |

## Conventions used throughout

**Node models.** RePyability consumes *fitted* models; fitting data is
surpyval's job. A node model is anything that exposes `sf(t)` and `ff(t)`:
surpyval parametric and non-parametric distributions,
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
See [Saving, reproducibility and performance](saving.md#reproducibility) for
the one exception (Kaplan–Meier nodes).

**Units.** Times are in whatever unit your models use; RePyability never
converts them.
