# System capacity

A diagram says whether a system works; a plant also asks how much it
delivers. With three pumps of half the demand each, one failure costs
nothing and two cost half the output, while the reliability of the pumps in
parallel says only that the plant still runs. Give each node a **capacity**,
its throughput while it works, and RePyability works out the exact
distribution of the system's capacity: how likely each output is, the
probability that it meets a demand, and the output to expect. Theory:
[Concepts](../concepts.md#capacity).

## Giving nodes a capacity

Every RBD class takes `capacity`, a dict of node capacities, alongside `k`:

```python
import numpy as np
import surpyval as surv
from repyability import NonRepairableRBD

pump = surv.Weibull.from_params([2000, 1.5])
pipe = surv.Exponential.from_params([1e-5])

#   in -> (p1 | p2 | p3) -> pipe -> out      pumps of 50, a pipe of 120
edges = [("in", "p1"), ("in", "p2"), ("in", "p3"),
         ("p1", "pipe"), ("p2", "pipe"), ("p3", "pipe"), ("pipe", "out")]
plant = NonRepairableRBD(
    edges,
    {"p1": pump, "p2": pump, "p3": pump, "pipe": pipe},
    capacity={"p1": 50, "p2": 50, "p3": 50, "pipe": 120},
)
```

The rules:

- A node carries at most its capacity while it works, and nothing once it
  has failed. Capacities are positive numbers in any unit (the same for
  every node): tonnes per hour, megawatts, percent of the design flow.
- The system's capacity is the most that can flow from the input to the
  output: the diagram's **maximum flow**. The edges carry any amount, so a
  series chain carries the least of its nodes' capacities and a parallel
  group the sum. Here all three pumps give 150, but the pipe passes 120.
- A node given no capacity limits nothing: it passes whatever reaches it
  while it works (a capacity of `inf`). Control systems and power supplies
  are usually of this kind: needed, but not carrying the flow.
- A node that works at several levels takes a dict of them instead (see
  [below](#components-with-several-levels)).
- A k-out-of-n node keeps its meaning (see
  [below](#k-out-of-n-nodes-and-demand)).
- A [repeated node](building.md) has the capacity of the node it repeats,
  wherever it is drawn: give that node's.

The capacity is positive exactly when the system works, so nothing about
reliability changes: capacity adds to the analysis, it does not replace it.

## At a time (non-repairable systems)

[`capacity_distribution(x)`][repyability.NonRepairableRBD.capacity_distribution]
gives the distribution at time `x`, a
[`CapacityDistribution`][repyability.CapacityDistribution]:

```python
capacity = plant.capacity_distribution(1000)
capacity.levels                     # array([  0.,  50., 100., 120.])
capacity.probabilities              # array([0.0361, 0.185 , 0.4361, 0.3428])
capacity.meets(100)                 # -> 0.7789   at least two pumps, and the pipe
capacity.mean()                     # -> 93.997   the expected capacity
capacity.delivered_fraction(100)    # -> 0.8714   E[min(capacity, 100)] / 100
capacity.meets(50)                  # -> 0.9639   the capacity is positive...
plant.sf(1000)                      # -> 0.9639   ...exactly when the plant works
```

- `levels` are the capacities the system can have, in increasing order
  (0 when it is down), and `probabilities` their probabilities.
- `meets(demand)` is the probability that the capacity is at least the
  demand: the system's reliability for that demand. A capacity within
  rounding of the demand meets it, so three units of 1/3 meet a demand of 1.
- `mean()` is the expected capacity.
- `delivered_fraction(demand)` is the fraction of the demand the system can
  be expected to deliver: all of it when the capacity is enough, and what it
  can when it is not.

An array of times gives one column per time, so each method returns an
array. The method also takes `working_nodes` and `broken_nodes`, and
honours [common-cause groups](common-cause.md):

```python
plant.capacity_distribution([500, 1000, 2000]).meets(100)   # array([0.957 , 0.7789, 0.3004])
plant.capacity_distribution(1000, broken_nodes=["p1"]).meets(100)  # -> 0.4882
```

## In the long run (repairable systems)

For a [`RepairableRBD`](repairable.md),
[`capacity_distribution()`][repyability.RepairableRBD.capacity_distribution]
gives the long-run distribution: the fraction of time the system spends at
each level. `delivered_fraction(demand)` is then the fraction of the demand
met over time, the **production availability** that plant owners contract
on:

```python
from repyability import RepairableRBD

E = surv.Exponential.from_params
unit = {"reliability": E([1 / 500]), "repairability": E([1 / 20])}  # MTTF 500, MTTR 20

pumps = RepairableRBD(
    [("in", "p1"), ("in", "p2"), ("in", "p3"),
     ("p1", "out"), ("p2", "out"), ("p3", "out")],
    {"p1": unit, "p2": unit, "p3": unit},
    capacity={"p1": 50, "p2": 50, "p3": 50},
)
long_run = pumps.capacity_distribution()
long_run.probabilities              # array([6.0000e-05, 4.2700e-03, 1.0668e-01, 8.8900e-01])
long_run.meets(100)                 # -> 0.99568   time with two pumps or more up
long_run.delivered_fraction(100)    # -> 0.99781   the production availability
long_run.mean()                     # -> 144.23    the average capacity
pumps.mean_availability()           # -> 0.99994   the time with any output at all
```

It is exact, with no simulation, from each component's long-run
availability, as [`mean_availability`][repyability.RepairableRBD.mean_availability]
is. Components inspected or replaced on a calendar are down together more
often than independent ones would be, and the distribution is averaged over
their schedules, as `mean_availability` is.

## Over a window (simulated)

The long run is where a plant settles; a contract year starts with every
unit new. [`availability()`](repairable.md#availability-over-time-simulated)
simulates the window, and when nodes have capacities the same simulations
follow what the system can deliver, against a `demand`:

```python
year = pumps.availability(8760, N=500, seed=1, demand=100)   # a year, in hours
year.capacity[:3]                   # array([150. , 149.9, 149.8])   the mean capacity curve
year.mean_capacity                  # -> 144.3     the long run: 144.23
year.delivered_fraction             # -> 0.99793   the long run: 0.99781
year.delivered_fraction_interval().upper   # -> 0.99807
year.capacity_time[100.0] / (500 * 8760)   # -> 0.10584   the time at 100
```

- `capacity_timeline` and `capacity` are the mean capacity over time, as
  `timeline` and `availability` are the mean availability. The capacity
  is worked out after every component failure and repair, not only those
  that change whether the system is up: here most failures cost 50 and
  leave the plant running.
- `capacity_time` is the time spent at each capacity, summed over the
  simulations.
- `delivered` holds each simulation's delivered fraction: what it delivered
  against the demand, `min(capacity, demand)` over the window, over the
  demand over the window. `delivered_fraction` is their mean, the
  production availability over the window, and
  `delivered_fraction_interval(confidence)` its simulation error.
- `demand` defaults to the design capacity, everything up (150 here).
  Contracts usually name a smaller one.

The simulation draws only when components fail and are repaired, exactly
as without capacities, so the availability results are unchanged. A node
working at several levels counts at each in proportion to its
probability. Nodes that take their capacity from their models (a
`DegradingNode`'s stages, or a nested RBD's capacities) are not followed:
give them a capacity, or use the exact long-run `capacity_distribution()`.

## Components with several levels

Some components degrade rather than fail outright: a pump at full, half or
no output. There are three ways to give a node several levels, and binary
nodes are the special case of each.

**Levels while it works.** A dict `{level: probability}` in place of a
number gives the levels a node works at and the probability of each while
it works. Its reliability model still says whether it works, so at a time
`t` it is at each level with the level's probability times `R(t)`, and down
with `F(t)`:

```python
derated = NonRepairableRBD(
    [("in", "p1"), ("in", "p2"), ("p1", "out"), ("p2", "out")],
    {"p1": pump, "p2": pump},
    capacity={"p1": {50: 0.9, 25: 0.1}, "p2": {50: 0.9, 25: 0.1}},   # 10% of the time at half output
)
derated.capacity_distribution(1000).levels       # array([  0.,  25.,  50.,  75., 100.])
derated.capacity_distribution(1000).meets(100)   # -> 0.3994
```

**A component that degrades over time.** A
[`DegradingNode`][repyability.DegradingNode] runs through stages, each at
its own capacity for a time from its own lifetime model, and has failed
once the last ends: a model of the component's states over time. Its
stages give its levels; its lifetime, the sum of its stages' times, gives
its reliability, so it is a node model like any other:

```python
from repyability import DegradingNode

E = surv.Exponential.from_params
worn = DegradingNode([
    (50, surv.Weibull.from_params([1500, 2])),   # full output until it wears
    (25, E([1 / 800])),                          # then half output until it fails
])
worn.stage_probabilities(1000)       # array([0.6412, 0.2379])   in each stage at 1000
worn.mean()                          # -> 2129.3   the stages' mean times, added

degrading = NonRepairableRBD(
    [("in", "p1"), ("in", "p2"), ("p1", "out"), ("p2", "out")],
    {"p1": worn, "p2": worn},
)
degrading.capacity_distribution(1000).meets(100)   # -> 0.4111   both still at full output
degrading.capacity_distribution(1000).meets(50)    # -> 0.9278
degrading.sf(1000)                                 # -> 0.9854   any output at all
```

The stage probabilities come from the same convolution of the stages' times
as the reliability: exact for identical exponential stages, and otherwise
accurate to about `1e-7`.

**A nested RBD with capacities.** A nested RBD that has capacities of its
own brings its whole distribution: a pump skid drawn as its own RBD, and
used as one node, gives the same distribution as its pumps drawn in place.

A capacity given for a node in the outer RBD takes the place of what its
model would give: the node is then a single component with that capacity.

In the long run, a `RepairableRBD` component with levels while it is up is
at each level for its probability of the time it is up. A degrading
component (a `DegradingNode` as its reliability) spends its up time in its
stages in proportion to their mean times, by the renewal-reward theorem;
one on a maintenance or inspection schedule has no exact long-run values
here. A nested `RepairableRBD` with capacities brings its own long-run
distribution.

```python
worn_unit = {"reliability": worn, "repairability": E([1 / 48])}   # MTTR 48
repaired = RepairableRBD(
    [("in", "p1"), ("in", "p2"), ("p1", "out"), ("p2", "out")],
    {"p1": worn_unit, "p2": worn_unit},
)
repaired.capacity_distribution().delivered_fraction(100)   # -> 0.7942
repaired.mean_availability()                               # -> 0.99951
```

## From node probabilities

[`RBD.system_capacity`][repyability.RBD.system_capacity] takes the
probability that each node works directly, as
[`system_probability`][repyability.RBD.system_probability] does, so a
structure with no models can be analysed too:

```python
from repyability import RBD

structure = RBD(
    [("in", "p1"), ("in", "p2"), ("p1", "pipe"), ("p2", "pipe"),
     ("pipe", "out")],
    capacity={"p1": 60, "p2": 40, "pipe": 80},
)
structure.system_capacity({"p1": 0.9, "p2": 0.8, "pipe": 1.0}).mean()  # -> 71.6
```

## k-out-of-n nodes and demand

A k-out-of-n node passes flow only while at least `k` of its inputs are
reached, as in the reliability analysis, and then carries the sum of what
they bring. So three pumps voting 2-out-of-3 at the output deliver nothing
with one pump running: the diagram says that the plant has failed.

```python
voting = RBD(
    [("in", "p1"), ("in", "p2"), ("in", "p3"),
     ("p1", "out"), ("p2", "out"), ("p3", "out")],
    k={"out": 2},
    capacity={"p1": 50, "p2": 50, "p3": 50},
)
voting.system_capacity({"p1": 0.9, "p2": 0.9, "p3": 0.9}).levels  # array([  0., 100., 150.])
```

To say instead that one pump gives half the output, draw the pumps in plain
parallel and ask what meets the demand: `meets(100)` of the plain parallel
pumps is the 2-out-of-3 reliability, and `delivered_fraction(100)` counts
the half output too.

## How it is computed

The distribution is exact, over every combination of the components'
states, assuming they are independent (apart from any common-cause groups).
The system's capacity is its maximum flow, which by the max-flow min-cut
theorem is the least total capacity of a cut: a set of nodes whose failure
would disconnect the output. It is worked out as the system reliability is
(see [Saving, reproducibility and performance](saving.md#performance)):
each series, parallel and k-out-of-n module's distribution in closed form
from its members' (the least of their capacities, their sum, their sum while
at least `k` work: the *universal generating function* of multi-state
systems), and what is left, such as a bridge, by conditioning on its parts'
capacities one at a time, keeping only each cut's running total. So it costs
about what `sf` does, and a series-parallel system of any size is quick.
