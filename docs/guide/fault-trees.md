# Fault trees

!!! tip "Learning this for the first time?"
    This page is the reference. The idea behind fault trees, and how they
    relate to block diagrams, is taught step by step in
    [Lesson 3](../learn/structure.md#the-same-logic-upside-down-fault-trees).

A reliability block diagram says how a system *works*. A **fault tree** says
how it *fails*: it starts from an undesired **top event** (the system
failing) and works down, through **gates**, to the **basic events** that
cause it (component failures). Safety and risk analyses are usually written
this way. [`FaultTree`][repyability.FaultTree] evaluates static fault trees
exactly, with the same engine as the diagrams, and converts them to and from
`NonRepairableRBD`s.

## Building a tree

A tree is its gates, keyed by name, and its basic events, keyed by name with
a probability or a lifetime model:

```python
import numpy as np
import surpyval as surv
from repyability import FaultTree

cooling = FaultTree(
    {
        "no cooling": ("or", ["no flow", "valve"]),       # either will do
        "no flow": ("and", ["pump 1", "pump 2"]),         # needs both
    },
    {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},        # probability each has occurred
)
cooling.top                        # 'no cooling'
cooling.top_event_probability()    # -> 0.0595   1 − (1 − 0.1²)(1 − 0.05)
cooling.ff()                       # -> 0.0595   the same, by an RBD's name
cooling.sf()                       # -> 0.9405   worked out in its own right
```

A tree answers to an RBD's names too, so the same code runs on either:
`ff` and `sf`, and the cut and path sets as a list
(`minimal_cut_sets()`, which an RBD has as well) or a set
(`get_min_cut_sets()`, as an RBD gives them).

| Gate | Written | Occurs when |
|---|---|---|
| OR | `("or", inputs)` | any input occurs |
| AND | `("and", inputs)` | every input occurs |
| VOTE | `("vote", k, inputs)` | at least `k` of the inputs occur (`k` out of `n`) |

An input is a gate or a basic event. The top event is the one gate that is
no other gate's input, or pass `top=`. Gates must not form a loop, every gate
and event must be below the top event, and a name cannot be both a gate and
an event; mistakes raise `ValueError`.

A basic event is a probability in [0, 1] that it has occurred, or a lifetime
model whose `ff(t)` is the probability that it has occurred by time `t`: a
surpyval distribution (`Weibull`, `FixedEventProbability`, ...) or any
RePyability node model. With lifetime models every method takes a time, a
number or an array:

```python
pump = surv.Weibull.from_params([8000, 1.8])
valve = surv.Weibull.from_params([20000, 1.5])
timed = FaultTree(
    {"no cooling": ("or", ["no flow", "valve"]),
     "no flow": ("and", ["pump 1", "pump 2"])},
    {"pump 1": pump, "pump 2": pump, "valve": valve},
)
timed.top_event_probability(5000)   # -> 0.2249   failed within 5000 h
timed.top_event_probability(np.array([1000, 5000, 10000]))
# array([0.01166, 0.22494, 0.72021])
```

The top event probability by time `t` is the unreliability of the system
the tree describes.

## Cut sets

A **minimal cut set** is a smallest set of basic events whose joint
occurrence makes the top event occur. They are the failure combinations the
tree allows, and a cut set of one event is a single point of failure:

```python
cooling.minimal_cut_sets()
# [frozenset({'valve'}), frozenset({'pump 1', 'pump 2'})]
cooling.ranked_cut_sets()           # most likely first
# [(frozenset({'valve'}), 0.05), (frozenset({'pump 1', 'pump 2'}), 0.01)]
timed.ranked_cut_sets(5000)         # at 5000 h the pumps dominate
# [(frozenset({'pump 1', 'pump 2'}), 0.1217), (frozenset({'valve'}), 0.1175)]
```

A cut set's probability is the product of its events'. `minimal_path_sets()`
gives the dual: the smallest sets of events whose not occurring keeps the top
event from occurring. `occurs(events)` evaluates the tree's logic for one
combination: whether the top event occurs when exactly `events` have.

## Repeated events

An event may feed several gates. Here two channels, either of which will do,
share a power supply:

```python
shared = FaultTree(
    {
        "no output": ("and", ["channel A", "channel B"]),
        "channel A": ("or", ["supply", "pump A"]),
        "channel B": ("or", ["supply", "pump B"]),
    },
    {"supply": 0.01, "pump A": 0.1, "pump B": 0.1},
)
shared.repeated_events           # frozenset({'supply'})
shared.top_event_probability()   # -> 0.0199   0.01 + 0.99 × 0.1²
```

Multiplying the channels' probabilities as if they were independent,
`(1 − 0.99 × 0.9)² = 0.0119`, would be 40% too low: one supply failure takes
out both channels at once. The tree is evaluated exactly, repeated events
included. Every gate below which nothing is shared with the rest of the tree
is a module with a closed form (an OR gate is `1 − ∏(1 − q_i)`, an AND gate
`∏ q_i`), and what the repeated events tie together is worked out by a
binary decision diagram built from the gates themselves, which branches on
one event at a time (the pivotal, or Shannon, decomposition, as for the core
of a block diagram: see
[Concepts](../concepts.md#how-the-system-quantity-is-computed)). The cut and
path sets are not listed to build it, as they multiply with the shared
events: an OR of fifty AND gates over twenty-five shared events is built and
evaluated in a fraction of a second. A tree whose diagram would pass two
million nodes (`repyability.fault_tree.DIAGRAM_LIMIT`) is refused, with the
advice to convert it with `to_rbd()` and simulate the diagram. There is no
rare-event or min-cut upper-bound approximation.

## Importance measures

Each measure returns `{event: value}` (floats for a number `t`, arrays for an
array), with `Q` the top event probability and `q_e` the event's:

| Method | Definition |
|---|---|
| `birnbaum_importance(t)` | `Q(e occurred) − Q(e did not)`: how much `Q` depends on the event. |
| `criticality_importance(t)` | `I_B(e) · q_e / Q`: the share of the top event the event accounts for. |
| `fussell_vesely(t)` | The probability that some minimal cut set containing `e` has occurred, over `Q` (`method="rare_event"`: their probabilities summed, which can pass 1). |
| `risk_achievement_worth(t)` | `Q(e occurred) / Q`. |
| `risk_reduction_worth(t)` | `Q / Q(e did not occur)`. |

```python
cooling.criticality_importance()
# {'pump 1': 0.1597, 'pump 2': 0.1597, 'valve': 0.8319}
cooling.risk_achievement_worth()["valve"]   # -> 16.81   top event certain once the valve fails
```

They are the same measures, with the same definitions, as the
[importance measures](importance.md) of a `NonRepairableRBD`: a tree and the
diagram of the same system give the same values.

## From diagrams to trees and back

`FaultTree.from_rbd(rbd)` turns a `NonRepairableRBD` into the tree of its
failure. Its series blocks become OR gates, parallel blocks AND gates, and
`k`-out-of-`n` blocks VOTE gates on `n − k + 1` failures; a part that is not
series-parallel (a bridge) becomes an OR gate over its minimal cut sets. The
nodes' models become the events', and nodes that can never fail
(`PerfectReliability` junctions) or never matter are left out:

```python
from repyability import NonRepairableRBD

fail = surv.FixedEventProbability.from_params
pumps_valve = NonRepairableRBD(
    [("in", "p1"), ("in", "p2"), ("p1", "v"), ("p2", "v"), ("v", "out")],
    {"p1": fail(0.1), "p2": fail(0.1), "v": fail(0.05)},
)
tree = FaultTree.from_rbd(pumps_valve)
tree.gates                        # {'G1': ('and', ['p1', 'p2']), 'TOP': ('or', ['G1', 'v'])}
tree.top_event_probability()      # -> 0.0595   = 1 − pumps_valve.sf()
```

`to_rbd()` goes the other way: an OR gate becomes its inputs in series, an
AND gate in parallel, and a VOTE gate a junction node (`PerfectReliability`,
named after the gate) that needs `n − k + 1` of its inputs working. A
repeated event becomes a [repeated node](building.md#one-component-in-several-places):

```python
diagram = cooling.to_rbd()
diagram.sf()                      # -> 0.9405
shared.to_rbd().repeated          # {'supply (2)': 'supply'}
```

An event or gate that feeds several gates is drawn once for each place,
its later appearances as repeated nodes that the diagram treats as the one
component, so every tree converts exactly: the diagram's reliability is one
minus the tree's top event probability. The tree of a bridge, for example,
whose events each appear in several cut sets, converts back to a diagram
with the bridge's logic.

## Saving

`to_json()` / `FaultTree.from_json()` (and `to_dict` / `from_dict`) save
the gates, the top event and the events, probabilities as numbers and
models as RePyability saves a diagram's node models.

## The model and its limits

- **Static gates only.** OR, AND and VOTE. Dynamic gates (priority-AND,
  spares, sequence dependence) and NOT gates are not supported; for standby
  and load sharing, see [Redundancy models](redundancy-models.md).
- **Independent basic events.** Events fail independently except through
  the events they share. For common-cause failures, model the cause as a
  repeated event, or use the diagram's
  [common-cause groups](common-cause.md) (`from_rbd` does not convert
  them).
- **Non-repairable.** A tree gives the probability that the top event has
  occurred by `t`. For availability with repair, build a
  [`RepairableRBD`](repairable.md).
