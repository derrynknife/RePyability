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
`NonRepairableRBD`s. It also takes [NOT gates](#not-gates-non-coherent-trees)
and the [dynamic gates](#dynamic-gates) (priority AND, spares, sequences and
functional dependencies), and writes the
[Open-PSA](#exporting-to-open-psa) exchange format.

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
| NOT | `("not", [input])` | the input has not occurred ([non-coherent](#not-gates-non-coherent-trees)) |
| PAND | `("pand", inputs)` | every input occurs, in order, left to right ([dynamic](#dynamic-gates)) |
| SPARE | `("spare", inputs)`, `("spare", dormancy, inputs)` | the primary (first) and every spare after it have failed |
| SEQ | `("seq", inputs)` | every input occurs; each starts ageing once the one before has occurred |
| FDEP | `("fdep", trigger, dependents)` | (no output) the trigger occurring makes each dependent occur |

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

A cut set's probability is the product of its events' (with
[common causes](#common-causes), that of its events occurring together).
`minimal_path_sets()` gives the dual: the smallest sets of events whose not occurring keeps the top
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

## Common causes

Redundant pumps of one design, in one room, also fail together: from one
flood, one bad batch of seals, one maintenance error. `ccf_groups` (#184)
declares such groups over basic events, with the
[common-cause models](common-cause.md) a block diagram takes: a
`BetaFactor` or `MGL` model splits each member's probability into the part
that is its own and the shared causes that fail several members at once.

```python
from repyability import BetaFactor, CCFGroup

grouped = FaultTree(
    {"no cooling": ("or", ["no flow", "valve"]),
     "no flow": ("and", ["pump 1", "pump 2"])},
    {"pump 1": 0.1, "pump 2": 0.1, "valve": 0.05},
    ccf_groups=[CCFGroup(["pump 1", "pump 2"], BetaFactor(0.1))],
)
grouped.top_event_probability()   # -> 0.0671   against 0.0595 independent
grouped.ranked_cut_sets()
# [(frozenset({'valve'}), 0.05), (frozenset({'pump 1', 'pump 2'}), 0.0180)]
grouped.criticality_importance()
# {'pump 1': 0.255, 'pump 2': 0.255, 'valve': 0.7315}
```

The pumps' pair now fails with probability `0.1 × 0.1 + 0.99 × 0.09² =
0.0180`, nearly twice the independent `0.01`: the shared cause fails both
with `β Q`, and without it each fails on its own with `(1 − β) Q`. A
`BetaFactor` group is that cause drawn as a repeated event (`OR(pump 1's
own, cause)` and `OR(pump 2's own, cause)`), which the group spares you
writing; an `MGL` group of three or more has a cause for each set of
members, as the [models](common-cause.md#what-each-model-assigns) assign.

- **Exact**, as for a diagram: the top event probability and every
  importance measure sum over the groups' shock outcomes, given each of
  which the events are independent; a member's importance is conditioned
  on its own state through them. A tree and its `to_rbd()` diagram, which
  carries the groups, give the same values.
- **The cut sets stay sets of basic events.** `ranked_cut_sets` gives each
  the probability that its events occur together, the shared causes
  included, so a pair of members often leads the ranking. (PRA tools that
  draw each shared cause as an event of its own list it as a cut set of
  one.)
- **Rules**, as a diagram's: a member is a basic event, in one group at
  most, and the members have one model. A group splitting a probability
  warns once its members' probability passes 0.1, and
  `basis="rate"` splits the failure rate over a lifetime
  ([Over a lifetime](common-cause.md#over-a-lifetime)).
- **Conversions keep them**: `to_rbd()` passes the groups to the diagram
  and `from_rbd` takes the diagram's, refusing a group with a member that
  cannot affect the system (it is no event of the tree, but the group's
  shocks would still strike it) and dropping one whose members none can.

## Importance measures

Each measure returns `{event: value}` (floats for a number `t`, arrays for an
array), with `Q` the top event probability and `q_e` the event's:

| Method | Definition |
|---|---|
| `birnbaum_importance(t)` | `Q(e occurred) − Q(e did not)`: how much `Q` depends on the event. |
| `criticality_importance(t)` | `I_B(e) · q_e / Q`: the share of the top event the event accounts for. |
| `improvement_potential(t)` | `Q − Q(e did not occur)`, which is `I_B(e) · q_e`: how far the top event would fall if the event could not occur. |
| `fussell_vesely(t)` | The probability that some minimal cut set containing `e` has occurred, over `Q` (`method="rare_event"`: their probabilities summed, which can pass 1). |
| `risk_achievement_worth(t)` | `Q(e occurred) / Q`. |
| `risk_reduction_worth(t)` | `Q / Q(e did not occur)`. |
| `structural_importance()` | The fraction of the other events' states in which `e` is pivotal: the Birnbaum importance with every event at 1/2, from the gates alone. |
| `differential_importance(t)` | Each event's share of a change in `Q` when every event's probability moves ([DIM](importance.md#shares-of-a-change-differential-importance)). |
| `joint_importance(t)` | Whether preventing two events together is worth more than preventing each ([second order](importance.md#pairs-joint-importance)). |
| `barlow_proschan_importance(t)` | The probability that the event's occurring is what brings the top event about, by `t` or ever: `I_B(e)` times the event's density, integrated, over the sum. |

```python
cooling.criticality_importance()
# {'pump 1': 0.1597, 'pump 2': 0.1597, 'valve': 0.8319}
cooling.risk_achievement_worth()["valve"]   # -> 16.81   top event certain once the valve fails
cooling.improvement_potential()
# {'pump 1': 0.0095, 'pump 2': 0.0095, 'valve': 0.0495}
cooling.structural_importance()             # the valve is pivotal unless both pumps have failed
# {'pump 1': 0.25, 'pump 2': 0.25, 'valve': 0.75}
timed.barlow_proschan_importance()          # which event brings cooling down, over its life
# {'pump 1': 0.3625, 'pump 2': 0.3625, 'valve': 0.2750}
```

They are the same measures, with the same definitions, as the
[importance measures](importance.md) of a `NonRepairableRBD`: a tree and the
diagram of the same system give the same values. The Barlow–Proschan
importance needs the events' models (an event given as a probability has
no density, so no share), and is integrated numerically, to about `1e-9`.

## Over a lifetime

With lifetime models the top event has a time, and the tree gives what a
diagram gives of its lifetime (see [Reliability of a
system](reliability.md)), worked out on the tree's own evaluation: the mean
time to the top event, the area under `sf` (to about `1e-10`); the time at
which `sf` falls to a reliability, or by which the top event has occurred
in `x` percent of systems, by root-finding; and the density, hazard and
cumulative hazard, the first two by differences of the exact probability.

```python
timed.mean()                        # -> 8014.98   the mean time to loss of cooling
timed.bx_life(10)                   # -> 3360.3    10% of systems have lost cooling by then
timed.time_to_reliability(0.5)      # -> 7676.3    half have
timed.hf(5000)                      # -> 0.000117  the hazard rate at 5000 h
timed.Hf(5000)                      # -> 0.2548    the cumulative hazard, −ln sf
```

A tree of probabilities alone has no time to its top event: `mean`,
`bx_life` and `time_to_reliability` refuse it, saying so. A common-cause
group that splits a probability is refused by `mean` too, as a diagram
refuses it (its model assumes a small probability, while over a lifetime it
runs to 1): split the failure rate instead (`basis="rate"`).

### Sensitivities and levers

`parameter_sensitivity(t)` gives how far `sf(t)` moves with each parameter
of each event's model, the event's Birnbaum importance times its model's
parameter derivative (an event given as a probability has the one lever
`p`; a common-cause group's parameters, and its model's `ccf_beta`, ...,
are reported once under the tuple of its members, by differences of the
exact probability). `levers()` lists the parameters, with their values and
ranges, and `with_levers` builds the tree with some moved, as a diagram's
do ([Sensitivities: the Greeks](greeks.md)):

```python
timed.parameter_sensitivity(5000)["valve"]
# {'alpha': 7.27e-06, 'beta': 0.1343}
[(lever.key, lever.name) for lever in timed.levers()][:2]
# [('pump 1', 'alpha'), ('pump 1', 'beta')]
timed.with_levers({("valve", "alpha"): 40000.0}).mean()   # -> 8850.3   a valve lasting twice as long
```

### Uncertainty from fitted event models

`sf_uncertainty` and `mean_uncertainty` draw plausible models for the
uncertain events, from their surpyval fits' covariance by default, or from
distributions over named parameters, or a list of models, and work the
top event probability, or the mean time to it, out exactly for each draw,
as a diagram's [`sf_uncertainty`](reliability.md#uncertainty-in-the-component-models)
does; the draws are those of the tree's diagram (`to_rbd()`, converted
once), so events of one population go together in a tuple, and a group's
members together. Here the pumps' Weibull scale is known only to lie
between 7000 and 9000 h:

```python
import scipy.stats as st

scale = {"alpha": st.uniform(7000, 2000)}
spread = timed.sf_uncertainty(5000, {("pump 1", "pump 2"): scale}, n_draws=2000, seed=1)
spread.nominal                      # -> 0.7751
spread.interval(0.9)                # (0.733, 0.804)
```

`analysis_routes()` says, without running anything, how each of the tree's
analyses is computed, as a diagram's does: exact, numerical, simulated, or
refused and why.

```python
timed.analysis_routes()["mean"].route     # 'numerical'
cooling.analysis_routes()["mean"].route   # 'refused'   no event has a lifetime
```

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
models as RePyability saves a diagram's node models, and the common-cause
groups as a diagram saves them. A tree built by `with_levers` saves the
same way: an event given as a probability is still one.

## NOT gates: non-coherent trees

A NOT gate occurs while its input has not occurred. With one, an event
occurring can make the top event *less* likely, and the tree is
*non-coherent*. Here an alarm is missed if the detector has failed, or if
a test is running while the bypass that should cover it is not in place:

```python
alarm = FaultTree(
    {
        "missed": ("or", ["detector", "masked"]),
        "masked": ("and", ["test running", "not bypassed"]),
        "not bypassed": ("not", ["bypass"]),
    },
    {"detector": 0.01, "test running": 0.05, "bypass": 0.2},
)
alarm.is_coherent                   # False
alarm.top_event_probability()       # -> 0.0496   0.01 + 0.99 × 0.05 × 0.8
alarm.birnbaum_importance()["bypass"]   # -> -0.0495   the bypass helps
[(sorted(i.occurred), sorted(i.not_occurred)) for i in alarm.prime_implicants()]
# [(['detector'], []), (['test running'], ['bypass'])]
```

- **Exact.** The tree is evaluated through its binary decision diagram,
  built from the gates with the NOT gates' inputs complemented, so the top
  event probability (and `sf`, worked out in its own right) is exact,
  repeated events included, with probabilities or lifetime models.
- **Prime implicants, not cut sets.** A non-coherent tree has no minimal
  cut sets: an event *not* occurring can be what makes the top event occur.
  `prime_implicants()` lists the smallest combinations of events occurring
  and not occurring that make it occur, each a
  [`PrimeImplicant`][repyability.PrimeImplicant], every one of them,
  including those the gates do not spell out (for `a b + a' c`, also
  `b c`). For a coherent tree they are the minimal cut sets.
- **Measures that still apply.** `birnbaum_importance` and
  `conditional_probabilities` are the top event conditioned on each
  event, which a non-coherent tree has; the Birnbaum importance of an
  event under a NOT gate can be negative. The other importance measures,
  the cut sets, the lifetime analyses (the top event can stop occurring,
  so it has no time to occur), the sensitivities and the uncertainty
  analyses, and the conversion to a block diagram, assume a coherent tree:
  each refuses, saying so, and `analysis_routes()` says which.
- Common-cause groups are not taken with NOT gates.

## Dynamic gates

A dynamic gate's output depends on the order in which things fail, so a
tree with one is worked out over time. A standby pump is a **SPARE** gate:
the duty pump is its primary (the first input), and the standby is
switched in when it fails. A standby that cannot fail while waiting (a
*cold* spare) makes loss of flow much less likely than two pumps running
together:

```python
pump = surv.Weibull.from_params([8000, 1.8])
standby = FaultTree(
    {"no flow": ("spare", ["duty", "standby"])},
    {"duty": pump, "standby": pump},
)
standby.ff(5000)                    # -> 0.0321   against 0.1217 for an AND
```

| Gate | Occurs when | Worked out exactly as |
|---|---|---|
| PAND `("pand", inputs)` | all inputs have occurred, in order, left to right (inputs that occur at one instant are in order) | `G_k(t) = ∫ G_{k−1}(s) dF_k(s)` over `[0, t]`, `G_1 = F_1` |
| SPARE `("spare", dormancy, inputs)` | the primary and every spare have failed; a spare waiting in reserve ages at the dormancy factor (0 cold, the default; 1 hot; between them warm), and one found failed when needed is passed over | the measure of each unit's failure time carried to the next: switched in at `s`, a spare has age `a s` and fails by `t` with `F(a s + t − s)` (for one cold spare, the convolution `∫ F_2(t − s) dF_1(s)`) |
| SEQ `("seq", inputs)` | all inputs have occurred; each starts ageing when the one before it occurs | the cold SPARE of its inputs: the sum of their lives |
| FDEP `("fdep", trigger, dependents)` | (no output) the trigger occurring makes each dependent occur at once | each dependent replaced by an OR of itself and the trigger: the static tree it is equal to |

A SPARE's spares and a SEQ's inputs are basic events (a SPARE's primary
may be a gate); an event may be a spare or a member of a sequence in one
gate only. A **warm** spare ages at `dormancy` units of age per unit of
time in reserve, and from its age then once at work:

```python
warm = FaultTree(
    {"no flow": ("spare", 0.2, ["duty", "standby"])},
    {"duty": pump, "standby": pump},
)
warm.ff(5000)                       # -> 0.0448
```

**PAND**: a guard that fails before the motor overheats lets the hazard
through; one that fails after it has already tripped the motor:

```python
plant = FaultTree(
    {
        "loss": ("or", ["hazard", "no flow"]),
        "hazard": ("pand", ["guard", "overheat"]),
        "no flow": ("spare", 0.2, ["duty", "standby"]),
    },
    {
        "guard": surv.Exponential.from_params([1 / 20000]),
        "overheat": surv.Weibull.from_params([12000, 2.5]),
        "duty": pump,
        "standby": pump,
    },
)
plant.ff(np.array([2000, 5000, 10000]))
# array([0.002755, 0.06105, 0.4108])
plant.analysis_routes()["ff"].route        # 'numerical'
```

**How it is worked out.** Each dynamic gate that is an independent module
(nothing below it is shared with the rest of the tree, and its inputs
share nothing with each other) is integrated over its inputs'
distributions, as in the table: on a grid fine at every scale below `t`,
halved until two Richardson-extrapolated passes, and the spline between
their points, agree (to about `1e-9` for smooth lives, `1e-7` or so for a
spare whose life has an unbounded density at 0, a Weibull shape below 1;
a warning says if a grid reached its limit first). Its inputs'
distributions are the events' models or an independent gate's own; the
static gates around the dynamic ones are exact, by the tree's own engine,
the dynamic gates its events.

A dynamic gate that shares events with the rest of the tree (here the duty
pump also feeds the PAND) is not a module, and cannot be integrated: the
exact method refuses, and `method="simulate"` simulates the smallest
independent part of the tree holding it, `mc_samples` times (seeded with
`seed`), keeping the rest exact:

```python
shared = FaultTree(
    {
        "loss": ("or", ["hazard", "no flow"]),
        "hazard": ("pand", ["guard", "duty"]),
        "no flow": ("spare", 0.2, ["duty", "standby"]),
    },
    {"guard": surv.Exponential.from_params([1 / 20000]), "duty": pump, "standby": pump},
)
shared.analysis_routes()["ff"].route       # 'refused'
shared.ff(5000, method="simulate", mc_samples=100_000, seed=1)   # ~> 0.0887
```

`random(size, seed)` simulates the whole tree's times to the top event:
each event's life drawn from its model (by its quantile function), the
spares aged at their dormancy, the sequences and priorities followed. Each
event has a random stream of its own, keyed by the seed and its place
among the events, so a part simulated alone draws what the whole tree
draws. A **functional dependency** is written out before any of this, so
a tree whose only dynamic gates are FDEPs is a static tree, with every
static analysis:

```python
powered = FaultTree(
    {
        "no flow": ("and", ["pump 1", "pump 2"]),
        "power": ("fdep", "supply", ["pump 1", "pump 2"]),
    },
    {"pump 1": pump, "pump 2": pump, "supply": surv.Exponential.from_params([1 / 50000])},
)
powered.ff(5000)                    # -> 0.2053
powered.minimal_cut_sets()
# [frozenset({'supply'}), frozenset({'pump 1', 'pump 2'})]
```

A tree with PAND, SPARE or SEQ gates gives the probability of its top event
over time (`top_event_probability`, `ff`, `sf`) and simulated times to it
(`random`); its other analyses (cut sequences, importance, the mean time to
the top event, the conversion to a block diagram) are not available yet,
and refuse, saying so. NOT gates and dynamic gates in one tree, and
common-cause groups with dynamic gates, are refused. For repairable
standby and load sharing, see [Redundancy models](redundancy-models.md).

## Exporting to Open-PSA

`to_open_psa()` writes the tree as an Open-PSA Model Exchange Format (MEF)
document, the XML that PRA tools read: the OR, AND, VOTE (`atleast`) and
NOT gates, an FDEP gate as its static equivalent, events given as
probabilities, exponential and Weibull lives as expressions of the mission
time, and common-cause groups of the beta-factor model splitting a
probability. A name that is not an MEF identifier is made one, the name as
given kept as its label. `to_open_psa(t)` writes every event's probability
at `t` instead, for models the format has no expression for. The format
has no dynamic gates, so a tree with PAND, SPARE or SEQ gates is refused:

```python
print(cooling.to_open_psa().splitlines()[3])
# <define-gate name="no_cooling"><label>no cooling</label><or><gate name="no_flow"/><basic-event name="valve"/></or></define-gate>
```

## The model and its limits

- **Gates.** OR, AND, VOTE and NOT, exactly; PAND, SPARE and SEQ over
  time, integrated where each is an independent module and simulated
  otherwise; FDEP as its static equivalent. Shared spares, and a SPARE's
  spare or a SEQ's member forced by an FDEP, are not supported; for
  standby and load sharing with repair, see
  [Redundancy models](redundancy-models.md).
- **Independent basic events,** but for the events they share and their
  [common causes](#common-causes).
- **Non-repairable.** A tree gives the probability that the top event has
  occurred by `t`. For availability with repair, build a
  [`RepairableRBD`](repairable.md).
