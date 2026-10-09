# Importance measures

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson
    4](../learn/importance.md), which builds every measure on this page from
    one idea. [Sensitivities: the Greeks](greeks.md) reads the sensitivity
    measures as one family and runs one system through all of them.

An importance measure ranks the nodes of a system by how much they matter.
"Matter" has several meanings, and the measures disagree on purpose;
[Concepts](../concepts.md#importance-measures-which-one-and-why) explains when
to reach for each. This page shows how to compute them. The examples use the
system from [Reliability of a system](reliability.md):

```python
import surpyval as surv
from repyability import NonRepairableRBD

rbd = NonRepairableRBD(
    [("s", "pump1"), ("s", "pump2"),
     ("pump1", "valve"), ("pump2", "valve"),
     ("valve", "t")],
    {
        "pump1": surv.Weibull.from_params([100, 2]),
        "pump2": surv.Weibull.from_params([100, 2]),
        "valve": surv.Weibull.from_params([200, 1.5]),
    },
)
```

## The probability-based measures

Each takes a time (or array of times) and returns a dict of node → value.

```python
rbd.birnbaum_importance(50)      # {'pump1': 0.1952, 'pump2': 0.1952, 'valve': 0.9511}
rbd.improvement_potential(50)    # {'pump1': 0.0432, 'pump2': 0.0432, 'valve': 0.1118}
rbd.risk_achievement_worth(50)   # {'pump1': 1.9461, 'pump2': 1.9461, 'valve': 6.2234}
rbd.risk_reduction_worth(50)     # {'pump1': 1.3675, 'pump2': 1.3675, 'valve': 3.284}
rbd.criticality_importance(50)   # {'pump1': 0.2687, 'pump2': 0.2687, 'valve': 0.6955}
rbd.fussell_vesely(50)           # {'pump1': 0.3045, 'pump2': 0.3045, 'valve': 0.7313}
rbd.birnbaum_importance(50)["valve"]   # -> 0.9511
```

With `R` the system reliability, `Q = 1 − R` its unreliability, `R_i` node
*i*'s reliability, and `R(1_i)` / `R(0_i)` the system reliability with node
*i* forced working / failed:

| Method | Definition |
|---|---|
| `birnbaum_importance` | `R(1_i) − R(0_i)`: how much the system reliability moves per unit change in `R_i` (`∂R/∂R_i`). |
| `improvement_potential` | `R(1_i) − R`: the reliability gained by making node *i* perfect. |
| `risk_achievement_worth` | `Q(0_i) / Q`: how many times more likely the system is to fail if node *i* has failed. |
| `risk_reduction_worth` | `Q / Q(1_i)`: by what factor perfecting node *i* would divide the system unreliability. |
| `criticality_importance` | `birnbaum_i · (1 − R_i) / Q`: the probability that node *i* has failed and is critical, given that the system has failed (the failure-oriented form; see below). |
| `fussell_vesely` | The probability that some minimal cut set containing *i* has failed (every member), divided by `Q`: the share of the system's unreliability that involves node *i*, between 0 and 1. |

`fussell_vesely` is exact: the union of the cut sets' failures is worked out
by the exact engine, not summed. Many PRA tools report the rare-event form,
the sum over the cut sets containing *i* of their probabilities, divided by
`Q`; `method="rare_event"` gives it, for comparison. The two agree while
failures are rare, but the sum over-estimates the union, and once the system
is likely to have failed it passes 1: on a bridge whose system is nearly
certain to have failed, it nears 2 while the exact share stays at most 1.

`fussell_vesely(t, fv_type="p")` substitutes the minimal *path* sets: the
probability that all the members of some path set containing *i* have
failed (or, with `method="rare_event"`, the sum over those path sets),
divided by `Q`. It is not bounded by 1. `fv_type="c"` (cut sets) is the
default and the standard measure.

### Failure- or success-oriented criticality

`criticality_importance` returns the failure-oriented form by default
(Rausand & Høyland): each node's share of the system failures. By t = 50 the
valve accounts for 70% of them, and each pump for 27% (the failures in which
both pumps are down and the valve works). The shares need not add to 1: both
pumps are critical in the same failures, and a failure of all three has no
single critical node.

`kind="success"` gives the success-oriented form, `birnbaum_i · R_i / R`: the
probability that node *i* is working and critical, given that the system
works. It is exactly 1 for every node in series with the rest of the system,
however unreliable, so it cannot rank the nodes in series:

```python
rbd.criticality_importance(50)["valve"]                  # -> 0.6955
rbd.criticality_importance(50, kind="success")["valve"]  # -> 1.0
```

The failure-oriented form is computed from the node unreliabilities through
the minimal cut sets, so it keeps its precision for a highly reliable
system, where `1 − R` would cancel. It is `nan` where the system cannot fail
(at t = 0, or with enough nodes held working), and the success-oriented form
is `nan` where the system cannot work.

An array of times gives arrays:

```python
rbd.birnbaum_importance([25, 50])
# {'pump1': array([0.058, 0.195]), 'pump2': array([0.058, 0.195]),
#  'valve': array([0.996, 0.951])}
```

### Conditioning on known states

Every measure accepts `working_nodes` and `broken_nodes`, which pin the other
nodes before the measure is computed:

```python
rbd.birnbaum_importance(50, broken_nodes=["pump2"])["pump1"]   # -> 0.8825
```

With pump2 down, pump1 is now in series with the valve, and its Birnbaum
importance rises from 0.195 to the valve's reliability, 0.883.

## Structural importance

`structural_importance()` is the Birnbaum importance with every node at
reliability ½: the fraction of the states of the other nodes in which a node
is pivotal. It depends only on the diagram, so it ranks where redundancy
matters before any life data exists.

```python
rbd.structural_importance()   # {'pump1': 0.25, 'pump2': 0.25, 'valve': 0.75}
```

It is the same for a `NonRepairableRBD` and a `RepairableRBD` on the same
diagram, takes no time argument, accepts `working_nodes`/`broken_nodes`, and
works on RBDs with common-cause groups. It also drives the structural
reliability allocation, `simple_allocation` (see
[Design and allocation](design.md#smallest-log-odds-change-a-structural-allocation)).

## Parameter sensitivity

`parameter_sensitivity(t)` gives, for each node with a parametric
distribution, the derivative of the system reliability with respect to each
of its parameters: `∂R/∂θ = birnbaum_i · ∂R_i/∂θ`.

```python
rbd.parameter_sensitivity(50)["valve"]   # {'alpha': 0.000787, 'beta': 0.145443}
rbd.parameter_sensitivity(50)["valve"]["beta"]   # -> 0.1454
```

The system here is far more sensitive to the valve's shape `β` than to its
scale `α` (per unit of each), so extra valve data should go to pinning down
the shape.

- The parameter derivative is a finite difference (relative step `rel_step`,
  default `1e-5`) that rebuilds the distribution with `from_params`, so it
  works for any surpyval parametric distribution.
- Composite nodes (nested RBDs, standby, repeated and load-sharing nodes)
  have no parameters and are left out.
- A node pinned by `working_nodes`/`broken_nodes` reports zeros.

## On a repairable system

A [`RepairableRBD`][repyability.RepairableRBD] has the same measures, with no
time argument: they are evaluated at the nodes' long-run availabilities
instead of reliabilities at a time.

```python
from repyability import RepairableRBD

def unit(rate):
    return {
        "reliability": surv.Exponential.from_params([rate]),
        "repairability": surv.Exponential.from_params([1.0]),
    }

plant = RepairableRBD(
    [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")],
    {"A": unit(0.1), "B": unit(0.1), "C": unit(0.02)},
)
plant.birnbaum_importance()      # {'A': 0.0891, 'B': 0.0891, 'C': 0.9917}
plant.risk_achievement_worth()   # {'A': 3.924, 'B': 3.924, 'C': 36.0877}
plant.risk_achievement_worth()["C"]   # -> 36.09
plant.criticality_importance()   # {'A': 0.2924, 'B': 0.2924, 'C': 0.7018}
plant.criticality_importance()["C"]   # -> 0.7018
```

`plant.node_availability()` gives the availabilities used. On a repairable
system the failure-oriented criticality is each node's share of the system's
downtime: C, in series, causes 70% of it. The simulation
also produces time-weighted criticality measures from the simulated histories;
see [Repairable systems](repairable.md#criticality-measures).

### Over time

The long-run measures describe the system once it has settled. From new,
or from the components' states now, the ranking can differ. Give `x`
(times from new) to evaluate a measure at the nodes' point availabilities
then (see `point_availability`), or `window` (a window's length) to
evaluate it over `[0, window)`; `state=` starts the components from their
current states, as for `point_availability`:

```python
critical = plant.criticality_importance(x=[0.5, 2.0, 20.0])["C"]
critical[0]   # -> 0.841   at 0.5: early on the pair rarely fails together
critical[2]   # -> 0.7018   settled: the long-run value
plant.criticality_importance(window=10.0)["C"]   # -> 0.7109
```

C causes 84% of the system's chance of being down at 0.5, against 70% in
the long run. Over a window, a ratio measure is the ratio of the system's
means over it, as `mission_availability` is its mean availability: C's
criticality over the first 10 time units is its share of the window's
expected downtime, not the mean of its shares at each time.

From a state: with A down now, in repair, B matters more for a while:

```python
from repyability import NodeState

down = {"A": NodeState(alive=False)}
plant.birnbaum_importance(x=1.0)["B"]               # -> 0.0599
plant.birnbaum_importance(x=1.0, state=down)["B"]   # -> 0.3886
```

With limited repair crews, the measures follow the crews' chain over time:
the Birnbaum importance, improvement potential and risk worths hold each
node working and failed in it, as in the long run, and the criticality and
Fussell–Vesely measures average over its states at each time (not yet
around nested RBDs). With common-cause groups, each time's point is split
by the groups' joint states then. A component whose curve over time is not
worked out (one repaired imperfectly, say) is refused, as by
`point_availability`.

### Levers of a repairable system

On a `RepairableRBD`, `parameter_sensitivity()` gives the derivative of the
long-run availability in each lever: each component's life and repair
models' parameters (`"reliability.<name>"`, `"repairability.<name>"`), its
preventive maintenance and tests (`"preventive.interval"`,
`"inspection.coverage"`, ...), a standby group's, a common-cause group's
(under its members together), and the change one more standby unit or
repair crew makes:

```python
levers = plant.parameter_sensitivity()
levers["C"]["reliability.failure_rate"]     # -> -0.9532
levers["C"]["repairability.failure_rate"]   # -> 0.01906
levers["A"]["repairability.failure_rate"]   # -> 0.00737
```

Per unit of repair rate, C's repairs are worth 2.6 times A's. Given what a
unit change of each costs, `unit_costs` ranks them by availability per unit
spent instead:

```python
costs = {
    ("C", "repairability.failure_rate"): 1000.0,
    ("A", "repairability.failure_rate"): 200.0,
}
ranked = plant.parameter_sensitivity(unit_costs=costs)
1e5 * ranked["A"]["repairability.failure_rate"]   # -> 3.683
1e5 * ranked["C"]["repairability.failure_rate"]   # -> 1.906
```

Faster repairs of A buy about twice the availability per unit spent. With
limited repair crews, the key None holds one more crew's gain:

```python
crewed = RepairableRBD(
    [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")],
    {"A": unit(0.1), "B": unit(0.1), "C": unit(0.02)},
    repair_crews=1,
)
crewed.parameter_sensitivity()[None]["repair_crews"]   # -> 0.0116
```

`x` (times) or `window` gives the sensitivity of the availability over
time, and `of="cost_rate"` (or both, as a tuple) the cost rate's. Each
continuous lever is a central difference of the system's own value with the
diagram rebuilt; with independent components, over time, a component's is
its Birnbaum importance times its own curve's difference, as exact and much
faster. In the long run, an interval of a component whose block
replacements or tests share a calendar with others' would move it off their
common calendar, where the long-run value jumps: its derivative takes its
schedule apart from theirs.

## Pairs: joint importance

The measures above rank improvements one at a time. Whether improving two
together is worth more than the sum of improving each is the joint
(second-order) importance, `JRI(i, j) = ∂²R/∂R_i ∂R_j = R(1_i, 1_j) −
R(1_i, 0_j) − R(0_i, 1_j) + R(0_i, 0_j)`. It is how much node *j*'s
Birnbaum importance rises when node *i* goes from failed to working:

```python
joint = rbd.joint_importance(50)
joint[("pump1", "pump2")]   # -> -0.8825
joint[("pump1", "valve")]   # -> 0.2212
```

Positive, the two are complements, as a pump and the valve in series are:
a better pump makes a better valve worth more, so they belong in one
campaign. Negative, they are substitutes, as the two pumps in parallel are:
either one does the other's job, so improving both buys less than the sum
of improving each. The structure being multilinear, the measure is exact
and cheap: each node's Birnbaum importance with the other held working,
less with it held failed. The result holds each pair once, its names in
order as text, and finds a pair either way round (`joint[("valve",
"pump1")]` too).

A `RepairableRBD` has it in the long run, or with `x`, `window` and
`state=` as its other measures take them; with limited repair crews, each
pair is held in the crews' chain:

```python
plant.joint_importance()[("A", "B")]                # -> -0.9804
plant.joint_importance(x=[0.5, 20.0])[("A", "C")]   # array([0.0385, 0.0909])
```

A `FaultTree` has it too, as `−∂²P/∂q_e ∂q_f` in its events'
probabilities, which is the same as its diagram's (`to_rbd()`): positive
under an OR gate (complements), negative under an AND gate (substitutes).
With common-cause groups, whose members cannot be held, it is refused.

## Shares of a change: differential importance

The measures above rank the nodes, but they do not add up: the two pumps'
Birnbaum importances, summed, are not the pumps' importance together.
`differential_importance` (DIM; Borgonovo & Apostolakis, 2001) gives each
node's share of the change in the system when every node changes together,
so the shares add up to 1, and a group's share (`groups`) is the sum of its
members':

```python
rbd.differential_importance(50)
# {'pump1': 0.1455, 'pump2': 0.1455, 'valve': 0.709}
both = {"pumps": ["pump1", "pump2"]}
rbd.differential_importance(50, groups=both)["pumps"]   # -> 0.291
rbd.differential_importance(50, change="proportional", groups=both)["pumps"]   # -> 0.4359
```

What changes together matters. `change="uniform"` (the default) moves every
node's probability of failing by as much, and shares out the Birnbaum
importance; `change="proportional"` moves each by the same fraction of
itself, and shares out the criticality importance (`kind="success"` moves
the probabilities of working in proportion instead, and shares out the
success-oriented form). By t = 50 the pumps are likelier than the valve to
have failed, so a fraction off each moves them more: they hold 44% of a
proportional change, against 29% of a uniform one.

`over="parameters"` shares out `parameter_sensitivity`'s derivatives
instead, keyed `(node, parameter)`. A uniform change adds the same amount
to parameters of different units (a scale in hours, a shape with none), so
over parameters the proportional change is the one to ask for:

```python
by_kind = {
    "scales": [("pump1", "alpha"), ("pump2", "alpha"), ("valve", "alpha")],
    "shapes": [("pump1", "beta"), ("pump2", "beta"), ("valve", "beta")],
}
rbd.differential_importance(
    50, over="parameters", change="proportional", groups=by_kind
)["shapes"]   # -> 0.5112
```

Half of the change from moving every parameter by the same fraction lies in
the shapes.

A `RepairableRBD` shares out its availability's change, in the long run, or
with `x`, `window` and `state` as its other measures take them; over its
levers, each lever's (one more standby unit or repair crew is no
derivative, and takes no part). Levers can pull against each other: a
failure rate lowers the availability, a repair rate raises it. Moved
together in proportion, an exponential unit's two rates cancel, since its
availability depends on their ratio alone, and there is no change to share
out: the shares are NaN. `improving=True` moves each lever the way that
raises the availability instead, so that every share is of a gain: what
share of the gain from improving every lever by the same fraction lies in
the lives, the repairs, or the maintenance:

```python
def wearing(alpha, beta, **more):
    return {
        "reliability": surv.Weibull.from_params([alpha, beta]),
        "repairability": surv.Exponential.from_params([1.0]),
        **more,
    }

maintained = RepairableRBD(
    [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")],
    {
        "A": wearing(10, 2.0),
        "B": wearing(10, 2.0),
        "C": wearing(
            50,
            3.0,
            preventive={
                "policy": "age",
                "interval": 20.0,
                "duration": surv.Exponential.from_params([4.0]),
            },
        ),
    },
)
kinds = {
    "lives": [(n, f"reliability.{p}") for n in "ABC" for p in ("alpha", "beta")],
    "repairs": [(n, "repairability.failure_rate") for n in "ABC"],
    "maintenance": [
        ("C", "preventive.interval"),
        ("C", "preventive.duration.failure_rate"),
    ],
}
gain = maintained.differential_importance(
    over="parameters", change="proportional", improving=True, groups=kinds
)
gain["lives"]         # -> 0.4502
gain["repairs"]       # -> 0.2928
gain["maintenance"]   # -> 0.2571
```

A `FaultTree` shares out its top event's change over its basic events, in
the same way (`change` and `groups`).

## What is moving the system: rates and Barlow–Proschan

The measures above say how much each component matters. Two more say which
component is moving the system *now*, or caused its failures *so far*.

### The rate of change, split by component

With independent components the system's reliability is multilinear in
theirs, so its rate of change is the sum of each component's Birnbaum
importance times its own rate, `dR/dt = Σ I_B^i dR_i/dt`. Each term is what
that component is doing to the system then, and the terms add up.
`reliability_rate(x)` gives the system's rate, which is `-df(x)`, and each
node's part:

```python
rate = rbd.reliability_rate(50)
rate.node_rate["valve"]   # -> -0.003147
rate.node_rate["pump1"]   # -> -0.00152
rate.rate                 # -> -0.006188
```

At 50 the valve is bringing the system down about as fast as the two pumps
together. On a `RepairableRBD`, `availability_rate(x)` splits the rate of
change of the point availability (from new, or from `state=`) the same way:

```python
rate = plant.availability_rate([0.5, 2.0])
rate.node_rate["C"][0]   # -> -0.01199
rate.node_rate["A"][0]   # -> -0.0022
```

Early on, C pulls the system down five times as fast as A, since A and B
back each other up. A component's rate is that of its point availability,
by differences on the grid it is solved on, numerical to about `1e-5` of
the rates; the down times kept off the grid (a repair or maintenance that
starts at a known time, however short) are differentiated on their own
scale, so the rate just after a scheduled event is as close.

Where a scheduled event makes an availability jump (a block replacement or
test that takes the component off line), the system's jumps are reported
apart, in `jump_times`, `jumps` and `node_jumps`. Components that jump
together share each jump along the straight path between their values
before and after, so the parts add up to it:

```python
serviced = RepairableRBD(
    [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")],
    {
        "A": wearing(10, 2.0),
        "B": wearing(10, 2.0),
        "C": wearing(
            50,
            3.0,
            preventive={
                "policy": "block",
                "interval": 20.0,
                "duration": surv.Exponential.from_params([4.0]),
            },
        ),
    },
)
rate = serviced.availability_rate(21.0)
rate.jump_times[0]         # -> 20.0
rate.node_jumps["C"][0]    # -> -0.9816
rate.node_rate["C"]        # -> 0.07491
```

At 20, C's block replacement takes the system down; at 21, C coming back
from it is what raises the availability.

### Which component caused the failures: Barlow–Proschan

Integrated over time, the parts give the Barlow–Proschan importance: the
probability that the system's failure is caused by the component's, the one
whose failure finds the system up and leaves it down.
`barlow_proschan_importance()` gives each component's share over the whole
life, and `barlow_proschan_importance(x)` its share of the failures by `x`:

```python
rbd.barlow_proschan_importance()["valve"]     # -> 0.3474
rbd.barlow_proschan_importance(50)["valve"]   # -> 0.7212
```

Over its whole life the valve causes a third of the system's failures, but
72% of those by 50: early failures are the valve's, while the pumps cause
one only once both have failed.

On a `RepairableRBD` it is each component's share of the system's failures
in the long run (its terms of `system_failure_frequency`), or over a window
from new or from `state=` (its terms of `expected_failures`). This is the
exact counterpart of the simulated
`failure_criticality_index.per_system_failure`:

```python
plant.barlow_proschan_importance()["C"]               # -> 0.5455
plant.barlow_proschan_importance(window=10.0)["C"]    # -> 0.5682
```

With common-cause groups, a cause that strikes several members at once is
counted for the group, under the tuple of its members.

With limited repair crews or common-cause groups the components do not fail
and recover independently, and the rates and shares come from the crews' or
the groups' Markov chains instead (#199). Each of the chain's transitions is
one component's failure or repair, or a cause's strike, and its part is
what those transitions do to the system: exact, from the same chain. With
one crew for the plant above:

```python
rate = crewed.availability_rate(2.0)
rate.node_rate["C"]                                    # -> -0.00349
crewed.barlow_proschan_importance(window=10.0)["C"]    # -> 0.5696
```

C pulls the availability down faster than with a crew each (-0.00258 at
2), as its repair can wait for a crew busy with A or B. A crew that finishes
a repair and takes the next job counts as the repair that freed it. In a
common-cause group, a member's own cause and its repairs are its part, and
the causes that strike more than one member the group's. A hidden group's
tests can find several members' failures at once, so a jump with a test in
it is split by the Shapley value of what changes then, worked out exactly.

## Limits

- A perfect junction node (`PerfectReliability`, such as the vote of a
  k-out-of-n arrangement) is a drawing device that never fails and cannot
  be improved: every importance measure leaves it out, the structural
  importance takes it as always working, and the allocations hold it at 1
  and leave it out of their results.
- With common-cause groups, a member's measures are conditioned on its state
  through the shared causes, and `parameter_sensitivity` reports a group's
  parameters (and its model's `ccf_beta`, ...) once, under the tuple of its
  members (see [Common-cause
  failures](common-cause.md#importance-sensitivity-uncertainty-and-allocation)).
  A member cannot be held working or broken.
- Otherwise the measures assume the nodes fail independently, as the exact
  engine does.
