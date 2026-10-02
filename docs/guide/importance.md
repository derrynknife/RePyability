# Importance measures

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson
    4](../learn/importance.md), which builds every measure on this page from
    one idea.

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
default and the standard measure. `fussel_vesely` (misspelled) is a
deprecated alias that warns.

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
  and non-parametric fits have no parameters and are left out.
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
