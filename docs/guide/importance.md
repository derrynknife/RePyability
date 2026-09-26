# Importance measures

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
rbd.criticality_importance(50)   # {'pump1': 0.1811, 'pump2': 0.1811, 'valve': 1.0}
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
| `criticality_importance` | `birnbaum_i · R_i / R`: Birnbaum weighted by the node's reliability relative to the system's (the success-oriented form). |
| `fussell_vesely` | Sum over the minimal cut sets containing *i* of the probability that every member has failed, divided by `Q` (the usual rare-event form). |

`fussell_vesely(t, fv_type="p")` substitutes the minimal *path* sets into the
same formula: the sum, over the path sets containing *i*, of the probability
that all their members have failed, divided by `Q`. `fv_type="c"` (cut sets)
is the default and the standard measure. `fussel_vesely` (misspelled) is a
deprecated alias that warns.

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
works on RBDs with common-cause groups.

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
```

`plant.node_availability()` gives the availabilities used. The simulation
also produces time-weighted criticality measures from the simulated histories;
see [Repairable systems](repairable.md#criticality-measures).

## Limits

- The probability-based measures and `parameter_sensitivity` raise
  `NotImplementedError` on an RBD with common-cause groups (structural
  importance does not).
- All measures assume the nodes fail independently, as the exact engine does.
