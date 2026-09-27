# Common-cause failures

Redundancy only helps while the redundant units fail for *independent*
reasons. In practice they often share a cause: one manufacturing batch, one
power supply, one maintenance error applied to every unit. The exact engine
assumes independence, so on its own it **over-estimates** a redundant group.
A common-cause (CCF) group adds the shared failures back. Theory, including
the model's assumptions: [Concepts](../concepts.md#common-cause-failures).

## Declaring a group

Wrap the coupled nodes and a model in a [`CCFGroup`][repyability.CCFGroup]
and pass the groups with `ccf_groups=`:

```python
import surpyval as surv
from surpyval import FixedEventProbability
from repyability import BetaFactor, CCFGroup, MGL, NonRepairableRBD, PerfectReliability

# Two redundant pumps, each failing with probability 0.01 on demand
edges = [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")]
pumps = {
    "p1": FixedEventProbability.from_params(0.01),
    "p2": FixedEventProbability.from_params(0.01),
}
independent = NonRepairableRBD(edges, pumps)
coupled = NonRepairableRBD(
    edges, pumps, ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.1))]
)
independent.ff()   # -> 0.0001    both must fail independently: 0.01²
coupled.ff()       # -> 0.001081  ten times worse: the shared cause dominates
```

The system unreliability is ten times the independent estimate, because one
shared cause (probability `β · Q = 0.001`) now fails both pumps at once.

## The two models

[`BetaFactor(beta)`][repyability.BetaFactor]: a fraction `beta` of each
member's failure probability `Q` comes from a cause shared by the **whole**
group; the rest, `(1 − beta) Q`, is independent. `beta` is in `[0, 1]`: `0`
reproduces the independent result exactly and `1` makes the group no better
than one unit.

```python
NonRepairableRBD(edges, pumps, ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.0))]).ff()
# -> 0.0001
NonRepairableRBD(edges, pumps, ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(1.0))]).ff()
# -> 0.01
```

[`MGL(beta, gamma, ...)`][repyability.MGL]: the Multiple Greek Letter model
also describes *partial* common causes, which fail some but not all of a
larger group. `beta` is the probability that a failure is shared with at
least one other member, `gamma` the probability that a shared failure takes
at least three, and so on. A group of `m` members takes exactly `m − 1`
letters (`MGL(0.1, 0.3).group_size == 3`), and `MGL(beta)` on two members is
exactly `BetaFactor(beta)`.

Partial causes matter for *k*-out-of-*n* groups. For three units where two
must work, a cause that fails a pair is enough to fail the system:

```python
def two_of_three(model=None):
    return NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("s", "c"),
         ("a", "v"), ("b", "v"), ("c", "v"), ("v", "t")],
        {
            "a": FixedEventProbability.from_params(0.01),
            "b": FixedEventProbability.from_params(0.01),
            "c": FixedEventProbability.from_params(0.01),
            "v": PerfectReliability,
        },
        k={"v": 2},
        ccf_groups=[CCFGroup(["a", "b", "c"], model)] if model else None,
    )

two_of_three().ff()                  # -> 0.000298   independent
two_of_three(BetaFactor(0.1)).ff()   # -> 0.001241   every shared cause takes all three
two_of_three(MGL(0.1, 0.3)).ff()     # -> 0.001591   70% of shared causes take a pair
```

With the same `beta`, the MGL group is *worse* here: a cause that fails two
of the three is as fatal as one that fails all three, and MGL says most
shared causes are of that kind.

### What each model assigns

`decompose(members, Q)` shows how a model splits each member's failure
probability `Q` into an independent part and mutually exclusive *shocks*
(each a set of members failing together, with its probability):

```python
import numpy as np

q_independent, shocks = MGL(0.1, 0.3).decompose(["a", "b", "c"], np.array([0.01]))
q_independent        # array([0.009])
[(sorted(members), float(p[0])) for members, p in shocks]
# [(['a', 'b'], 0.00035), (['a', 'c'], 0.00035), (['b', 'c'], 0.00035),
#  (['a', 'b', 'c'], 0.0003)]
```

`required_group_size()` is the group size a model needs (`None` for
`BetaFactor`, which fits any group of two or more).

## Rules for groups

- **Symmetric.** Every member must carry an identical model (the standard CCF
  assumption); otherwise the constructor raises `ValueError`.
- **Disjoint.** A node can be in at most one group.
- Members must be component nodes of the RBD: not the input or output node,
  and not a [repeated component](building.md#one-component-in-several-places).
- An `MGL` model must match its group's size.

## Time-dependent members

With lifetime models, `Q` is each member's failure probability at the time
asked for:

```python
unit = surv.Weibull.from_params([1000, 2])
pair = NonRepairableRBD(edges, {"p1": unit, "p2": unit})
pair_ccf = NonRepairableRBD(
    edges, {"p1": unit, "p2": unit},
    ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.1))],
)
pair.ff(100)       # -> 9.9e-05
pair_ccf.ff(100)   # -> 0.001075
```

**Keep each member's `Q` small.** The models are the probabilistic-risk
assessment basic-event models: they split a failure *probability*, and they
are rare-event models. Use them over a mission or a proof-test interval, where
each member's `Q` stays small, not over a whole life. For a `β = 0.3` pair,
the result is within 3.5% of a rate-based treatment at `Q = 0.1`, but from
about `Q = 0.5` it comes out *more* reliable than an independent pair
([Concepts](../concepts.md#common-cause-failures) has the numbers).

## What honours a CCF group

| Method | With CCF groups |
|---|---|
| `sf`, `ff`, `reliability`, `unreliability`, and what is derived from them (`df`, `hf`, `Hf`, `cs`, `time_to_reliability`, `bx_life`) | Include the common cause, exactly. |
| `structural_importance` | Unaffected (it does not use probabilities). |
| `random`, `mean`, `mean_time_to_failure`, `mean_time_to_failure_interval` | Sample members **independently**: the common cause is not included, since a lifetime runs to `Q = 1`, outside the model. |
| The probability-based importance measures, `parameter_sensitivity`, and the condition-based methods | Raise `NotImplementedError`. |
| `working_nodes` / `broken_nodes` naming a group member | Raises `NotImplementedError`. |
| `allocate_redundancy` | Not supported (duplicating a member would have to extend its group). |

```python
pair_ccf.mean(seed=0)   # -> 1147.4   the same as without the group
pair.mean(seed=0)       # -> 1147.4
```

Groups are saved with the RBD. An *alpha-factor* model, a data-estimable
reparameterisation of MGL, is a planned extension.
