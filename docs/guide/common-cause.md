# Common-cause failures

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson
    5](../learn/dependence.md) (when redundancy disappoints).

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
than one unit. Over a lifetime, split the failure rate instead,
`BetaFactor(beta, basis="rate")` (see [Over a lifetime](#over-a-lifetime)).

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

**Keep each member's `Q` small.** By default the models split a failure
*probability*: they are the probabilistic-risk assessment basic-event models,
rare-event models for a mission or a proof-test interval, where each
member's `Q` stays small, not for a whole life. For a `β = 0.3` pair, the
result is within 3.5% of a rate-based treatment at `Q = 0.1`, but from about
`Q = 0.5` it comes out *more* reliable than an independent pair
([Concepts](../concepts.md#common-cause-failures) has the numbers). So the
diagram warns, once for each group, when a member's `Q` passes 0.1:

```python
pair_ccf.sf(1000)   # -> 0.6336   more reliable than the independent pair's 0.6004
# UserWarning: Common-cause group ['p1', 'p2'] (BetaFactor(beta=0.1)): its members'
# probability of failing reaches Q = 0.632 at the times evaluated, beyond the 0.1
# the probability split is meant for. [...] Over a lifetime, split the failure
# rate: BetaFactor(beta=0.1, basis='rate').
```

## Over a lifetime

`basis="rate"` splits each member's failure **rate** instead. The shared
cause is a shock that has not struck by `t` with probability `R(t)^β`, and
each member survives its own causes with `R(t)^(1 − β)`, `R` the members'
reliability:

- every member keeps its own life distribution, whatever it is: its
  reliability is `R(t)^β · R(t)^(1 − β) = R(t)`;
- the model holds over the whole life, so the system's reliability falls
  to 0 as its members' do, and the exact MTTF and the simulations include
  the group;
- while `Q` is small it agrees with the probability split to first order,
  so over a mission or a proof-test interval the two give about the same.

```python
pair_rate = NonRepairableRBD(
    edges, {"p1": unit, "p2": unit},
    ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.1, basis="rate"))],
)
pair_rate.ff(100)    # -> 0.00108   as the probability split's 0.001075
pair_rate.sf(1000)   # -> 0.5862    below the independent pair's 0.6004
pair_rate.sf(3000)   # -> 0.000247  where the probability split still gives 0.1712
pair_rate.mean()     # -> 1129.5    exact; the independent pair's is 1145.8
pair_rate.mean(method="simulate", seed=0)   # -> 1128.6   the shocks are drawn
pair_rate.bx_life(10)   # -> 581.9
```

`MGL(..., basis="rate")` gives each specific set of members a cause of its
own, which strikes independently of the others with the share of the
members' hazard that the probability split gives its probability (`Q_k / Q`
above). Several causes may have struck by a time, and a member has failed if
any of its causes has, so `decompose` lists the distinct sets the struck
causes fail between them, mutually exclusive as before:

```python
q_independent, shocks = MGL(0.1, 0.3, basis="rate").decompose(["a", "b", "c"], np.array([0.01]))
q_independent[0]     # -> 0.0090045   1 − 0.99^0.9, against 0.009 by probability
[(sorted(members), round(float(p[0]), 7)) for members, p in shocks]
# [(['a', 'b'], 0.0003513), (['a', 'c'], 0.0003513), (['b', 'c'], 0.0003513),
#  (['a', 'b', 'c'], 0.0003018)]
```

The simulations draw a rate-split group through its members' quantile
function: in the members' cumulative hazard `H = −log R`, each cause strikes
at an exponential time of rate its share, and a member fails at the first
of its causes to strike. Members whose model has no quantile function (one
that draws its own random numbers) cannot be drawn so, and the simulations
refuse the group; its `sf`, `ff` and `mean` stay exact.

## What honours a CCF group

| Method | With CCF groups |
|---|---|
| `sf`, `ff`, `reliability`, `unreliability`, and what is derived from them (`df`, `hf`, `Hf`, `cs`, `time_to_reliability`, `bx_life`) | Include the common cause, exactly. A probability split warns once its members' `Q` passes 0.1. |
| `structural_importance` | Unaffected (it does not use probabilities). |
| `mean`, `mean_time_to_failure` (exact by default) | Include a group splitting the rate. Raise `NotImplementedError` for a group splitting a probability: the exact MTTF integrates the reliability over whole lifetimes, where `Q` runs to 1. |
| `random`, `mean(method="simulate")`, `mean_time_to_failure_interval`, `compare`, `unreliability_interval` | Draw a group splitting the rate with its shared shocks. Sample the members of a group splitting a probability **independently**, leaving the common cause out (`unreliability_interval` refuses such a group). |
| The probability-based importance measures, `parameter_sensitivity`, and the condition-based methods | Raise `NotImplementedError`. |
| `working_nodes` / `broken_nodes` naming a group member | Raises `NotImplementedError`. |
| `allocate_redundancy` | Not supported (duplicating a member would have to extend its group). |

```python
pair_ccf.mean(method="simulate", seed=0)   # -> 1147.4   the same as without the group
pair.mean(method="simulate", seed=0)       # -> 1147.4
pair.mean()                                # -> 1145.8   exact, without the group
```

Groups are saved with the RBD, their basis with them. An *alpha-factor*
model, a data-estimable reparameterisation of MGL, is a planned extension.
