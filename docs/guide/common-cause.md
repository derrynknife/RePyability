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

### Exclusive or independent shocks

By default the shocks are mutually exclusive, as `decompose` gives them:
one shared cause strikes the group at most, so each member fails with
probability `Q` exactly. PRA codes (SAPHIRE, CAFTA, RiskSpectrum) take each
shock's `Q_k` as a basic event of its own instead, independent of the
others, so several can strike at once. The two differ at second order in
`Q`, so to check a result against such a tool, give
`MGL(..., shocks="independent")`:

```python
W = surv.Weibull.from_params([1000, 1.5])          # Q = 0.031 at t = 100
parallel = [("s", x) for x in "abc"] + [(x, "t") for x in "abc"]

def vote(model):
    return NonRepairableRBD(parallel, {x: W for x in "abc"}, k={"t": 2},
                            ccf_groups=[CCFGroup(list("abc"), model)])

vote(MGL(0.2, 0.3)).ff(100.0)                         # -> 0.010219   exclusive
vote(MGL(0.2, 0.3, shocks="independent")).ff(100.0)   # -> 0.010192   as PRA codes
```

A `BetaFactor`, or an `MGL` group with one shared cause, has a single shock,
and both conventions agree. By rate (`basis="rate"`, below) the causes
always strike independently.

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

## Importance, sensitivity, uncertainty and allocation

A group's members fail together, so a member's state says something about
the others'. The importance measures condition on it: `R(1_i)` and
`R(0_i)`, the system's reliability given member `i` works and given it has
failed, are summed over the groups' shock outcomes, each weighted by the
member's chance of that state under it. A node outside the groups is held
working and failed, as without them. Birnbaum, improvement potential, RAW,
RRW and criticality follow from them as usual, and Fussell–Vesely sums,
outcome by outcome, the probability that a minimal cut set containing the
node has failed. Each is a sum of products, so a small probability keeps
its precision, and `beta = 0` gives the measures without the group. With a
valve in series with the pumps, the shared cause moves the importance from
the valve to the pumps:

```python
valve_edges = [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")]
valve_nodes = {
    "p1": FixedEventProbability.from_params(0.01),
    "p2": FixedEventProbability.from_params(0.01),
    "v": FixedEventProbability.from_params(0.001),
}
group = CCFGroup(["p1", "p2"], BetaFactor(0.1))
with_valve = NonRepairableRBD(valve_edges, valve_nodes, ccf_groups=[group])
with_valve.fussell_vesely()["p1"]          # -> 0.5197   independent: 0.0909
with_valve.fussell_vesely()["v"]           # -> 0.4808   independent: 0.9092
with_valve.risk_achievement_worth()["p1"]  # -> 52.45    independent: 9.99
```

A member held working or broken (`working_nodes`, `broken_nodes`) is
refused, as it is for `sf`.

`parameter_sensitivity` reports a group's parameters once, under the tuple
of its members. They carry one model, so each value is the derivative of
the system reliability as the parameter moves for all of them at once. The
parameters of the group's own model come with them, named `ccf_beta` (and,
for an `MGL` model, `ccf_gamma`, `ccf_delta`, ...):

```python
sensitivity = with_valve.parameter_sensitivity()
sensitivity[("p1", "p2")]["p"]          # -> -0.11606   the pumps' probability of failing
sensitivity[("p1", "p2")]["ccf_beta"]   # -> -0.00981
sensitivity["v"]["p"]                   # -> -0.99892
```

The parameter-uncertainty methods (`sf_uncertainty`, `mean_uncertainty`,
`time_to_reliability_uncertainty`, `bx_life_uncertainty`) work out each
draw with the groups. The members' uncertainty is given together, in one
tuple, and the group's own model can be uncertain too: give the group as a
key, with distributions over its parameters (or a list of models, drawn with
replacement):

```python
import scipy.stats as st

result = with_valve.sf_uncertainty(
    uncertainty={
        ("p1", "p2"): {"p": st.beta(2, 198)},   # about 0.01
        group: {"beta": st.beta(2, 18)},        # about 0.1
    },
    n_draws=10_000,
    seed=1,
)
result.interval(0.9)   # (0.99564, 0.9989); (0.99616, 0.99882) with beta known
```

`mean_uncertainty` refuses a group that splits a probability, as `mean`
does.

In `allocate_redundancy` and `redundancy_front` a copy of a member joins its
group, as it would in the plant: the shared cause fails it too. A
`BetaFactor` group's `beta` holds at any size, so its members can be copied:
active copies of their own model, with any number of them required. An
`MGL` model's letters are for its group's size, so copying its member is
refused, as are options and cold spares for a member. The shared cause can
change what is worth buying. With two pumps that fail with probability 0.05
and a valve that fails with 0.002, a third pump is the better buy if the
pumps are independent, and a second valve if a fifth of their failures are
shared:

```python
weak = {
    "p1": FixedEventProbability.from_params(0.05),
    "p2": FixedEventProbability.from_params(0.05),
    "v": FixedEventProbability.from_params(0.002),
}
costs = {"p1": 1.0, "p2": 1.0, "v": 1.0}
NonRepairableRBD(valve_edges, weak).allocate_redundancy(costs, budget=4).units
# {'p1': 1, 'p2': 2, 'v': 1}
NonRepairableRBD(
    valve_edges, weak, ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.2))]
).allocate_redundancy(costs, budget=4).units
# {'p1': 1, 'p2': 1, 'v': 2}
```

`allocate_reliability_redundancy` takes the groups in for the nodes outside
them, and refuses to choose a member's component reliability, which is the
group's.

## What honours a CCF group

| Method | With CCF groups |
|---|---|
| `sf`, `ff`, `reliability`, `unreliability`, and what is derived from them (`df`, `hf`, `Hf`, `cs`, `time_to_reliability`, `bx_life`) | Include the common cause, exactly. A probability split warns once its members' `Q` passes 0.1. |
| `structural_importance` | Unaffected (it does not use probabilities). |
| `mean`, `mean_time_to_failure` (exact by default) | Include a group splitting the rate. Raise `NotImplementedError` for a group splitting a probability: the exact MTTF integrates the reliability over whole lifetimes, where `Q` runs to 1. |
| `random`, `mean(method="simulate")`, `mean_time_to_failure_interval`, `compare`, `unreliability_interval` | Draw a group splitting the rate with its shared shocks. Sample the members of a group splitting a probability **independently**, leaving the common cause out (`unreliability_interval` refuses such a group). |
| The importance measures (Birnbaum, improvement potential, RAW, RRW, criticality, Fussell–Vesely) | Include the groups, exactly: a member conditioned on its state through the shock outcomes, a node outside them held. |
| `parameter_sensitivity` | A group's parameters, and its model's (`ccf_beta`, ...), reported once, under the tuple of its members. |
| `sf_uncertainty`, `time_to_reliability_uncertainty`, `bx_life_uncertainty`, `mean_uncertainty` | Each draw worked out with the groups; the members drawn together, and the group's model too if it is given. `mean_uncertainty` refuses a group splitting a probability, as `mean` does. |
| `allocate_redundancy`, `redundancy_front` | A member's copies join its group: a `BetaFactor` group's, active copies of the member's own model. Copies of an `MGL` group's member, and options or cold spares for a member, raise `NotImplementedError`. |
| `allocate_reliability_redundancy` | Includes the groups; a member in `uses` raises `NotImplementedError` (its reliability is the group's). |
| The condition-based methods (`sf_given_state`, `remaining_life`, `importances_given_state`) | Raise `NotImplementedError`: they would need members of different ages. |
| `working_nodes` / `broken_nodes` naming a group member | Raises `NotImplementedError`. |

```python
pair_ccf.mean(method="simulate", seed=0)   # -> 1147.4   the same as without the group
pair.mean(method="simulate", seed=0)       # -> 1147.4
pair.mean()                                # -> 1145.8   exact, without the group
```

Groups are saved with the RBD, their basis with them. An *alpha-factor*
model, a data-estimable reparameterisation of MGL, is a planned extension.

A [`FaultTree`](fault-trees.md#common-causes) takes the same groups over its
basic events, with the same exact top event probability and importance
measures, and its conversions to and from a diagram keep them.

## Repairable systems

A [`RepairableRBD`][repyability.RepairableRBD] takes `ccf_groups` too. A
repairable component's failures are a rate, so a group's model always splits
the rate (whatever its `basis`): each cause, a member's own or a shared one,
strikes at its share of the failure rate and fails the members it names that
are up, at once. Each member on its own still fails at its rate, so
`node_availability` is as without the group; what changes is which members
are down together. With two pumps, each down a tenth of the time, a fifth of
their failures shared:

```python
from repyability import RepairableRBD

pump = {
    "reliability": surv.Exponential.from_params([0.01]),
    "repairability": surv.Exponential.from_params([0.1]),
}
independent = RepairableRBD(edges, {"p1": pump, "p2": pump})
shared = RepairableRBD(
    edges,
    {"p1": pump, "p2": pump},
    ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.2))],
)
independent.mean_unavailability()   # -> 0.008264   (1/11)^2
shared.mean_unavailability()        # -> 0.01585
```

The long-run values are exact: a group's members form a Markov chain of
which of them are down, its long-run distribution found without
subtraction, so a small probability keeps its precision. The members need
exponential lives, and either revealed failures with exponential repairs, as
here, or hidden failures found by tests (instant, as the long-run values of
tests need, with one coverage for the group), at offsets of their own if
they are staggered; see [the PFDavg of a safety
function](costs.md#common-cause-staggered-tests-and-test-coverage).
`mean_availability`, `mean_unavailability`, `system_failure_frequency`,
MTBF, MUT and MDT, the cost rate, `capacity_distribution`, and the interval
choices built on them take the groups in. So do the importance measures:
each long-run time's points are split by the members' joint states, and a
member's measures are conditioned on its state at each time, then averaged
over the times as every node's are. With `beta = 0` they are the measures
without the group:

```python
shared.birnbaum_importance()["p1"]       # -> 0.1743   P(p2 down | p1 down)
independent.birnbaum_importance()["p1"]  # -> 0.0909
```

Over time from new, each group's chain is followed from every member up at
0 (by uniformization, or through the members' tests, after whose first
period it repeats its long run), and each time is split by the groups'
joint states as the long-run times are. `point_availability`,
`mission_availability`, `expected_failures`, `expected_events`,
`expected_cost`, `point_capacity` and `mission_capacity` take the groups in,
and settle into the long-run values. A member's own curve, and so its own
events, are as without its group; the system's failures count each cause
that takes it down, a shared one several members at once:

```python
shared.point_availability(10.0)       # -> 0.98891   independent: 0.99632
shared.mission_availability(1000.0)   # -> 0.98429   settling to 0.98415
shared.expected_failures(1000.0)      # -> 3.158     independent: 1.639
```

The simulations (`availability`, `cost`, `compare`, `simulate_timelines`,
event stepping and `spares_demand(method="simulate")`) draw each group's
causes: each strikes as a Poisson process at its share of the members'
failure rate, and fails the members it names that are up, at once. A test
that can miss a failure tosses one coin for all the failures a cause makes.
They run in Python (the compiled engine does not draw the causes, as yet),
and they take in what the chains cannot: tests and repairs that take time,
and repairs of any distribution:

```python
run = shared.availability(1000.0, mc_samples=2000, seed=0)
run.system_uptime / (2000 * 1000.0)          # -> 0.9841   simulated
run.mean_availability_interval().estimate    # -> 0.98429  exact, as the chains give it
```

In `allocate_redundancy` a member's copies join its group, as for a
non-repairable diagram: a `BetaFactor` group's, each copy alike in every way,
struck by the shared cause too, and tested with its member when its failures
are hidden. Copies of an `MGL` group's member, or of a train holding one, are
refused. Each design is scored exactly by its groups' chains, which count
how many of a member's copies are down rather than telling them apart, so a
design with many copies is quick to score. The copies are repaired at once
(the allocations assume as many repair crews as jobs), so a shared failure
ends with the first copy repaired, and a shared cause can make more pumps
worth buying, not fewer:

```python
costed = {
    "p1": {**pump, "acquisition_cost": 10000.0},
    "p2": {**pump, "acquisition_cost": 10000.0},
    "v": {
        "reliability": surv.Exponential.from_params([0.001]),
        "repairability": surv.Exponential.from_params([0.5]),
        "acquisition_cost": 3000.0,
    },
}
plant = RepairableRBD(valve_edges, costed, downtime_cost_rate=100.0)
plant.allocate_redundancy(87600.0, nodes=["p1", "v"]).units
# {'p1': 2, 'v': 2}
plant_shared = RepairableRBD(
    valve_edges,
    costed,
    ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.2))],
    downtime_cost_rate=100.0,
)
plant_shared.allocate_redundancy(87600.0, nodes=["p1", "v"]).units
# {'p1': 3, 'v': 2}
```

`availability_allocation` and `mttf_mttr_allocation` keep the members'
availability, as their MTTF and MTTR are their group's, and allocate the
other components', the system scored over the groups' joint states. The
members then bound what the others can bring: with the valve never down, the
shared failures hold the plant to 0.98415, and a higher target is refused:

```python
plant_shared.availability_allocation(0.983).mttf["v"]   # -> 1704.4   the valve's MTTF, at its 2 h repairs
```

What the groups' chains need (exponential lives, and revealed failures with
exponential repairs or tests and repairs in no time) holds for the exact and
numerical values; the simulations need exponential lives alone. A member
held working or broken, a member started from a current state (`state=`), a
member maintained on a schedule, repaired imperfectly or in a maintenance
group, and a group with limited repair crews are refused, with the reason;
`analysis_routes()` says which.
