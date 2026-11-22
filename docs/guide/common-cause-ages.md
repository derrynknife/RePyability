# Common causes with wear-out lives and ages

A [common-cause group](common-cause.md) is symmetric: its members carry
one model. This page is about members that wear out and have been running
for a while, which asks what the shared cause is:

| The shared cause | How to model it | Members |
|---|---|---|
| A shock from outside, at a rate that does not depend on the members' ages, that fails every member it reaches (a flood, a surge, a wrong setting applied to all) | An exponential node in series with the members: no group | Any lives, each at its own age |
| The same shocks, each failing each member it reaches with a probability `p` (the binomial failure rate model) | `shock_nodes`, which writes a node for each set of members a shock can fail | Any lives, each at its own age |
| A share `β` of each member's own hazard, which ages as the members do (a batch defect, a shared environment that wears them together) | A group splitting the failure rate, `BetaFactor(beta, basis="rate")` | One life and one age: installed and replaced together |

Both are taken by every analysis of a
[`NonRepairableRBD`][repyability.NonRepairableRBD], the
[condition-based](condition-based.md) ones among them.

## A shock that fails every member

A shock process of rate `λc` that fails every member it strikes is one more
component: it "fails" at the first shock, at an exponential time, and the
members it strikes cannot work past it. Put it in series with them. Each
member keeps its own life and its own age, and the shock has no age (an
exponential life has no memory).

Two pumps in parallel, of Weibull lives, with a shock every 50,000 hours on
average:

```python
import numpy as np
import surpyval as surv
from repyability import BetaFactor, CCFGroup, NodeState, NonRepairableRBD

pump = surv.Weibull.from_params([8000, 2.5])
shock = surv.Exponential.from_params([2e-5])

parallel = [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")]
with_shock = [("s", "p1"), ("s", "p2"), ("p1", "shock"), ("p2", "shock"), ("shock", "t")]

alone = NonRepairableRBD(parallel, {"p1": pump, "p2": pump})
pair = NonRepairableRBD(with_shock, {"p1": pump, "p2": pump, "shock": shock})

alone.sf(2000)   # -> 0.99905
pair.sf(2000)    # -> 0.95988   = exp(-2e-5 × 2000) × 0.99905
alone.mean()     # -> 8816.9
pair.mean()      # -> 8022.8    exact
pair.mean(method="simulate", seed=0)   # -> 8016.3
```

**Members of different ages.** One pump has run 6000 hours and the other
1000. Each is conditioned on its own age, and the shock on nothing:

```python
ages = {"p1": NodeState(age=6000), "p2": NodeState(age=1000)}

alone.sf_given_state(2000, ages)   # -> 0.96893   the next 2000 hours, no shared cause
pair.sf_given_state(2000, ages)    # -> 0.93094

# The closed form: R_sys = exp(-λc x) × R_independent(x | ages)
float(np.exp(-2e-5 * 2000) * alone.sf_given_state(2000, ages))   # -> 0.93094

alone.remaining_life(0.9, ages)   # -> 3118.0   hours until the reliability is 0.9
pair.remaining_life(0.9, ages)    # -> 2471.6
```

The importance measures name the shock as they name any component, which
says how much of the risk is the shared cause's:

```python
worth = pair.importances_given_state(2000, ages)
worth["criticality"]["shock"]   # -> 0.5501   over half of the risk in the next 2000 hours
worth["criticality"]["p1"]      # -> 0.4322
worth["birnbaum"]["p2"]         # -> 0.3855   the newer pump matters more: the older is likelier to fail
```

### Members that are not side by side

Where the members sit in different branches, no one place in the diagram is
in series with all of them. Draw the shock once in each branch, as [one
component in several places](building.md#one-component-in-several-places):
the second node names the first, and the two fail as one.

```python
valve = surv.Weibull.from_params([20000, 1.5])

#   s -> c1 -> p1 -> v1 -> t
#   s -> c2 -> p2 -> v2 -> t        c2 is the same shock as c1: it fails both pumps
trains = NonRepairableRBD(
    [("s", "c1"), ("c1", "p1"), ("p1", "v1"), ("v1", "t"),
     ("s", "c2"), ("c2", "p2"), ("p2", "v2"), ("v2", "t")],
    {"p1": pump, "p2": pump, "v1": valve, "v2": valve, "c1": shock, "c2": "c1"},
)
trains.repeated   # {'c2': 'c1'}
trains.sf_given_state(2000, {"p1": 6000, "p2": 1000, "v1": 3000, "v2": 3000})   # -> 0.90282
```

Several shock processes (one that fails the pumps, another the valves) are
several such nodes. A shock that fails only some of the members is a node in
series with those members only.

## A shock that fails each member with a probability

A shock need not fail everything it reaches. In the binomial failure rate
model, shocks arrive at a rate and each fails each member independently
with probability `p`, so one shock may fail one member, another two, a
third none. By thinning the shocks, that is one independent cause for
each set of members: the cause that fails exactly a set of `j` of the `m`
members strikes at rate `rate · p^j · (1 − p)^(m − j)`. Each is an
exponential node in series with the members it fails, so
[`shock_nodes`][repyability.shock_nodes] writes them into a diagram's edges
and nodes: a node after each member a cause fails, one component in
several places.

```python
from repyability import shock_nodes

edges, nodes = shock_nodes(
    parallel, {"p1": pump, "p2": pump},
    members=["p1", "p2"], rate=2e-5, p=0.6,
)
partial = NonRepairableRBD(edges, nodes)
len(nodes)   # -> 6   the pumps, and three causes: of each pump alone (one node) and of both (two)

partial.sf_given_state(2000, ages)   # -> 0.95111   between no shocks (0.96893) and p = 1 (0.93094)
partial.mean()                       # -> 8404.2

# The closed form: the cause of both has not struck, and each pump has
# survived its own life from its age and the cause of it alone
both = np.exp(-2e-5 * 0.6 * 0.6 * 2000)
alone_ = np.exp(-2e-5 * 0.6 * 0.4 * 2000)
old = pump.sf(8000) / pump.sf(6000) * alone_
young = pump.sf(3000) / pump.sf(1000) * alone_
float(both * (1 - (1 - old) * (1 - young)))   # -> 0.95111
```

The members keep their own lives, which may differ, and their own ages; the
causes have none. `p=1` (the default) is the one node of the section above,
and `lethal_rate=` adds shocks of a second kind that fail every member, as
the model's authors have it. A process has a cause for each set of its
members (`2^m − 1`), so it takes up to ten. The importance measures name
each cause, `("shock", the members it fails, the member it follows)`, and a
member's own entry is that of its own failures. Give each process of a
diagram a name of its own (`name=`).

The same shocks can be given as a group's model,
[`Shocks`][repyability.Shocks], which the diagram writes into its structure
exactly as `shock_nodes` does (the same nodes and the same values, to the
last bit) and saves as given:

```python
from repyability import Shocks

grouped = NonRepairableRBD(
    parallel, {"p1": pump, "p2": pump},
    ccf_groups=[CCFGroup(["p1", "p2"], Shocks(rate=2e-5, p=0.6))],
)
grouped.sf_given_state(2000, ages) == partial.sf_given_state(2000, ages)   # True
list(grouped.nodes) == list(partial.nodes)                                 # True
grouped.to_dict()["ccf_groups"][0]["model"]["kind"]                        # 'shocks'
```

Unlike the other models, which split the failures the members' own model
describes, `Shocks` adds failures to it, so the members need not be alike.
The first group's causes are named `"shock"`; with several, `"shock 1"`,
`"shock 2"`, ... in the groups' order.

For members with exponential lives this is an
[alpha-factor](common-cause.md#the-three-models) group by rate: a member
fails at its own rate plus `rate · p`, and the cause of each set of `j` is
its share of that.

## A share of each member's hazard

Where the shared cause wears as the members do, a share `β` of each member's
hazard `h(t)` is the shared cause's and the rest the member's own. The
shared cause then has survival `R(t)^β` and each member's own causes
`R(t)^(1 − β)`, `R` the members' reliability, so every member still has the
life `R(t)^β · R(t)^(1 − β) = R(t)`. That is a group splitting the failure
rate, [`BetaFactor(beta, basis="rate")`](common-cause.md#over-a-lifetime),
for members of any life:

```python
beta = 0.1
group = NonRepairableRBD(
    parallel, {"p1": pump, "p2": pump},
    ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(beta, basis="rate"))],
)
alone.sf(5000)   # -> 0.92941
group.sf(5000)   # -> 0.91250
group.mean()     # -> 8705.3
```

**Members of one age.** The pumps were installed together 4000 hours ago.
Give every member that state, and each cause takes its share of the
members' hazard from that age on:

```python
installed = {"p1": NodeState(age=4000), "p2": NodeState(age=4000)}
group.sf_given_state(2000, installed)    # -> 0.91186
group.remaining_life(0.9, installed)     # -> 2136.8
group.mean_residual_life(installed)      # -> 5278.3
group.importances_given_state(2000, installed)["birnbaum"]["p1"]   # -> 0.3303
```

It only means something for members of **the same age**. A share of "the
members' hazard" has no one value when one member's hazard is that of 6000
hours and another's that of 1000: the model does not say whose the shared
cause follows. So a group whose members are given different states (or one
of them failed) is refused, and so is a group splitting a probability,
which says nothing of age:

```python
try:
    group.sf_given_state(2000, {"p1": 6000, "p2": 1000})
except NotImplementedError as error:
    refusal = str(error)
"node in series" in refusal   # True
```

For members of different ages, model the shared cause as a shock
([above](#a-shock-that-fails-every-member)), whose rate is its own.

### The same model drawn as nodes

The split is itself a node in series: each member's own causes as its node,
with survival `R^(1 − β)`, and the shared cause as one node with `R^β`. For
a Weibull or an exponential life the two parts have a closed form, which
checks the group:

| The members' life | Each member's own causes, `R^(1 − β)` | The shared cause, `R^β` |
|---|---|---|
| Weibull, scale `α` and shape `k` | Weibull, scale `α (1 − β)^(−1/k)`, shape `k` | Weibull, scale `α β^(−1/k)`, shape `k` |
| Exponential, rate `λ` | Exponential, rate `(1 − β) λ` | Exponential, rate `β λ` |

```python
alpha, k = 8000, 2.5
own = surv.Weibull.from_params([alpha * (1 - beta) ** (-1 / k), k])   # scale 8344.4
shared = surv.Weibull.from_params([alpha * beta ** (-1 / k), k])      # scale 20095.1
split = NonRepairableRBD(with_shock, {"p1": own, "p2": own, "shock": shared})

split.sf(5000)   # -> 0.91250   the group's
split.mean()     # -> 8705.3
split.sf_given_state(2000, {"p1": 4000, "p2": 4000, "shock": 4000})   # -> 0.91186
```

For an exponential life the two models of this page are one: a share `β` of
a constant hazard is a shock of rate `βλ`, which is why the members' ages
do not matter there.

```python
rate = 1e-4
exponential = NonRepairableRBD(
    parallel,
    {"p1": surv.Exponential.from_params([rate]), "p2": surv.Exponential.from_params([rate])},
    ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(beta, basis="rate"))],
)
exponential.sf(5000)   # -> 0.82632
exponential.sf_given_state(5000, {"p1": 30000, "p2": 30000})   # -> 0.82632   no memory
exponential.mean()     # -> 14736.8
```

## What these do not say

- **A share of the hazard for members of different ages**: there is no one
  hazard to share. Model the cause as a shock.
- **Repairable diagrams, exactly.** A node in series is down, and
  repaired, as one component, where a shock leaves each member it failed
  to be repaired on its own. Give a
  [`RepairableRBD`][repyability.RepairableRBD] the shocks as a group's
  model, `CCFGroup(members, Shocks(...))`: it
  [simulates them](common-cause.md#shocks-at-a-rate-of-their-own), for
  members of any lives, and its exact methods refuse, naming the
  simulation.
