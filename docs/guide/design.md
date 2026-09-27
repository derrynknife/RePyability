# Design and allocation

Importance measures say *where* a system is weak. The methods on this page
help decide what to do about it: how many redundant copies of each component
to buy (redundancy allocation), or what reliability each component must reach
for the system to meet a target (reliability allocation). Theory:
[Concepts](../concepts.md#allocation).

## Redundancy allocation

`allocate_redundancy(costs, ...)` chooses how many identical, independent,
active copies of each costed node to fit: the classic Redundancy Allocation
Problem. It works on any diagram, not only a series of subsystems, because
each candidate design is scored with the exact system computation.

```python
import surpyval as surv
from repyability import NonRepairableRBD

line = NonRepairableRBD(
    [("s", "pump"), ("pump", "valve"), ("valve", "ctrl"), ("ctrl", "t")],
    {
        "pump": surv.Weibull.from_params([8000, 1.8]),
        "valve": surv.Weibull.from_params([20000, 1.5]),
        "ctrl": surv.Weibull.from_params([40000, 1.2]),
    },
)
line.sf(5000)   # -> 0.5291   one of each, at a 5000 h mission

costs = {"pump": 4000, "valve": 900, "ctrl": 12000}   # per copy
```

### Most reliable design within a budget

```python
best = line.allocate_redundancy(costs, budget=40_000, t=5000)
best.units         # {'pump': 3, 'valve': 4, 'ctrl': 2}
best.reliability   # -> 0.9513
best.cost          # -> 39600.0   every copy, including the original
best.method        # 'exact'
```

### Cheapest design that meets a target

```python
cheapest = line.allocate_redundancy(costs, target=0.95, t=5000)
cheapest.units   # {'pump': 3, 'valve': 4, 'ctrl': 2}
cheapest.cost    # -> 39600.0
```

The result is a [`RedundancyAllocation`][repyability.RedundancyAllocation]
(`units`, `reliability`, `cost`, `method`, and `resources`, the totals of
every resource used).

### The whole trade-off

A budget or a target gives one design. `redundancy_front` gives every design
that no other beats, being no more expensive (in every resource) while at
least as reliable: the Pareto front of cost against reliability, so a design
can be chosen by looking at the whole curve rather than guessing a budget.
It takes the same arguments as `allocate_redundancy` (except `target`), and
the budget, or `max_units`, bounds it.

```python
front = line.redundancy_front(costs, budget=50_000, t=5000)
len(front)       # -> 41   designs from 16,900 (one of each) up
[(d.cost, round(d.reliability, 5)) for d in front[26:33]]
# [(37800.0, 0.93838), (38700.0, 0.94998), (39600.0, 0.95134),
#  (40500.0, 0.9515), (41400.0, 0.95152), (41800.0, 0.96549),
#  (42700.0, 0.97742)]
```

Each point is a `RedundancyAllocation` (`units`, `reliability`, `cost`,
`resources`, …). The curve shows where money stops buying reliability: past
39,600 two more valves add 0.0002, while 2,200 more buys a fourth pump (in
place of two valves) and 0.0142. The best design within any budget, and the
cheapest reaching any
target, are on it:

```python
next(d for d in front if d.reliability >= 0.95).units
# {'pump': 3, 'valve': 4, 'ctrl': 2}   as allocate_redundancy(target=0.95)
```

With several resources the front holds the best reliability for every
combination of them. It is exact: on a series of costed nodes it comes from
the dynamic program, and on other structures from evaluating every design
within the budget (with the same size limit as the exact search).

### Options

- **`t`** is the mission time at which reliability is scored. It is not
  needed when every node is a fixed probability.
- **`max_units`** caps the copies: an int for every costed node, or a dict
  for particular ones (space, weight, or a supplier limit).

  ```python
  line.allocate_redundancy(costs, budget=40_000, t=5000, max_units={"pump": 2}).units
  # {'pump': 2, 'valve': 8, 'ctrl': 2}
  ```

- **`method`**: `"exact"` (the default) returns a proven optimum. When
  every costed node is in series with the rest of the system, as in `line`,
  the system reliability is a product over them, and a dynamic program over
  the nodes solves even a long series of subsystems in a fraction of a
  second. On other structures, because adding a copy never lowers a coherent
  system's reliability, a budget search only needs to score the designs that
  cannot afford another copy, which keeps typical problems (a handful of
  nodes) fast. Either way, a problem too large to solve raises an
  explanatory error instead of running for ever.
  `"greedy"` adds one copy at a time, always the one with the largest gain in
  log-reliability per unit cost. It is fast at any size but not guaranteed
  optimal, even on a series system:

  ```python
  line.allocate_redundancy(costs, budget=40_000, t=5000, method="greedy").reliability
  # -> 0.919   against 0.9513 for the exact optimum at the same cost
  ```

- The "cost" can be any additive resource: money, weight, volume, power;
  or several at once (below).
- Nodes not in `costs` are left as they are.

### Several resources

Copies use more than money: weight, volume, power, space. Give what one copy
of each node uses as a dict of resources, and the budget as a dict of limits
on any of them (the classic multi-constraint problem of Fyffe, Hines & Lee,
1968). A resource that is not limited is still totalled.

```python
kit = {
    "pump": {"cost": 4000, "weight": 30},
    "valve": {"cost": 900, "weight": 12},
    "ctrl": {"cost": 12000, "weight": 5},
}
light = line.allocate_redundancy(
    kit, budget={"cost": 40_000, "weight": 120}, t=5000
)
light.units         # {'pump': 2, 'valve': 4, 'ctrl': 2}
light.reliability   # -> 0.8726
light.resources     # {'cost': 35600.0, 'weight': 118.0}
```

The weight limit costs reliability: 0.8726, against 0.9513 for the best
design on money alone, which weighs 148. With several limits the greedy
heuristic measures a copy by its total share of them, and falls further
short here (0.8695).

A **target and a budget together** ask for the cheapest design that meets
the target without breaking any limit:

```python
line.allocate_redundancy(kit, target=0.9, t=5000).resources
# {'cost': 30700.0, 'weight': 161.0}
line.allocate_redundancy(
    kit, target=0.9, budget={"weight": 130}, t=5000
).resources
# {'cost': 37800.0, 'weight': 124.0}
```

A target minimises `"cost"` (or the only resource) unless **`minimise`**
names another. `cost` in the result is always the total of the minimised
resource:

```python
lightest = line.allocate_redundancy(
    kit, target=0.95, minimise="weight", t=5000
)
lightest.units       # {'pump': 3, 'valve': 3, 'ctrl': 3}
lightest.cost        # -> 141.0   the weight
lightest.resources   # {'cost': 50700.0, 'weight': 141.0}
```

The cheapest design meeting 0.95 weighs 148 (it is the 39,600 design above);
the lightest weighs 141 but costs 50,700.

### Choosing between component types

A node can often be built from one of several component types, each with its
own reliability and cost. Give it a list of
[`ComponentOption`][repyability.ComponentOption] instead of a cost. Its model
in the diagram is then not used, so list it as an option too if it is a
candidate.

```python
from repyability import ComponentOption

pumps = [
    ComponentOption("standard", surv.Weibull.from_params([8000, 1.8]), cost=4000),
    ComponentOption("premium", surv.Weibull.from_params([15000, 2.2]), cost=9000),
]
choice = {"pump": pumps, "valve": 900, "ctrl": 12000}
mixed = line.allocate_redundancy(choice, budget=40_000, t=5000)
mixed.units         # {'pump': 2, 'valve': 3, 'ctrl': 2}
mixed.mix           # {'pump': {'standard': 1, 'premium': 1}}
mixed.reliability   # -> 0.9626
```

By default the copies of a node may mix types (Coit & Smith, 1996): here one
premium pump backed by a standard one. With `mixing=False` every copy of a
node is of one type (Fyffe, Hines & Lee, 1968), which can be simpler to stock
and maintain. The best single-type design here is three standard pumps, the
same as without the premium option:

```python
single = line.allocate_redundancy(choice, budget=40_000, t=5000, mixing=False)
single.mix           # {'pump': {'standard': 3}}
single.reliability   # -> 0.9513
```

`mix` gives the number of each type used, for the nodes with options.
`max_units` caps a node's copies of all types together, and options can use
several resources (every option then names the same ones). The exact search
is checked against the classic benchmark of 14 subsystems with three or four
component types each, with mixing (Fyffe, Hines & Lee's data, in Coit &
Smith's form): it returns the best reliabilities published for it, such as
0.986811 within a cost of 130 and a weight of 191, in seconds.

### Several units required, and standby spares

**`required`** sets how many of a node's copies must work (k-out-of-n; Coit &
Liu, 2000). If the line needs two pumps running, the budget buys fewer spares
for the rest:

```python
two = line.allocate_redundancy(costs, budget=40_000, t=5000, required={"pump": 2})
two.units         # {'pump': 6, 'valve': 4, 'ctrl': 1}
two.reliability   # -> 0.9004
```

**`strategy`** arranges a node's spares: `"active"` (the default) runs every
copy; `"cold"` keeps the spares unpowered until one is switched in to replace a
failed unit (Coit, 2001), which makes the node a
[`StandbyModel`][repyability.StandbyModel] of its copies; `"choose"` lets the
optimiser pick, node by node (Coit, 2003). Cold spares do not age, so with
perfect switching they always beat active ones:

```python
cold = line.allocate_redundancy(
    costs, budget=40_000, t=5000, strategy={"pump": "cold"}
)
cold.strategy      # {'pump': 'cold', 'valve': 'active', 'ctrl': 'active'}
cold.reliability   # -> 0.9922   the same design as active: 0.9513
```

With imperfect switching (**`switching_probability`**, the chance that
switching onto a spare succeeds) that is no longer so, and `"choose"` weighs
one against the other. Here cold spares still win, narrowly:

```python
unsure = line.allocate_redundancy(
    costs,
    budget=40_000,
    t=5000,
    strategy={"pump": "choose"},
    switching_probability={"pump": 0.9},
)
unsure.strategy["pump"]   # 'cold'
unsure.reliability        # -> 0.955
```

Cold standby needs lifetime models (not fixed probabilities). Its reliability
comes from `StandbyModel`: exact for identical Exponential units, a numerical
convolution (accurate to about 1e-3) for one unit required, and simulated
(seeded, so reproducible) otherwise; imperfect switching needs one unit
required. Each cold design is evaluated this way, which is slower than active
copies, so give cheap cold-standby nodes a `max_units`. A node without spares
(only the copies it requires) is the same either way, and is reported as
active.

### The model and its limits

`n` copies of a node with reliability `p`, all active and independent, have
reliability `1 − (1 − p)ⁿ` (with mixed types, `1 − (1 − p₁)(1 − p₂)…`). That
is the right model for parts added in active parallel; standby spares are
modelled with `strategy` (above), but common-cause coupling is not (an RBD
with common-cause groups raises `NotImplementedError`). A budget that
cannot buy one of each costed node, a target that no design within
`max_units` (or the budget) reaches, or a node that uses none of any limited
resource and has no `max_units` (so could be copied without limit) raises
`ValueError` with the numbers involved.

## Reliability allocation

Reliability allocation works the other way round from analysis: given a
target for the system, what must each component achieve? Many allocations
meet a target (in a series of three, any three reliabilities whose product is
the target will do), so each method is a rule for picking one. The rules
differ in what they take into account: the structure, the components'
current reliabilities, and how hard each component is to improve.

These helpers work on **probabilities** (each node's reliability at the
mission time, or its availability), not on lifetime models, and each returns
a dict of node → required probability. They are methods of every RBD class,
so they also work on a bare `RBD` structure. Theory:
[Concepts](../concepts.md#allocation).

### The methods at a glance

| Method | Named method | Needs | Structure | How it picks |
|---|---|---|---|---|
| `equal_allocation` | Equal apportionment | The target | Any | Every node the same |
| `simple_allocation` | None: a structural allocation | The target; optional weights | Any | The least weighted change in log-odds from 0.5 |
| `improvement_allocation` | ARINC-style proportional apportionment | Current reliabilities | Any | Every failure probability scaled by one factor |
| `minimum_effort_allocation` | Minimization of effort (Albert, 1958) | Current reliabilities | Series only | The weakest nodes raised to one common level |
| `cost_based_allocation` | Cost-based allocation (Mettas, 2000) | Current reliabilities; optional maxima and feasibilities | Any | The least total cost of improvement |

### Choosing a method

- **Only the diagram is known** (early design, no component data):
  `simple_allocation` sets each node's requirement by its place in the
  structure; `equal_allocation` is the structure-blind baseline.
- **Current reliabilities are known** (predictions, test or field data) and
  the system falls short:
    - `improvement_allocation` cuts every node's failure probability by the
      same fraction, so the most frequent failures get the largest
      improvements;
    - `minimum_effort_allocation`, for a series system whose components are
      about equally hard to improve, lifts only the weakest;
    - `cost_based_allocation`, on any structure, takes into account that
      some components are harder to improve than others, or can only get so
      good.
- **Some components cannot change** (bought-in parts, a frozen design): pass
  them in `fixed` to `improvement_allocation`, or give them a maximum equal
  to their current value in `cost_based_allocation`.

### What they share

- **One mission time.** A probability is a reliability at one time (or an
  availability), so an allocation holds at that time only. For an
  exponential component, an allocated reliability `R` at time `t` is a
  failure rate `λ = −ln(R) / t`.
- **Exact, with independent nodes.** Every method meets the target exactly,
  scoring candidates with the exact engine, on any structure it supports.
  The nodes are treated as independent: a `NonRepairableRBD`'s common-cause
  groups are ignored, and a repeated component (one component drawn in
  several places) is a single node.
- **A node is a block.** A node that stands for a subsystem gets one
  requirement; allocate within the subsystem afterwards, with its own
  diagram (top-down apportionment).
- **Requirements, not designs.** An allocation says what each component must
  achieve; better components or redundancy (see
  [Redundancy allocation](#redundancy-allocation)) achieve it.
- **Errors.** A target outside [0, 1] raises `ValueError`, and so does a
  target the method cannot reach, with the reachable range in the message.
- **Solver results.** Every method but `minimum_effort_allocation`, which is
  closed-form, keeps its solver's result in `rbd.res`.

The examples use three components in series, currently at 0.99, 0.97 and
0.90 (0.864 for the system), and a target of 0.95:

```python
from surpyval import FixedEventProbability

three_in_series = NonRepairableRBD(
    [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
    {n: FixedEventProbability.from_params(0.05) for n in "abc"},
)
current = {"a": 0.99, "b": 0.97, "c": 0.90}
three_in_series.system_probability(current)[0]   # -> 0.8643
```

### Equal apportionment

`equal_allocation(target)` gives every node the same reliability:
`target ** (1/n)` for `n` nodes in series, `1 − (1 − target) ** (1/n)` for
`n` in parallel, and whatever the exact engine finds for other structures:

```python
three_in_series.equal_allocation(0.95)   # {'a': 0.98305, 'b': 0.98305, 'c': 0.98305}
three_in_series.equal_allocation(0.95)["a"]   # -> 0.98305   = 0.95 ** (1/3)
```

**Use it** for a first cut when nothing distinguishes the components, or as a
baseline to compare the other methods with.

**Limits.** It uses nothing but the target: not the components' current
reliabilities, not their place in the structure, not what improving them
costs. A redundant component is asked for as much as one that every path
goes through, and a long chain in series asks every component for a very
high reliability.

### Proportional improvement (ARINC-style)

`improvement_allocation(target, node_probabilities, fixed=None,
weights=None)` starts from the current reliabilities and scales every
node's **failure probability** by one common factor until the system meets
the target, so nodes that are already good stay proportionally good. This is
the ARINC apportionment, generalised: ARINC scales every failure *rate* by a
common factor, which is the same for small failure probabilities, and this
works on any structure:

```python
three_in_series.improvement_allocation(0.95, current)
# {'a': 0.99639, 'b': 0.98917, 'c': 0.96389}: every failure probability × 0.361
```

- `fixed` lists nodes that cannot change (they keep their current value, and
  the others make up the difference).
- `weights` makes some nodes improve faster: node *i*'s failure probability
  is scaled by `exp(−x · weight_i)` for a common `x`, so a weight of 0 holds
  a node at its current value.
- A node missing from `node_probabilities` starts at 0.5, and
  `equal_allocation(target)` is `improvement_allocation` starting from 0.5
  for every node.

```python
three_in_series.improvement_allocation(0.95, current, fixed=["a"])
# {'a': 0.99, 'b': 0.99061, 'c': 0.96869}
three_in_series.improvement_allocation(0.95, current, fixed=["a"])["c"]   # -> 0.96869
```

A target below the current system reliability gives the *lowest* node
reliabilities that still meet it (each failure probability is capped at 1). A
target the scaling cannot reach, because the fixed nodes limit the system,
raises `ValueError` with the reachable range:

```python
try:
    three_in_series.improvement_allocation(0.999, current, fixed=["a", "b"])
except ValueError as error:
    print(error)
# target 0.999 cannot be reached: with the fixed nodes and weights given,
# the system probability can only range from 0 to 0.9603.
```

**Use it** when the components' current reliabilities are known and the
system falls short, and improving each component in proportion to how often
it fails is reasonable: the most frequent failures get the largest absolute
improvements, and the components keep their ranking.

**Limits.** Every adjustable node must improve, however good or unimportant
it already is. The factor is the same wherever a node sits, so a redundant
node takes the same proportional cut as a node in series with everything.
Nothing says what an improvement costs (the weights are the only lever), and
it scales probabilities rather than rates, which matches ARINC only while the
failure probabilities are small.

### Minimum effort

`minimum_effort_allocation(target, node_probabilities)` is Albert's
minimization-of-effort algorithm (MIL-HDBK-338B) for a series system. It
raises the least reliable nodes to one common level, just high enough to meet
the target, and leaves the others as they are:

```python
three_in_series.minimum_effort_allocation(0.95, current)
# {'a': 0.99, 'b': 0.97959, 'c': 0.97959}: c and b raised together, a untouched
three_in_series.minimum_effort_allocation(0.95, current)["c"]   # -> 0.97959
```

This is the least total effort for any effort function the nodes share that
meets Albert's conditions (the effort to raise a reliability from `x` to `y`
grows with `y` and adds up over successive steps, among others), so no
effort function has to be chosen.

**Use it** for a series system whose components are about equally hard to
improve, to find the least total effort: only the weakest components are
raised, together, and the rest are left alone. It is closed-form and
instant.

**Limits.** It applies to series systems only; any other structure raises
`ValueError`. One effort function is shared by every component, so it cannot
express that one is harder to improve than another, or can only get so good:
the common level it asks for may be out of reach for some of them. Effort is
abstract, not a cost.

### Cost-based allocation

`cost_based_allocation(target, node_probabilities, max_probabilities=None,
feasibility=None)` is Mettas's (2000) method, and works on any structure. It
finds the cheapest node reliabilities that meet the target, where raising a
node from its current reliability `R_min` towards the most it can reach,
`R_max`, costs

```
c(R) = exp((1 − f) · (R − R_min) / (R_max − R))
```

The cost is 1 at the current reliability and grows without bound towards the
maximum. The feasibility `f`, in [0, 1), says how easily a node can be
improved relative to the others: the lower it is, the faster the cost rises.

```python
three_in_series.cost_based_allocation(0.95, current)
# {'a': 0.99352, 'b': 0.98667, 'c': 0.96911}
three_in_series.cost_based_allocation(0.95, current)["c"]   # -> 0.96911
three_in_series.cost_based_allocation(0.95, current, feasibility={"c": 0.2})
# {'a': 0.99483, 'b': 0.98906, 'c': 0.9655}: c is hard to improve, so a and b do more
three_in_series.cost_based_allocation(0.95, current, max_probabilities={"c": 0.97})
# {'a': 0.99738, 'b': 0.99395, 'c': 0.95829}
```

- A node without a `max_probabilities` entry can approach 1; one whose
  maximum equals its current reliability is held fixed. A node without a
  `feasibility` entry gets 0.5.
- Every node stays below its maximum, which is only approached, at an
  ever-growing cost. A target the maxima cannot reach, or can only
  approach, raises `ValueError` with the reachable range.
- At the optimum, every improved node buys system reliability at the same
  marginal cost per unit of its Birnbaum importance; a node not worth
  improving is left as it is.
- It stays exact at any size: the system probability is handled on the
  log-odds scale from both ends of the exact engine, with exact gradients.

On a structure that is not a series, minimum effort does not apply, but the
cost-based method does:

```python
from repyability import RBD

pumps_and_valve = RBD([("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")])
pumps_and_valve.cost_based_allocation(0.99, {"p1": 0.9, "p2": 0.9, "v": 0.9})
# {'p1': 0.98081, 'p2': 0.98081, 'v': 0.99036}: the valve must reach 0.99 on its own
```

**Use it** on any structure when improving some components is harder than
others, or limited, and you want the cheapest allocation in those terms. It
is the approach ReliaSoft's BlockSim uses for allocation.

**Limits.** It needs a feasibility and a maximum for each component, usually
engineering judgement, and the answer depends on them, so try a range. The
cost is a relative penalty for comparing components, not money. It is solved
numerically, in a fraction of a second for a few hundred nodes, and the
maxima are only approached, never reached.

### Smallest log-odds change: a structural allocation

`simple_allocation(target, weights=None)` starts every node at 0.5 and finds
the node reliabilities that meet the target with the smallest weighted change
on the log-odds scale, `log(p / (1 − p))`: it minimises `Σ s_i² / w_i`, with
`s_i` node *i*'s log-odds and `w_i` its weight (1 by default). With no
weights, similar nodes end up equal:

```python
three_in_series.simple_allocation(0.95)   # {'a': 0.98305, 'b': 0.98305, 'c': 0.98305}
three_in_series.simple_allocation(0.95, weights={"a": 1.0, "b": 1.0, "c": 3.0})
# {'a': 0.97893, 'b': 0.97893, 'c': 0.99133}
three_in_series.simple_allocation(0.95, weights={"a": 1.0, "b": 1.0, "c": 3.0})["c"]   # -> 0.99133
```

It uses no component data, only the diagram, the target and the weights,
which makes it a **structural** allocation. At the solution, each node's
change is proportional to its weight times the sensitivity of the system's
log-odds to it, and that sensitivity comes from the node's Birnbaum
importance. With every node at 0.5, the Birnbaum importance *is* the
[structural importance](importance.md#structural-importance): the fraction of
the other nodes' states in which the node decides whether the system works.
So for a small change, each node moves in proportion to its weight times its
structural importance. For larger changes the probabilities move away from
0.5, each node's sensitivity becomes its Birnbaum importance at the new
probabilities, and the proportions drift, but they are still fixed by the
structure, the target and the weights alone:

```python
import math

pumps_and_valve.structural_importance()   # {'p1': 0.25, 'p2': 0.25, 'v': 0.75}


def log_odds(p):
    return math.log(p / (1 - p))


# Every node at 0.5 gives the system 0.375; just above it the valve moves
# three times as far as each pump, as its structural importance says.
small = pumps_and_valve.simple_allocation(0.3751)
log_odds(small["v"]) / log_odds(small["p1"])   # -> 3.0
large = pumps_and_valve.simple_allocation(0.99)
log_odds(large["v"]) / log_odds(large["p1"])   # -> 1.84
```

A weight of 0 holds a node at 0.5, and a target that such nodes put out of
reach raises `ValueError` with the reachable range. The answer is exact at
any size, as the method works on the log-odds scale from both ends of the
exact engine; a target of 0 or 1 is approached to within 1e-12. On a series
system with equal weights it is equal apportionment, since structure alone
cannot tell components in series apart.

**Use it** in early design, when the diagram is all there is: unlike equal
apportionment, it asks more of the components that the structure makes
matter more, and weights can add judgement (a component that is easier to
make reliable can take a larger weight).

**Limits.** It ignores the current reliabilities: a component already better
than its requirement has margin that the allocation does not use to relax
the others, and a weak one may be asked for a large jump. It is not one of
the classic named methods, and it follows structural importance exactly only
for small changes.

### The methods side by side

On the series system, target 0.95 (the minimum-effort and cost-based results
start from the current values):

| Method | a (0.99 now) | b (0.97 now) | c (0.90 now) |
|---|---|---|---|
| Equal apportionment | 0.98305 | 0.98305 | 0.98305 |
| Structural (`simple_allocation`) | 0.98305 | 0.98305 | 0.98305 |
| Proportional improvement | 0.99639 | 0.98917 | 0.96389 |
| Minimum effort | 0.99 | 0.97959 | 0.97959 |
| Cost-based | 0.99352 | 0.98667 | 0.96911 |

- Equal apportionment and the structural allocation cannot tell components in
  series apart; they ignore that `a` is already at 0.99 and `c` only at 0.90.
- Proportional improvement cuts every failure probability by the same
  factor, so even `a` must improve.
- Minimum effort leaves `a` alone and lifts `b` and `c` together.
- The cost-based allocation improves all three, `c` the most, but asks less of
  `c` than minimum effort does, because closing a node's gap to its maximum
  gets ever more expensive.

On the pumps and valve, with the pumps at 0.8 and the valve at 0.95 (0.912
for the system), target 0.97:

```python
pumps_now = {"p1": 0.8, "p2": 0.8, "v": 0.95}
pumps_and_valve.equal_allocation(0.97)["v"]                     # -> 0.97083
pumps_and_valve.improvement_allocation(0.97, pumps_now)["v"]    # -> 0.97775
pumps_and_valve.cost_based_allocation(0.97, pumps_now)["v"]     # -> 0.98053
pumps_and_valve.simple_allocation(0.97)["v"]                     # -> 0.98108
```

| Method | p1 (0.8 now) | p2 (0.8 now) | v (0.95 now) |
|---|---|---|---|
| Equal apportionment | 0.97083 | 0.97083 | 0.97083 |
| Structural (`simple_allocation`) | 0.89374 | 0.89374 | 0.98108 |
| Proportional improvement | 0.91099 | 0.91099 | 0.97775 |
| Minimum effort | (series only) | | |
| Cost-based | 0.89638 | 0.89638 | 0.98053 |

- Equal apportionment asks the redundant pumps for as much as the valve that
  every path goes through.
- Proportional improvement cuts the pumps' failure probability by the same
  factor as the valve's, although the pumps back each other up.
- The cost-based and structural allocations both put the effort on the
  valve, one from the current values and costs, the other from the structure
  alone. Here they nearly agree, because the structure is what makes the
  valve matter.
