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
(`units`, `reliability`, `cost`, `method`).

### Options

- **`t`** is the mission time at which reliability is scored. It is not
  needed when every node is a fixed probability.
- **`max_units`** caps the copies: an int for every costed node, or a dict
  for particular ones (space, weight, or a supplier limit).

  ```python
  line.allocate_redundancy(costs, budget=40_000, t=5000, max_units={"pump": 2}).units
  # {'pump': 2, 'valve': 8, 'ctrl': 2}
  ```

- **`method`**: `"exact"` (the default) returns a proven optimum. Because
  adding a copy never lowers a coherent system's reliability, a budget search
  only needs to score the designs that cannot afford another copy, which
  keeps typical problems (a handful of nodes) fast; a problem too large to
  search raises an explanatory error instead of running for ever.
  `"greedy"` adds one copy at a time, always the one with the largest gain in
  log-reliability per unit cost. It is fast at any size but not guaranteed
  optimal, even on a series system:

  ```python
  line.allocate_redundancy(costs, budget=40_000, t=5000, method="greedy").reliability
  # -> 0.919   against 0.9513 for the exact optimum at the same cost
  ```

- The "cost" can be any additive resource: money, weight, volume, power.
- Nodes not in `costs` are left as they are.

### The model and its limits

`n` copies of a node with reliability `p`, all active and independent, have
reliability `1 − (1 − p)ⁿ`. That is the right model for identical parts added
in parallel; it does not model standby spares or common-cause coupling (an
RBD with common-cause groups raises `NotImplementedError`). A budget that
cannot buy one of each costed node, or a target that no design within
`max_units` reaches, raises `ValueError` with the numbers involved.

## Reliability allocation

Reliability allocation works the other way round: given a system target, what
must each component achieve? These helpers work on **probabilities** (each
node's reliability at the mission time, or its availability), not on
lifetime models, and return a dict of node → required probability. Each is
one of many allocations that meet the target, chosen by a particular rule,
and all but one are the classic named methods:

| Method | Named method | Structure | Starts from |
|---|---|---|---|
| `equal_allocation` | Equal apportionment | Any | Nothing (every node alike) |
| `improvement_allocation` | ARINC-style proportional apportionment | Any | Current reliabilities |
| `minimum_effort_allocation` | Minimization of effort (Albert, 1958) | Series | Current reliabilities |
| `cost_based_allocation` | Cost-based allocation (Mettas, 2000) | Any | Current reliabilities |
| `simple_allocation` | None: a least-squares heuristic | Any | 0.5 for every node |

```python
from surpyval import FixedEventProbability

three_in_series = NonRepairableRBD(
    [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
    {n: FixedEventProbability.from_params(0.05) for n in "abc"},
)
```

### Equal apportionment

`equal_allocation(target)` gives every node the same reliability:

```python
three_in_series.equal_allocation(0.95)   # {'a': 0.98305, 'b': 0.98305, 'c': 0.98305}
three_in_series.equal_allocation(0.95)["a"]   # -> 0.98305   = 0.95 ** (1/3)
```

### Proportional improvement (ARINC-style)

`improvement_allocation(target, node_probabilities, fixed=None,
weights=None)` starts from the current reliabilities and scales every
node's **failure probability** by one common factor until the system meets
the target. Nodes that are already good stay proportionally good. This is the
ARINC apportionment, generalised: ARINC scales every failure *rate* by a
common factor, which is the same for small failure probabilities, and it
works on any structure:

```python
current = {"a": 0.99, "b": 0.97, "c": 0.90}
three_in_series.improvement_allocation(0.95, current)
# {'a': 0.99639, 'b': 0.98917, 'c': 0.96389}: every failure probability × 0.361
```

- `fixed` lists nodes that cannot change (they keep their current value, and
  the others make up the difference).
- `weights` makes some nodes improve faster: node *i*'s failure probability
  is scaled by `exp(−x · weight_i)` for a common `x`.
- A node missing from `node_probabilities` starts at 0.5.

```python
three_in_series.improvement_allocation(0.95, current, fixed=["a"])
# {'a': 0.99, 'b': 0.99061, 'c': 0.96869}
three_in_series.improvement_allocation(0.95, current, fixed=["a"])["c"]   # -> 0.96869
```

`equal_allocation(target)` is `improvement_allocation` starting from 0.5 for
every node.

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
effort function has to be chosen. It is closed-form, so there is no solver
result in `rbd.res`, and it applies only to series systems: any other
structure raises `ValueError`.

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

On a structure that is not a series, only the cost-based method applies:

```python
from repyability import RBD

pumps_and_valve = RBD([("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")])
pumps_and_valve.cost_based_allocation(0.99, {"p1": 0.9, "p2": 0.9, "v": 0.9})
# {'p1': 0.98081, 'p2': 0.98081, 'v': 0.99036}: the valve must reach 0.99 on its own
```

### Least-squares allocation (a heuristic)

`simple_allocation(target, weights=None)` searches the node reliabilities
(on a logistic scale, starting from 0.5 each) for any combination whose
system reliability equals the target, minimising the squared shortfall.
`weights` scales how strongly each node is moved; with no weights, similar
nodes end up equal:

```python
three_in_series.simple_allocation(0.95)   # {'a': 0.98305, 'b': 0.98305, 'c': 0.98305}
three_in_series.simple_allocation(0.95, weights={"a": 1.0, "b": 1.0, "c": 3.0})
# {'a': 0.97468, 'b': 0.97468, 'c': 1.0}
```

The weighted answer shows the method's character: it finds *an* allocation
that meets the target, and heavily weighted nodes can be driven to
perfection. It is not a named method, and which of the many allocations that
meet the target it returns depends on the optimiser, so prefer the named
methods above.

`simple_allocation` checks the allocation it finds and raises `ValueError`
if it misses the target (by more than one part in a million). That happens
when the target is out of reach (a node with weight 0 stays at 0.5), and on
large systems: the search starts with every node at 0.5, which puts the
probability of, say, 30 nodes in parallel so near 1 that it cannot move.
`equal_allocation` and `improvement_allocation` are exact at any size.

All of them raise `ValueError` for a target outside [0, 1]. All but
`minimum_effort_allocation` work on any structure and keep the solver's
result in `rbd.res` for inspection. They treat the nodes as independent.
