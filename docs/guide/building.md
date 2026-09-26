# Building an RBD

A reliability block diagram is a directed graph from one **input** node to one
**output** node, plus a model for every node in between. The system works
while at least one path of working nodes connects the input to the output.

## Edges and node models

Give the edges as `(from, to)` pairs and a dict mapping each intermediate node
to its model. Node names can be any hashable (strings, integers, tuples).

```python
import surpyval as surv
from repyability import NonRepairableRBD

#   s -> (pump1 | pump2) -> valve -> t
edges = [
    ("s", "pump1"), ("s", "pump2"),
    ("pump1", "valve"), ("pump2", "valve"),
    ("valve", "t"),
]
models = {
    "pump1": surv.Weibull.from_params([100, 2]),
    "pump2": surv.Weibull.from_params([100, 2]),
    "valve": surv.Weibull.from_params([200, 1.5]),
}
rbd = NonRepairableRBD(edges, models)
rbd.sf(50)   # -> 0.8393   system reliability at t = 50
```

The input and output nodes are inferred as the unique node with no
predecessors and the unique node with no successors, and need no model. You
can name them with `input_node=`/`output_node=`, but they must be that source
and sink: naming any other node raises `ValueError`. Either way, the diagram
must have exactly one source and one sink.

A node model is anything that exposes `sf(t)` and `ff(t)`:

| Model | Use it for |
|---|---|
| A surpyval parametric distribution (`Weibull`, `Exponential`, `LogNormal`, …) | An ordinary component with a fitted lifetime. |
| A surpyval non-parametric fit (`KaplanMeier`, `NelsonAalen`, …) | A component described directly by its data. |
| `surpyval.FixedEventProbability` | A component with a fixed probability of failure (a demand, a mission). |
| [`PerfectReliability`][repyability.PerfectReliability] / [`PerfectUnreliability`][repyability.PerfectUnreliability] | A node that never fails / has always failed (a junction, a placeholder). |
| [`StandbyModel`][repyability.StandbyModel], [`RepeatedStandbyNode`][repyability.RepeatedStandbyNode] | Standby redundancy (see [Redundancy models](redundancy-models.md)). |
| [`RepeatedNode`][repyability.RepeatedNode] | *n* identical independent copies in series or parallel. |
| [`LoadSharingModel`][repyability.LoadSharingModel] | Units that share a load and fail dependently. |
| [`RegressionNode`][repyability.RegressionNode] | A component whose life depends on its operating conditions (see [Condition-based evaluation](condition-based.md#covariate-dependent-components)). |
| Another `NonRepairableRBD` | A subsystem, nested as a single node. |

## k-out-of-n nodes

A node with `k` greater than 1 works only while at least `k` of its incoming
branches are working. Pass `k={node: k}`:

```python
from repyability import PerfectReliability

#   three pumps, of which two must run, feed a perfect junction "v"
unit = surv.Weibull.from_params([100, 2])
two_of_three = NonRepairableRBD(
    [("s", "a"), ("s", "b"), ("s", "c"),
     ("a", "v"), ("b", "v"), ("c", "v"),
     ("v", "t")],
    {"a": unit, "b": unit, "c": unit, "v": PerfectReliability},
    k={"v": 2},
)
two_of_three.sf(50)   # -> 0.8749   = 3p² − 2p³ with p = unit.sf(50)
```

`k` must be a positive integer no larger than the node's number of incoming
edges; `k` equal to that number is allowed but warned about in the structure
check (it is just a series arrangement).

## One component in several places

Some diagrams need the *same physical component* in more than one place: a
power supply feeding two otherwise separate branches, say. Draw it as two
nodes and point the second one's entry at the first node's name instead of at
a model. The RBD then treats both as one component, which fails once for
both:

```python
from surpyval import FixedEventProbability

#   s -> ps  -> a -> t
#   s -> ps2 -> b -> t        ps2 is the same supply as ps
shared_supply = NonRepairableRBD(
    [("s", "ps"), ("ps", "a"), ("a", "t"),
     ("s", "ps2"), ("ps2", "b"), ("b", "t")],
    {
        "ps": FixedEventProbability.from_params(0.05),
        "ps2": "ps",
        "a": FixedEventProbability.from_params(0.1),
        "b": FixedEventProbability.from_params(0.1),
    },
)
shared_supply.sf()        # -> 0.9405   = 0.95 × (1 − 0.1²)
shared_supply.repeated    # {'ps2': 'ps'}
```

Treating the two supply nodes as independent would give `(1 − 0.05 · 0.1)² =
0.99`, a large over-estimate. A repeated node cannot be forced working or
broken on its own, and cannot be a member of a common-cause group. This is
different from [`RepeatedNode`][repyability.RepeatedNode], which models *n
distinct, identical* copies.

## Nested RBDs

A `NonRepairableRBD` can be a node of another, so a subsystem can be built and
tested once and reused:

```python
pair = NonRepairableRBD(
    [("s", "x"), ("s", "y"), ("x", "t"), ("y", "t")],
    {"x": unit, "y": unit},
)
plant = NonRepairableRBD(
    [("s", "pumps"), ("pumps", "valve"), ("valve", "t")],
    {"pumps": pair, "valve": surv.Weibull.from_params([300, 1.5])},
)
plant.sf(50)          # -> 0.8885
plant.node_names()    # ['pumps', 'valve']
```

A `RepairableRBD` can likewise contain `RepairableRBD` components (see
[Repairable systems](repairable.md#nested-repairable-rbds)).

## Validation and the structure check

The constructor checks the diagram and, by default, raises `ValueError` if it
is not a valid RBD. It checks for:

- cycles;
- more than one node without predecessors, or more than one without
  successors;
- a `k` of zero, or a `k` larger than the node's number of inputs, or a `k`
  given for a node that is not in the graph;
- a node without a model.

Pass `on_infeasible_rbd="warn"` to get a warning instead, or `"ignore"` to
build it silently, and inspect the findings in `structure_check`:

```python
broken = NonRepairableRBD(
    [("s", "a"), ("a", "t"), ("b", "t")],   # "b" is a second source
    {"a": unit, "b": unit},
    on_infeasible_rbd="ignore",
)
broken.structure_check["is_valid"]                     # False
broken.structure_check["nodes_with_no_predecessors"]   # ['s', 'b']
```

`structure_check` is a dict of findings: `is_valid`, `has_cycles` and
`cycles`, `nodes_with_no_predecessors` / `nodes_with_no_successors`,
`koon_errors` and `koon_warnings`, `irrelevant_nodes`,
`all_distributions_fixed`, and `is_analytically_solvable` with
`non_analytic_nodes`. An RBD built with errors can give meaningless results;
use `"warn"`/`"ignore"` to diagnose a diagram, not to analyse it.

## Path sets, cut sets and irrelevant nodes

```python
two_of_three.get_min_path_sets(include_in_out_nodes=False)
# {frozenset({'a', 'b', 'v'}), frozenset({'a', 'c', 'v'}),
#  frozenset({'b', 'c', 'v'})}
two_of_three.get_min_cut_sets()
# {frozenset({'a', 'b'}), frozenset({'a', 'c'}),
#  frozenset({'b', 'c'}), frozenset({'v'})}
```

- `get_min_path_sets(include_in_out_nodes=True)` returns the minimal path
  sets: minimal sets of nodes whose working guarantees the system works. By
  default each set includes the input and output nodes.
- `get_min_cut_sets(include_in_out_nodes=False)` returns the minimal cut
  sets: minimal sets of nodes whose failure guarantees the system fails.
- `get_all_path_sets()` iterates over every simple path from input to output
  (minimal or not). The number of paths can grow very fast with the size of
  the diagram.
- `find_irrelevant_components()` returns the nodes that appear in no minimal
  path set: their state never changes whether the system works, which usually
  means an edge is missing or misplaced.

```python
#   b only ever leads into a, which is reachable directly
redundant_b = NonRepairableRBD(
    [("s", "a"), ("a", "t"), ("s", "b"), ("b", "a")],
    {"a": unit, "b": unit},
)
redundant_b.find_irrelevant_components()   # {'b'}
list(redundant_b.get_all_path_sets())      # [['s', 'a', 't'], ['s', 'b', 'a', 't']]
```

Path and cut sets depend only on the structure and are computed once per RBD.

## Is the system time-dependent, and is it exact?

- `is_fixed` is `True` when every node is a fixed probability. Then time plays
  no part: `sf()` can be called without a time, and methods that invert
  reliability to a time raise. `is_time_varying` is its complement.
- `is_analytically_solvable()` is `False` when a node's reliability comes
  from a simulation-backed model: a `StandbyModel`, `RepeatedStandbyNode` or
  `LoadSharingModel` (always counted as non-analytic, even when that model
  has a closed form), or a nested RBD containing one.
  `get_non_analytic_nodes()` names them. The system value is still computed
  exactly *from* the node reliabilities; it is only as good as those nodes'
  own estimates.

```python
shared_supply.is_fixed                      # True
plant.is_fixed                              # False
from repyability import StandbyModel
standby = NonRepairableRBD(
    [("s", "sb"), ("sb", "t")], {"sb": StandbyModel([unit, unit])}
)
standby.is_analytically_solvable()          # False
standby.get_non_analytic_nodes()            # {'sb': 'StandbyModel'}
```

## Low-level building blocks

These are what the higher-level methods are made of; they are public for
custom analyses.

- `is_system_working(component_status, method)` evaluates the structure
  function: given `{node: True/False}` for every node, is the system up?
- `system_probability(node_probabilities, method="p")` is the exact engine:
  given each node's probability of working (a number or an array, all the
  same length), it returns the probability the system works.
- `path_set_probabilities(node_probabilities)` returns an array holding, for
  each minimal path set, the product of its members' probabilities. The
  entries are not labelled and come in no fixed order; pair them with the
  sets yourself if you need to know which is which.

```python
status = {"pump1": False, "pump2": True, "valve": True}
rbd.is_system_working(status, "p")                                  # True
rbd.system_probability({"pump1": 0.9, "pump2": 0.9, "valve": 0.95})  # array([0.9405])
```
