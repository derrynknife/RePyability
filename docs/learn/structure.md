# Lesson 3: Paths, cuts and the exact engine

!!! abstract "In this lesson"
    You will learn:

    - why some systems cannot be reduced to series and parallel pieces;
    - the **structure function**, which defines a system's reliability as a
      sum over every combination of working and failed components;
    - **minimal path sets** and **minimal cut sets**, and what they reveal,
      such as single points of failure;
    - why path probabilities cannot be added, and the **rare-event
      approximation** that safety analysts use instead;
    - the **pivotal decomposition**, and how RePyability builds its exact
      engine on it.

    **Before you start:** [Lesson 2](systems.md) (series, parallel and the
    reduction of series-parallel systems). About 30 minutes.

## The question

A cooling system has two pumps, `a` and `b`, each feeding its own heat
exchanger, `c` and `d`. A cross-tie valve, `e`, joins the two lines, so that
either pump can feed either exchanger through it. Each of the five
components works (over the period of interest) with probability 0.9. What is
the probability that the plant gets cooling?

```mermaid
flowchart LR
    s((in)) --> a[pump a] & b[pump b]
    a --> c[exchanger c]
    b --> d[exchanger d]
    a --> e[cross-tie e]
    b --> e
    e --> c & d
    c & d --> t((out))
```

Try the reduction of [Lesson 2](systems.md). The pump `a` is not in series
with `c` (water from `a` can also leave through the cross-tie), nor in
parallel with anything. The cross-tie is neither in series nor in parallel
with any block. There is no group to replace by one block, so the reduction
stops before it starts. This arrangement is the classic **bridge**, and it is
common: cross-ties, ring mains and meshed networks all contain bridges.

In a block diagram the arrows say which way success can flow. The cross-tie
can carry water from pump `a`'s line to exchanger `d`, or from pump `b`'s line
to exchanger `c`, so it has arrows in from both pumps and out to both
exchangers:

```python
import itertools
import numpy as np
import surpyval as surv
from repyability import NonRepairableRBD

fail = surv.FixedEventProbability.from_params   # takes the FAILURE probability
bridge_edges = [
    ("in", "a"), ("in", "b"),       # two pumps
    ("a", "c"), ("b", "d"),         # each feeds its own exchanger
    ("a", "e"), ("b", "e"),         # either pump can feed the cross-tie
    ("e", "c"), ("e", "d"),         # which can feed either exchanger
    ("c", "out"), ("d", "out"),
]
bridge = NonRepairableRBD(bridge_edges, {n: fail(0.1) for n in "abcde"})
bridge.sf()   # -> 0.97848
```

RePyability has the answer: 0.97848. This lesson shows where it comes from,
and three ways of thinking about a system that work for any structure.

## The definition: the structure function

Describe the state of the components by a vector $x$, with $x_i = 1$ if
component $i$ works and $x_i = 0$ if it has failed. The **structure function**
$\varphi(x)$ is 1 if, in that state, the working components connect the input
to the output, and 0 otherwise.

With independent components, the probability of a particular state is the
product of $p_i$ for the working components and $q_i = 1 - p_i$ for the
failed ones. The system's reliability is the total probability of the states
in which it works:

$$
R = \sum_{x} \varphi(x) \prod_{i} p_i^{x_i}\, q_i^{1 - x_i}
$$

For two components in parallel, each working with probability 0.9, there are
four states:

| $x_1$ | $x_2$ | $\varphi(x)$ | Probability |
|---|---|---|---|
| 1 | 1 | 1 | $0.9 \times 0.9 = 0.81$ |
| 1 | 0 | 1 | $0.9 \times 0.1 = 0.09$ |
| 0 | 1 | 1 | $0.1 \times 0.9 = 0.09$ |
| 0 | 0 | 0 | $0.1 \times 0.1 = 0.01$ |

$R = 0.81 + 0.09 + 0.09 = 0.99$, the parallel formula of Lesson 2.

The same definition works for the bridge; it only needs 32 states. The
library can evaluate $\varphi$ for any state (`is_system_working`), so you
can carry out the definition literally:

```python
total, working_states = 0.0, 0
for states in itertools.product([True, False], repeat=5):
    state = dict(zip("abcde", states))
    probability = np.prod([0.9 if up else 0.1 for up in states])
    if bridge.is_system_working(state, method="p"):
        total += probability
        working_states += 1
working_states   # -> 16   of the 32 states work
total            # -> 0.97848
```

This **state enumeration** is exact and makes the meaning of system
reliability plain. It is also hopeless beyond a few dozen components: $n$
components have $2^n$ states, over a million million for 40 components. The
rest of the lesson finds better ways.

## Paths and cuts

Two kinds of component set describe a structure.

A **path set** is a set of components whose working guarantees that the
system works, whatever the others do. A **minimal path set** has no smaller
path set inside it: every component in it is needed. For the bridge they are
the four routes the water can take:

```python
sorted(sorted(p) for p in bridge.get_min_path_sets(include_in_out_nodes=False))
# [['a', 'c'], ['a', 'd', 'e'], ['b', 'c', 'e'], ['b', 'd']]
```

`{a, c}` is pump `a` through its own exchanger; `{a, e, d}` is pump `a`
through the cross-tie to exchanger `d`.

A **cut set** is a set of components whose failure guarantees that the
system fails. A **minimal cut set** has no smaller cut set inside it:

```python
sorted(sorted(c) for c in bridge.get_min_cut_sets())
# [['a', 'b'], ['a', 'd', 'e'], ['b', 'c', 'e'], ['c', 'd']]
```

Both pumps; both exchangers; or one pump, the cross-tie and the *other* line's
exchanger. The two lists are dual: a cut set must break every path, so it
contains at least one component from each minimal path set, and a path set
contains at least one component from each minimal cut set.

Cut sets are how engineers find weak spots. A minimal cut set with a single
component is a **single point of failure**: that component alone can fail the
system. The bridge has none, which is what the cross-tie buys. The pumps and
valve of the running example have one:

```python
pumps_valve = NonRepairableRBD(
    [("in", "p1"), ("in", "p2"), ("p1", "v"), ("p2", "v"), ("v", "out")],
    {"p1": fail(0.1), "p2": fail(0.1), "v": fail(0.05)},
)
sorted(sorted(c) for c in pumps_valve.get_min_cut_sets())
# [['p1', 'p2'], ['v']]
```

Series and parallel systems are the two extremes. A series system of $n$
components has one minimal path set (all of them) and $n$ minimal cut sets of
one component each. A parallel system is the reverse.

## Why you cannot add paths

"The system works if at least one minimal path set works." It is tempting to
add up the probabilities of the paths:

| Path | Probability that every component in it works |
|---|---|
| $\{a, c\}$ | $0.9^2 = 0.81$ |
| $\{b, d\}$ | $0.81$ |
| $\{a, e, d\}$ | $0.9^3 = 0.729$ |
| $\{b, e, c\}$ | $0.729$ |
| **Sum** | **3.078** |

A "probability" of 3.078 is impossible. The paths share components, so their
events overlap: a state in which every component works is counted four times.
The probability of a union is the sum of the probabilities *minus* the
overlaps. For two events,

$$
\Pr(A \cup B) = \Pr(A) + \Pr(B) - \Pr(A \cap B).
$$

With the two routes `{a, c}` and `{b, d}` alone (the cross-tie failed), the
overlap is both routes working, $0.81 \times 0.81$:
$0.81 + 0.81 - 0.6561 = 0.9639$. For four overlapping paths, the full
**inclusion–exclusion** formula has fifteen terms, and for real systems it
runs to millions.

### The rare-event approximation

Safety analysts, who work with fault trees and cut sets, use the other side.
The system fails if at least one minimal cut set has entirely failed, and
when failures are rare the overlaps between cut-set events are tiny, so

$$
Q = 1 - R \approx \sum_{\text{minimal cut sets } C}\ \prod_{i \in C} q_i .
$$

For the bridge, $0.1^2 + 0.1^2 + 0.1^3 + 0.1^3 = 0.022$, against the exact
$1 - 0.97848 = 0.02152$: 2% too high. Make the components ten times better
($q = 0.01$) and the approximation gives $0.000202$, against the exact
$0.00020195$: 0.02% too high.

```python
better = NonRepairableRBD(bridge_edges, {n: fail(0.01) for n in "abcde"})
better.ff()                    # -> 0.00020195
2 * 0.01**2 + 2 * 0.01**3      # -> 0.000202   the rare-event approximation
```

The approximation always errs on the high side (the sum of the probabilities
of events is never less than the probability of their union), so it is a
safe, conservative number, and a good one when the $q_i$ are small. It is
also the basis of the Fussell–Vesely importance of [Lesson 4](importance.md).
RePyability does not need it: it computes the exact value.

## Pivotal decomposition: divide and conquer

The idea that cracks the bridge is to *condition* on one awkward component.
Either the cross-tie works or it has failed. By the law of total probability
(see the [refresher](index.md#a-probability-refresher)),

$$
R = p_e\, R(\text{system} \mid e \text{ works})
  + q_e\, R(\text{system} \mid e \text{ failed}).
$$

Each conditional system is series-parallel, and Lesson 2 solves it:

- **Cross-tie working.** Either pump can reach either exchanger, so the
  system needs at least one pump and at least one exchanger:
  $(a \parallel b)$ in series with $(c \parallel d)$, giving
  $(1 - 0.1^2)(1 - 0.1^2) = 0.9801$.

    ```mermaid
    flowchart LR
        s((in)) --> a[pump a] & b[pump b]
        a & b --> c[exchanger c] & d[exchanger d]
        c & d --> t((out))
    ```

- **Cross-tie failed.** The two lines are separate: $(a - c)$ in parallel
  with $(b - d)$, giving $1 - (1 - 0.81)^2 = 0.9639$.

    ```mermaid
    flowchart LR
        s((in)) --> a[pump a] --> c[exchanger c] --> t((out))
        s --> b[pump b] --> d[exchanger d] --> t
    ```

Weighting the two cases by the cross-tie's probabilities:

$$
R = 0.9 \times 0.9801 + 0.1 \times 0.9639 = 0.88209 + 0.09639 = 0.97848.
$$

RePyability can hold any component working or failed, so each case can be
checked separately:

```python
bridge.sf(working_nodes=["e"])   # -> 0.9801   cross-tie working
bridge.sf(broken_nodes=["e"])    # -> 0.9639   cross-tie failed
0.9 * bridge.sf(working_nodes=["e"]) + 0.1 * bridge.sf(broken_nodes=["e"])   # -> 0.97848
```

This is the **pivotal decomposition** (also called the Shannon expansion),
with `e` as the pivot. It works for any component and any structure: pivot on
a component, and each of the two smaller systems has one component fewer to
worry about. Pivoting on a different component gives the same answer by a
different route (exercise 3). It also underlies the importance measures of
[Lesson 4](importance.md): the difference between the two cases,
$R(e \text{ works}) - R(e \text{ failed})$, is the cross-tie's Birnbaum
importance.

## How RePyability computes exactly

RePyability's engine applies the pivotal decomposition over and over, to the
minimal path sets:

1. Pick the component that appears in the most path sets, and split into two
   cases.
2. If it works, remove it from every path set that contains it (it no longer
   needs to be satisfied). If it has failed, delete every path set that
   contains it (those paths are broken).
3. Repeat on each case until it is trivial: a path set that has become empty
   means the system works (probability 1); no path sets left means it has
   failed (probability 0).

Many branches of this tree lead to the same sub-problem, and the engine
solves each one only once and reuses its answer. The whole decomposition
depends only on the diagram, not on the probabilities, so it is worked out
once per RBD and then replayed for any probabilities: every time in an array,
every importance measure, every step of an allocation search. The result is
exact, with no approximation and no simulation.

You can hand the engine any component probabilities directly:

```python
bridge.system_probability({"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9, "e": 0.9})
# array([0.97848])
bridge.system_probability({"a": 0.95, "b": 0.8, "c": 0.9, "d": 0.99, "e": 0.5})[0]   # -> 0.979425
```

The same computation can run over the minimal cut sets with the components'
unreliabilities (`method="c"`); the two give the same value.

!!! note "What exactness costs"
    The engine's work grows with the number of minimal path sets and how
    they overlap, not with $2^n$. For most real diagrams that is modest:
    hundreds of components in series or in parallel are routine. Densely
    meshed structures are the hard case, because their path sets multiply. A
    chain of $k$ bridges in series has $4^k$ minimal path sets (4, 16, 64,
    ...), so long chains of meshes become expensive to evaluate exactly. The
    [performance notes](../guide/saving.md#performance) in the guide say
    more.

## Pitfalls

!!! warning "Common mistakes"
    - **Adding path probabilities.** Paths overlap; their probabilities can
      sum to more than 1. Use the union, not the sum.
    - **Forcing a structure into series and parallel.** Cross-ties, shared
      supplies and ring mains usually are not series-parallel. Draw the
      diagram as it is, and let the exact engine evaluate it.
    - **Drawing one physical component as two independent blocks.** If one
      power supply feeds both pumps, it is a single component that appears
      in several path sets, not two independent copies; treating it as two
      makes the system look far more reliable than it is. RePyability lets
      [one component appear in several places](../guide/building.md#one-component-in-several-places).
    - **Trusting the rare-event approximation for unreliable parts.** It is
      conservative, and accurate only when failure probabilities are small.

## Summary

!!! success "Key ideas"
    - The structure function $\varphi(x)$ defines system reliability as the
      total probability of the working states; enumerating the $2^n$ states
      is exact but only feasible for tiny systems.
    - Minimal path sets are the minimal ways to succeed; minimal cut sets the
      minimal ways to fail. A cut set of one component is a single point of
      failure.
    - Path probabilities cannot be added (the paths overlap). The rare-event
      approximation, the sum over minimal cut sets of the product of their
      unreliabilities, is conservative and accurate when failures are rare.
    - The pivotal decomposition,
      $R = p_i R(i \text{ works}) + q_i R(i \text{ failed})$, splits any
      system into two simpler ones.
    - RePyability applies it recursively to the path sets, reusing repeated
      sub-problems, and computes every system reliability exactly.

## Exercises

**1.** List the minimal path sets and minimal cut sets of the pumps-and-valve
system (two pumps in parallel feeding a valve). Which component is a single
point of failure?

??? success "Answer"
    Path sets: `{p1, v}` and `{p2, v}`. Cut sets: `{v}` and `{p1, p2}`. The
    valve is a cut set on its own: a single point of failure.

    ```python
    sorted(sorted(p) for p in pumps_valve.get_min_path_sets(include_in_out_nodes=False))
    # [['p1', 'v'], ['p2', 'v']]
    len(pumps_valve.get_min_cut_sets())   # -> 2
    ```

**2.** A system has two minimal path sets, `{a, b}` and `{a, c}`, and every
component works with probability 0.9. Find its reliability by
inclusion–exclusion, and check it by reducing the diagram.

??? success "Answer"
    Both paths work only when `a`, `b` and `c` all work, so
    $R = 0.81 + 0.81 - 0.9^3 = 0.891$. The diagram is `a` in series with
    `b` and `c` in parallel: $0.9 \times (1 - 0.1^2) = 0.891$.

    ```python
    shared = NonRepairableRBD(
        [("in", "a"), ("a", "b"), ("a", "c"), ("b", "out"), ("c", "out")],
        {n: fail(0.1) for n in "abc"},
    )
    shared.sf()   # -> 0.891
    ```

**3.** Solve the bridge again, pivoting on pump `a` instead of the
cross-tie. Do you get the same answer?

??? success "Answer"
    With `a` working, the system works if `c` works, or if `d` works and
    either `b` or `e` does: $1 - 0.1 \times [1 - 0.9 \times (1 - 0.1^2)]
    = 0.9891$. With `a` failed, only `b` can pump, through `d` or through
    the cross-tie to `c`: $0.9 \times [1 - 0.1 \times (1 - 0.81)] =
    0.8829$. Then $R = 0.9 \times 0.9891 + 0.1 \times 0.8829 = 0.97848$, the
    same: any pivot works.

    ```python
    bridge.sf(working_nodes=["a"])   # -> 0.9891
    bridge.sf(broken_nodes=["a"])    # -> 0.8829
    ```

**4.** Every component of the bridge now works with probability 0.99. How
far off is the rare-event approximation? And if they work with probability
0.7?

??? success "Answer"
    At 0.99 the approximation, 0.000202, is 0.02% above the exact 0.00020195.
    At 0.7 it gives $2 \times 0.09 + 2 \times 0.027 = 0.234$, against an
    exact 0.1984: 18% too high. The approximation is only good when failures
    are rare.

    ```python
    poor = NonRepairableRBD(bridge_edges, {n: fail(0.3) for n in "abcde"})
    poor.ff()                      # -> 0.19836
    2 * 0.3**2 + 2 * 0.3**3        # -> 0.234
    ```

**5.** How many states would state enumeration need for a system of 40
components? Why is the exact engine not limited in the same way?

??? success "Answer"
    $2^{40} \approx 1.1 \times 10^{12}$ states. The engine never lists
    states: it pivots on components and works on the path sets, whose
    number depends on the structure. A series or parallel system of 40
    components has 1 or 40 path sets, and is solved instantly.

    ```python
    2**40   # -> 1099511627776
    ```

## Where next

- [Lesson 4: Which component matters?](importance.md) uses the pivotal
  decomposition to measure how much each component matters.
- In the user guide, [Building an RBD](../guide/building.md) covers every way
  to describe a structure, including path and cut sets, and
  [Concepts](../concepts.md#how-the-system-quantity-is-computed) summarises
  the exact engine.
