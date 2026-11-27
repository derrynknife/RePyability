# Event trees: from an initiating event to each outcome

An event tree follows an **initiating event** (a loss of cooling, an
overpressure, a release) through the barriers that should stop it. Each
**heading** asks whether one barrier works on the demand: a safety
function, a protection layer, an operator's action. A **sequence** is one
path through the headings, each succeeding, failing or not asked, and an
**outcome** gathers the sequences that end alike. The frequency of an
outcome is the initiator's frequency times the probability of its
sequences (NASA's *Probabilistic Risk Assessment Procedures Guide*;
IEC 62502; LOPA, IEC 61511-3 Annex F).

[`EventTree`][repyability.EventTree] takes each heading as a
[`FaultTree`][repyability.FaultTree] (its top event, the barrier failing)
or as a probability (an independent protection layer), and works out
every sequence **exactly, where the barriers share basic events**.

## Why shared events matter

Barriers often need the same support: one power supply, one instrument
air header, one operator. Then a barrier failing says something about the
next one, and a sequence's probability is not the product of its
branches'. Each sequence here is written as one fault tree: its top is
the AND of each failed heading's top event and the NOT of each heading it
succeeds, over the headings' gates and their events, an event of one name
being one event in every heading. The tree with NOT gates is evaluated by
its decision diagram (see
[NOT gates](fault-trees.md#not-gates-non-coherent-trees)), so nothing is
approximated, and the sequences' frequencies sum to the initiator's.

## A loss of cooling

A loss of cooling occurs 0.1 times a year. It must be detected (an
operator, missing it with probability 0.01), then injection and later
recirculation must work. Each pump needs the one power supply:

```python
from repyability import EventTree, FaultTree

injection = FaultTree(
    {"no injection": ("or", ["pump A", "power"])},
    {"pump A": 0.05, "power": 0.01},
)
recirculation = FaultTree(
    {"no recirculation": ("or", ["pump B", "power"])},
    {"pump B": 0.1, "power": 0.01},
)
tree = EventTree(
    0.1,
    [
        ("detection", 0.01),
        ("injection", injection),
        ("recirculation", recirculation),
    ],
)
tree.shared_events            # -> frozenset({'power'})
```

By default every combination of the headings is a sequence of its own,
every heading succeeding first. The sequence in which detection works and
both pumping stages are lost:

```python
exact = tree.sequence_frequencies()
naive = tree.sequence_frequencies(approximation="independent")
path = ("success", "failure", "failure")
exact.frequencies[path]   # -> 1.480050e-03 a year
naive.frequencies[path]   # -> 6.420645e-04
```

By hand: injection and recirculation both fail when the power fails, or
when it does not and both pumps fail, so the sequence's frequency is
`0.1 × 0.99 × (0.01 + 0.99 × 0.05 × 0.1) = 1.480050e-3`. The product of
the branches, `0.1 × 0.99 × 0.0595 × 0.109 = 6.420645e-4`, counts the
power failure twice as independent and is 57% too low.

| Sequence (detection, injection, recirculation) | Exact, a year | Product of the branches |
|---|---|---|
| success, success, success | 8.379855e-02 | 8.296056e-02 |
| success, success, failure | 9.310950e-03 | 1.014894e-02 |
| success, failure, success | 4.410450e-03 | 5.248436e-03 |
| success, failure, failure | 1.480050e-03 | 6.420645e-04 |
| failure, success, success | 8.464500e-04 | 8.379855e-04 |
| failure, success, failure | 9.405000e-05 | 1.025145e-04 |
| failure, failure, success | 4.455000e-05 | 5.301450e-05 |
| failure, failure, failure | 1.495000e-05 | 6.485500e-06 |

The exact column sums to 0.1; the product's does too here, but only
because a heading's success and failure are each other's complement: each
sequence on its own is wrong, by up to a factor of 2.3.

## Paths, outcomes and headings not asked

Give the paths as `{path: outcome}`, each path a tuple with an entry per
heading: `"success"`, `"failure"`, or None where the path does not ask
the heading (once detection fails, nothing else is tried). The paths must
cover every combination exactly once; a gap or an overlap is refused,
naming a combination on none or on both.

```python
tree = EventTree(
    0.1,
    [("detection", 0.01), ("injection", injection),
     ("recirculation", recirculation)],
    {
        ("success", "success", "success"): "ok",
        ("success", "success", "failure"): "core damage",
        ("success", "failure", None): "core damage",
        ("failure", None, None): "core damage",
    },
)
outcomes = tree.outcome_frequencies()
outcomes.frequencies      # -> {'ok': 0.08379855, 'core damage': 0.01620145}
outcomes.sequences["core damage"]   # the three paths that end there
```

`sequence_probabilities` gives each path's probability given the
initiator, `sequence_frequencies` its frequency, and
`outcome_frequencies` the sums by outcome. Each takes `t` where an event
is a lifetime model (its probability of having failed by `t`) or a
repaired component (its probability of being down at `t`, its tree's
point unavailability).

## LOPA: protection layers alone

A layer of protection analysis is the simplest event tree: every heading a
probability of failing on demand, each layer asked only where the ones
before it failed.

```python
lopa = EventTree(
    0.2,                                   # overpressures a year
    [("alarm", 0.1), ("relief", 0.01), ("dike", 0.05)],
    {
        ("success", None, None): "safe",
        ("failure", "success", None): "safe",
        ("failure", "failure", "success"): "contained",
        ("failure", "failure", "failure"): "release",
    },
)
lopa.outcome_frequencies().frequencies["release"]   # -> 1e-05 a year
```

A heading given as a probability stands as a basic event named for the
heading, so it can be asked for its importance as any other event.

## The initiator as a fault tree

The initiator can be a fault tree of repaired events, whose long-run
frequency of its top event
([`system_failure_frequency`][repyability.FaultTree.system_failure_frequency])
is the initiator's. Its events are taken apart from the headings': the
headings are probabilities on demand, given that the initiator has
occurred.

## Reconciling with other tools

Many tools quantify a sequence from its failed headings alone, as the
probability that they fail together, and drop the successes (about 1 when
failures are rare). `approximation="rare_event"` gives that value, so a
result can be set beside such a tool's. It overstates each sequence, and
the sequences then sum to more than the initiator (here 0.01785 of core
damage a year for the exact 0.01620). `approximation="independent"` gives
the product of the branches. Both return the same result types, with the
approximation named in them.

`sequence_tree(path)` returns a sequence's fault tree itself, whose
[`prime_implicants`][repyability.FaultTree.prime_implicants] are the
sequence's smallest combinations of events occurring and not occurring
(its cut sets, with the successes it needs).

## Importance

`birnbaum_importance(outcome, t)` gives each basic event's Birnbaum
importance to an outcome's frequency: the outcome's frequency given that
the event occurred less that given that it did not, the rate at which the
frequency rises with the event's probability. `fussell_vesely(outcome,
t)` gives the share of the frequency the event's occurring takes part in,
`(F − F given the event did not occur) / F`, the form PSA codes report as
`1 − 1/RRW`. Both are exact, on each of the outcome's sequences' trees
conditioned on the event. An event that a success branch needs not to
occur can lower an outcome, so either can be negative.

```python
tree.fussell_vesely("core damage")
# -> {'pump A': 0.2722, 'power': 0.05225, 'pump B': 0.5747,
#     'detection': 0.05225}
```

## What an event tree refuses

- A heading tree with dynamic gates (PAND, SPARE, SEQ), as a NOT over an
  order of failures has no agreed meaning: give its top event probability
  as the heading, if it shares no event.
- A heading tree with common-cause groups: write each shared cause as a
  basic event of its own under the members it fails.
- An event with different models in two headings, named: an event of one
  name is one event.
- An FDEP gate that forces an event another heading has.

Drawing the tree (or a bow-tie) is left to the application that shows it.
