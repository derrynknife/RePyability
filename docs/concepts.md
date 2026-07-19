# Concepts

The reference behind the numbers: what an RBD means, how the system quantities
are computed, and how to choose among the importance measures. The
[tutorial](tutorial.md) shows these in action; this page explains them.

## Reliability block diagrams

A **reliability block diagram** models a system as a directed graph from a
single input (source) to a single output (sink). Each intermediate node is a
component with a reliability model. The system **works** at time `t` if there is
a path of working components from input to output.

Two structures underlie everything:

- A **minimal path set** is a minimal set of components whose simultaneous
  working guarantees the system works. The system is up iff *at least one* path
  set is fully up.
- A **minimal cut set** is a minimal set of components whose simultaneous
  failure guarantees the system fails. The system is down iff *at least one* cut
  set is fully down.

```python
rbd.get_min_path_sets(include_in_out_nodes=False)
rbd.get_min_cut_sets()
```

Series, parallel, and *k*-out-of-*n* are just special cases: a series system is
one path set of every component (and each component its own cut set); a parallel
system is the reverse.

### How the system quantity is computed

Given each node's reliability, the system reliability is computed **exactly**,
not by simulation. RePyability evaluates the probability that at least one path
set is satisfied (or, equivalently with `method="c"`, that no cut set is), using
a pivotal (Shannon) decomposition with memoisation so repeated substructure is
not recomputed. The two methods return the same value; the path-set method is
the default because it avoids deriving the cut sets.

The identity that drives the decomposition — and the importance measures — is
**pivotal decomposition** around any node *A*:

```
R_sys = R_A · R_sys(A working) + (1 − R_A) · R_sys(A failed)
```

Nodes whose reliability is only available by simulation (a standby arrangement)
make the *system* non-analytic; those quantities fall back to Monte-Carlo (and
take a `seed`).

## Reliability vs availability

- **Reliability** `R(t)` — the probability the system has *never* failed by `t`.
  The right question for a mission or a non-repairable item. Lives on
  [`NonRepairableRBD`][repyability.NonRepairableRBD].
- **Availability** `A(t)` — the probability the system is *up at* `t`, allowing
  for repair. The right question for a serviced, long-running system. Lives on
  [`RepairableRBD`][repyability.RepairableRBD], which needs a repairability
  distribution per component. Long-run availability is closed-form
  (`mean_availability`); the time-resolved curve comes from a seeded
  discrete-event simulation (`availability`).

## Importance measures — which one, and why

An importance measure ranks components by *how much they matter* — but "matter"
has several meanings, and they disagree on purpose. All are available at time(s)
`t` and accept `working_nodes`/`broken_nodes` conditioning.

| Measure | Answers | Reach for it when |
|---|---|---|
| **Birnbaum** `birnbaum_importance` | How much does system reliability move per unit change in this node's reliability? (`∂R_sys/∂R_i`) | Ranking where a reliability improvement pays off most *right now*. |
| **Improvement potential** `improvement_potential` | How much system reliability could I gain by making this node perfect? (`R_sys(1_i) − R_sys`) | Bounding the upside of fixing one component. |
| **Risk achievement worth** `risk_achievement_worth` | How much worse does system *un*reliability get if this node fails? (`Q_sys(0_i)/Q_sys`) | Finding components you must *keep working* — surveillance/protection targets. |
| **Risk reduction worth** `risk_reduction_worth` | How much would perfecting this node reduce system unreliability? (`Q_sys/Q_sys(1_i)`) | Prioritising which single fix removes the most risk. |
| **Criticality** `criticality_importance` | Weighs Birnbaum by the node's own reliability relative to the system. | Ranking that accounts for how reliable each node already is, not just its structural leverage. |
| **Fussell–Vesely** `fussell_vesely` | What fraction of system-failure probability involves this node? | A cut-set-based culprit ranking; standard in PRA/PSA. |

Two more measures answer *design-time* and *data-targeting* questions rather
than ranking at an operating point:

- **Structural importance** `structural_importance` — Birnbaum with every node
  reliability set to ½. It is **model-free**: it depends only on the diagram, so
  you can rank redundancy needs *before any data exists*. Being purely
  structural, it is identical for a `NonRepairableRBD` and a `RepairableRBD` on
  the same layout.
- **Parameter sensitivity** `parameter_sensitivity` — the derivative of system
  reliability with respect to each node's *distribution parameters*
  (`∂R_sys/∂θ = birnbaum_i · ∂sf_i/∂θ`), computed numerically. Where Birnbaum
  says *which component* matters, this says *which fitted parameter* matters —
  so you know where more data would most change the answer.

A rule of thumb: **Birnbaum** for "where does an improvement help most",
**risk achievement worth** for "what must not be allowed to fail",
**structural importance** at the whiteboard, and **parameter sensitivity** when
deciding where to spend a testing budget.

## Condition-based evaluation

The measures above assume every component is new. The condition-based methods
instead take each component's *current life* `Xᵢ` and condition on it:

```
Rᵢ(x | Xᵢ) = Rᵢ(Xᵢ + x) / Rᵢ(Xᵢ)
```

This is the survival of a further `x` given the component has already reached
`Xᵢ`. Feeding the conditioned per-node reliabilities through the same exact
system computation gives `sf_given_state`; inverting it gives `remaining_life`
(remaining useful life); and evaluating the importance measures at the
conditioned reliabilities gives `importances_given_state`.

Because the structure is static and only the state changes, each sensor update
is a cheap, exact re-evaluation — the heterogeneous generalisation of the
`cs` (conditional survival) method, which conditions the *whole system* on one
age, to *per-component* ages. Only lifetime (time-varying) distributions age; a
fixed-probability component's reliability does not depend on `Xᵢ`.

## Scope

- **Fitting failure data to distributions lives in
  [surpyval](https://github.com/derrynknife/SurPyval), not here.** RePyability
  consumes fitted models (anything exposing `sf`/`ff`) as node inputs.
- **Plotting and dashboards live in the Reliafy app, not here.** RePyability is
  the computational engine; it returns numbers and typed result objects.
