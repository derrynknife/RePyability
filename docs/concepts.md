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

## Covariate-dependent and time-varying reliability

A component's reliability often depends on the *conditions it runs under*, not
only on elapsed time. If you have fitted a **regression** survival model in
surpyval — accelerated-failure-time (AFT), proportional-hazards (Cox, PH),
proportional-odds (PO), … — a [`RegressionNode`][repyability.RegressionNode]
uses it as an ordinary RBD node.

**Fixed covariates.** Pin the component's operating point `Z` (temperature,
load, duty cycle) and its reliability is the model's survival there:

```
R(x) = model.sf(x, Z)
```

That is a single univariate curve, so the node takes part in system reliability,
importance, MTTF and the condition-based (`age`) layer with no special handling
— a hotter-running unit is just a node with different covariates. This is
family-agnostic: AFT, PH and PO all expose `sf(x, Z)`.

**A time-varying covariate path.** When the load itself changes over the
component's life, the reliability is no longer the survival at one covariate
value but the survival *along the whole path* `Z(t)`. For a piecewise-constant
path (a surpyval `StepSchedule`) this is

```
R(x) = model.sf_tvc(x, schedule)
```

the probability of surviving each segment in turn under its own covariate.
Conditioning on an `age` needs no special case, because

```
R(x | age) = sf_tvc(age + x) / sf_tvc(age) = sf_tvc(x, schedule, given=age)
```

is exactly the go-forward survival from the component's current life under the
schedule — the load-dependent-aging / "digital twin" node. Whether a family
composes along a path is a property of the model: **AFT** (the path rescales the
clock) and **proportional-/additive-hazards** (the path accumulates hazard) do;
**proportional-odds** does not, and is refused in schedule mode. The
fixed-covariate node is the special case of a constant path.

## Dependent failures: load sharing

Redundant units that *share a load* do not fail independently. While all are up
each carries its share; when one fails the survivors pick up the slack, run
harder, and age faster — so the failures are positively correlated, and treating
them as `n` independent parallel nodes over-counts the redundancy.

A [`LoadSharingModel`][repyability.LoadSharingModel] captures the coupling as a
single node. Each unit is a fitted AFT model with **load as its covariate**, so a
unit running under load `ℓ` ages on its baseline clock at an acceleration factor
`φ(ℓ)` (the AFT time-scaling). With `s` survivors sharing a total load `L`, each
carries `L / s` and ages at `φ(L / s)`; as siblings fail, `s` falls, `L / s`
rises, and the survivors' clocks speed up. The group works while at least `k` of
the `n` units survive. This is the **cumulative-exposure** model: a unit's
*virtual age* is the integral of `φ(load(t))` over real time, and it fails when
that virtual age reaches its baseline failure age.

Two regimes:

- **Closed form.** Identical units with an **Exponential** baseline give a group
  lifetime that is a sum of exponential stages — each stage the time for the
  next unit to fail at the current shared load — i.e. a **hypoexponential**
  distribution, evaluated exactly with no simulation (`is_simulated == False`).
- **Simulation.** Otherwise the survival curve is a Kaplan–Meier fit to
  lifetimes drawn from the cumulative-exposure event loop (seeded;
  `is_simulated == True`).

As a check, with no load effect (`φ ≡ 1`) the survivors do not accelerate and the
group reduces *exactly* to the ordinary k-out-of-n parallel result. Load sharing
is the "self-loading" sibling of the condition-based layer: there the load is
streamed in from sensors, here it is computed from the group's own survivors.

## Common-cause failures

Redundancy only buys reliability if the redundant units fail for *independent*
reasons. In practice they often share a root cause — a common manufacturing
batch, a shared power supply, one miscalibration applied to every unit — and a
single event takes them all down together. Because the exact engine assumes
independence, it **over-estimates** a redundant group; a common-cause model
injects the shared coupling. (This is the mirror image of load sharing: there the
coupling is mechanical load transfer, here it is a shared shock.)

A [`CCFGroup`][repyability.CCFGroup] declares the coupled (symmetric) members and
the model, passed via `ccf_groups`. Two models:

- [`BetaFactor(beta)`][repyability.BetaFactor] — a fraction `β` of each unit's
  failure probability `Q` comes from a cause shared across the **whole** group
  (which fails every member at once); the remaining `(1 − β) Q` is independent.
  The workhorse of probabilistic-risk assessment.
- [`MGL(beta, gamma, ...)`][repyability.MGL] — the **Multiple Greek Letter**
  model, which also resolves *partial* common causes (a cause failing some but
  not all of the group) through a cascade of conditional probabilities:
  `β = P(shared by ≥ 2 | failed)`, `γ = P(≥ 3 | ≥ 2)`, and so on. The probability
  that a cause fails a *specific* set of `k` of the `m` members is the standard
  basic-event probability

```
Q_k = [ 1 / C(m−1, k−1) ] · (ρ₁ ρ₂ ⋯ ρ_k) · (1 − ρ_{k+1}) · Q
```

  with `ρ₁ = 1, ρ₂ = β, ρ₃ = γ, …, ρ_{m+1} = 0`; these partition each unit's `Q`
  exactly. A group of `m` members takes `m − 1` letters, and `MGL(β)` on two
  members is exactly `BetaFactor(β)`.

**The evaluation is exact, not a correction factor.** Each model's `decompose`
splits the group's failure into an independent part plus a set of
**mutually-exclusive shocks** (each a subset of members failing together). The
system reliability is then computed by conditioning on the shock outcome of every
group — in each branch the shocked members are down and the rest fail only
independently, so it is an ordinary independent system-reliability evaluation —
and blending the branches by their probabilities. Hence `β = 0` reproduces the
independent result exactly, and `β = 1` makes a redundant group no better than a
single unit.

Common cause is currently reflected in `sf()` / `ff()` (and quantities derived
from them) and persists through serialisation. Groups must be symmetric
(identical member models) and disjoint. The probability-dependent
importance/sensitivity and the condition-based methods do not yet account for it
and raise a clear error on a CCF RBD; `structural_importance`, being
probability-free, is unaffected. **Alpha-factor**, a data-estimable
reparameterisation of the same multiplicities, is a planned extension.

## Scope

- **Fitting failure data to distributions lives in
  [surpyval](https://github.com/derrynknife/SurPyval), not here.** RePyability
  consumes fitted models (anything exposing `sf`/`ff`) as node inputs.
- **Plotting and dashboards live in the Reliafy app, not here.** RePyability is
  the computational engine; it returns numbers and typed result objects.
