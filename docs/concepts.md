# Concepts

The reference behind the numbers: how RePyability computes each quantity,
what each model assumes, and which method to choose. The
[tutorial](tutorial.md) shows these in action and the
[user guide](guide/index.md) shows how to call them; this page explains how
they are worked out.

!!! tip "New to the theory?"
    This page is the reference for how RePyability computes each analysis.
    The [Learn](learn/index.md) course teaches the ideas themselves, step by
    step: each one worked out by hand with small numbers, then with
    RePyability, with exercises.

## Reliability block diagrams

A diagram's system works at `t` when working components connect its input
to its output, and its minimal path and cut sets describe that structure
(see [Systems](learn/systems.md#reliability-block-diagrams) and
[Paths and cuts](learn/structure.md#paths-and-cuts)). A *k*-out-of-*n* node
is up when at least `k` of its incoming branches are. A component in no
minimal path set is **irrelevant**: its state never decides whether the
system works.

The theory assumes a **coherent** system (repairing a component never makes
it worse) with **independent** components, unless a dependency is modelled
explicitly (a shared component, load sharing, standby, or a common-cause
group).

### How the system quantity is computed

Given each node's reliability, the system reliability is computed
**exactly**, not by simulation, in two stages.

1. **Reduction.** The diagram is reduced to *modules*: a series chain (a
   node whose only successor has it as its only predecessor), a parallel
   group (nodes with the same predecessors and successors, feeding nodes
   that need one working input) and a *k*-out-of-*n* group (all the inputs
   of a node with `k > 1`, when they share their own inputs) each become one
   block, with the closed forms `R = ∏ R_i`, `R = 1 − ∏ (1 − R_i)` and a sum
   over how many members work. A node that its own input bypasses by a
   direct edge is dropped, as it is irrelevant. This repeats until nothing
   more reduces; a series-parallel diagram becomes a single module.
2. **Pivotal decomposition.** Whatever is left, the *core* (a bridge, a
   shared node), is evaluated by a pivotal (Shannon) decomposition over its
   modules and nodes: from its minimal path sets when it has few, and
   otherwise from a binary decision diagram built from its graph, whose
   size grows with how wide the mesh is rather than with how many paths it
   has. Only the core pays the combinatorial price.

Both stages depend only on the structure, so they are worked out once per
RBD and replayed for every evaluation: repeated evaluations (arrays of times,
importance measures, allocation searches) cost little more than arithmetic.
A series-parallel diagram is never expanded into its path sets, however many
it has: thirty duplicated stages in series have `2^30` of them, and are
evaluated in milliseconds. With `method="c"` the probability that the system
fails is computed instead, and its complement returned; the two methods give
the same value. Every step is a sum of products of node probabilities and
their complements, so both keep their full relative precision.

A core so meshed that even its decision diagram would take more than
`repyability.rbd.bdd.STEP_LIMIT` steps to build (25 million, a few seconds)
is not worked out. The RBD is still built, with
`structure_check["is_too_meshed"]` set: its simulations follow the graph
itself (a node works when it has not failed and enough of its inputs work),
and the exact analyses refuse, saying why, as `analysis_routes()` reports.
Raise the limit to try harder.

Both stages rest on
[pivotal decomposition](learn/structure.md#pivotal-decomposition-divide-and-conquer),
which also gives the importance measures.

The engine is exact *given the node reliabilities*. A node with no exact or
numerical reliability (a standby or load-sharing arrangement only a
simulation works out) has none to give it: the analyses that need it
refuse, the system's simulations draw its lifetimes, and
`is_analytically_solvable()` flags it.
`analysis_routes()` says how each analysis is computed, and why.

The other time functions follow from the reliability: `F = 1 − R`, the
cumulative hazard `H = −ln R` (exact), the density `f = −dR/dt` (a numerical
derivative of the exact reliability) and the hazard `h = f / R`.

### Lifetimes by simulation

A coherent system fails when its last intact path set breaks, so its lifetime
is

```
T_sys = max over minimal path sets P of ( min over i in P of T_i )
```

`random()` draws each component's lifetime and applies this rule, through the
modules (a series module fails at its first failure, a parallel one at its
last, a *k*-out-of-*n* one when fewer than `k` are left) so that the path sets
are only needed for the core. `mean_time_to_failure()` integrates the exact
reliability by default. With `method="simulate"` it averages such
lifetimes, with standard error `s / √n`
([why](learn/systems.md#mean-time-to-failure-of-a-system)), which
`mean_time_to_failure_interval()` reports.

### Simulation error, and making it smaller

The error of a simulated mean shrinks like `1/√N`
([Lesson 6](learn/availability.md#precise-enough-sooner)). Given a
`tolerance`, a run adds `N` results at a time until the confidence
interval's half-width, `z · s / √N`, is at most the tolerance. (The
stopping point depends on the estimated `s`, which makes a sequential
rule's coverage slightly below the nominal level when `N` is small;
checking only after each batch of `N` keeps the effect small.)

Two classical variance-reduction techniques make the error smaller for the
same `N`, without biasing the estimate (the
[lesson](learn/availability.md#precise-enough-sooner) gives the intuition;
here is why they work):

- **Antithetic variates.** A result is a function `f(U)` of uniform random
  numbers. `f(U)` and `f(1 − U)` have the same distribution, and when `f` is
  monotone in each number (in either direction) their covariance is at most
  zero, so the mean of the pair varies at most half as much as one result.
  A coherent system's lifetime increases with every component's lifetime,
  and each lifetime with its uniform (by inverse transform), so pairing
  always helps a lifetime. The availability and cost of a window are not
  monotone in the draws (a longer up time moves a component's later
  repairs, which may then overlap another component's), so they gain less,
  though usually still a useful amount.
- **Common random numbers.** The difference of two designs' results has
  variance `Var(A) + Var(B) − 2 Cov(A, B)`. Simulated independently, the
  covariance is zero; driven by the same random numbers wherever the
  designs share a component, the results move together, and the covariance
  removes most of the variance.

In a `RepairableRBD` both work component by component: each component draws
each quantity (its times to failure, its repairs, ...) from a stream of its
own, keyed by the seed, its place in the diagram and the quantity, and
counter-based, so that the stream's `k`-th draw in simulation `r` (or pair
`r`) is fixed by those alone. Its `k`-th draw is then matched, or paired, however
the components' events interleave, and every simulation is the same however
the run is split up: over processes or threads, or in a run to a tolerance.

A parallel run of a `NonRepairableRBD` splits the lifetimes into blocks
seeded in turn from one `numpy.random.SeedSequence`, whose spawned seeds
give independent streams. The block, not the process that runs it, fixes
the random numbers, so the results do not depend on the number of
processes.

### Fault trees

A fault tree is the diagram's dual, with OR, AND and VOTE gates for series,
parallel and *k*-of-*n* blocks
([Lesson 3](learn/structure.md#the-same-logic-upside-down-fault-trees)). A
tree is evaluated by the same engine: each gate below which no event or gate is
shared with the rest of the tree is a module with a closed form, and what the
repeated events tie together is a core, solved exactly by the pivotal
decomposition, on a binary decision diagram built from its gates (its cut
sets are listed only when asked for). The measures of importance are the
diagram's, with the top event as the system failing.

### Parameter uncertainty

The node models are estimates, so the system's values carry *epistemic*
uncertainty ([Lesson 2](learn/systems.md#how-sure-are-you-of-the-inputs)).
`sf_uncertainty` propagates it by
Monte Carlo over the parameters: each draw gives every uncertain node a
plausible model and the system reliability is computed exactly, all draws
at once (the node probabilities are arrays over draws and times), and the
percentiles of the draws form an uncertainty interval. A maximum-likelihood
fit's own estimate of its parameters' uncertainty (the inverse Hessian of
the log-likelihood, from surpyval) gives the draws on a transformed scale
(log for a positive parameter, logit for one in (0, 1)), which is the delta
method's normal approximation. Nodes of one population share one draw (a
tuple of nodes), as they share their parameters. The same draws give the
MTTF's, a B*X* life's and the time to a reliability's uncertainty
(`mean_uncertainty`, `bx_life_uncertainty`,
`time_to_reliability_uncertainty`): each draw's value is the exact one for
its models, the area under its reliability or the root of its reliability
less the target. A repairable system's availability and
cost rate are uncertain in the same way, through its components' lives,
repairs and maintenance times: each draw rebuilds the diagram with its
models and works its value out as the diagram's own
(`mean_availability_uncertainty` and the rest). The draws can be
quasi-random (`sampling="sobol"`), the points of a scrambled Sobol
sequence, which cover the parameters more evenly than random draws and so
settle the summaries with fewer of them.

## Reliability vs availability

- **Reliability** `R(t)`: the probability the system has *never* failed by
  `t` ([Lesson 1](learn/lifetimes.md#the-time-to-failure-is-a-random-variable)).
  The right question for a mission or a non-repairable item. Lives on
  [`NonRepairableRBD`][repyability.NonRepairableRBD].
- **Availability** `A(t)`: the probability the system is *up at* `t`,
  allowing for repair ([Lesson 6](learn/availability.md#a-new-component-starts-up)).
  The right question for a serviced, long-running system. Lives on
  [`RepairableRBD`][repyability.RepairableRBD], which needs a repairability
  distribution per component.

## Importance measures — which one, and why

An importance measure ranks components by *how much they matter*, but
"matter" has several meanings, and they disagree on purpose. All are
available at time(s) `t` and accept `working_nodes`/`broken_nodes`
conditioning. Below, `R` is the system reliability, `Q = 1 − R`, `R_i` node
*i*'s reliability, and `R(1_i)`/`R(0_i)` the system reliability with node *i*
forced working/failed.

| Measure | Answers | Reach for it when |
|---|---|---|
| **Birnbaum** `birnbaum_importance` | How much does system reliability move per unit change in this node's reliability? `∂R/∂R_i = R(1_i) − R(0_i)` | Ranking where a reliability improvement pays off most *right now*. |
| **Improvement potential** `improvement_potential` | How much could I gain by making this node perfect? `R(1_i) − R` | Bounding the upside of fixing one component. |
| **Risk achievement worth** `risk_achievement_worth` | How much more likely is system failure if this node fails? `Q(0_i) / Q` | Finding components you must *keep working*: surveillance and protection targets. |
| **Risk reduction worth** `risk_reduction_worth` | By what factor would perfecting this node reduce system unreliability? `Q / Q(1_i)` | Prioritising which single fix removes the most risk. |
| **Criticality** `criticality_importance` | Given that the system has failed, how likely is it that this node has failed and is critical? `I_B · (1 − R_i) / Q`, its share of the system failures (`kind="success"` gives `I_B · R_i / R`, which is 1 for every node in series) | Ranking the culprits, series nodes included, by how much each contributes to system failure; on a repairable system, its share of the downtime. |
| **Fussell–Vesely** `fussell_vesely` | What fraction of system-failure probability involves this node? `P(some cut set C ∋ i has failed) / Q`, exactly (`method="rare_event"` gives the usual approximation `Σ_{cut sets C ∋ i} Π_{j ∈ C} (1 − R_j) / Q`) | A cut-set-based culprit ranking; standard in PRA/PSA. |

Two more answer *design-time* and *data-targeting* questions rather than
ranking at an operating point:

- **Structural importance** `structural_importance`: Birnbaum with every node
  at ½, which depends only on the diagram
  ([Lesson 4](learn/importance.md#structural-importance-before-you-have-any-data)).
- **Parameter sensitivity** `parameter_sensitivity`: the derivative of system
  reliability with respect to each node's *distribution parameters*,
  `∂R/∂θ = I_B · ∂R_i/∂θ`, computed numerically. Where Birnbaum says *which
  component* matters, this says *which fitted parameter* matters, so you know
  where more data would most change the answer.
- **Differential importance** `differential_importance` (DIM): each node's
  or parameter's share of the change in the system when they all change
  together, `I_i dθ_i / Σ_j I_j dθ_j`. The other measures do not add up; the
  shares do, so a group's share is the sum of its members': what share of a
  possible gain lies in the pumps, or in the repair times against the
  maintenance intervals. A uniform change (every `dθ` equal) shares out the
  Birnbaum importance, a proportional one (every `dθ / θ` equal) the
  criticality.
- **Uncertainty importance** `uncertainty_importance`: each uncertain
  input's share of the variance of a system quantity (the reliability, the
  MTTF, a B-life, the availability, the cost rate) over the fitted models'
  parameter uncertainty, by the
  delta method or as Sobol indices: where more data would narrow the answer
  most.
- **Joint importance** `joint_importance`: the second-order Birnbaum
  measure, `∂²R/∂R_i ∂R_j`, for each pair. Positive for complements
  (series: improving one makes improving the other worth more), negative
  for substitutes (parallel): whether to bundle two improvements.
- **Rates and Barlow–Proschan** `availability_rate`, `reliability_rate`,
  `barlow_proschan_importance`: with independent components the system's
  rate of change is the sum of each one's Birnbaum importance times its own
  rate, so it splits into what each component is doing to the system now.
  Integrated over time, a component's part gives the probability that it
  caused the system's failure (Barlow–Proschan), the exact counterpart of
  the simulated failure criticality index.

A rule of thumb: **Birnbaum** for "where does an improvement help most",
**risk achievement worth** for "what must not be allowed to fail",
**structural importance** at the whiteboard, and **parameter sensitivity**
when deciding where to spend a testing budget.

On a repairable system they take long-run availabilities
([Lesson 4](learn/importance.md#on-a-repairable-system)), or point
availabilities over time, from new or from the components' current states.

Read together, the sensitivity measures are the system's *Greeks*, named
after an option's: delta (Birnbaum), the levers' deltas (parameter
sensitivity), their shares (differential importance), gamma (joint
importance), theta (the rate of change and Barlow–Proschan) and vega
(uncertainty importance). [Sensitivities: the Greeks](guide/greeks.md)
runs one pumping station through all of them.

## Condition-based evaluation

The measures above assume every component is new. The condition-based methods
instead take each component's *current life* `Xᵢ` (a repairable system's
analyses over time take its components' states too: see
[From the present](#availability)). Each component's reliability is
conditioned on its age, `Rᵢ(Xᵢ + x) / Rᵢ(Xᵢ)`
([Lesson 1](learn/lifetimes.md#the-exponential-failures-that-ignore-age)).
Feeding the conditioned per-node reliabilities through the same exact
system computation gives `sf_given_state`; inverting it gives
`remaining_life` (remaining useful life); and evaluating the importance
measures at the conditioned reliabilities gives `importances_given_state`.

Because the structure is static and only the state changes, each sensor update
is a cheap, exact re-evaluation: the heterogeneous generalisation of the `cs`
(conditional survival) method, which conditions the *whole system* on one
age, to *per-component* ages. Only lifetime (time-varying) distributions age;
a fixed-probability component's reliability does not depend on `Xᵢ`.

## Covariate-dependent and time-varying reliability

A component's reliability often depends on the *conditions it runs under*,
not only on elapsed time. If you have fitted a **regression** survival model
in surpyval (accelerated failure time (AFT), proportional hazards (Cox, PH),
proportional odds (PO), …), a [`RegressionNode`][repyability.RegressionNode]
uses it as an ordinary RBD node.

**Fixed covariates.** Pin the component's operating point `Z` (temperature,
load, duty cycle) and its reliability is the model's survival there:

```
R(x) = model.sf(x, Z)
```

That is a single univariate curve, so the node takes part in system
reliability, importance, MTTF and the condition-based (`age`) layer with no
special handling: a hotter-running unit is just a node with different
covariates. This is family-agnostic: AFT, PH and PO all expose `sf(x, Z)`.

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

is exactly the go-forward survival from the component's current life under
the schedule: the load-dependent-ageing ("digital twin") node. Whether a
family composes along a path is a property of the model: **AFT** (the path
rescales the clock) and **proportional-/additive-hazards** (the path
accumulates hazard) do; **proportional odds** has no single natural
extension: surpyval switches to the new covariate's hazard at each step (the
survival does not jump to the new covariate's curve). The fixed-covariate
node is the special case of a constant path.

## Standby and repeated nodes

**Repeated nodes.** `n` independent, identical copies of a component with
reliability `p` have reliability `pⁿ` in series and `1 − (1 − p)ⁿ` in
parallel. A `RepeatedNode` is exactly that, as one node. It is not the same as
**one component drawn in several places** (a shared power supply feeding two
branches): that is a single component, which fails once for every place it
appears, and treating its appearances as independent copies over-states
reliability.

**Cold standby.** With one unit operating, cold spares add their lifetimes
([Lesson 5](learn/dependence.md#cold-standby)), so the arrangement's
survival function is the convolution of theirs. RePyability computes it by
numerical convolution (deterministic, no sampling). For identical Exponential
units the sum is Erlang, in closed form, for any `k` operating units.

**Imperfect switching.** If each switch-over succeeds with probability `s`,
the lifetime is the sum of the first `j` units' lifetimes with probability
that exactly `j − 1` switches succeeded before one failed (or all succeeded):
a mixture of partial sums, again computed by convolution.

**Warm and hot standby.** A warm spare ages at `κ` (the `dormancy_factor`)
of the operating rate
([Lesson 5](learn/dependence.md#warm-and-hot-standby)). It fails when its
virtual age reaches its baseline failure age, so it can fail before it is
switched in. Identical Exponential units give a hypoexponential lifetime,
exactly, each stage (from `j` to `j − 1` surviving units) exponential with
rate `λ (k + (j − k) κ)`. Hot standby (`κ = 1`) is *k*-out-of-*n* active
parallel, and is worked out so for any units. With one unit operating and
any units, a spare switched in at `τ` has aged `κτ` and runs until its
failure age, so the lifetime follows a recursion over the switch-ins, which is computed on a time grid; with two
units, `R(t) = S₁(t) + ∫₀ᵗ f₁(u) S₂(t − (1 − κ)u) du`.

With `k ≥ 2` operating units, cold, each operating position runs a renewal
process of the lives of the units put into it; for identical units these are
independent, so the number of failures by `t` is a sum of `k` renewal counts,
whose distributions come from the convolution, and the arrangement fails at
the `n − k + 1`-th. With different units, which unit goes where depends on
the order of failures. With two operating, after each failure the state is
its time and when the other operating unit started (the newcomer starts
new), a recursion on a grid of the two; with three or more, and warm with
`k ≥ 2` (the spares' ages too), they are simulated.

## Dependent failures: load sharing

Units that share a load fail sooner as their siblings fail, so independent
parallel nodes over-count the redundancy
([Lesson 5](learn/dependence.md#a-shared-load)).

A [`LoadSharingModel`][repyability.LoadSharingModel] captures the coupling as
a single node. Each unit is a fitted AFT model with **load as its covariate**,
so a unit running under load `ℓ` ages on its baseline clock at an acceleration
factor `φ(ℓ)` (the AFT time scaling). With `s` survivors sharing a total load
`L`, each carries `L / s` and ages at `φ(L / s)`; as siblings fail, `s` falls,
`L / s` rises, and the survivors' clocks speed up. The group works while at
least `k` of the `n` units survive. This is the **cumulative-exposure** model:
a unit's *virtual age* is the integral of `φ(load(t))` over real time, and it
fails when that virtual age reaches its baseline failure age.

Three regimes:

- **Closed form.** Identical units with an **Exponential** baseline give a
  group lifetime that is a sum of exponential stages (each stage the time for
  the next unit to fail at the current shared load), i.e. a
  **hypoexponential** distribution, evaluated exactly with no simulation
  (`is_simulated == False`).
- **Numerical.** Other identical units all age alike, so they fail in the
  order of their exposures to failure, and a recursion over the failures
  gives the lifetime's distribution (`is_simulated == False`).
- **Simulation only.** Different units have no survival curve: the group
  draws lifetimes from the cumulative-exposure event loop for the system's
  simulations, and the analyses that need its reliability refuse
  (`is_simulated == True`).

As a check, with no load effect (`φ ≡ 1`) the survivors do not accelerate and
the group reduces *exactly* to the ordinary *k*-out-of-*n* parallel result.
Load sharing is the "self-loading" sibling of the condition-based layer: there
the load is streamed in from sensors, here it is computed from the group's own
survivors. Warm standby uses the same virtual-age machinery with a fixed
dormant rate instead of a load-dependent one.

## Common-cause failures

A shared cause can fail redundant units together, which the independent
engine over-rates ([Lesson 5](learn/dependence.md#one-cause-every-unit)); a
common-cause model adds that coupling. (This is the mirror image of load
sharing: there the coupling is load transfer, here it is a shared shock.)

A [`CCFGroup`][repyability.CCFGroup] declares the coupled (symmetric) members
and the model, passed via `ccf_groups`. Two models:

- [`BetaFactor(beta)`][repyability.BetaFactor]: a share `β` of each
  member's `Q` fails the whole group at once, and the remaining `(1 − β) Q`
  is independent ([Lesson 5](learn/dependence.md#the-beta-factor-model)).
- [`MGL(beta, gamma, ...)`][repyability.MGL]: the **Multiple Greek Letter**
  model, which also resolves *partial* common causes (a cause failing some
  but not all of the group) through a cascade of conditional probabilities:
  `β = P(shared by ≥ 2 | failed)`, `γ = P(≥ 3 | ≥ 2)`, and so on. The
  probability that a cause fails a *specific* set of `k` of the `m` members
  is the standard basic-event probability

```
Q_k = [ 1 / C(m−1, k−1) ] · (ρ₁ ρ₂ ⋯ ρ_k) · (1 − ρ_{k+1}) · Q
```

  with `ρ₁ = 1, ρ₂ = β, ρ₃ = γ, …, ρ_{m+1} = 0`; the `Q_k` of the sets
  holding a unit sum to its `Q`, so, its own failure and the shared causes
  being separate events, it fails with probability `Q` to first order (as
  in PRA's basic events). A group of `m` members takes `m − 1` letters, and `MGL(β)` on
  two members is exactly `BetaFactor(β)`.

**The evaluation is exact, not a correction factor.** Each model's
`decompose` splits the group's failure into an independent part plus a set of
**mutually exclusive shocks** (each a subset of members failing together).
The system reliability is then computed by conditioning on the shock outcome
of every group (in each branch the shocked members are down and the rest fail
only independently, so it is an ordinary independent system-reliability
evaluation) and blending the branches by their probabilities. Hence `β = 0`
reproduces the independent result exactly, and `β = 1` makes a redundant
group no better than a single unit.

**By default the model assumes each member's `Q` is small.** "Exact" means
the evaluation of the model is exact; the default model is the PRA
basic-event one, which splits each member's failure *probability* (`βQ`
shared, `(1 − β)Q` independent) and is a rare-event model. Use it over
periods in which each member's failure probability stays small, such as a
mission or a proof-test interval. For a parallel pair with `β = 0.3`, the
system unreliability is within 0.3% of a rate-based beta-factor treatment at
`Q = 0.01`, 3.5% at `Q = 0.1` and about 10% at `Q = 0.3`. From about
`Q = 0.5` the pair comes out *more* reliable than an independent pair. Over
a whole life (`Q → 1`) the model stops describing a lifetime at all: under
the beta factor, each member would only ever fail with probability
`1 − β(1 − β)` (0.79 at `β = 0.3`). So a diagram warns, once for each group,
when a member's `Q` passes 0.1.

**Over a lifetime, split the rate.** With `basis="rate"` the model splits
each member's failure *rate*: in the members' cumulative hazard
`H(t) = −log R(t)`, the shared cause is a shock of hazard `β H(t)`, which has
not struck by `t` with probability `R(t)^β`, and each member's own causes
have hazard `(1 − β) H(t)`. A member fails at the first of its causes, so
its reliability is `R(t)^β · R(t)^(1 − β) = R(t)`: every member keeps its own
life distribution, whatever its shape, and the model holds over the whole
life. To first order in `Q` the shock strikes with `βQ` and each member fails
on its own with `(1 − β)Q`, as the probability split has it. Under MGL each
specific set of `k` members has a cause of its own, of hazard
`(Q_k / Q) H(t)`, striking independently of the others; several may strike,
so the outcomes are the sets the struck causes fail between them.

Common cause is reflected in `sf()` / `ff()` (and quantities derived from
them) and persists through serialisation, basis included. Groups must be
symmetric (identical member models) and disjoint. An MTTF integrates over the
whole life: the exact `mean()` integrates a group that splits the rate, and
refuses one that splits a probability, where `Q` is no longer small. The
Monte-Carlo `random()`, `mean(method="simulate")` and MTTF interval draw a
rate-split group's shocks (each cause strikes at an exponential time in
`H`), and sample the members of a probability-split group independently,
leaving the common cause out. The importance measures condition a member
on its state through the shock outcomes (a node outside the groups is held
working and failed, as without them); parameter sensitivity and parameter
uncertainty take a group's members together, with its model's parameters;
and a copy of a beta-factor group's member joins the group in a redundancy
allocation. The condition-based methods would need members of different
ages, and raise a clear error on a CCF RBD; `structural_importance`, being
probability-free, is unaffected.
**Alpha-factor**, a data-estimable reparameterisation of the same
multiplicities, is a planned extension.

A `RepairableRBD` takes `ccf_groups` too. There a member's failures are a
rate, so the model splits the rate: each cause, a member's own or a shared
one, strikes at its share of it and fails the members it names that are up.
Each member alone fails as before; which are down together is a Markov chain
of the group (its members repaired at exponential rates, or found by their
tests), and the long-run values average the structure function over its
states, exactly (see [Common-cause failures](guide/common-cause.md#repairable-systems)).

## Availability

Each component alternates up and down, as good as new after each repair,
independently of the others and of the system's state
([Lesson 6](learn/availability.md#up-down-up-again)).

**Long-run availability and frequencies.** The long-run availability is the
exact system probability at `A_i = MTTF_i / (MTTF_i + MTTR_i)`, and the
failure frequency is `ω = Σ_i I_B^i ω_i`, with `ω_i = 1 / (MTTF_i + MTTR_i)`
and `MTBF = 1 / ω`, `MUT = A / ω` and `MDT = (1 − A) / ω` from it, all exact
([derivations](learn/availability.md#a-failure-of-a-critical-component)). No
simulation is needed. An instantly repaired component (`MTTR = 0`) has
`A_i = 1`; a nested repairable RBD contributes its own system frequency.

**Availability over time.** From new, `point_availability` solves each
component's renewal equation
([Lesson 6](learn/availability.md#a-new-component-starts-up)) numerically,
on a grid of 1,000 steps over the component's typical up time (an error of
about `4e-7`). Components that fail and are repaired
independently are up or down independently at every time, so the system's
`A(t)` is its system probability at the `A_i(t)`, and
`mission_availability` is its mean over `[0, T]`. For a long mission that
mean is the long-run value plus about `b/T`: for one component
`b = A (E[C²]/(2E[C]) − E[U²]/(2E[U]))`, `C = U + D`, positive for a life
that wears out, and for a system `Σ_i I_B^i b_i` to first order.

**Expected events over a window.** The frequency formula holds at every
time, not only in the long run: a component failing at `t` fails the system
if it is critical then, with probability `I_B^i(t)` at the availabilities
`A_j(t)`. So the system's expected failures in `[0, T)` from new are

```
E[N(T)] = ∫₀ᵀ Σ_i I_B^i(t) dM_i(t)
```

with `M_i(t)` component `i`'s expected failures by `t`, its renewal
function, which the renewal equation gives alongside `A_i(t)`.
`expected_failures` and `expected_events` compute it, with the system's
planned outages counted the same way and the expected downtimes as
integrals of the unavailabilities; their long-run rates are `ω` and the
like. Events at exact times (replacements due at the same age or block
time) are taken together: they take the system down at most once.

**From the present.** Given each component's current state, only its first
period changes. A unit up at age `a` has a first life with survival
`R(a + s) / R(a)` (its replacement due at `T − a` under age replacement);
one down for `r` has a first down time with survival `G(r + s) / G(r)`, and
starts new after it. The later units are new: their renewals are the
first period's end convolved with the renewal measure from new, so
`A_i(t)`, `M_i(t)`, and everything built on them above, follow as before.
Under block replacement the unit's own curve runs to the first block time,
which hands the block intervals' recursion the probability that it is up
there (and replaced) and the repairs going on; a calendar is shifted by the
phase. A component long in service whose state is not known starts in its
long-run (stationary) state: off a calendar it stays at `A_i` throughout,
with its events at their long-run rates; on one, it follows its settled
cycle from its phase. In the simulation, the first draw is the inverse
transform of the conditional distribution, `qf(F(a) + u (1 − F(a))) − a`,
worked through the cumulative hazard so that it keeps its precision at
great ages: still one uniform per draw.

`availability()` simulates `mc_samples` independent histories instead (each
component's alternating failures and repairs, merged in time order, with the
system's state re-evaluated at every event) and reports the fraction of
histories up at each time. The histories also give what the exact methods
do not: how much the counts, downtimes and costs over the window vary, and
the criticality measures below. Each point's confidence band is the Wilson
score interval ([why](learn/availability.md#how-much-to-trust-a-simulation)).
For exponential components the simulation is held to the exact Markov
solution in the test suite, and on the benchmark diagrams to
`point_availability`.

A nested repairable RBD runs its own history on the same clock, and the outer
system sees a state change when the nested system's state changes.

**Criticality measures.** From the simulated histories:

- The **operational criticality index** relates time: of the time the system
  was down, the fraction a node was also down (and likewise for up time).
- **Intersection over union** compares a node's up (or down) intervals with
  the system's: the time both were up over the time either was up.
- The **failure criticality index** counts events: the fraction of system
  failures a node's failure caused (the failure that took the system from up
  to down), and the fraction of the node's own failures that did so.
- The **restoration criticality index** does the same for repairs that
  restored the system.

## Capacity

A reliability block diagram answers a yes-or-no question: does the system
work? A plant also asks how much it delivers. Give each node a capacity
`c_i`, the throughput it passes while it works (0 once it has failed), and
the system's capacity is the most that can flow from the input to the output
through the working nodes, each passing at most its capacity: the diagram's
**maximum flow**. The edges carry any amount, so

```
series:    C = min(c_1, c_2, ...)
parallel:  C = c_1 + c_2 + ...
```

and in general, by the **max-flow min-cut theorem**, the capacity is the
least total capacity of a cut, a set of nodes whose failure disconnects the
output:

```
C = min over cuts K of  Σ_{i in K} c_i · [node i works]
```

A k-out-of-n node passes flow only while at least `k` of its inputs are
reached, as in the reliability analysis, so the capacity is positive exactly
when the system works: `P(C > 0)` is the reliability (or availability).

**The distribution.** Over the components' states, `C` takes finitely many
values. The probability of each follows as the system probability does. A
module's distribution comes from its members' in closed form: the
distribution of the least of independent capacities for a series chain, of
their sum for a parallel group, and of their sum while at least `k` work for
a k-out-of-n group. Combining distributions this way is the **universal
generating function** of multi-state systems (Ushakov; Lisnianski and
Levitin). What is left, such as a bridge, is conditioned on its parts'
capacities one at a time, carrying only each cut's running total and the
least complete total, and merging the states that agree on them.

**What it gives.** From the distribution:

- `P(C ≥ d)`, the probability of meeting a demand `d`: the system's
  reliability for that demand (its availability, in the long run);
- `E[C]`, the expected capacity;
- `E[min(C, d)] / d`, the expected fraction of the demand delivered. In the
  long run this is the fraction of the demand met over time: the
  **production availability**, the figure plant owners contract on.

In the long run the probability of each level is the fraction of time spent
at it: the components' long-run availabilities stand in for their
reliabilities, as for the long-run availability. Over time from new, their
point availabilities `A_i(t)` do (`point_capacity`), and the fraction of the
demand delivered over a window is the time average of `E[min(C, d)] / d`
(`mission_capacity`). The availability simulation follows the capacity too:
after every component event it works out the capacity the components that
are up give, and the spread of the delivered fraction from window to window
comes from it.

**Multi-state components.** A component can itself have several levels: a
pump at full, half or no output. The distribution of each node's capacity
enters the calculation the same way, whether it has two levels or many, so
binary components are the special case. A component's levels can be fixed
(each with a probability while it works), come from a nested system, or come
from a model of its states over time: a component that degrades through
stages is in stage `j` at time `t` with probability
`P(S_{j-1} ≤ t < S_j)`, where `S_j` is the sum of its first `j` stages'
times. In the long run, renewed after each failure, it spends its up time in
each stage in proportion to the stage's mean time (the renewal-reward
theorem); from new, it is in stage `j` at `t` if its first unit is, or a
unit put into service at `s` is at age `t − s`, summed over its renewals.

## Costs

`expected_cost_rate` is the renewal-reward rate, each price times its rate
(`ω_i`, `1 − A_i`, `1 − A_sys`), with a cost distribution entering through
its mean ([Lesson 7](learn/costs.md#the-whole-plant-rates-times-prices)). By
linearity of expectation the expected cost of a finite window from new is
exact: `expected_cost` sums each category's expected events over the window (see
availability above) times its mean cost. The cost of a window is random,
though, and its distribution has no closed form, so `cost()` simulates it:
each history accumulates the charges at its failures and the downtime it
incurs. Its spread belongs to the system, while its mean's error shrinks
like `1/√N` ([Lesson 7](learn/costs.md#two-different-uncertainties)).

The **total cost of ownership** is `Σ a_i + H ×` the cost rate
([Lesson 7](learn/costs.md#buying-redundancy)); with a continuous
`discount_rate` `r`, the present value, `H` counts as `(1 − e^{−rH}) / r`.
Redundancy that minimises it trades copies against downtime:
`n_i` independently repaired active copies of component *i* each cost
`a_i + H · r_i` (`r_i` its own running cost rate) and are all down
`(1 − A_i)^{n_i}` of the time, so a design costs
`Σ n_i (a_i + H r_i) + H · downtime_cost_rate · (1 − A_sys)`. The bound
`H · downtime_cost_rate · U^k (1 − U)` on the saving of the `k+1`-th copy of
a component of unavailability `U` limits the copies tried; the search then
works as for redundancy allocation below.

## Allocation

**Redundancy allocation** chooses integers `n_i ≥ 1` (copies of each costed
node) to maximise system reliability subject to `Σ c_i n_i ≤ budget`, or to
minimise `Σ c_i n_i` subject to reliability `≥ target`. With several
resources (cost, weight, volume, …) each has its own limit,
`Σ c_ij n_i ≤ B_j`, and a target may be combined with limits. With `n_i`
active independent copies, node *i*'s reliability is `1 − (1 − p_i)^{n_i}`,
and each candidate is scored by the exact engine, so the structure is
arbitrary. A node may instead choose among component types: `k_j` copies of
each type `j` give `1 − ∏_j (1 − p_j)^{k_j}`, all of one type or, with
mixing, any combination. A node may need `k` of its copies working
(k-out-of-n: the probability that at least `k` work, a binomial tail for
identical copies), and its spares may be *cold standby*, unpowered until
switched in, so that they do not age: the node is then a standby arrangement
of its copies. With perfect switching cold spares always beat active ones;
with imperfect switching they need not, and choosing the strategy node by node
is part of the optimisation (Coit, 2003). Both exact methods work on each
node's list of designs, less those another design of the node beats (using no
more of any resource while being at least as reliable), which can never be
part of a better system. Instead of one optimum, the whole trade-off can be
had: the designs no other beats on every resource and on reliability (the
Pareto front), which the dynamic program yields directly and which is found
by enumeration on other structures.

When every costed node is in series with the rest of the system (it lies on
every minimal path), the reliability factorises,
`R = R_rest · ∏ (1 − (1 − p_i)^{n_i})`, and a dynamic program over the nodes
finds the optimum exactly. It keeps only the partial designs that no other
beats on every resource and on log-reliability (dominance, as in Kettelle's
1962 method): a dominated partial design can never be completed into a
better one than the design dominating it. On other structures, because adding
a copy never lowers a coherent system's reliability, the best design within a
budget can always be found among the designs that cannot afford another copy,
which is what makes the exhaustive search practical. The greedy alternative
adds the copy with the best gain in log-reliability per unit cost (per unit
of its total share of the limits, with several) until the budget runs out;
it is fast but can stop short of the optimum.

**Reliability-redundancy allocation** chooses both at once: each node's
component reliability `r_i`, within bounds, and its copies `n_i`, to maximise
system reliability when what the copies use depends on both (a more reliable
component costs more). It is a mixed-integer nonlinear problem. For fixed
copies it is a continuous problem in the `r_i`, solved with the exact gradient
of the system reliability; over the copies, a vector can be skipped when even
the reliability with every node at the best component it could afford alone
does not beat the best design found, which keeps the exact search short.

**Reliability allocation** apportions a system target among components.
There are many allocations that meet a target; each rule picks one.

- *Equal apportionment* gives every component the same reliability
  ([Lesson 9](learn/design.md#equal-apportionment)).
- *Proportional improvement* (ARINC-style) scales every adjustable
  component's failure probability by a common factor (`q_i → q_i · e^{−x w_i}`
  with optional weights `w_i`) and solves for `x`. The ARINC method scales
  failure rates instead, which is the same for small failure probabilities
  ([Lesson 9](learn/design.md#proportional-improvement-arinc)).
- *Minimization of effort* (Albert, 1958) is for a series system. If raising a
  component's reliability from `x` to `y` takes effort `G(x, y)`, the same
  function for every component, growing with `y` and adding up over
  successive steps (with Albert's regularity condition), the least total
  effort raises the `k` least reliable components to one common level
  `R_0 = (R* / ∏_{i>k} R_i)^{1/k}` and leaves the others, with `k` the largest
  number for which the `k`-th reliability is below that level.
- *Cost-based allocation* (Mettas, 2000) minimises `Σ c_i(R_i)` subject to
  the system reliability reaching the target and
  `R_min,i ≤ R_i < R_max,i`, with
  `c_i(R) = exp((1 − f_i)(R − R_min,i)/(R_max,i − R))`. At the optimum, every
  improved component buys system reliability at the same marginal cost,
  `c_i'(R_i) / I_B(i) = λ` with `I_B(i)` its Birnbaum importance, and a
  component left unchanged would cost more than `λ` per unit: the
  improvement goes where it is cheapest per unit of system reliability.
- The *smallest log-odds change* (not a named method) starts every
  component at 0.5 and minimises `Σ s_i² / w_i`, with `s_i` component *i*'s
  log-odds `log(p_i / (1 − p_i))` and `w_i` its weight, subject to the
  target. At the optimum `s_i ∝ w_i · p_i (1 − p_i) · I_B(i)`, so components
  that matter more, and more heavily weighted ones, move further. It is a
  *structural* allocation: it uses no component data, and with every
  component at 0.5 the Birnbaum importance is the structural importance, so
  small changes follow `w_i` times the structural importance exactly; larger
  ones drift as the importances are re-evaluated along the way.

**Availability allocation** applies the same rules to the components'
long-run availabilities `A_i = MTTF_i / (MTTF_i + MTTR_i)`, with the system's
availability computed exactly as in [Availability](#availability). An
allocated `A_i` fixes only the ratio `MTTF_i / MTTR_i = A_i / (1 − A_i)`: it
is met by an MTTF of `MTTR_i · A_i / (1 − A_i)` at the current MTTR, or by an
MTTR of `MTTF_i · (1 − A_i) / A_i` at the current MTTF. Choosing between the
two is a cost-based allocation with two variables per component, the failure
rate `λ_i = 1 / MTTF_i` and the repair time, each with Mettas's cost
`exp((1 − f)(x₀ − x)/(x − x_min))` as it falls from `x₀` towards its floor
`x_min`. The odds that component *i* is down are `λ_i · MTTR_i`, so both
levers act on its log-odds of being up, `−log λ_i − log MTTR_i`, in the same
way: at the optimum every lever in use buys system availability at the same
marginal cost, the cheaper lever is used first, and at equal costs both move
by the same factor. Holding the failure rates gives *maintainability
allocation*. In a series system, cutting component *i*'s MTTR by a factor
`e^v` raises the system's log-odds of being up at `U_i / (1 − A_sys)` per unit
of `v`, `U_i` its unavailability, so, at equal costs, the components down
most often get the largest cuts.

## Maintenance models

**Age replacement** (`NonRepairable`) uses the renewal-reward rate
`C(t) = (c_p R(t) + c_u F(t)) / ∫₀ᵗ R(u) du`, which has a finite optimum
only for a unit that wears out; otherwise the rate is `c_u / MTTF`
([Lesson 8](learn/maintenance.md#age-replacement-the-cost-per-hour-from-first-principles)).

**Minimal repair and overhaul.** A minimal repair returns the unit to the
state just before it failed ("as bad as old"), so failures follow a
non-homogeneous Poisson process with cumulative intensity `Λ(t)`. Overhauling
every `t` to as good as new, at cost `c_o`, with minimal repairs at `c_r`,
gives the Barlow–Hunter cost rate

```
C(t) = (c_r Λ(t) + c_o) / t
```

For a power-law (Crow–AMSAA) process, `Λ(t) = (t/α)^β`, and a finite optimum
needs `β > 1`. The expected time to the *n*-th failure has the closed form
`E[T_n] = α Γ(n + 1/β) / Γ(n)`.

**Imperfect repair.** The Kijima *virtual-age* models sit between the two.
The unit fails as a new unit of its virtual age would, and each repair
resets the virtual age `v` after an operating interval `x`: in Kijima I to
`v + q·x` (the repair keeps a fraction `q` of the age added since the last
repair), in Kijima II to `q·(v + x)` (it keeps a fraction `q` of the whole
age). `q = 0` is perfect repair (a renewal) and `q = 1` minimal repair.
`E[N(t)]` then has no closed form and is estimated by simulating the process,
and the same overhaul policy applies. Replacing at the *N*-th failure instead
gives `C(N) = (c_r (N − 1) + c_o) / E[T_N]`.

## Scope

- **Fitting failure data to distributions lives in
  [surpyval](https://github.com/derrynknife/SurPyval), not here.** RePyability
  consumes fitted models (anything exposing `sf`/`ff`) as node inputs.
- **Plotting and dashboards live in the Reliafy app, not here.** RePyability
  is the computational engine; it returns numbers and typed result objects.
