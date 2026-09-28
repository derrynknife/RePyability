# Concepts

The reference behind the numbers: what an RBD means, how each quantity is
computed, what each model assumes, and how to choose among the importance
measures. The [tutorial](tutorial.md) shows these in action and the
[user guide](guide/index.md) shows how to call them; this page explains them.

!!! tip "New to the theory?"
    This page is a compact summary. The [Learn](learn/index.md) course
    teaches the same ideas step by step: each one worked out by hand with
    small numbers, then with RePyability, with exercises.

## Reliability block diagrams

A **reliability block diagram** models a system as a directed graph from a
single input (source) to a single output (sink). Each intermediate node is a
component with a reliability model. The system **works** at time `t` if there
is a path of working components from input to output.

Two structures underlie everything:

- A **minimal path set** is a minimal set of components whose simultaneous
  working guarantees the system works. The system is up iff *at least one*
  path set is fully up.
- A **minimal cut set** is a minimal set of components whose simultaneous
  failure guarantees the system fails. The system is down iff *at least one*
  cut set is fully down.

Series, parallel, and *k*-out-of-*n* are special cases: a series system is one
path set of every component (and each component its own cut set); a parallel
system is the reverse. A *k*-out-of-*n* node is up when at least `k` of its
incoming branches are. A component that appears in no minimal path set is
**irrelevant**: its state never decides whether the system works.

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
   shared node), is evaluated from its minimal path sets by a pivotal
   (Shannon) decomposition, over its modules and nodes. Only the core pays
   the combinatorial price.

Both stages depend only on the structure, so they are worked out once per
RBD and replayed for every evaluation: repeated evaluations (arrays of times,
importance measures, allocation searches) cost little more than arithmetic.
A series-parallel diagram is never expanded into its path sets, however many
it has: thirty duplicated stages in series have `2^30` of them, and are
evaluated in milliseconds. With `method="c"` the probability that the system
fails is computed instead, and its complement returned; the two methods give
the same value. Every step is a sum of products of node probabilities and
their complements, so both keep their full relative precision.

The identity that drives the decomposition, and the importance measures, is
**pivotal decomposition** around any node *A*:

```
R_sys = R_A · R_sys(A working) + (1 − R_A) · R_sys(A failed)
```

The engine is exact *given the node reliabilities*. When a node's reliability
is itself an estimate (a simulated standby or load-sharing arrangement, or a
Kaplan–Meier fit), the system value inherits that estimate's error, and
`is_analytically_solvable()` flags the simulation-backed nodes.

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
are only needed for the core; `mean_time_to_failure()` is the average of many
such lifetimes. By the central
limit theorem the average is approximately normal with standard error
`s / √n` (the sample standard deviation over the square root of the number of
samples), which gives `mean_time_to_failure_interval()`. The mean is estimated
rather than integrated because the system lifetime distribution of a general
diagram, especially with composite nodes, has no convenient closed form.

### Simulation error, and making it smaller

A simulated mean of `N` independent results has standard error `s / √N`, so
its error halves when `N` is quadrupled. Given a `tolerance`, a simulation
adds `N` results at a time until the confidence interval's half-width,
`z · s / √N`, is at most the tolerance. (The stopping point depends on the
estimated `s`, which makes a sequential rule's coverage slightly below the
nominal level when `N` is small; checking only after each batch of `N`
keeps the effect small.)

Two classical variance-reduction techniques make the error smaller for the
same `N`, without biasing the estimate:

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
from a stream of its own, keyed by the seed, its place in the diagram and
the simulation (or the pair), so its `k`-th draw is matched, or paired,
however the components' events interleave.

A parallel run splits the simulations into blocks seeded in turn from one
`numpy.random.SeedSequence`, whose spawned seeds give independent streams.
The block, not the process that runs it, fixes the random numbers, so the
results do not depend on the number of processes.

### Fault trees

A fault tree describes the same structure from the side of failure: the top
event occurs through OR gates (any input), AND gates (every input) and VOTE
gates (at least `k` of `n` inputs) over the basic events. Its logic is the
dual of a diagram's: an OR gate is a series block, an AND gate a parallel
block, and a VOTE gate on `k` of `n` failures a block needing `n − k + 1` of
`n` working; the tree's minimal cut sets are the diagram's. A tree is
evaluated by the same engine: each gate below which no event or gate is
shared with the rest of the tree is a module with a closed form, and what the
repeated events tie together is a core, solved exactly by the pivotal
decomposition over its minimal path sets. The measures of importance are the
diagram's, with the top event as the system failing.

### Parameter uncertainty

The node models are estimates, so the system reliability computed from them
is uncertain too: *epistemic* uncertainty, about the models, as opposed to
the *aleatory* variability they describe. `sf_uncertainty` propagates it by
Monte Carlo over the parameters: each draw gives every uncertain node a
plausible model and the system reliability is computed exactly, all draws
at once (the node probabilities are arrays over draws and times), and the
percentiles of the draws form an uncertainty interval. A maximum-likelihood
fit's own estimate of its parameters' uncertainty (the inverse Hessian of
the log-likelihood, from surpyval) gives the draws on a transformed scale
(log for a positive parameter, logit for one in (0, 1)), which is the delta
method's normal approximation. Nodes of one population share their
parameters and so their draws: drawing them independently averages part of
the uncertainty away.

## Reliability vs availability

- **Reliability** `R(t)`: the probability the system has *never* failed by
  `t`. The right question for a mission or a non-repairable item. Lives on
  [`NonRepairableRBD`][repyability.NonRepairableRBD].
- **Availability** `A(t)`: the probability the system is *up at* `t`,
  allowing for repair. The right question for a serviced, long-running
  system. Lives on [`RepairableRBD`][repyability.RepairableRBD], which needs
  a repairability distribution per component.

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
| **Fussell–Vesely** `fussell_vesely` | What fraction of system-failure probability involves this node? `Σ_{cut sets C ∋ i} Π_{j ∈ C} (1 − R_j) / Q` | A cut-set-based culprit ranking; standard in PRA/PSA. |

Two more answer *design-time* and *data-targeting* questions rather than
ranking at an operating point:

- **Structural importance** `structural_importance`: Birnbaum with every node
  reliability set to ½, i.e. the fraction of the other nodes' states in which
  the node is pivotal. It is **model-free**: it depends only on the diagram,
  so you can rank redundancy needs *before any data exists*.
- **Parameter sensitivity** `parameter_sensitivity`: the derivative of system
  reliability with respect to each node's *distribution parameters*,
  `∂R/∂θ = I_B · ∂R_i/∂θ`, computed numerically. Where Birnbaum says *which
  component* matters, this says *which fitted parameter* matters, so you know
  where more data would most change the answer.

A rule of thumb: **Birnbaum** for "where does an improvement help most",
**risk achievement worth** for "what must not be allowed to fail",
**structural importance** at the whiteboard, and **parameter sensitivity**
when deciding where to spend a testing budget.

On a repairable system the same measures are evaluated with long-run
availabilities in place of reliabilities.

## Condition-based evaluation

The measures above assume every component is new. The condition-based methods
instead take each component's *current life* `Xᵢ` and condition on it:

```
Rᵢ(x | Xᵢ) = Rᵢ(Xᵢ + x) / Rᵢ(Xᵢ)
```

This is the survival of a further `x` given the component has already reached
`Xᵢ`. Feeding the conditioned per-node reliabilities through the same exact
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
extension: surpyval 0.20 refuses it in schedule mode, and later versions
switch to the new covariate's hazard at each step (the survival does not jump
to the new covariate's curve). The fixed-covariate node is the special case of a constant
path.

## Standby and repeated nodes

**Repeated nodes.** `n` independent, identical copies of a component with
reliability `p` have reliability `pⁿ` in series and `1 − (1 − p)ⁿ` in
parallel. A `RepeatedNode` is exactly that, as one node. It is not the same as
**one component drawn in several places** (a shared power supply feeding two
branches): that is a single component, which fails once for every place it
appears, and treating its appearances as independent copies over-states
reliability.

**Cold standby.** With one unit operating and spares that do not age while
waiting, the arrangement's lifetime is the *sum* of the units' lifetimes, so
its survival function is the convolution of theirs. RePyability computes it by
numerical convolution (deterministic, no sampling). For identical Exponential
units the sum is Erlang, in closed form, for any `k` operating units.

**Imperfect switching.** If each switch-over succeeds with probability `s`,
the lifetime is the sum of the first `j` units' lifetimes with probability
that exactly `j − 1` switches succeeded before one failed (or all succeeded):
a mixture of partial sums, again computed by convolution.

**Warm and hot standby.** A dormant spare that ages at a fraction `κ` of the
operating rate (the `dormancy_factor`) accumulates *virtual age* at rate `κ`
while waiting and `1` once operating, and fails when its virtual age reaches
its baseline failure age. It can therefore fail *latent*, before it is ever
switched in. For identical Exponential units the memoryless property makes
each stage (from `j` to `j − 1` surviving units) exponential with rate
`λ (k + (j − k) κ)`, so the lifetime is **hypoexponential**; `κ = 0` gives
the cold Erlang and `κ = 1` the parallel order statistic. Other units are
simulated and fitted with Kaplan–Meier. Hot standby (`κ = 1`) is exactly
*k*-out-of-*n* active parallel.

With `k ≥ 2` operating units and general lifetimes, which unit fails next
depends on the order of failures, so the lifetime is not a simple sum and is
simulated.

## Dependent failures: load sharing

Redundant units that *share a load* do not fail independently. While all are
up each carries its share; when one fails the survivors pick up the slack,
run harder, and age faster, so the failures are positively correlated, and
treating them as `n` independent parallel nodes over-counts the redundancy.

A [`LoadSharingModel`][repyability.LoadSharingModel] captures the coupling as
a single node. Each unit is a fitted AFT model with **load as its covariate**,
so a unit running under load `ℓ` ages on its baseline clock at an acceleration
factor `φ(ℓ)` (the AFT time scaling). With `s` survivors sharing a total load
`L`, each carries `L / s` and ages at `φ(L / s)`; as siblings fail, `s` falls,
`L / s` rises, and the survivors' clocks speed up. The group works while at
least `k` of the `n` units survive. This is the **cumulative-exposure** model:
a unit's *virtual age* is the integral of `φ(load(t))` over real time, and it
fails when that virtual age reaches its baseline failure age.

Two regimes:

- **Closed form.** Identical units with an **Exponential** baseline give a
  group lifetime that is a sum of exponential stages (each stage the time for
  the next unit to fail at the current shared load), i.e. a
  **hypoexponential** distribution, evaluated exactly with no simulation
  (`is_simulated == False`).
- **Simulation.** Otherwise the survival curve is a Kaplan–Meier fit to
  lifetimes drawn from the cumulative-exposure event loop (seeded;
  `is_simulated == True`).

As a check, with no load effect (`φ ≡ 1`) the survivors do not accelerate and
the group reduces *exactly* to the ordinary *k*-out-of-*n* parallel result.
Load sharing is the "self-loading" sibling of the condition-based layer: there
the load is streamed in from sensors, here it is computed from the group's own
survivors. Warm standby uses the same virtual-age machinery with a fixed
dormant rate instead of a load-dependent one.

## Common-cause failures

Redundancy only buys reliability if the redundant units fail for
*independent* reasons. In practice they often share a root cause (a common
manufacturing batch, a shared power supply, one miscalibration applied to
every unit) and a single event takes them all down together. Because the
exact engine assumes independence, it **over-estimates** a redundant group; a
common-cause model injects the shared coupling. (This is the mirror image of
load sharing: there the coupling is mechanical load transfer, here it is a
shared shock.)

A [`CCFGroup`][repyability.CCFGroup] declares the coupled (symmetric) members
and the model, passed via `ccf_groups`. Two models:

- [`BetaFactor(beta)`][repyability.BetaFactor]: a fraction `β` of each
  unit's failure probability `Q` comes from a cause shared across the
  **whole** group (which fails every member at once); the remaining
  `(1 − β) Q` is independent. The workhorse of probabilistic-risk
  assessment.
- [`MGL(beta, gamma, ...)`][repyability.MGL]: the **Multiple Greek Letter**
  model, which also resolves *partial* common causes (a cause failing some
  but not all of the group) through a cascade of conditional probabilities:
  `β = P(shared by ≥ 2 | failed)`, `γ = P(≥ 3 | ≥ 2)`, and so on. The
  probability that a cause fails a *specific* set of `k` of the `m` members
  is the standard basic-event probability

```
Q_k = [ 1 / C(m−1, k−1) ] · (ρ₁ ρ₂ ⋯ ρ_k) · (1 − ρ_{k+1}) · Q
```

  with `ρ₁ = 1, ρ₂ = β, ρ₃ = γ, …, ρ_{m+1} = 0`; these partition each unit's
  `Q` exactly. A group of `m` members takes `m − 1` letters, and `MGL(β)` on
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

**The model assumes each member's `Q` is small.** "Exact" means the
evaluation of the model is exact; the model itself is the PRA basic-event
one, which splits each member's failure *probability* (`βQ` shared,
`(1 − β)Q` independent) and is a rare-event model. Use it over periods in
which each member's failure probability stays small, such as a mission or a
proof-test interval. For a parallel pair with `β = 0.3`, the system
unreliability is within 0.3% of a rate-based beta-factor treatment (which
splits each member's failure *rate* instead, making the shared cause a shock
with reliability `R(t)^β`) at `Q = 0.01`, 3.5% at `Q = 0.1` and about 10% at
`Q = 0.3`. From about `Q = 0.5` the pair comes out *more* reliable than an
independent pair. Over a whole life (`Q → 1`) the model stops describing a
lifetime at all: under the beta factor, each member would only ever fail with
probability `1 − β(1 − β)` (0.79 at `β = 0.3`).

Common cause is currently reflected in `sf()` / `ff()` (and quantities derived
from them) and persists through serialisation. Groups must be symmetric
(identical member models) and disjoint. The Monte-Carlo `random()`, `mean()`
and MTTF interval sample the members independently and do not include CCF: an
MTTF integrates over the whole life, where `Q` is no longer small, so it is
outside this model. The probability-dependent importance/sensitivity and the
condition-based methods do not yet account for it and raise a clear error on a
CCF RBD; `structural_importance`, being probability-free, is unaffected.
**Alpha-factor**, a data-estimable reparameterisation of the same
multiplicities, is a planned extension.

## Availability

A repairable component alternates between up periods (drawn from its
reliability model) and down periods (drawn from its repairability model). Each
repair restores it **as good as new**, so its history is an *alternating
renewal process*, and components do this independently of each other and of
the system's state.

**Long-run availability.** By the renewal-reward theorem a component is up a
fraction

```
A_i = MTTF_i / (MTTF_i + MTTR_i)
```

of the time in the long run, and, the components being independent, the
system's long-run availability is the exact system probability evaluated at
the `A_i`. No simulation is needed. An instantly repaired component
(`MTTR = 0`) has `A_i = 1`.

**Failure frequency and MUT/MDT.** In steady state a component fails
`ω_i = 1 / (MTTF_i + MTTR_i)` times per unit time. A component failure fails
the system when the component is *critical*, which happens with probability
equal to its Birnbaum importance `I_B^i` (at the availabilities), so the
system fails

```
ω = Σ_i I_B^i · ω_i
```

times per unit time (the Birnbaum/Vesely frequency formula, exact for
independent components). From it: the mean time between failures
`MTBF = 1 / ω`, the mean up time `MUT = A / ω`, and the mean down time
`MDT = (1 − A) / ω`, with `MTBF = MUT + MDT`. A nested repairable RBD
contributes its own system frequency.

**Availability over time.** Before the long run, availability depends on
time: a new system starts up (`A(0) = 1`) and settles towards the long-run
value, possibly overshooting. There is no general closed form, so
`availability()` simulates `N` independent histories (each component's
alternating failures and repairs, merged in time order, with the system's
state re-evaluated at every event) and reports the fraction of histories up
at each time. Each point is a proportion, so its standard error is
`√(A(1 − A)/N)`; the confidence band uses the Wilson score interval, which
stays sensible at `A = 1`. For exponential components the simulation is held
to the exact Markov solution in the test suite.

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

## Costs

The long-run cost rate follows from the **renewal-reward theorem**: in the
long run, the cost per unit time is the expected cost per cycle over the
expected cycle length, and rates add over independent contributors. Each
component's corrective costs are charged at its failure frequency `ω_i`;
downtime costs are charged at the rate of time spent down:

```
cost rate = downtime_cost_rate · (1 − A_sys)
          + Σ ω_i · (repair_cost_i + replace_cost_i)
          + Σ (1 − A_i) · downtime_cost_i
```

A cost distribution enters through its mean, by linearity of expectation. The
cost over a finite window is random, and its distribution has no closed form,
so `cost()` simulates it: each history accumulates the charges at its failures
and the downtime it incurs. Its **spread** (standard deviation, percentiles)
is a property of the system; the **uncertainty of its mean** shrinks like
`1/√N`. As the window grows, the simulated cost per unit time converges to the
exact rate.

The **total cost of ownership** over a horizon `H` adds the one-off cost of
buying the components, `Σ a_i`, to `H` times the long-run cost rate
(undiscounted). Redundancy that minimises it trades copies against downtime:
`n_i` independently repaired active copies of component *i* each cost
`a_i + H · r_i` (`r_i` its own running cost rate) and are all down
`(1 − A_i)^{n_i}` of the time, so a design costs
`Σ n_i (a_i + H r_i) + H · downtime_cost_rate · (1 − A_sys)`. The total is not
monotone in the copies, but the `k+1`-th copy of a component of
unavailability `U` saves at most `H · downtime_cost_rate · U^k (1 − U)`, which
bounds the copies worth trying; the search then works as for redundancy
allocation below.

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

- *Equal apportionment* gives every component the same reliability.
- *Proportional improvement* (ARINC-style) scales every adjustable
  component's failure probability by a common factor (`q_i → q_i · e^{−x w_i}`
  with optional weights `w_i`) and solves for `x`. The ARINC method scales
  failure rates instead, which is the same for small failure probabilities.
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

**Age replacement.** Replace at age `t` or at failure, whichever is first,
with a planned cost `c_p` and an unplanned cost `c_u > c_p`. Each replacement
renews the unit, so by the renewal-reward theorem the long-run cost rate is

```
C(t) = (c_p R(t) + c_u F(t)) / ∫₀ᵗ R(u) du
```

A finite optimum exists only if the unit wears out (an increasing hazard);
otherwise replacing early never pays, and the rate is `c_u / MTTF`.

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
