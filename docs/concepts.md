# Concepts

The reference behind the numbers: what an RBD means, how each quantity is
computed, what each model assumes, and how to choose among the importance
measures. The [tutorial](tutorial.md) shows these in action and the
[user guide](guide/index.md) shows how to call them; this page explains them.

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
**exactly**, not by simulation. RePyability evaluates the probability that at
least one path set is satisfied (or, equivalently with `method="c"`, that no
cut set is), by a pivotal (Shannon) decomposition over the sets. The
decomposition depends only on the structure, so it is worked out once per
RBD and replayed for every evaluation: repeated evaluations (arrays of times,
importance measures, allocation searches) cost little more than arithmetic.
The two methods return the same value; the path-set method is the default
because it does not need the cut sets.

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

`random()` draws each component's lifetime and applies this rule;
`mean_time_to_failure()` is the average of many such lifetimes. By the central
limit theorem the average is approximately normal with standard error
`s / √n` (the sample standard deviation over the square root of the number of
samples), which gives `mean_time_to_failure_interval()`. The mean is estimated
rather than integrated because the system lifetime distribution of a general
diagram, especially with composite nodes, has no convenient closed form.

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
| **Criticality** `criticality_importance` | Birnbaum weighted by the node's reliability relative to the system's: `I_B · R_i / R` | Ranking that accounts for how reliable each node already is, not just its structural leverage. |
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
accumulates hazard) do; **proportional odds** does not, and is refused in
schedule mode. The fixed-covariate node is the special case of a constant
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

## Allocation

**Redundancy allocation** chooses integers `n_i ≥ 1` (copies of each costed
node) to maximise system reliability subject to `Σ c_i n_i ≤ budget`, or to
minimise `Σ c_i n_i` subject to reliability `≥ target`. With `n_i` active
independent copies, node *i*'s reliability is `1 − (1 − p_i)^{n_i}`, and each
candidate is scored by the exact engine, so the structure is arbitrary.
Because adding a copy never lowers a coherent system's reliability, the best
design within a budget can always be found among the designs that cannot
afford another copy, which is what makes the exact search practical. The greedy
alternative adds the copy with the best gain in log-reliability per unit cost
until the budget runs out; it is fast but can stop short of the optimum.

**Reliability allocation** apportions a system target among components. The
proportional-improvement rule scales every adjustable component's failure
probability by a common factor (`q_i → q_i · e^{−x w_i}` with optional
weights `w_i`) and solves for `x`; equal allocation is the special case of
identical starting points, so every component gets the same reliability. The
least-squares rule searches any combination that meets the target. There are
many allocations that meet a target; each rule picks one.

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
