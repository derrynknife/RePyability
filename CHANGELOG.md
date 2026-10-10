# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions have two parts, major.minor (until 0.10.1 they had three): from
1.0, a release that breaks compatibility raises the major number, and any
other release, fixes included, the minor.

## [Unreleased]

### Added

- **Fitted repairable-unit models as components (#269).** A spec's
  `"reliability"` may be what surpyval fits to a repairable unit's failure
  history. A Poisson process (`CrowAMSAA`, `Duane`, `HPP`) is minimal
  repair of the life whose cumulative hazard is its cumulative intensity
  (a Weibull, or an exponential life), so its expected failures with
  repairs in no time are exactly the fitted ones; a `GeneralizedRenewal`
  (Kijima I or II) is its life distribution with the `"repair"` of its
  Kijima model and restoration factor. The spec is saved as that life and
  repair. surpyval's ARA, ARI and G1 renewal models are refused, their
  repairs not being Kijima's (SurPyval#833).
- **A regression node's covariates are levers (#272).** A
  `RegressionNode` at fixed covariates (a load, a temperature) lists each
  covariate in `NonRepairableRBD.levers()` as `"covariate.<name>"` (the
  model's feature name, or its place), `parameter_sensitivity` reports the
  system's derivative in it, and `with_levers` builds the diagram with the
  component run at another value. A covariate is unbounded: running
  outside the fitted conditions is extrapolation, as an accelerated life
  test's use level is, and is not refused.
- **A simulation starts an imperfectly repaired unit from its virtual age
  (#269).** `NodeState(age=a, virtual_age=v)`: its virtual age `v` at its
  last repair and its operating time `a` since; its life left is drawn
  from virtual age `v + a`, and its next repair takes the virtual age on
  by the whole time since its last. A fitted `GeneralizedRenewal`'s
  `unit_states()` gives both. The exact methods still refuse such a state,
  as do the simulations for a component also replaced after some
  failures, maintained or tested, which would need its history since it
  was renewed. Seeded runs from other states are unchanged.
- **Fitted degradation processes as lives (#271).** A surpyval
  `WienerProcess` or `GammaProcess` fit (with its failure `threshold`) is
  a node's life in `NonRepairableRBD` and a component's in
  `RepairableRBD`: the first-passage time from its starting level, which
  every exact method and both simulation engines take, and saving keeps.
  One fitted with stress covariates is refused: fit it at the stress the
  component runs at.
- **Replacement on condition by the measured level (#271).**
  `"preventive": {"policy": "condition", "level": x, ...}`, in place of
  `"threshold"`, for a component whose life is a degradation process:
  each inspection measures its level and replaces it at or past `x`. The
  level an inspection finds is drawn given that the unit has not failed
  since the last (a gamma increment truncated at the threshold, or a
  Wiener path killed there); surpyval does not give it yet (SurPyval#836).
  Simulated only, on the Python engine: the exact methods and a start
  state refuse such a component. Seeded runs without it are unchanged.
- **A component that operates part of the time (`"duty"`, #276).** A
  `RepairableRBD` spec's `"duty"`, the fraction of the time the component
  operates, puts a life fitted in operating time on the diagram's clock:
  `R(d t)`, the same surpyval distribution with its scale moved, which
  every method takes. Repairs, maintenance and tests stay on the clock;
  levers, draws and saving keep the life as given. The rescaling rule of
  each distribution is RePyability's until surpyval has one (SurPyval#845).

### Changed

- **Parameter uncertainty draws a fit's shares too (#267).** `"fit"` drew
  only a distribution's own parameters (from surpyval's `hess_inv`) and
  kept a limited failure population's share that ever fails (`lfp_p`) and
  a zero-inflated fit's share dead on arrival (`f0`) at their fitted
  values. Long after the units that fail have failed, the reliability is
  that share alone, so its spread came out as nothing: with 42% of units
  ever failing, known to 0.029, every draw had 42%. It now draws every
  parameter the fit estimated from surpyval's `covariance()`, the shares on
  the logit scale, in `sf_uncertainty` and the other uncertainty methods,
  vega and both classes' `uncertainty_importance`, and seeded uncertainty
  results of such fits change. An offset is still held where it was
  fitted, as surpyval's covariance leaves it out (SurPyval#830); `"fit"`
  now warns that it does.
- **`lfp_p` and `f0` are parameters (#267).** Where a node's model has a
  limited failure population or zero inflation, `parameter_sensitivity`,
  `levers()` and `with_levers` take the shares as parameters, after the
  distribution's own (the order of surpyval's `covariance()`), with the
  range (0, 1), and an uncertainty `{parameter: distribution}` may name
  them.
- **A `RegressionNode`'s mean and draws are exact.** Its `mean()`
  integrated the survival curve tabulated on 4,096 points and stopped
  where the survival fell to 1e-4, so it came out low (8e-5 for a
  LogNormal AFT model; the heavier the tail, the lower), and its draws
  were read off that grid, so none fell past its end. The mean is now
  integrated as an RBD's MTTF is, to a relative 1e-10, on pieces split at
  the model's quantiles or the schedule's change points, and `inf` where
  some lifetimes never end (a `ValueError` before). A draw is the model's
  quantile `qf(u, Z)` at fixed covariates (surpyval 0.24), and along a
  schedule the time at which the cumulative hazard reaches `-log(1 - u)`,
  by bisection. Seeded draws of a diagram with a regression node change.
  surpyval's own `mean(Z)` is not used: on an additive-hazards model with
  a Normal, Logistic or Gumbel baseline it integrates a survival function
  above 1 before 0 (SurPyval#828), a model `RegressionNode` refuses.
- **surpyval 0.24 or later is required** (0.23 was). It gives a mixture
  life a survival function that keeps its precision in the far tail, and
  draws a life given an age (imperfect repair) reading a limited failure
  population's share by its own name, `lfp_p`, which a mixture lacks. The
  mixture's new quantile function loses the long lives' precision and is
  slow (SurPyval#821), so the simulations keep drawing a mixture's lives
  with RePyability's own; seeded results are unchanged.
- **The documentation is easier to find one's way in.** The API reference
  is a page per section (the largest 0.9 MB, where the one page was
  5.7 MB with its source inlined), each class's page lists what it
  inherits, and docstrings no longer show bare issue numbers. Maintaining
  a system (preventive maintenance in a diagram, replacement on
  condition, opportunistic maintenance, hidden failures and their
  intervals) has its own guide page, out of Costs. Concepts is the
  reference for how RePyability computes each value, and links the Learn
  lesson that teaches an idea where it restated it. Three statements were
  wrong: Concepts had `mean_time_to_failure()` average simulated
  lifetimes (it integrates exactly by default), and the Learn lesson on
  dependence had load sharing of identical non-exponential units
  simulated (it is numerical) and the standby arrangements without a
  reliability simulated with `seed=0` (since 0.12 they refuse, and the
  system's simulations draw them).
- **`RepairableRBD`'s code is in modules by analysis.**
  `repyability/rbd/repairable_rbd.py` (some 20,500 lines) holds the class
  alone (some 7,100): its constructor, its public methods, each calling a
  function of the same name in a module of its kind (`_runs`,
  `_event_loop`, `_curves`, `_windows`, `_long_run`, ...), and the hooks
  `RBD` calls. Results are unchanged. Code that imported private names
  from `repyability.rbd.repairable_rbd` (`_PythonRunner`, `_aged_life`,
  the worker functions, ...) finds them in those modules; the criticality
  indices of a run's counts are still importable from there. A
  simulation's state is one object (`_events._Run`), which a run's end
  forgets whole (it had left the common-cause groups' state behind), and
  `system_state`, `component_status`, `t_simulation` and
  `last_change_planned`, which `initialize_event_queue` and `next_event`
  document as kept on the diagram, are read-only properties of the run in
  progress.
- **A fault in a model is raised, not taken for something it cannot do.**
  Where RePyability probes a node's model (its quantiles to split a grid
  at, its mean, whether its rate is constant, whether a group's members
  save alike), it caught any error as "the model cannot do this" and went
  on without it. It now catches only what a model that cannot take the
  values raises (an arithmetic, type, value, lookup, attribute,
  not-implemented or runtime error; a save's type, value, attribute or
  not-implemented error), so a fault in a user's model, a `NameError` or
  an `AssertionError` say, is raised where it happens. `analysis_routes()`
  likewise reports only refusals, and raises any other error a check
  meets.

### Fixed

- **Importance measures with common-cause groups whose tests take time
  (#294).** While a member's test kept it off line it could not be up,
  and the system given it up was 0/0, so Birnbaum's measure, the
  improvement potential, the risk reduction worth, the criticality and
  the differential importance were NaN for every node, with no warning.
  Where a member's state has no chance, the system given it is now the
  system with the member held in that state, as for a node outside the
  groups: with no shared cause the measures are the independent members'
  own.
- **A component that never fails is a junction again.** A `RepairableRBD`
  component whose life is a probability of failing of 0
  (`FixedEventProbability` at 0, as a vote point is often drawn) was
  refused by the exact methods since a probability per demand was (and
  before that was read as a mean life of 0, an availability of 0). It is
  now a junction, as a life of `PerfectReliability` is: it never fails,
  every analysis leaves it out, and a cost or schedule given it is
  refused.
- **Conditional survival keeps its precision at old ages (#268).** The
  chance of surviving a further `x` given survival to an age `X` was the
  ratio `R(X + x) / R(X)`, which came out 0 once both were too small for a
  float: for a Weibull(100, 3) unit 1000 hours old, the next 10 hours'
  survival is 6.9e-14, where it gave 0. It is now worked out from the
  model's cumulative hazard, `exp(-(H(X + x) - H(X)))`, wherever the model
  has one (surpyval's models, and a diagram's own `Hf`), in
  `NonRepairableRBD.cs` and a node's age in `sf_given_state`. A diagram's
  own `cs` is still 0 once its reliability at `X` is below the smallest
  float, as its `Hf` is then infinite.
- **A probability per demand has no long run in a `RepairableRBD`.** A
  `FixedEventProbability` life (a unit that fails at once with that
  probability, or never) was read as a mean life by the long-run methods:
  `p = 0.1` gave an availability of 0.09 where the simulations give 0.999.
  The long-run values, frequencies, cost rate and importance now refuse
  it, as the values over time did, and the routes say so.
- **`RegressionNode` takes surpyval's accelerated-life models.** An
  Arrhenius, Eyring or power-law life was refused, its covariate count
  taken from its life model's parameters (two, for one stress); it is now
  the number of stresses the model was fitted with.
- **Results no longer depend on the unit of time.** A cold-standby
  group's grid started at time 1 whatever its lives' scale, and the
  density and hazard (and the reliability rate's parts) took steps of
  `1e-6` of time below time 1: in a diagram whose lives were about 1e-5
  of the unit long, a cold standby's probability of failing was off by
  up to 27% (and its density by 5%). Both
  are relative now: the grid starts at the typical life, a step is `1e-6`
  of the time (of the shortest typical life at time 0). A regression
  node's check for a proper survival curve looks just after 0, not at
  `1e-9`.
- **Proportional differential importance over parameters does not depend
  on the unit.** A LogNormal's `mu` (the log of a time) was moved by a
  share of itself, which changes with the unit (in hours its share of one
  diagram was 0.89, in years 19); a proportional change of it is now a
  proportional change of the time (`d mu = epsilon`), as for a Cox-Lewis
  process's `alpha`. A regression node's covariate, whose zero is its
  unit's, is refused (`change="uniform"` takes it).

## [0.13] - 2026-10-09

Simpler, faster and harder to misuse. Each computation now has one
implementation (#204): a simulation's random numbers are counter-based, a
function of the seed, the stream, the simulation and the draw alone
(#209); every timeline is recorded by the event loop (#205); numba's engine
takes every level's events in one loop (#206) and keeps whether each level
works up to date without truth tables (#255, #254); a decision diagram has
one search and one replay (#207); and a run's changes are put in time order
by numpy alone (#208). The exact curves are built several times as fast,
an MTTF takes a few hundred evaluations where it took thousands (#229),
and capacity runs, timelines and conditional runs spend less time around
the compiled loop (#246, #247, #248). Common-cause groups are worked out
module by module, linear in the groups rather than doubling per group, in
fault trees and non-repairable diagrams (#219) and in repairable ones
(#218), whose members' tests and repairs may now take time (#220); limited
repair crews refuse common-cause groups where they dropped them (#251,
#252); and `availability_rate` is exact just after a scheduled maintenance
(#240). From the 0.12 persona check (#213): a mixture as a repairable life
(#227), times first in the measures over time (#224), replacement intervals
from a calendar (#230), discounted costs from new and endless horizons
(#231), chunks of a plain run (#236), unavailability over time to its own
precision (#237), `to_dict()` on every result (#235), public levers
(#244), simulations of one diagram from several threads (#216), a shared
fitted model drawn once in parameter uncertainty (#214), messages that say
what to do and inputs checked where they are given (#232, #233), and docs
brought up to date (#238). What 0.12 deprecated is gone.

Behaviour changes: a `RepairableRBD` simulation's seeded results differ
from 0.12's, within their standard errors, as its random numbers are
counter-based (#209); exact curves move in their seventh or eighth
significant figure (a grid of 1,000 steps); a cost result's `mean`,
`cost_rate` and breakdowns, and the new `mean_availability`, are the run's
estimate (exact or conditional by default), the simulations' own being
`sample_mean` (#223); a model given for the input or output node is
refused (#217); costs, intervals and times given as text or booleans,
surpyval's distribution class in place of a model, and seeds that are not
whole numbers or lists of them are refused (#233, #232), as are unknown
nodes in `allowed=`, `offset_shares=` and `with_intervals` (#222) and a
component that fails and is repaired at once; a fixed-probability node's
simulated lifetime is 0 or infinite, so seeded draws of such a diagram
change; `compare(quantity="cost")` counts acquisition, and `cost()` with
only an acquisition cost gives a result (#234); a test offset within a
billionth of the interval is 0 (#237); a capacity run's probabilities can
differ from 0.12's in their last digits (#246); what 0.12 deprecated is
removed: calling `SparesDemand.mean()` or `std()` raises `TypeError` (they
are properties), as do `StandbyModel`'s and `LoadSharingModel`'s
`mc_samples`, `lower` and `seed`, and `StandbyModel`'s
`switching_probability` and `dormancy_factor` are given by name; and
surpyval 0.23 is required. Deprecated, to go in 0.14:
`optimal_inspection_intervals(offsets=)` (now `offset_shares=`), calling
`CapacityDistribution.mean()`, and the simulation options of an exact
`mean`.

### Added

- **A surpyval `MixtureModel` as a repairable component's life (#227).**
  A two-mode population (infant mortality and wear-out) fitted with
  surpyval's `MixtureModel`, which a `NonRepairableRBD` took, was refused
  by `NonRepairable`, and so by `RepairableRBD`, with "Unknown reliability
  function". It is now taken wherever a parametric life is: the exact and
  numerical analyses use its survival function, distribution, density
  and mean; the simulations draw its lives from their streams, one uniform
  a draw, by inverting its distribution function (it has no quantile
  function, surpyval #651), on both engines; and imperfect repair and a
  start from an age draw its life given the age through its cumulative
  hazard. Its parameters cannot be drawn by the uncertainty methods
  (expectation-maximisation leaves no covariance), which now say so,
  where they said to fit it with surpyval; a list of models can stand in.
  Any other life is refused naming the component and what it was given.
  A `NonRepairableRBD` draws a mixture as surpyval does, as before.
- **A repairable diagram's measures take their times first (#224)**, as a
  non-repairable diagram's do: `birnbaum_importance(5.0)` is
  `birnbaum_importance(x=5.0)`, and `barlow_proschan_importance(100.0)`
  takes its `window`, where the time was taken for node names ("'float'
  object is not iterable"). A number, or numbers none of which is a node,
  given first are the times, for the importance measures (Birnbaum,
  criticality, improvement potential, risk achievement and reduction
  worth, Fussell–Vesely, joint and differential) and
  `parameter_sensitivity`; node names are node names, as before.
- **Replacement intervals from a calendar (#230).**
  `optimal_replacement_intervals(allowed=...)` chooses each component's
  age-replacement interval from a list (one for every component, or a dict
  of a list each; `inf` among them for never), every combination tried
  when there are at most 2000, as `optimal_inspection_intervals` chooses
  tests': replacement every four weeks, say, rather than at 497 hours. The
  0.12 notes said it took `allowed` already; it did not. Its docstring now
  says the plan is the best for the long run, and that a new plant, whose
  units' first replacements fall together, can cost more in its first years
  (#234).
- **The discounted expected cost from new (#231).** `expected_cost(t,
  discount_rate=r)` discounts each cost when it falls, where `total_cost`
  spends the long-run rate from the start: `exp(-r t) C(t) + r * integral
  of exp(-r s) C(s)`, by parts from the expected cost from new, on
  Gauss-Kronrod pieces halved until within 1e-8 (closing in on the jumps
  of scheduled tests and replacements). The ten-year pump of `total_cost`'s
  example is worth 114,528 from new against 114,538 at the long-run rate.
  The result's `discount_rate` says it is a present value.
- **Endless and many horizons (#231).** `total_cost` (and
  `allocate_redundancy`) take an endless horizon when the costs are
  discounted: the acquisition and `rate / r`. `total_cost` takes an array of
  horizons too, as `expected_cost` does.
- **Chunks of a plain run (#236).** `simulate_chunk` and
  `availability_from_chunks` take `control_variate=False` and
  `conditional=False`, so a split run gives the means
  `availability(..., control_variate=False)` gives, where merged chunks
  always took the default ones. Given to `simulate_chunk`, it is kept with
  the chunk (and its saved form); chunks made with different ones do not
  merge. True, which simulates a twin or the modules alone, is refused.
- **Unavailability over time to its own precision (#237).**
  `RepairableRBD.point_unavailability(x)` and `mission_unavailability(t)`
  take what `point_availability` and `mission_availability` take and work
  out the probability of being down itself, where one less the
  availability rounds to 0 below about 1e-16: a component with an
  exponential life and repair by its closed form, the others from their
  curves of being down, and repair crews and common-cause groups through
  their chains. For a pair of valves failing once in 10^9 hours, repaired in
  8, `1 - point_availability(1000.0)` gave 1.1e-16 and
  `1 - mission_availability(8760.0)` 0; `point_unavailability(1000.0)`
  gives the exact 6.4e-17 and `mission_unavailability(8760.0)` 6.391e-17,
  so a small PFD(t) can be plotted on a log scale.
- **`to_dict()` on every result (#235)**, ready for `json.dumps`, for a
  service that hands results on: each field by name, arrays as lists,
  numpy numbers as Python ones, the results a result holds (its
  criticalities, cost or control variate) as their own `to_dict()`, and
  keys JSON cannot hold (tuple node names) as their text. An
  `AnalysisRoute`, `joint_importance`'s pairs (nested, `{first: {second:
  value}}`), a `Timeline` and `Timelines` have one too.
- **A simulated check of an exact mean (#233).** `StandbyModel`,
  `LoadSharingModel` and `DegradingNode`'s `mean(method="simulate")`
  estimates the mean from `mc_samples` draws (10,000 by default) with
  `seed`, as `NonRepairableRBD.mean` does; `method="exact"` refuses the
  draws' options.
- **A diagram's levers are public (#244).** `RepairableRBD.levers()` and
  `NonRepairableRBD.levers()` list what `parameter_sensitivity` moves, in
  the order it reports them, as `Lever` results: whose each is and its
  name, as the sensitivities key them, its value, the range of its values
  (one outside it is refused), whether it is discrete (one more standby
  unit or repair crew), and whether it moves a calendar its component
  shares with others (whose long-run sensitivity takes its schedule
  apart). `with_levers({lever: value})` builds the diagram with levers
  moved, as the sensitivities move them (a test interval taking its full
  tests with it, a common-cause group's members moved together), so that
  a what-if agrees with them. A report or an app no longer needs the
  private `repyability.rbd._sensitivity` to name, show or move the levers.

### Changed

- **Behaviour change: a `RepairableRBD` simulation's random numbers are
  counter-based (#209), so seeded results differ from 0.12's.** Uniform `k`
  of simulation `r` from a stream is now the `k`-th of numpy's Philox
  generator keyed from the run's seed and the stream's name, from the
  counter `(0, r, 0, 0)`: a function of those alone. Each block of
  simulations had a PCG64 generator of its own, whose numbers went to a
  simulation by the block's width, which followed from the component's
  models and the window, so a component's draws hung on how they were laid
  out. Now they do not: the layout (the blocks' widths and chunks) only
  decides how fast they are worked out, and `compare` and the control
  variate no longer give two systems' streams the same widths, which could
  make a seeded run's simulations differ with and without them. Results
  are as accurate as before, with other numbers: a seeded run gives
  different estimates, within their standard errors. With numba, Philox
  is compiled and a run takes as long as before; without it, working the
  numbers out takes longer (about 90 ns a uniform rather than 5), and a run
  of eight components on the Python engine took about a sixth longer.
  Engines from other
  packages read the run's blocks as before, but their numbers changed:
  `engines.API` was raised to 2 (and to 3 with #255).
- **`simulate_timelines` always records its histories in the event loop
  (#205).** On the Python engine, independent components' histories were
  drawn from their streams by a second implementation of their lives and
  repairs; the event loop now records them as it runs, as it did for every
  other diagram. The histories are the same, bit for bit, and
  `TimelineSimulation.method` is always `"event loop"` (it was `"streams"`
  for those diagrams). Without numba, such a run is slower: a bridge of
  five units over 20,000 simulations takes 7.6 s where it took 3.0 s, and
  a system with a nested diagram 9.7 s where it took 0.8 s. The numba
  engine, which records in its own loop, is unchanged; recording in the
  Python loop now costs about a quarter of the loop's own time, where it
  cost nearly a half.
- **A component that fails at once and is repaired at once is refused**
  when the `RepairableRBD` is built (an `ExactEventTime` of 0 for both its
  life and its repair). A simulation would change its state without end:
  `availability()` never returned, and `simulate_timelines` refused it.
- **numba's engine takes every level's events in one loop (#206).** The
  system's own events and a nested RBD's were written out twice in the
  compiled loop; one loop now takes both, a nested RBD's to its next change
  as it is wanted. Results are the same, bit for bit. A system with nested
  RBDs runs faster (a three-level system in about half the time), the
  others as before.
- **A decision diagram has one search and one replay (#207).** The search
  (`bdd.search`) and the replay (`shannon.replay`, `replay_gradient`) are
  each written once and run as Python or compiled by numba as written,
  where numba is installed and a core is large; the compiled search's
  separate design, its packing of states into integers and its fall back
  to Python for a wide frontier are gone. Plans are the same, step for
  step, and values the same, bit for bit, checked against plans and values
  recorded before the change. A core's compiled search takes about a third
  less time (a 12 by 24 grid, 0.15 s rather than 0.23 s); a compiled
  replay of 1,000 sets of probabilities takes 0.45 s rather than 1.0 s,
  and of its gradient 1.8 s rather than 2.9 s.
- **A run's changes are put in time order by numpy alone (#208).** The
  compiled radix sort and merge (`_time_order`, taken where numba was
  installed and a run had a million changes or more) are gone, with their
  tuning constants. The +1 and -1 changes of state are netted by one sort
  of integers (each time's bits above the change's sign), and the
  capacity's changes by one sort of (time, change) pairs. Results are the
  same, bit for bit. Without numba, a run of a million simulations puts
  its changes in order about 2.5 times faster than before; with numba, a
  capacity run of a million simulations takes about a tenth longer, and an
  availability run as long.
- **The compiled engine keeps whether the system works up to date, at
  every level, and builds no table of states (#255).** For a system of up
  to 20 components, and for every nested RBD, it built a table of whether
  the system works in each of the `2^n` states of its components, again
  on every run: a tenth of a second for 20 components, half of a short
  run. It now keeps the structure up to date as components change, as it
  did above 20 components, at every level in one structure; that is as
  fast as the table at every size measured (and a seventh faster at 20).
  A 1,000-simulation run of two lines of ten components takes 0.036 s
  rather than 0.119 s. Results are the same, bit for bit. A nested RBD of
  more than 20 components, which ran in Python, is now compiled.
  `engines.API` is now 3: the run's arrays an engine is handed changed
  (the system's no longer ends with a table, and `_System.kept` lays out
  every level's structure).
- **A run works out each stream's key once (#254)**, not once for each
  block of its draws: 40 keys rather than 800 for 20,000 simulations of
  20 components, about 0.035 s of a 0.65 s run. `_streams.Block` takes
  the key (`Plan.block` gives it). Laying the draws out more tightly was
  tried and left out: giving each simulation its own limit, extended
  alone, and starting nearer the expected draws worked out a third fewer
  values, but running again the simulations that ran out cost what that
  saved. Most of the draws' time is the models' quantile functions,
  whose overhead per value is raised in surpyval (SurPyval#769).
- **Faster capacity runs, timelines and conditional runs (#246, #247,
  #248)**. Timelines and conditional runs give the same results, bit for
  bit; a capacity run's probabilities can differ in their last digits.
  - A capacity run adds up equal levels with a sorted reduction rather
    than `np.add.at`, which sums a level's rows in another order (its
    probabilities equal to rounding, its levels the same), and tidies each
    level once: 25 redundant pairs with capacities (50 components),
    2,000 h, 500 simulations with numba, 0.31 s rather than 0.39 s; 70
    pairs on the Python loop, 3.5 s rather than 3.8 s.
  - A timeline's changes' histories and positions are worked out once,
    where every measure worked them out again, and a unit's own changes
    no longer store their index (their position): `simulate_timelines`
    on a 12-component bridge feeding a vote, 20,000 simulations, 0.69 s
    rather than 0.76 s, and the measures over its result 0.68 s rather
    than 0.79 s.
  - A conditional run puts the modules' changes in order with one sort
    (of a key that grows with the simulation and the time; equal keys
    then put in order exactly) rather than two multi-key sorts, and finds
    each joint state's stretches once: on a standby system of five
    components, 20,000 simulations, ordering the changes takes 0.11 s
    rather than 0.43 s, and a default `availability()` 7.1 s rather than
    7.5 s.
- **surpyval 0.23 or later is required** (0.22 was). Its next release drops
  0.22's name for a limited-failure population's share that ever fails,
  `p`, for `lfp_p`, which 0.22 does not know, so no example could be
  written for both; 0.23 also pickles its fits (SurPyval#573) and takes
  `success_run`'s level as `alpha_ci` (SurPyval#580). RePyability's
  workarounds for 0.22 are gone: a system of fitted models goes to `n_jobs`'
  worker processes as it is, rather than in its saved form, and models are
  read by `lfp_p` alone. Files saved before 0.10 with `p` in a model's
  extras still load.
- **The exact curves are built several times as fast** (`point_availability`,
  `mission_availability`, `expected_events`, `expected_cost` and the exact
  means a simulation run takes by default, #187). A 12-component system's
  exact mission availability over 5,000 h takes 0.33 s rather than 1.76 s,
  and a 70-component one's 0.86 s rather than 3.7 s:
  - each component's curve is built on a grid of 1,000 steps over its
    typical up time, rather than 2,000. The error falls as the square of
    the step: about 4e-7 at a point (up to 4e-6 soon after the start), and
    about 4e-8 in a mission average, four times what it was. Exact values
    change in their seventh or eighth significant figure;
  - a curve that has not settled at its long-run value is followed to
    where, judging by how fast it is settling, it will have, rather than
    four times as far: most of the work went on curves that had long since
    settled;
  - identical components (the same life and repair models, the same state
    at 0, with no schedule, tests or imperfect repair) share one curve;
  - each series in a curve's convolutions is transformed once.
- **A model given for the input or output node is refused (#217).** The
  input and output nodes are inferred from the edges and never fail, so a
  model given for one was dropped: forgetting a component's edge to the
  output node made it the output node, and the answer changed with no
  warning. Both RBD classes now refuse it as an invalid structure, naming
  the edge most likely missing ("'belt' is the output node ..., so the
  model given for it would be ignored: did you forget an edge from 'belt'
  to the output node?"). `PerfectReliability`, or a fixed probability of
  failing of 0 (as a `node_availability()` of 1 gives), may still be given
  for an end, and `on_infeasible_rbd="warn"` keeps the old behaviour.

- **A cost result's `mean` is its run's estimate of the expected cost
  (#223)**, the value its `mean_interval()` gives: exact where the exact
  methods work it out (by default since 0.12), taken given the modules'
  histories where those apply, and otherwise the simulations' own.
  `mean_se`, `cost_rate`, `by_category` and `by_component` follow it, the
  breakdowns exact or conditional too (but under `control_variate=True`,
  whose twin controls only the total). The simulations' own mean and its
  error are the new `sample_mean` and `sample_se`. Likewise the new
  `AvailabilityResult.mean_availability` is the estimate
  `mean_availability_interval()` gives, and `sample_mean_availability`
  the simulations' own, `system_uptime / (n_simulations *
  time_simulated_to)`. In 0.12 a result carried two expected values, and
  `mean`, `cost_rate` and the breakdowns were the noisier one, up to 3%
  off the exact value in a persona's study. Both results' reprs now
  summarise the run (the estimate, its standard error and method, the
  simulations' own mean and spread), where they printed every sample.
  `control_variate=False` keeps the simulations' own values, as before.

- **A fixed-probability node's simulated lifetime is 0 or infinite.**
  `NonRepairableRBD.random()` took surpyval's draw for it, a 0/1 event
  indicator, as a time, so the samples of a diagram with one did not
  follow its `sf` (the docstring called them "not meaningful"): the node
  now fails at the start with its probability, and otherwise never, as
  `sf` takes it. Seeded draws of such a diagram change. An MTTF estimate
  (`mean_time_to_failure_interval`, `mean(method="simulate")`) refuses a
  diagram of fixed probabilities alone, as the exact `mean` does: it fails
  at the start or never, and has no lifetimes to average.
- **An MTTF takes a few hundred evaluations of the survival function
  (#229),** where it took thousands: the integral split at every knot of
  every node model (127 quantiles each) and doubled into the tail up
  to 1e300 at once. It now starts from a few pieces a decade (one up to
  where the curve first falls by 1e-6), the knots thinned to two a decade
  but every kink kept (a numerical curve's grid, where a support starts),
  each piece integrated by the Gauss-Kronrod (7, 15) rule, whose Gauss
  points bound its error at no extra cost, and the tail followed only
  until it is 0. A diagram of 20 different Weibulls in series takes 513
  points where it took 63,100, to the same 1e-10. `Network.mean()` on the
  issue's 10x10 grid takes 7 s, where it took 209: its replay evaluates
  more times at once too (11 a chunk, at 12 ms a time, where two took 28
  ms), as does `sf` at many times.
- **A network refused as too meshed is refused at once after (#229)**:
  each call repeated the search that refused it (seven seconds on an 11x11
  grid).
- **Fussell-Vesely on a meshed structure takes half the memory (#229)**:
  the decision diagrams' table of combinations worked out starts again
  past half a million, where it kept 1.6 million (250 MB) for a mesh of
  50 nodes; a cofactor leaves the nodes after its variable alone.
- **Choosing the test intervals of a common-cause group, or of a unit
  whose tests take time, is faster (#229).** A group's chain keeps
  `exp(G dt)` for the steps it takes again, period after period, where it
  summed the series anew each time (48,000 times for one plan of a 2oo3
  group); a plan keeps its groups' states for its cost rate and its
  availability, and the plans of one search share the tested units'
  models. Each plan of the issue's 2oo3 group takes 0.2 to 0.3 s where it
  took 1.4 to 6.5, and its choice among five intervals 32 s where it took
  506. A tested unit's walk convolves directly or by FFT without scipy's
  choosing each time.
- **Of interval plans as good, the search keeps the first it tried
  (#229)**: plans whose cost rates are within 1e-10 of each other's, or
  whose availabilities are within 1e-10, are as good, where the last bits
  of their arithmetic chose between them. Which of identical members is
  tested more often no longer turns on them, and of plans that cost the
  same, the most available is chosen however their costs' last bits fall
  (as a stagger's offsets, which change no cost, are chosen).
- **`compare` is exact where both designs' expected values are (#236).**
  It simulated both designs with common random numbers even where the
  exact methods give the difference: two pump trains' availability
  0.005787 ± 0.000172 against the exact 0.005676. Now, as `availability`
  and `cost` take their means, where both `RepairableRBD`s' mission
  availability (and expected cost) are worked out, `compare` gives their
  difference with no error and no simulation (`method="exact"` on the
  result); `control_variate=False` simulates it as before
  (`method="simulated"`). A `NonRepairableRBD`'s `compare` gives the exact
  difference of the MTTFs where `mean` works both out, and simulates on
  request (`method="simulate"`, or where a `mean` is refused).
  `analysis_routes()` and the README's table say so.
- **`compare(quantity="cost")` counts the components' acquisition (#234).**
  It compared the running costs alone, so "is a second pump worth buying?"
  overstated the saving by its price: a second 20,000 pump "saved" 4,273 a
  year where it costs some 15,750 more to own. The cost compared is now what
  owning each design for the window costs, as `total_cost` and
  `expected_cost(...).total` count it, and a design priced by its purchase
  alone can be compared.
- **`cost()` with only an acquisition cost (#234)** gives a `CostResult`
  with no running cost, exactly, and the acquisition beside it, where it
  gave None (and `.mean` raised); so does `availability()`'s result. A
  system with nothing priced at all still gives None.
- **Inputs are refused where they are given (#233).** A cost given as
  text or as True or False (`repair_cost: "800"` was priced at 800, `True`
  at 1) is refused, as other non-numbers were, and so are an interval
  given as text (an inspection's `"interval": "8760"`) and times given as
  text (`sf("8760")`), with a TypeError naming them. surpyval's
  distribution itself (`surv.Weibull`) given where a model of it goes,
  which built and failed in surpyval at the first evaluation, is refused at
  once by a `NonRepairableRBD`, `RepairableRBD`, `FaultTree` or
  `NonRepairable`, saying to fit it or give its parameters; a repairable
  diagram given a life alone says it needs its repairs too. A part of
  `spares_demand` or `spares_stock` that lists a component twice, or a
  component in both a part and `nodes`, which stocked its spares on two
  shelves, is refused, as `allocate_redundancy` refuses a node in a train
  and `nodes`.
- **A test offset within a billionth of the interval is no offset
  (#237).** An offset of 0 puts a component's first test at its interval,
  but any positive one put a test at the offset, near the start: one more
  test, its cost and its outage, and a unit off line at once, so an offset
  worked out as a share of the interval fell on either side of the jump.
  An offset no more than `1e-9 * interval` is now 0. The constructor's
  docstring and the costs guide say what 0 and a small offset do.
- **Seeds are checked where they are given (#232).** A seed is a whole
  number from 0 to 2**32 - 1, a list of them, or None; a numpy
  `Generator`, refused deep in numpy before, is refused saying how to draw
  a seed from it (`seed=int(rng.integers(2**32))`).
- **Shorter reprs (#235).** A `SparesDemand` prints its mean, standard
  deviation and how many probabilities it holds, and a `ControlVariate` and
  a `ConditionalRun` their values and how many simulations, rather than
  their arrays. (A diagram's repr was already a summary, and so was the
  bound method printed when `()` is left off.)
- **Warnings point at the line that called the package (#232)**, where
  several pointed inside it.

### Deprecated

- **`optimal_inspection_intervals(offsets=)` is renamed `offset_shares=`
  (#222).** Its values are shares of the interval, where
  `with_intervals(offsets=)` and the plan's `offsets` are times, so a
  share passed to `with_intervals` undid a stagger without a word.
  `offsets=` still works, with a `FutureWarning`; 0.14 refuses it.
- **`CapacityDistribution.mean` is a property (#235)**, as the other
  results' values are and as 0.12 made `SparesDemand.mean`. Called,
  `mean()`, it still gives the mean, with a `FutureWarning`; 0.14 refuses
  it.
- **The simulation options of an exact mean (#233).** `StandbyModel`,
  `LoadSharingModel` and `DegradingNode`'s `mean(mc_samples, seed)`
  ignored them without a word where the mean is exact (or numerical); they
  now warn, with a `FutureWarning`, and 0.14 refuses them.
  `mean(method="simulate", ...)` simulates it.
- **The private names callers used for the levers (#244).**
  `repyability.rbd._sensitivity`'s `levers` and `Lever`, and its
  `_calendar_lever` and `_as_spec`, stay for 0.13 (its `Lever` tuple with
  a `bounds` field at its end), and may change or go in 0.14: use
  `levers()`, `Lever` (its `calendar` and `value`) and `with_levers`.

### Removed

- **What 0.12 deprecated.** `SparesDemand.mean` and `std` are properties
  alone: calling them, `mean()`, raises `TypeError` (#184). `StandbyModel`
  and `LoadSharingModel` no longer take `mc_samples`, `lower` or `seed`,
  which set a fit to simulated lifetimes that 0.12 removed (#149):
  passing them raises `TypeError`. `StandbyModel`'s `switching_probability`
  and `dormancy_factor` came after them, so they are now given by name, and
  a call that gave them by position is refused rather than misread.

### Fixed

- **`availability_rate` just after a scheduled maintenance** (#240). A
  component's rate was its whole curve differenced over its grid's step,
  which smoothed over a down time far shorter than a step: the greeks
  guide's valve, maintained for about 6 h on a grid of 0.04 to 0.08 days,
  had its rate a day after its replacement 1-2% low (0.0514, or 0.0509 on
  the coarser grid, for 0.0518). The down times a curve keeps off its grid
  (its dips, the later maintenances of an age-replaced unit and the return
  from a block replacement) are now differentiated on their own scale, and
  only the grid is differenced; a nested RBD's rate is its nodes' times
  their importance. The two guides' quoted rates change (0.0509 to 0.0518,
  0.07493 to 0.07491), each now within 1e-4 of its value on grids sixteen
  times finer.
- **`optimal_inspection_intervals` chooses the intervals of tests that can
  miss a failure (#221),** which 0.12 listed as numerical but refused: a
  component whose tests have a `coverage` below 1 keeps its full tests'
  interval, and its test interval is chosen among those that divide it,
  from `allowed` (an interval that does not is refused, by name, with
  some that do) or, left out, from every one in range.
- **Offsets and per-node options are checked (#222).** An offset given to
  `with_intervals` for a component with no tests is refused also when its
  interval is given (it was dropped); `allowed=` and `offset_shares=`
  dicts that name a node not chosen are refused, naming it, where a typo
  was dropped; a node that is not a component, in `nodes=` or
  `with_intervals`, is refused listing the components; and a plan from
  `optimal_inspection_intervals` has its `offsets` whether they were
  searched or not.
- **Parameter uncertainty draws a shared fitted model once (#214).** In a
  `RepairableRBD`, nodes holding the same fitted life but repair fits of
  their own (one fleet's life fit, repairs recorded by site) had their
  life drawn per node, by default, and no spec could share it: the
  intervals of `*_uncertainty` were some 40% too narrow, and the vega
  split wrong. Draws are now shared per fitted object in each role: a
  node may come under several inputs, a role of it under one
  (`{("a", "b"): {"reliability": "fit"}, "a": {"repairability": "fit"},
  ...}`), which the default now gives. In both diagram classes, a spec that
  draws one node and leaves others holding the same fitted object as they
  are is warned about. Seeded draws change where models were shared in
  some roles only.
- **A conditional run whose modules never changed state no longer reports
  a certain answer (#215).** When the dependent modules met only the state
  they started in, in every simulation (a standby pair that never went
  down), each simulation's expected values given them were the same: the
  run reported a standard error of 0, and a `tolerance` was met after the
  first batch, for the value of the rest of the system alone. Such a run
  now warns; a default run's mean intervals are its simulations' own
  (`method="simulated"`), and judge its `tolerance`; a run of the modules
  alone (`conditional=True`) has no error to give (`nan`), and runs on to
  `max_samples`. A run with no modules stays exact.
- **Common-cause groups no longer double the time per group (#219).** A
  `FaultTree`'s or `NonRepairableRBD`'s exact values conditioned on every
  combination of every group's shock outcomes: 15 groups took 2 s for the
  top event, and 8 over two minutes for the importance measures. A group is
  now conditioned on only within the smallest module of the structure
  holding its members, so groups in separate modules cost a sum each (the
  issue's 40 groups take milliseconds); where a module's groups would
  multiply past 64 outcomes, as a group of each kind of component across
  redundant trains does, their shared causes are written out as events of
  their own, repeated under every member they strike, which the decision
  diagram works out with the rest (30 groups across three trains in a
  tenth of a second). An MGL model's exclusive shocks are taken as
  independent causes that fail the same sets of members as often, which
  exist unless the model leaves out a set two pair shocks would fail
  together (`gamma = 0`), when such groups are conditioned on together, as
  before. The values are those of every combination, to rounding: the top
  event, `sf`, `ff`, the importance measures (Fussell–Vesely conditioning
  within the modules), `ranked_cut_sets` and the rare-event Fussell–Vesely
  (a product over the groups). The capacity distribution and the
  redundancy allocations still condition on every combination.
- **Common-cause groups in a `RepairableRBD` no longer take memory that
  doubles per group (#218).** The long-run values, the importance
  measures, the failure frequency (MTBF, MUT, MDT) and the values over
  time split each time by every combination of every group's members up
  or down: ten groups (pairs of tested units in series) were killed by the
  operating system for memory, with no message. Each group is now
  conditioned on only within the smallest module holding its members, as
  for a non-repairable diagram (#219), an owner's combinations worked out
  together a bounded chunk at a time: fifty such pairs take a third of a
  second, in flat memory. Where groups meet in one module (kinds of
  component across redundant trains) the values over time take far less
  too: with five kinds across three trains, `point_availability` at 200
  times took 42 s and 4 GB, and takes 0.1 s. Where a module's combinations
  would still be too many, the exact values refuse before working anything
  out, saying to simulate; the capacity distribution and the allocations,
  which still take every combination at once, refuse where that would
  take too much memory. The values are those of every combination, to
  rounding.
- **A common-cause group's members' tests and repairs may take time
  (#220).** The most common SIL calculation, a 1oo2 or 2oo3 with a beta
  factor and a mean repair time, was refused by every exact method. A
  tested member's level in its group's chain may now be off line for a test
  while working (where it neither ages nor is struck), under a test while
  failed, or under repair: a test or repair of a fixed length ends a fixed
  time after its test, and one of an exponential length at a rate, and a
  test that falls in a member's own test or repair is not done, as in the
  simulations. The long-run values, the importance measures and the values
  over time are numerical (to the members' own models' grids): the issue's
  1oo2, with an eight-hour MRT, has a PFDavg of 9.591e-4 against IEC
  61508-6's estimate of 9.609e-4. Refused, each saying what to do: tests or
  repairs of another distribution, a test of an exponential length
  followed by a repair of a fixed one, fixed lengths as long as the test
  interval, copies of such members in the allocations, and the failure
  frequency where their tests take time (planned outages). The chains'
  other refusals now say what to do too, and name a member as "the life of
  member 'v1'" where they printed "member 'v1''s".
- **Common-cause groups are no longer dropped with limited repair
  crews (#251).** With fewer `repair_crews` than jobs, `mean_availability`,
  `mean_unavailability`, the importance measures, the failure frequency
  and the capacity distribution gave the system without its groups (the
  crews' chain does not take common causes in), and `analysis_routes`
  called them exact. They now refuse, as the values over time did, and so
  does `node_availability`, which gave each component's value in the
  crews' chain without the groups, where a shared failure leaves one
  member waiting for the crew. The refusal now says to simulate, with
  `availability()` or `cost()`, which take both.
- **`ConfidenceInterval.method` names every case of a repairable run's
  mean (#223):** `"simulated"`, `"control_variate"`, `"conditional"` or
  `"exact"`, where the simulations' own mean and a controlled one were
  both `None`. The docs' examples are now checked for the `True` and
  `False`, strings, tuples and lists they quote, as well as their
  numbers: the costs guide's check that the interval holds the expected
  cost printed `False`, and the stepped simulation's first outage was
  quoted at a stale time.
- **Simulations of one diagram from several threads (#216)** crashed
  (`AttributeError: ... '_cancelled'`) or returned another seed's result:
  the event loop keeps a run's state on the diagram, and draws that cannot
  be streamed come from numpy's global RNG. The simulations now take turns,
  one run at a time in the process, so each call gives what it gives
  alone; the exact methods, and `n_jobs`' processes and the compiled
  engine's threads, are not held up.
- **`NonRepairableRBD.node_mttf()` leaves junctions out (#228)**, as the
  importance measures do, where it raised an `AttributeError` on a
  `PerfectReliability` node though `analysis_routes()` reported it exact;
  a `PerfectUnreliability` node's is 0. Both helpers have a `mean()`, as
  surpyval's models do: `inf` and 0.
- **`analysis_routes()` says what each method does on every kind of
  diagram (#239).** The routes test now calls every analysis the report
  lists (27 were never called) on a catalogue of 22 non-repairable and 49
  repairable diagrams, with junctions, k-out-of-n votes, every node class,
  every spec key, policy and option, and a guard that fails when a new one
  has no diagram. It found the report calling exact or simulated what the
  method refuses: `NonRepairableRBD`'s parameter-uncertainty methods on a
  diagram with a node that has no reliability or a structure too meshed to
  work out; `minimum_effort_allocation` off a series system (now found
  from the graph, a pass per node, rather than from the cut sets); a
  `system_capacity` whose nodes take their capacity from their models;
  and a repairable availability allocation with no component to allocate.
  Each is now refused in the report with the method's own message. The
  catalogue is shared by the engines' test, which checks that numba's loop
  agrees with the Python one on every diagram it runs, and the exact and
  numerical routes are checked against the simulation of the same
  quantity on each diagram (MTTF, reliability, availability over a window,
  expected cost).
- **A surpyval `MixtureModel` node** no longer breaks `analysis_routes()`
  or saving: its `dist` is only its components' distribution, so it was
  taken for a plain distribution of that name. It is saved with
  surpyval's `to_dict()`.
- **A node's name given alone is that node (#225).** `working_nodes="belt"`
  was taken as the nodes `"b"`, `"e"`, `"l"` and `"t"`, so the error
  changed from run to run with the strings' hashes, and a word made of
  one-letter node names was taken without a word. A string given to
  `working_nodes`, `broken_nodes` or `nodes`, on every method of either
  diagram class, is now one node, as `FaultTree.occurs("ab")`'s is one
  event; the unknown nodes are listed, in order, in one message.
- **The uncertainty methods take what the point methods take (#226).**
  `bx_life_uncertainty([1, 10])`, `time_to_reliability_uncertainty([0.99,
  0.9])` and `sf_uncertainty` of a 2-d array of times failed with numpy's
  errors, and `uncertainty_importance([10, 50], of="mission_availability")`
  blamed the window for its `x`: each now gives one row of draws per
  target, from the same parameter draws (each element what that target
  alone gives), with the times' shape; a percentage or reliability out of
  range is refused naming its argument. A life some of whose units never
  fail has an infinite mean in every draw, and `mean_uncertainty` infinite
  bounds, where it gave `nan` with numpy's warning.
- **A discount rate given per year with models in hours is warned of
  (#231).** `discount_rate=0.07` with models in hours discounts every cost
  after the first fourteen hours away: owning a 20,000 pump for ten years
  came to 20,021 where it is 114,538 at 7% a year. `total_cost`,
  `allocate_redundancy` and `expected_cost` now warn when the rate
  discounts the components' shortest mean life by more than `exp(-20)`, or
  a horizon by more than `exp(-1000)`, saying how to convert an annual
  rate (`math.log(1 + i) / 8760`). A long horizon at a high rate is no
  mistake, and is not warned of.
- **0.12's notes corrected (#230).** Its behaviour changes said an MGL group
  splitting a probability takes PRA's independent shocks: the default is
  mutually exclusive shocks, as before, and `shocks="independent"` is asked
  for. Its staggered-tests entry gave `optimal_replacement_intervals` an
  `allowed` it did not have (it has now, see Added), and the MGL example's
  probability is 0.0311.
- **Messages that say what to do (#232).** The names 0.12 removed (`N`,
  `max_N`, `n_sims`, `n_simulations`) are refused naming what took their
  place, where Python said only "unexpected keyword argument", and
  `fussel_vesely` names `fussell_vesely`. A node name with a typo (in a
  spares part, a train, a common-cause group, a state or an allocation)
  is answered with the closest name, whatever its case, and the
  components; a junction given where a component goes is called a
  junction. A role or parameter with a typo in an uncertainty spec is
  answered with the closest; a spec mixing the two says so; and
  `uncertainty_importance(of="cost_rate")` is `expected_cost_rate`. An
  outage log says which outage is wrong and how (not a pair, a start or
  end that is not a time, outside the window, ending before it starts)
  and, out of order, to give `merge=True`. Gates that form a loop say so,
  naming one, where the tree asked for its top event. `spares_stock` with
  repair crews says it has no simulation to fall back on, and what gives
  the stock when a crew is always free. And no message reads "'a''s".
- **`PerfectReliability()` is `PerfectReliability` (#232).** An instance,
  as a repairable diagram's node or a spec's `reliability`, is taken as
  the class, where the diagram refused it (a `NonRepairableRBD` took it
  already); a junction's spec without repairs is saved and loaded; and a
  `NonRepairable` given one says a life must end.
- **A single common-cause group (#232)** is taken by `ccf_groups=` as a
  list of one, by both diagram classes and `FaultTree`, where it was
  refused as "not iterable".
- **`NonRepairable(life, "instant")` (#233)** replaces in no time, as a
  spec's `"instant"` repairs, where it was taken as a model and failed at
  its first use; any other text is refused.
- **`UncertaintyResult.percentile` and `CostResult.percentile` warn of a
  share (#233)**: a `q` between 0 and 1 is taken on numpy's 0 to 100
  scale, as before, but warns that `percentile(5)` is the 5th percentile,
  where `interval()` takes shares.
- **An infinite risk reduction worth (#235)** no longer warns of dividing
  by zero in a `NonRepairableRBD` or `RepairableRBD`, as it did not in a
  `FaultTree`.
- **`FaultTree.from_rbd` keeps a common-cause member the logic makes
  irrelevant (#237)**, `b` in `a OR (a AND b)` with `a` and `b` in one
  group, where it was refused: the tree keeps it under the top event as
  `OR(G, AND(G, b))`, which is `G`, for the group's shared cause to
  strike it with `a`. 39 of 150 random trees with one common-cause pair
  were refused; every one now goes round.
- **Staggered tests in a common-cause group restore only the member
  tested (#237)**: each member is restored by its own test, as in the
  simulations; the costs guide said a shared failure was "found by
  whichever test comes first" and "twice as soon", as if a test restored
  both, which gives a little less (practice that restores both channels
  at the first test is not modelled).
- **The docs after 0.12 (#238).** The saving guide said a standby or
  load-sharing node with no exact reliability was fitted to simulated
  lifetimes, which 0.12 removed; the spares guide described `mean()` and
  `std()`, properties since 0.12; the common-cause guide, the concepts and
  `MGL`'s docstring said a member fails with probability `Q` exactly, where
  it is `Q` to first order (`Q - β(1 - β)Q²` for one shared cause, as in
  PRA's basic events: 0.0991 at `Q = 0.1`, `β = 0.1`); and
  `sf_given_state`, `remaining_life` and `mean_residual_life` said a
  number given as a node's state is refused, where it is its age.
- **0.12's features are easier to find (#238).** Junctions, pooled spares,
  discounting, whole trains, plans that keep both risks, common causes in
  fault trees and the unavailability over time are in the README's list,
  the docs' home table and the guide's index; the glossary defines
  junction, delta, gamma, theta, vega, differential and Barlow–Proschan
  importance, discount rate, fill rate, lead time, producer's and
  consumer's risk, and unavailability.
- **`help()` reads well (#238).** The package's docstring says where to
  start (the diagram classes, `analysis_routes()`, the docs); the API
  reference's cross-references read in `help()` as the names they link,
  where they showed as markup (the web pages, built from the source, keep
  their links); the sensitivity
  measures name the Greeks the guide calls them by; and `availability`'s
  longest options (`control_variate`, `conditional`, `engine`) point to the
  guide rather than repeat it. `initialize_event_queue` stays public, as
  the guide steps a simulation by hand with it, and the wheel keeps its
  tests (#167).
- **The values over time of a tested component with an exponential life
  are no longer refused on some platforms.** Where its tests or repairs
  take time, its curve follows its state from new until it is the long
  run's, which it took to be within 1e-15 of it. The state settles a few
  roundings from the long run's, how few depending on the platform's
  numerical libraries, and on some (CI's Python 3.12 and 3.13) it did not
  come that close: `point_availability`, `mission_availability`,
  `expected_events` and its other values over time were refused as not
  settled at times later than its curve can follow (some 2,000 tests). It
  now has to be within 1e-12, as a component's with any other life
  already did.

## [0.12] - 2026-10-04

Exact where there were estimates, and the sensitivities as one family. A
simulation run's expected availability and cost are exact by default where
the exact methods give them, and otherwise taken given its dependent
modules' histories, which a conditional run simulates alone, the rest exact
(#187, #189); a run controlled by the system itself is exact (#186).
Common-cause groups in a repairable diagram are followed over time, in the
simulations and in the allocations (#158); hidden failures whose tests or
repairs take time, or whose tests miss them, are numerical for any life
(#159); replacement on condition (#161) and repair crews around nested RBDs
(#162) are exact over time, and so is minimal repair in no time over a
window (#179); the spares of block-replaced components are counted, also
when their repairs or replacements take time (#160), and interchangeable
parts share one shelf (#183). The sensitivity measures read as one family,
the Greeks (#197): a repairable diagram's importance over time (delta,
#191), the sensitivity of its availability and cost to each lever (#192),
the shares of a change (#193), complements and substitutes (gamma, #194),
which component is moving the availability now and which caused the
failures (theta and Barlow–Proschan, #195, #199), and whose parameter
uncertainty widens the answer (vega, #196), with parameter uncertainty in a
repairable system and quasi-random parameter draws (#200). A repairable
diagram takes junctions, which are in no cut or path set (#175, #182,
#198); simulated timelines start from the plant as it is (#163); fault
trees take common-cause groups, demonstration plans keep both risks,
intervals are chosen for limited crews and staggered tests, total costs are
discounted and redundancy is allocated a train at a time (#184). Meshed
diagrams, networks and fault trees are worked out in a fraction of the
time, and a core too meshed to work out is simulated rather than built
without end (#171, #172, #173); with numba, large decision diagrams are
built and replayed compiled (#202); the analyses over a window cost little
more than the point curve (#164); and a capacity run's curve follows
`curve_points` (#190) and is built in linear passes (#201). Everything 0.11
deprecated is gone (#149).

Behaviour changes: what 0.11 deprecated is removed (#149): the old
simulation-count names and ignored arguments raise `TypeError`, a diagram
refuses a non-parametric node, and a standby or load-sharing model with no
exact or numerical reliability refuses `sf`, `ff`, `cs` and `mean()`,
leaving it to the simulations, where a fit to simulated lifetimes stood in;
`availability()` and `cost()` give the exact or conditional mean by default
(`method="exact"` or `"conditional"` in their mean intervals), and
`control_variate=False` keeps the simulations' own (#187, #189, #186);
`availability_from_chunks` refuses chunks with simulations missing, unless
`allow_gaps=True` (#176); a simulation over a window that is not positive
and finite is refused (#174), and so are bad counts and structures (#168,
#169, #178) and a count of `True` in the demonstration functions (#179);
junctions leave the minimal cut and path sets (#198); the window analyses
move by a few parts in 10^9, and the fixed planned outages after age
replacement and expected failures of hidden failures by up to a few in
10^4 (#164); `SparesDemand.mean` and `std` are properties, and calling
them, like `StandbyModel`'s and `LoadSharingModel`'s `mc_samples`, `lower`
and `seed`, warns until 0.13 (#184, #149); and surpyval 0.22 is required.

### Added

- **Junctions in a `RepairableRBD` (#182, #175).** A node given
  `PerfectReliability` (or a spec whose `"reliability"` it is) is a
  junction: it never fails, so a k-out-of-n vote can sit anywhere, such as
  two 2-of-3 stages in series, where before it could only be the output
  node's. It is no component: it is folded out of the structure
  (`modular.fold`: a series module leaves it out, a parallel one with it
  always works, a k-out-of-n one needs one fewer of the rest), so every
  analysis and both simulation engines see the components alone, and give
  what the same diagram drawn without the junction gives. The capacity
  analysis lets it pass what reaches it, up to a capacity if it has one; it
  takes no costs or maintenance, and cannot be held working or broken.
  `PerfectReliability` as a repairable component's life failed with
  "Unknown reliability function". A `NonRepairableRBD` node in the edges
  with no model is now told that a junction takes `PerfectReliability`, and
  `NonRepairable(PerfectReliability)` says to give it as the node itself.

- **One shelf for interchangeable parts (#183).** `spares_demand` and
  `spares_stock` sized each component's spares apart, where identical parts
  share one bin: the seals of a station's three pumps. `parts={part:
  [nodes]}` pools them under the part's name. The positions' demands are
  independent, so a part's is their sum; for its stock, a demand comes
  from each position with its share of the long-run replacement rates,
  finding its own position's spares on order as that position's demands
  do and the others' as at a random time. The positions may differ (one
  under age replacement, the others not). Thirteen stations of three
  wear-out seals, six weeks to restock, need 21 seals on three shelves for
  a 95% fill rate, and 17 on one. Members of one common-cause group, which
  their shared causes replace together, are refused; the results say which
  `members` a part holds.
- **Demonstration plans that keep both risks (#184).** Each demonstration
  function answered one question, so designing a test that a design at the
  target passes rarely and a good design passes often meant searching by
  hand; a success run that keeps the consumer's risk fails a design 50%
  better two times in three. `demonstration_plan(reliability,
  good_reliability, confidence, producer_risk)` gives the fewest units and
  the failures to allow (or, given `n` and the Weibull `shape`, the
  shortest test), and `mtbf_demonstration_plan` the least test time, for
  a constant failure rate: MIL-HDBK-781's fixed-length plans, keeping both
  risks. Both return a `DemonstrationPlan` with the plan's risks.
- **Intervals with limited repair crews, and plans applied (#184).**
  `optimal_replacement_intervals` and `optimal_inspection_intervals`
  failed with limited `repair_crews`, saying only to simulate, though
  crew-limited plants are the norm. They now refuse with the way through,
  and `assume_unlimited_crews=True` chooses the intervals as if every
  repair started at once. `RepairableRBD.with_intervals(plan)` gives the
  diagram with a plan's intervals (and test offsets), built as the
  diagram was, to simulate with its crews (`cost()`, `availability()`)
  or `compare()` with the schedules it has.
- **Staggered tests chosen with their intervals (#184).**
  `optimal_inspection_intervals` chose intervals but tested every
  component from its offset as given (usually 0), where testing redundant
  components apart finds a common-cause failure sooner: a 1oo2 with a 10%
  common cause has half the PFDavg with yearly tests six months apart.
  `offsets=` chooses the first tests' times too, as shares of each
  interval (a list, a dict per node, or `"stagger"` for even spreads),
  searched with the intervals; the plan's `offsets` gives them. At a
  PFDavg of 5e-4 that plan costs a third less than the best tested
  together. Of plans that cost the same, the most available is now chosen
  (it was the first found).
- **Discounted total costs (#184).** `total_cost` and
  `allocate_redundancy` were undiscounted, so over a 20-year life a copy
  bought now weighed the same as the running costs it saves later.
  `discount_rate=r`, a continuous rate per unit time of the models
  (`math.log(1.07) / 8760` for 7% a year in hours), gives the present
  value: the components bought at the start, the running costs discounted,
  the horizon counting as `(1 - exp(-r H)) / r`. A fourth pump train that
  pays undiscounted no longer does at 15% a year. `TotalCostAllocation`
  reports the `discount_rate`. Undiscounted by default, as before.
- **Redundancy a train at a time (#184).** `RepairableRBD.allocate_redundancy`
  copied single nodes, each in parallel with its own, so "should we add a
  fourth pump train?" could not be asked of it. `trains={name: [nodes]}`
  names chains of components in series that may be given copies as a
  whole: a copy is another path alongside the train into the node it
  feeds, which keeps its `k`, so copies of a train of a 2-out-of-3 vote
  make it 2-out-of-4. Each design is scored exactly, the copies drawn out
  as trains of their own, and `units` counts a train and its copies under
  its name (the result's `trains` lists their nodes). Given `trains`,
  `nodes` is by default none.
- **Common causes in fault trees (#184).** `FaultTree(..., ccf_groups=)`
  takes the `CCFGroup`s a diagram takes, over basic events: a tree could
  only draw a shared cause as a repeated event of its own. The top event
  probability and every importance measure sum over the groups' shock
  outcomes, exactly, as the diagram's do (a member conditioned on its own
  state through them); `ranked_cut_sets` gives each cut set's probability
  of its events occurring together, the shared causes included. `to_rbd()`
  and `from_rbd` keep the groups, where `from_rbd` refused a diagram with
  any, and the tree saves them. `from_rbd` drops a group none of whose
  members can affect the system, and refuses one with a member that cannot
  and one that can.
- **PRA's independent shocks for MGL groups (#180).** Splitting the
  probability, an MGL group's shocks were mutually exclusive (one shared
  cause at most), where PRA codes (SAPHIRE, CAFTA, RiskSpectrum) take each
  specific set's `Q_k` as an independent basic event: the two differ at
  second order in `Q` (a 2-out-of-3 group of `MGL(0.2, 0.3)` at
  `Q = 0.0311` fails with probability 0.010219 one way and 0.010192 the
  other).
  `MGL(..., shocks="independent")` combines them as PRA codes do (exact,
  over the unions of the causes that strike), to check a result against
  one; the default is as before. It is saved, kept by parameter changes
  and draws, and a `BetaFactor` or a single-shock MGL agrees either way.
  The guide and `MGL`'s docstring say so where the model is described.
- **`mean_residual_life(state)` on `NonRepairableRBD` (#179).** The mean
  remaining life given the components' states, the area under
  `sf_given_state` from now: `remaining_life` gives a percentile of it (the
  time to a reliability target), and this its mean.
- **Minimal repair in no time is exact over a window (#179).** A
  component repaired imperfectly (Kijima) was refused by every exact
  method. Minimally repaired (`q = 1`) and instantly, with no
  `replace_after`, preventive maintenance or tests, it is up throughout and
  its failures are a Poisson process whose rate is its life's hazard at its
  age, so it fails `H(t)` times by `t` on average, `H` its life's
  cumulative hazard. `point_availability`, `mission_availability`,
  `expected_failures`, `expected_events` and `expected_cost` take it in,
  its repairs charged their `repair_cost` alone, as the simulation charges
  them; `analysis_routes()` and the README's table say so. Its long-run
  values still refuse, as its rate of failures need not settle, and other
  imperfect repair is still simulated, the refusal naming what stands in
  the way.
- **Simulated timelines from the plant as it is now (#163).**
  `simulate_timelines` always started the components new.
  `state={node: NodeState}` now starts them as `availability(state=...)`
  does: their ages, a repair or maintenance under way, a nested RBD's own
  states. The histories are that run's simulations, each kept whole: a
  component down at 0 starts its history down, and the system its own in
  its state then, with no change at 0, so the time to the next failure
  and its cause can be read off today's plant. The event loop records
  them, in Python, as the compiled engine starts no component part way
  through a life.
- **Hidden failures whose tests or repairs take time, or whose tests miss
  them, for any life (#159).** A component with hidden failures had exact
  values only with tests and repairs in no time, and with tests that can
  miss a failure only for an exponential life; otherwise every exact
  method refused, and a proof test of hours on a valve tested yearly was
  simulated. Now a test of a working unit takes it off line (a planned
  outage) without ageing it, a failure is repaired once its test is over,
  the tests in the repair are not done, and a missed failure waits for the
  next full test, as in the simulation, numerically: the tests that find
  failures are the unit's regeneration points, and the cycle from one to
  the next is followed test by test, the unit's age at its tests a random
  walk on a grid (`_hidden_tests`), or, for an exponential life, its chain
  of states from test to test. The long run follows by renewal-reward over
  the cycle (with tests that miss, a Markov renewal over where the cycle
  starts in the full tests' period), and from new or from a state (up at
  an age, in a repair, or in its long-run state) the tests that find
  failures are a renewal process on the tests. So the long-run values, the
  frequencies and the planned outages at the tests, the cost rate (each
  test done charged), the importance measures, the availability over time,
  the expected events and costs of a window and
  `optimal_inspection_intervals` are numerical, to about 1e-8 (when a test
  is over, and a unit back in service, exactly). They agree with the
  closed forms and the sums of #144 where those apply, and with the
  simulation. Two valves of a 1oo2 pair proof-tested together for about
  3.6 hours a year have a PFDavg of 4.28e-4, the tests taking the function
  off line; tested half a year apart, 7.12e-5. `spares_demand` and
  `spares_stock` count such a component's spares too: its replacements
  still fall on the tests that find failures, a renewal process on the
  tests, or with tests that can miss a failure, a Markov renewal process
  over where each cycle starts between the full tests. A test that can
  last as long as its interval stays simulated (the refusal says how
  likely it is to), as do a common-cause group's members whose tests or
  repairs take time (#158).
- **The Greeks: the sensitivity measures as one family (#197).** A guide
  page, *Sensitivities: the Greeks*, reads them as one family, named after
  an option's Greeks: delta (`birnbaum_importance`, over time on a
  repairable diagram, #191), the levers' deltas (`parameter_sensitivity`,
  #192), their shares (`differential_importance`, #193), gamma
  (`joint_importance`, #194), theta (`availability_rate`,
  `reliability_rate` and `barlow_proschan_importance`, #195) and vega
  (`uncertainty_importance`, #196). It says what they share (the same `x`,
  `window`, `state`, `working_nodes` and `broken_nodes`, exact where the
  structure is, shares that add up) and runs one pumping station through
  every one. Rho, a plan's value's sensitivity to the discount rate, waits
  for the plan's net present value.
- **Uncertainty importance: whose uncertainty widens the answer (#196).**
  `NonRepairableRBD.uncertainty_importance(x, uncertainty)` gives each
  uncertain input's share of a system quantity's variance over the
  parameter uncertainty `sf_uncertainty` and the like propagate: the
  reliability at `x`, the MTTF, a B-life or the time to a reliability
  (`of=`), the inputs as those methods take them (a node, a population of
  nodes, a common-cause model). By default by the delta method, the
  quantity's derivative in each input's parameters (central differences
  of the exact value) with their covariance (a fit's `hess_inv`, or the
  variances of the distributions given), whose parts add up to the
  variance; `method="sobol"` estimates the first-order and total Sobol
  indices from draws (Jansen's estimators), which take nonlinearity and
  interactions in. An `UncertaintyImportance` holds the variance and the
  shares. On a `RepairableRBD` it came with #200, below.
- **Parameter uncertainty in a repairable system, and its vega (#200).**
  `RepairableRBD.mean_availability_uncertainty`,
  `point_availability_uncertainty(x)`, `mission_availability_uncertainty(t)`
  and `expected_cost_rate_uncertainty` give the spread of the availability
  and of the cost rate over plausible models of the components, as an
  `UncertaintyResult`. A component's models are its roles: its life
  (`"reliability"`), its repair (`"repairability"`), and the durations of
  its preventive maintenance and of its tests (`"preventive.duration"`,
  `"inspection.duration"`), named as `parameter_sensitivity`'s levers. Each
  may be uncertain (`{node: {role: uncertainty}}`, the uncertainty as
  `sf_uncertainty` takes it: `"fit"`, distributions over its parameters, or
  a list of models), and so may a common-cause group's model, keyed by the
  `CCFGroup`. By default every model that is a surpyval fit with a
  parameter covariance is drawn, the nodes holding the same fitted object
  sharing its draws. Each draw is the diagram rebuilt from its
  constructor's arguments with the drawn models, and its value worked out
  as the diagram's own (exactly or numerically), so what the diagram
  refuses, the draws do too. `uncertainty_importance(x, uncertainty, of=)`
  splits the variance of the long-run availability, the availability at
  `x` or over missions `x` (from new or from `state=`), or the cost rate
  among the inputs: by the delta method, `parameter_sensitivity`'s
  derivatives with the parameters' covariance, or by Sobol indices from
  draws. Checked against a repair rate known to a range (whose mean
  availability has a closed form), the delta method's variance against the
  draws', and its shares against the Sobol indices.
- **Quasi-random parameter draws (#200).** Every method that draws models
  from their uncertainty, on both diagram classes, takes
  `sampling="sobol"`: the draws are the points of a scrambled Sobol
  sequence (seeded by `seed`), each parameter or choice from a list one of
  its dimensions, rather than random numbers. They cover the parameters
  more evenly, so the mean, the percentiles and the Sobol indices settle
  with fewer draws: for a repair rate known to a range, 512 points put the
  mean availability within `1e-5` of its value, ten times closer than
  random draws. The default, `sampling="random"`, draws as before.
- **Joint importance: complements and substitutes (#194).**
  `joint_importance` (on `NonRepairableRBD`, `RepairableRBD` and
  `FaultTree`) gives the second-order Birnbaum measure of each pair,
  `∂²R/∂R_i ∂R_j = R(1_i, 1_j) − R(1_i, 0_j) − R(0_i, 1_j) + R(0_i, 0_j)`
  (Hong & Lie, 1993): positive where the two are complements (in series,
  improving one makes improving the other worth more), negative where
  they are substitutes (in parallel). Exact, as each node's Birnbaum
  importance with the other held working less with it held failed (twice
  as many evaluations as nodes, not as pairs), keyed by each pair once
  and found either way round. A
  `RepairableRBD`'s is long-run, or with `x`, `window` and `state`, each
  pair held in the crews' chain where there are crews; a `FaultTree`'s,
  `−∂²P/∂q_e ∂q_f`, is its diagram's. Refused with common-cause groups,
  whose members cannot be held.
- **Which component is moving the system, and which caused its failures
  (#195).** `RepairableRBD.availability_rate(x)` gives how fast the
  system's point availability is changing at each time (from new or from
  `state=`), split among the components: with independent components it is
  multilinear in theirs, so its rate is the sum of each one's Birnbaum
  importance times its own rate, and each term is what that component is
  doing to the system then. A component's rate is its point availability's,
  by differences on the grid it is solved on. Where a scheduled event makes
  an availability jump (a block replacement or test that takes a component
  off line), the system's jumps are reported apart, split among the
  components that jump together along the straight path between their
  values before and after, so that the parts add up. A `RateBreakdown`
  holds the rates and jumps. `NonRepairableRBD.reliability_rate(x)` splits
  the system's density the same way (from each node's own density; a
  common-cause group's part, under the tuple of its members, by
  differences). `barlow_proschan_importance` gives the probability that the
  system's failure is caused by each component's: on a `NonRepairableRBD`,
  over its whole life or given that it fails by `x` (its parts of the
  density, integrated by adaptive quadrature); on a `RepairableRBD`, each
  component's share of the system's failures in the long run (its terms of
  `system_failure_frequency`, from the crews' and common-cause groups'
  chains too) or over a window (its terms of `expected_failures`), the exact
  counterpart of the simulated `failure_criticality_index`.
- **Theta and Barlow–Proschan with repair crews and common-cause groups
  (#199).** `availability_rate` and `barlow_proschan_importance(window=)`
  refused a system whose components wait for crews or share common causes,
  as they do not fail and recover independently. Their parts now come from
  the crews' and the groups' Markov chains. The system's rate is `p(t) Q
  u`; each transition is one component's failure or repair (a crew taking
  the next job belongs to the repair that freed it), so the generator
  splits by component, and each part, `p(t) Q_i u`, is one more vector of
  the same uniformized chain: exact. A nested RBD, with crews of its own,
  takes its part as an independent component, its importance over the
  chain. In a common-cause group, a member's own cause and repairs are its
  part, and the causes that strike more than one member the group's, under
  the tuple of its members, each at the system's availability with the
  group in each of its states. A hidden group's tests change the members'
  joint states, so a jump with a test in it is split by the Shapley value
  of each test and node that changes then, worked out exactly (up to 12 at
  once). Over a window, the shares integrate the chains' failure rates by
  component and by cause. Checked against `expm` for a pair with one crew,
  the parts adding up to the system's rate and jumps, long windows reaching
  the long-run shares, and the simulated failure criticality with a crew.
- **Differential importance: shares that add up (#193).**
  `differential_importance` (on `NonRepairableRBD`, `RepairableRBD` and
  `FaultTree`) gives each node's (or basic event's) share of the change in
  the system when they all change together: the differential importance
  measure of Borgonovo & Apostolakis,
  `I_i dθ_i / Σ_j I_j dθ_j`. Unlike the other measures, the shares add up,
  so `groups` (`{name: keys}`) gives a group's share as the sum of its
  members': what share lies in the pumps, or in the repair times against
  the maintenance intervals. `change="uniform"` (every `dθ` equal) shares
  out the Birnbaum importance, `"proportional"` (every `dθ / θ` equal) the
  criticality importance (of either `kind`, the probabilities of failing or
  of working moved in proportion). `over="parameters"` shares out
  `parameter_sensitivity`'s derivatives instead, keyed `(node, parameter)`
  (a repairable diagram's continuous levers: one more standby unit or
  crew takes no part), and `improving=True` moves each the way that
  improves the system, so that every share is of a gain: by default the
  parameters move together, as the measure is defined, and opposing
  effects give shares of either sign, NaN where they cancel (an
  exponential unit's two rates, moved in proportion). A repairable
  diagram's shares are long-run, or over time with `x`, `window` and
  `state`.
- **Parameter sensitivity of a repairable diagram (#192).**
  `RepairableRBD.parameter_sensitivity()` gives the derivative of the
  long-run availability in each lever: each component's life and repair
  models' parameters (`"reliability.<name>"`, `"repairability.<name>"`), its
  preventive maintenance (`"preventive.interval"`, `"threshold"`,
  `"opportunity"`, its duration's parameters), its tests
  (`"inspection.interval"`, `"coverage"`, `"offset"`, their duration's), its
  standby group's (`"dormancy_factor"`, `"switching_probability"`, and one
  more unit), its imperfect repair's `"repair.q"`, a common-cause group's
  (its members' parameters together, under the tuple of their names, and
  `"ccf_beta"`, ...), and one more repair crew (under the key None). Each
  continuous lever is a central difference of the system's own value, the
  diagram rebuilt with the lever moved (one-sided at a bound, NaN where
  neither side is valid). `x` (times, from new or `state=`) and `window`
  give the sensitivities over time, where with independent components a
  component's lever is its Birnbaum importance times its own curve's
  difference, exact and a fraction of the cost; `of="cost_rate"` (or a
  tuple of both) the cost rate's; `unit_costs` ranks levers by
  availability per unit spent. The step defaults to `1e-5` in the long run
  and `1e-2` over time, whose curves are numerical. In the long run an
  interval of a component sharing a calendar with others' tests or block
  replacements would move it off their common calendar, where the value
  jumps: its derivative takes its schedule apart from theirs. As for the
  diagram's other measures, `working_nodes` and `broken_nodes` come first,
  and `x`, `window`, `state` and `rel_step` are keyword-only.
- **A repairable diagram's importance measures over time (#191).**
  `birnbaum_importance`, `improvement_potential`, `risk_achievement_worth`,
  `risk_reduction_worth`, `criticality_importance` and `fussell_vesely` of a
  `RepairableRBD` were long-run only, though from new, around a planned
  outage or from the components' states now the ranking can differ. Each
  now takes `x` (times from new), evaluating it at the nodes' point
  availabilities then, or `window` (a length), evaluating it over
  `[0, window)` as a ratio of the system's means over the window (as
  `mission_availability` is its mean availability), with `state=` to start
  from the components' current states. Without them the measures are the
  long-run ones, as before. With common-cause groups each time is split by
  the groups' joint states then; with limited repair crews the Birnbaum
  importance, improvement potential and risk worths hold each node in the
  crews' chain over time, and the criticality and Fussell–Vesely measures
  average over its states at each time (not yet around nested RBDs). The
  window's integrals are on the pieces `mission_availability` uses, and a
  window past the time the curves settle or repeat costs no more than one
  to it. `x`, `window` and `state` are keyword-only.
- **Conditional runs: simulate only what needs it (#189).** A system with
  one dependency (a standby group of other lives, a nested RBD sharing a
  crew, a unit repaired imperfectly, a maintenance group) was simulated
  whole, though the rest of it is independent components whose values over
  time are exact, and whose randomness is most of a run's error.
  `availability(conditional=True)` and `cost(conditional=True)` simulate
  only the *modules*, the nodes the exact methods over time do not take,
  and take every other node exactly given their joint states (worked out
  once per state met, on a grid, to about 1e-8 of the window): each
  simulation contributes its expected up time, failures, restorations,
  planned outages, curve and cost given its modules' histories. The
  estimates are unbiased, and vary less: twelve units in a line with a
  Weibull standby pair, 43 times less for the same simulations in the same
  time. The modules draw what they draw in a plain run with the seed, on
  either engine and any `n_jobs`, with their own costs from the same
  simulations (the tally now keeps each simulation's cost beside its
  histories). Each simulation's values are expected values, whose spread
  is less than a window's own: the cost's `percentile` and `std` refuse,
  `criticalities` is None, and `result.conditional` (a `ConditionalRun`)
  names the modules. `analysis_routes()` names them for `availability` and
  `cost` when a conditional run applies. Changes at one instant are taken
  as the event loop takes them: a module's before the other nodes' (a
  scheduled replacement shared with one, a unit dead on arrival), and a
  failure before the opportunistic stop it opens. Limited repair crews make
  every node they serve a module, simulated with the crews (a nested RBD,
  with crews of its own, stays exact). A run can start from the components'
  states (`state=`), follow capacities (each level's expected time and the
  delivered fraction, from the exact capacity over time with the modules
  held), run its modules as shards (`shard_map`, the run's result to the
  last bit), and be controlled (`control_variate=True`) by the exact twin's
  stand-ins for the modules, simulated alongside them with common random
  numbers. Refused when the modules would be every component (crews
  serving them all) or a maintenance group stops at every outage of the
  system (`system_down`). By default a plain run of such a system takes the
  same means from its own simulations (see Changed).
- **Common-cause groups in a repairable diagram over time, in the
  simulations and in the allocations (#158).** A `RepairableRBD` with
  `ccf_groups` had exact long-run values and importance, but refused the
  values over time, the simulations and the allocations.
  - Over time from new, each group's Markov chain is followed from every
    member up (by uniformization, or through the members' tests, after
    whose first period it repeats its long run), and each time is split by
    the groups' joint states: `point_availability`,
    `mission_availability`, `expected_failures`, `expected_events`,
    `expected_cost`, `point_capacity` and `mission_capacity` take the
    groups in, a nested RBD's too. Two pumps sharing a fifth of their
    failures are up 98.43% of their first 1,000 hours and the pair fails
    3.16 times, against 99.18% and 1.64 independent.
  - The simulations (`availability`, `cost`, `compare`, `simulate_chunk`,
    `shards`, `simulate_timelines`, event stepping and
    `spares_demand(method="simulate")`) draw each shared cause as a
    Poisson process with its own random stream, failing the members it
    names that are up at once, and a member's own failures at its own
    share of the rate. They run in Python (the compiled engine leaves the
    groups to it), and take in what the chains do not: tests and repairs
    that take time, and repairs of any distribution. A control variate's
    exact twin keeps the groups.
  - `allocate_redundancy` gives a `BetaFactor` group's member copies that
    join its group, each design scored exactly by chains that count how
    many of a member's copies are down rather than telling them apart, so
    a design with many copies is quick to score. Copies of an `MGL` group's
    member, or of a train holding one, are refused. The copies are repaired
    at once, so a shared failure ends with the first repaired, and a
    shared cause can make more copies worth buying, not fewer.
    `availability_allocation` and `mttf_mttr_allocation` keep the members'
    availability (their MTTF and MTTR are their group's) and score the
    system over the groups' joint states.

  Members of other lives, held working or broken, or started from a current
  state are refused with the reason, as are groups with limited repair
  crews; `analysis_routes()` says which.
- **Repair crews around nested RBDs, over time (#162).** With limited
  repair crews, the expected events and cost and the capacity over time
  refused a nested RBD, though its availability over time was worked out.
  A nested RBD has crews of its own, so it is independent of the crews'
  chain. Its failures now count as another node's would, at its
  importance over the chain and the other nested RBDs' patterns, beside
  the chain's own components' failures at their rates in each state.
  The capacity is worked out for each combination of the nested RBDs'
  levels, weighted by their own distributions. A crew RBD with nested
  RBDs gives its events to an RBD it is nested in, and one with
  capacities its capacity. With a crew for every component, the chain
  gives the independent values (to 1e-7); otherwise they agree with the
  simulation.
- **Replacement on condition over time (#161).** The availability over
  time and the expected events, cost and capacity of a window refused a
  component replaced on condition, simulated only, though its long run was
  numerical. They now follow the long run's recursion from new over every
  inspection interval, rather than over one cycle: each interval starts
  with what the inspection at its start replaced, the repairs carried into
  it and the units it kept, by age, and after some intervals it repeats
  from one to the next, its long-run cycle. From a state, the unit in
  service is decided on at each inspection at its own age, and from its
  long-run state the cycle is shifted to its phase. A year of a weekly
  inspected pump costs 114,491 from new (114,442 ± 447 simulated), less
  than the long run's 117,139. A threshold of 0 gives block replacement's
  values to 1e-12; the control variate's twin keeps the policy, so such a
  system is its own twin, and its inspections are counted for their cost
  as the simulation counts them, at each one it is up at.
- **The stock of a block-replaced component (#160).** `spares_stock`
  refused a component under block replacement, whose demand in a lead time
  depends on where in the block interval the lead time falls. With its
  repairs and block replacements in no time, each interval starts with a
  new unit and the demand repeats every interval: from a random time it is
  averaged over the phase, and before a replacement over where the
  replacements fall (a failure, at the renewal density, or a block
  replacement), which by exchanging the order of the integrals takes 1-D
  sums on one grid of the interval, to about 1e-6. Twenty pumps swapped
  in no time every 1,000 hours need 52 on the shelf for a 95% fill rate
  with 12 weeks to restock, against 46 swapped at 1,000 hours of their
  age. A fleet's systems are taken as out of step; two block-replaced
  components in one part, whose block times keep step, are refused.

  With repairs or block replacements that take time, a unit down at a
  block time is not replaced there, and its work carries over into the
  next interval, which need not start with a new unit. The demand is then
  counted from a typical replacement in the long run (its Palm
  distribution): a failure at each phase of the interval or a block
  replacement, weighted as intervals followed one after another from new
  settle, each followed block interval by block interval, all phases at
  once on one grid. Before a replacement, the chance of `s` on order is
  that the `s`-th replacement after a typical one falls within the lead
  time; from a random time, Campbell's formula integrates those chances
  over the lead time at the long-run rate. Only the life is rounded: the
  repairs and block replacements are split between the grid points either
  side, keeping their mean (rounded, 8-hour repairs on a yearly interval
  were still 7e-6 out on the finest grid), and a lead time of whole
  intervals, on a jump in the replacement times' distribution, is read
  from each rounding on its own. The grids are extrapolated as the step
  squared, to about 1e-6, in a few seconds. The pumps above, repaired
  in about 8 hours and swapped in about 4, need the same 52, for a 96.7%
  fill rate. 58 million simulated replacements agree within their noise
  (fewer than two pumps on order at a demand: 6.05e-5 ± 1.0e-6, against
  5.99e-5). A unit dead on arrival while its repairs or block
  replacements may take no time is refused, as its replacements can then
  come several at one instant.
- **Smaller API additions (#179, #184).** `"paths"` and `"cuts"` name the
  structure methods wherever `"p"` and `"c"` do. A `Network` takes a number
  as a link's or node's probability of failing, as a `FaultTree` takes an
  event's. The demonstration functions answer for each element of arrays
  given for their numbers, and `time_to_reliability`, `bx_life` and
  `remaining_life` for each of several targets. `demonstrated_mtbf`,
  `mtbf_test_time` and `mtbf_pass_probability` take
  `failure_terminated=True` for a test that stops at its last failure
  (`2r` degrees of freedom). `PhasedMission` has `sf` and `ff`, the names
  the diagrams use, beside `reliability` and `unreliability`.
  `Timeline.from_outages(merge=True)` joins records that overlap or touch
  (two work orders on one outage) into one outage, which is otherwise
  refused, or counted as two failures when they touch. The uncertainty
  methods draw, when no `uncertainty` is given, every node whose model is
  a surpyval fit with a parameter covariance, those sharing a model object
  (or a common-cause group) together, where `None` was refused, and an
  `UncertaintyResult` shows a summary (the nominal value, the median, the
  90% interval) rather than its thousand samples.

### Changed

- **A capacity run's curve follows `curve_points` (#190).** The capacity
  curve kept every change of every simulation's expected capacity, millions
  for a large run, and sorting them took most of the run: 3.4 s of a
  compiled run of three pumps over 2,000 h, 5,000 simulations, whose loop
  took under half a second. With `curve_points` the changes are now summed
  in the grid's steps as they come, exactly (each step's total the same
  however the run is split), so `capacity_timeline` is the grid and the
  curve is the full one at its times: the same run takes 1.2 s. Without a
  grid, changes that come in order are no longer sorted.
- **A run's exact timelines are built in linear passes (#201).** Without
  `curve_points`, a capacity run spent most of its time after the
  simulations, putting their changes in time order: a stable `argsort`,
  the gathers after it, `np.unique` and two `searchsorted` over every
  change, and several full-size temporaries. Where numba is installed and
  a run has a million changes or more, they are now put in order by a
  stable radix sort of the times' bits, eleven bits a pass, in blocks on
  numba's threads; grouped by time in one pass; and the capacity's times,
  running totals and limits merged in one more, the curve worked out in
  place. The availability curve's changes are sorted the same way. The
  results are the same to the last bit (the same order, the totals added
  in `np.cumsum`'s order, the curve by numpy's own `maximum` and `where`),
  checked against numpy's path for whole runs and their chunks, and the
  sort with blocks of every size. Three pumps with capacities over
  2,000 h, 20,000 simulations (21.8 million changes): 16.5 s, of which
  12.1 s built the result, now 5.2 s, of which 1.3 s. Without numba, or
  for fewer changes, numpy's path is as before.

- **A run's expected values are exact, or taken given its modules, by
  default (#187, #189).** `availability()` and `cost()` estimated a
  window's mean availability and cost from the simulations' own values,
  even where the exact methods give them: for 12 components over 5,000 h,
  ±1e-6 took some 19 million simulations against seconds for
  `mission_availability`. Now, by default, where the exact methods work out
  a system's expected values over the window (independent components, and
  crews, standby groups and common-cause groups where their chains do),
  `mean_availability_interval` and the cost's `mean_interval` are those
  values, with no error and `method="exact"` (the run's `control_variate`
  is the system itself), and a run to a `tolerance` stops after its first
  `mc_samples`. Where a system has dependent modules (#189), the whole
  system is still simulated, and the intervals are those of each
  simulation's expected values given its modules' histories, which are
  simulated again alone, drawing what they drew, the rest exact given their
  states (`method="conditional"`; `result.conditional`, a `ConditionalRun`
  with `whole=True`, holds them): twelve units in a line with a Weibull
  standby pair, a standard error 6.6 times smaller for a quarter more time,
  and a run to a tolerance judged on them stops that much sooner. Everything
  else in a result (each simulation's values, the curve, the totals, the
  percentiles, the criticalities) is the simulations' own, and the
  simulations are as before. The exact part takes a few tenths of a second,
  more than a quick run of a few hundred simulations of a small system:
  `control_variate=False` keeps the simulations' own means and skips it
  (`conditional=False` keeps them where they would be taken given the
  modules).
  `control_variate=True` is the twin control, as before, and runs with
  `shard_map` where the twin is the system itself; merged chunks
  (`availability_from_chunks`) take the same means as the run. The
  `availability` and `cost` routes say which a system takes, and the
  simulation guide has a table of which questions are exact and which
  simulated.

- **The analyses over a window cost little more than the point curve
  (#164).** `mission_availability`, `mission_capacity`, `expected_failures`,
  `expected_events` and `expected_cost` summed their integrals between every
  point of every component's grid: two million pieces for 80 components over
  ten years, and four times the time the curves themselves take. They are now
  summed over pieces cut where the curves bend (a scheduled replacement, a
  test, a down time from a known instant) and a few steps long of the grid
  of the finest curve still changing, each halved until its 4-point
  Gauss-Legendre quadrature agrees with its halves' (`_quadrature`); a
  component's expected events, against the Birnbaum importance taken as a
  cubic on each piece. On 80 components over ten years,
  `mission_availability` takes 4.4 s rather than 20.5 s, about as long as
  `point_availability` at 200 times, and `expected_failures` 5.8 s rather
  than 35 s; on 36, 1.9 s and 2.5 s rather than 3.9 s and 6.9 s; and
  `mission_capacity` on six components over a year 0.07 s rather than
  0.48 s. A system that took a second or less takes about as long as
  before. The mission availability moves by a few parts in 10^9 at most,
  and the expected events within the curves' accuracy, but for the fixes
  below.
- **`availability_from_chunks` refuses chunks with simulations missing
  between or before them (#176).** Chunks of simulations 0 to 499 and 600
  to 999 gave a run of 900 simulations without a word, so a shard that never
  came back passed unnoticed. They must now hold simulations `0` to `N - 1`,
  none missing (with `mc_samples=N`, as before, also the last), and
  `allow_gaps=True` takes the result of whichever simulations they hold, as
  before.
- **`SparesDemand.mean` and `std` are properties (#184)**, as the other
  results' values are (`UncertaintyResult.mean`, `CostResult.mean`): what
  takes an argument is a method (`stock(probability)`), and a value a
  property. Calling them, `demand.mean()`, still gives the value, with a
  `FutureWarning`, and goes in 0.13 (`deprecation.NEXT_REMOVAL`).
- **A count given as `True` is refused by the demonstration functions
  (#179)**, as elsewhere (a fault tree's vote), where it was taken as 1.
- **A run controlled by the system itself is exact (#179, #185, #186).** With
  `control_variate=True`, a system whose exact twin is itself (nothing ties
  its components together, and the exact methods take all of it) gave an
  interval of width 1e-17 around its exact value;
  `mean_availability_interval` and the cost's `mean_interval` now give
  that value, with no error and `method="exact"`. Such a run simulated
  the system twice, as itself and as its twin, the same draws to the last
  bit; it now runs once (#186). And the twin's exact cost and
  availability build each component's curve once between them, not once
  each (#185), with the same values: 39 s instead of 68 for 24 maintained
  components over 4,000 simulations. A component minimally repaired in no
  time no longer makes the twin differ.
- **A meshed diagram's core is worked out in a fraction of the time
  (#172).** A core that does not reduce, decided by its binary decision
  diagram, knew each sub-problem by which of the decided nodes that still
  feed undecided ones had been reached; it is now known by how many reached
  inputs each node still to come has, up to its `k`, which makes one of the
  sub-problems that differ only in which inputs were reached, and gives the
  same diagram. A random mesh of 35 nodes and 129 links took 44 seconds to
  build, and one of 40 nodes did not finish; they take a twentieth and a
  tenth of a second, one of 50 nodes and 250 links a second, and one of 60
  nodes and 345 links two. The exact
  `fussell_vesely` no longer lists the core's minimal cut sets and works
  out, for each node, the union of those through it: it builds that union's
  decision diagram from the core's (the node's critical states, closed
  upwards), so a ladder of thirty bridges takes a fifth of a second rather
  than 19 seconds, and one of a hundred, which took more than a minute, 1.7.
  And the minimal cut and path sets are read off the core's plan with one
  lookup each, rather than each checked against all the others (the
  structure being coherent, a set with the pivot working that holds one
  with it failed is that one): a ladder of a hundred bridges' 40,000 cut
  sets take 0.7 seconds rather than three minutes. Every result is as it
  was.
- **A core too meshed to work out is simulated, where building the diagram
  ran on without end (#172).** Building a core's decision diagram stops
  after `repyability.rbd.bdd.STEP_LIMIT` steps (25 million, a few seconds:
  a random mesh of 70 nodes and 485 links needs more). The RBD is then
  still built, with `structure_check["is_too_meshed"]` set, and its
  simulations follow the graph itself (`modular.GraphStructure`: a node
  works while it has not failed and enough of its inputs work), giving what
  the structure worked out gives, to the last bit: a `NonRepairableRBD`'s
  `random`, `mean(method="simulate")` and the other simulations, and a
  `RepairableRBD`'s `availability`, `cost` and `simulate_timelines`, on the
  Python engine. The exact and numerical analyses refuse, saying why, as
  `analysis_routes()` reports, and so does the compiled engine. Raise the
  limit to try harder.
- **Large decision diagrams are built and replayed compiled, with numba
  (#202).** With numba installed, a meshed core whose search may be long
  (`bdd.COMPILED`, `"auto"`) has its decision diagram built by the same
  search compiled (`_bdd_kernel.build`), each state two integers: each
  undecided vertex's count of reached predecessors in a field of its own,
  laid out alike for every state at a step, and the values of the
  repeated components still to come in bits. A plan of 5,000 steps or more
  is replayed compiled for its probabilities and their gradient, 16
  columns at a time, into buffers kept with the decomposition
  (`modular.COMPILED_STEPS`). The plan is the same step for step, the
  steps counted against `STEP_LIMIT` are the same (so a core is too meshed
  on both paths or neither), and the values are the same to the last bit.
  A 12 × 24 grid of 288 nodes is built in 0.19 s rather than 0.9, its
  reliability at 200 times takes 0.03 s rather than 0.17, its Birnbaum
  importances 0.03 s rather than 0.65, and its mean time to failure 5.2 s
  rather than 53. Without numba nothing changes.
- **A network's decision diagram is worked out a decision at a time, so a
  10 × 10 grid is exact (#173).** The states before each decision (each a
  way the frontier can be joined up) were found one by one; they are now
  worked out together, as the rows of an array, equal ones merged, and the
  diagram is reduced from the last decision up. It is the same diagram,
  and its values are as they were, to the last bit (the importance to
  rounding). Its size was not a merging fault: a 10 × 10 grid, corner to
  corner, has 1.9 million states in all, though 42,000 at most before any
  one decision, so it passed the limit of a million and was refused after
  about ten seconds. It now takes a second and a half, a 9 × 9 grid 0.4
  seconds rather than 3.6, and an 8 × 8 one 0.12 rather than 0.9. The
  diagram is held, and evaluated, a decision at a time, so `sf` of the 10
  × 10 grid takes 0.02 seconds. The limit, `network.MAX_STATES`, is five
  million, which an 11 × 11 grid (7.7 million, eight seconds) passes; a
  network beyond it is refused in about five seconds.
- **A fault tree with shared events is built from its gates (#171).** The
  core that the repeated events tie together was given by its minimal path
  sets, listed as the tree was built, and they multiply with the shared
  events: an OR of fifty AND gates over twenty-five shared events took 18
  seconds to build and four more for `ff`, and one of 300 AND gates three
  and a half minutes. The core is now the binary decision diagram of the
  gates themselves, built in a twentieth of a second and 0.7 seconds, and
  the cut and path sets are found from it only when asked for.
  A tree whose diagram passes `repyability.fault_tree.DIAGRAM_LIMIT` (two
  million nodes) is refused, with the advice to simulate it as a diagram;
  `PATH_SET_LIMIT`, which limited the listing, is gone.
- **Requires surpyval 0.22** (was 0.21).
  - A distribution's parameters are read by surpyval 0.22's name for them,
    `parameter_names`, in `parameter_sensitivity` and in the uncertainty
    methods' parameter distributions. 0.11 reads `param_names`, which
    surpyval 0.23 removes: with it, 0.11's sensitivities would name the
    parameters `param0`, `param1`, ... and its parameter distributions
    would be refused.
  - The simulations' own copy of the Normal and LogNormal quantiles is
    gone, as surpyval 0.22 computes them directly
    ([SurPyval#469](https://github.com/derrynknife/SurPyval/issues/469)).
    Seeded results are the same, draw for draw, and take as long.
  - A `RegressionNode` with a proportional-odds model keeps the precision
    of a small `ff` along a covariate schedule too, as surpyval 0.22's
    `Hf_tvc` does
    ([SurPyval#528](https://github.com/derrynknife/SurPyval/issues/528)):
    0.21's was 6e-4 off at a probability near 1e-6.
  - The tests fail on a surpyval deprecation that RePyability or its tests
    run into (`filterwarnings` in `pyproject.toml`). The upstream workflow,
    which runs them on surpyval's development branch, then shows a name
    surpyval will remove while it still works.
  - The tests take surpyval's `success_run` bound with `alpha_ci`, the
    name surpyval 0.23 gives it, where the installed surpyval has it, and
    with `confidence` otherwise, so the installed tests also pass with
    surpyval's next release, which deprecates `confidence`
    ([SurPyval#580](https://github.com/derrynknife/SurPyval/issues/580)).
  - A model's limited-failure proportion is read by surpyval 0.23's name
    for it, `lfp_p`, where the model has it, and by `p` otherwise
    ([SurPyval#608](https://github.com/derrynknife/SurPyval/issues/608)).
    Under surpyval 0.23, which renames `p` (the argument, the attribute and
    the key of `extras`), the exact methods would otherwise take a
    limited-failure Exponential for a plain one, and the simulations read
    the proportion with a deprecation warning. A file saved before 0.10
    that holds `p` loads under either, and the tests build limited-failure
    models by the name the installed surpyval takes.


### Deprecated

- **`StandbyModel`'s and `LoadSharingModel`'s `mc_samples`, `lower` and
  `seed` (#149).** They set the fit to simulated lifetimes that 0.12
  removes, so they are ignored: passing them warns, with a `FutureWarning`,
  and 0.13 will refuse them. A simulated model's mean is estimated with
  `mean(mc_samples=..., seed=...)`.

### Removed

- **What 0.11 deprecated is gone (#149)**, after its release's notice:
  - The old simulation-count names: `N` and `max_N`
    (`RepairableRBD.availability()`, `cost()`, `compare()`), `n_sims`
    (`StandbyModel`, `LoadSharingModel`), `N` (the node models' `mean()`)
    and `n_simulations` (`Repairable`'s methods). Use `mc_samples` and
    `max_samples`: an old name raises `TypeError`.
  - Ignored arguments, which now raise `TypeError`: simulation options
    given to `NonRepairableRBD.mean()`, `mean_time_to_failure()` or
    `RepeatedNode.mean()` without `method="simulate"` (the answer is
    otherwise exact, and the message says so), `node_mttf()`'s
    `mc_samples` and `seed`, `RepeatedStandbyNode`'s `N` and `lower` (its
    `switching_probability` is now keyword-only), and
    `NonRepairable.find_optimal_replacement()`'s `options`.
  - The misspelt `fussel_vesely()` alias of `fussell_vesely()`.
  - **Non-parametric nodes.** A diagram of either kind with a surpyval
    `KaplanMeier`, `NelsonAalen` or other non-parametric fit as a node's
    life or repair time, directly or inside a standby, repeated or
    degrading node or a `NonRepairable`, is refused with a `ValueError`
    naming the nodes, and so is loading a saved one. Their curves end at
    the data, so the MTTF, B-lives and long-run values beyond it were
    artefacts. Fit a parametric distribution in surpyval instead. A
    standalone `NonRepairable`'s maintenance policies keep them.
  - **The fit to simulated lifetimes.** Warm standby with two or more units
    operating, cold standby with three or more different units operating
    (`StandbyModel`), and load sharing of different units
    (`LoadSharingModel`) have no exact or numerical reliability. A
    Kaplan–Meier fit to simulated lifetimes stood in for one, and every
    exact analysis of a system with such a node inherited its Monte-Carlo
    error. Their `sf`, `ff`, `cs` and `mean()` now raise
    `NotImplementedError`, as do the analyses of a diagram that need their
    reliability (its `sf`, importance measures and `node_mttf`, or a
    `RepairableRBD`'s long-run and over-time values), naming the
    simulations that take them: `random`, `mean(method="simulate")` and
    `unreliability_interval` of a `NonRepairableRBD`, or `availability`
    and `cost` of a `RepairableRBD`. `analysis_routes()` reports those
    analyses as refused, with the same reasons. Such a model draws its
    lifetimes as before, so seeded simulations are unchanged, and
    `mean(mc_samples=..., seed=...)` estimates its mean from new draws.
    `allocate_redundancy`, which scored cold standby that needs three or
    more copies working with such a fit when the copies are of different
    models, refuses those designs.

### Fixed

- **Asking a compiled engine for what it does not simulate says so,
  installed or not.** `engine="numba"` (or another package's engine) on a
  diagram that engine does not simulate (common-cause groups, imperfect
  repair, a run from a state, ...) asked for numba to be installed when it
  was not, though installing it would not have helped. The refusal, which
  names what is not simulated and that `engine="python"` or `"auto"` runs
  it, now comes first.
- **An age-replaced unit's later maintenance from new starts a piece of its
  integrals.** A unit replaced on age from new is maintained again, unit
  after unit, at nearly fixed times, which its curve follows on a grid of
  its own; the start of each such window, where the chance of being down
  for it begins to rise, was not among the curve's breaks, so a piece of a
  window's integral (or a difference for its rate, #195) could straddle the
  kink there.
- **Schedules that repeat together only after a very long time are refused
  rather than exhausting memory.** The long-run values with block
  replacements or tests average over their schedules' common period, on a
  grid of every block-replaced profile across it, whose size was checked
  only once it was built: a block interval of 99.999 against tests every 50
  asked for 7.5 GB. The grid's size is now worked out first, and such a
  calendar refused with the reason, as before for larger ones.
- **A junction is in no cut set or path set (#198).** A
  `NonRepairableRBD` keeps a node given `PerfectReliability` (a k-out-of-n
  vote point, or any junction drawn for the layout) as a node that never
  fails, and its minimal cut sets listed it: a 2-out-of-3 vote showed as a
  single-point cut set beside the controller it fed, and was in every path
  set, though `sf` held it perfect. The sets are now read from the
  structure with the junctions folded in as always working, as a
  `RepairableRBD`'s are and `FaultTree.from_rbd` already gave them: a cut
  set with a junction never happens, and a path set needs nothing of one.
  So the path-set Fussell–Vesely importance (`fv_type="p"`), which was 0
  for every node of a diagram with a junction (each path set needed the
  junction to fail), is now worked out on those path sets; and
  `minimum_effort_allocation` no longer refuses a series diagram drawn
  with a junction in it.
- **Long renewal sums no longer wait on BLAS threads.** OpenBLAS splits a
  dot product of more than about 10,000 terms across its threads, which
  here cost about 5 milliseconds a call against 2 microseconds on one, and
  several loops make thousands of them: the from-new curve of a component
  tested often against its life (#144), the stock of one under block
  replacement with a long interval (#160) and the long run of replacement
  on condition. They now sum their products in numpy's own loop
  (`repyability.utils.vectors.dot`): a weekly test of a 20-year life went
  from about a minute to 10 seconds, a block-replaced stock from about 10
  to 2, and the test suite from 310 seconds to 240. The values agree to
  rounding.
- **`FaultTree.from_rbd` of a diagram with events the logic absorbs
  (#170).** A module of nodes that cannot affect the system (in an AND of
  `e4`, `e6` and a vote that, given them, needs only `e5`, the `e1 AND e2`
  input) became a gate that nothing used, and the tree it made was refused
  (`'G1', 'e1', 'e2' are not below the top event`), so such a tree did not
  convert back from its own diagram: two in a thousand random trees of
  seven events and six gates. Only the gates below the top event are kept,
  numbered in turn.
- **A phased mission with an equal standby group in each phase (#179).**
  A component must keep one model through a mission, and two separately
  built but equal `StandbyModel`s (or other node models of RePyability's)
  were taken as different, so the mission was refused; models that save
  the same way are now the same model, as equal surpyval distributions
  were.
- **A simulation over a window that is not positive and finite is refused
  (#174).** `availability`, `cost`, `simulate_timelines`, `simulate_chunk`,
  `shards` and `compare` took a negative window (and gave negative uptimes),
  a zero one (and an availability of nan), and an infinite one; they now
  raise a `ValueError`, as the exact analyses over a window do. A simulated
  `spares_demand` over no time is no spares, as the exact one says.
- **Bad counts and structures are refused with a message that says what to
  give (#168, #169, #178).** A `RepeatedNode` (and `RepeatedStandbyNode`)
  takes a whole number of copies, at least 1: -1 gave a reliability of
  -0.58, and 2.5 two and a half units. A repeated node that repeats another
  repeat (`{"a": W, "b": "a", "c": "b"}`) is refused when the diagram is
  built, naming the component to repeat, where `sf` crashed with a
  `KeyError`. A node's `k` and a `StandbyModel`'s `k` must be whole numbers
  from 1, rather than failing inside the evaluation (`k=1.5`, `"2"`, `-1`;
  `StandbyModel(k=0)` divided by zero). An empty diagram says it has no
  edges, a diagram with no input or output node says so, and a diagram with
  a cycle built with `on_infeasible_rbd="ignore"` says it has a cycle when
  evaluated, rather than recursing until Python stopped it.
- **A system of models fitted in surpyval runs with `n_jobs` (#181).** A
  surpyval fit (Weibull, Gamma, Gumbel, Logistic, LogLogistic, ExpoWeibull,
  Beta) holds a closure, so pickle cannot send it to a worker process
  (SurPyval#573), and `random`, `mean(method="simulate")`, `availability`,
  `cost` and `simulate_timelines` with `n_jobs` failed with an error from
  inside surpyval. A model that pickle refuses is now sent in its saved
  form (`to_dict`) and rebuilt in the worker, so the results are those of a
  run in one process, to the last bit; whatever else cannot be sent is
  refused with a message that says to run without `n_jobs` or with
  `shard_map`.
- **The tests run from an installed copy (#167).** The wheel holds the tests
  but not the recorded results 13 of their modules read, which errored on
  collection: they are package data now, and `test_packaging` checks that
  every data file in the package is listed as such. The tests that check the
  repository's docs, README or `pyproject.toml` skip in an installed copy,
  where they had read another package's `docs` folder in site-packages.
- **The package's licence metadata (#177).** `LICENSE` named the holder of
  the sample project it was copied from; it is Derryn Knife's. The licence
  is given as an SPDX expression (`license = "MIT"`, `license-files`, PEP
  639), which building needs setuptools 77 for, and the wheel is marked as
  typed (`py.typed`). SciPy's floor is 1.13, the first built for NumPy 2,
  which RePyability already required.
- **Planned outages after age replacement were overcounted, by about one
  in 10^4 (#164).** The chained maintenance of units that each reach their
  age, when it takes a random time, was counted from running sums that
  overshot below 0 where the time's density jumps (at 0, for an exponential
  time), and from a cubic between them that undershot there too: the counts
  dipped just before each maintenance and rose again. The window's sums,
  which clipped each piece's count at 0, counted each dip's rise again: 3.9e-4
  too many of 3.44 planned outages by 1,000 hours in the tests. The running
  sums and the cubic are held within 0 and their total, and a piece's count
  is no longer clipped, so that the totals no longer depend on the pieces.
  `expected_events`' `system_planned_outages` are about 1e-4 lower for such
  components; the costs, which count each component's own maintenance, are
  as they were.
- **The expected system failures of components with hidden failures of any
  life were off by up to about 2 in 10^4 (#164).** Each piece's failures
  were weighed by the components' mean importance over it, which is exact
  only to the square of the pieces' length, and their tests' pieces are
  long: two tested pumps' 1.8242 failures in 500 hours are 1.8238, which
  the old sum reaches as its pieces are cut finer. The importance is now
  taken as a cubic on each piece (see above).
- **A block-replaced component's spares to a horizon on a block time
  (#160).** `spares_demand` read the count there as at any time, half a
  step on, taking in part of the first step of the unit put in at that
  block time, which fails in it often for a falling hazard: a unit with a
  Weibull shape of 0.5 was counted 0.6379 failures in a block interval of
  30 hours where it has 0.6371, and the error shrank only as the square
  root of the step (as the step for other lives). It is now read just
  before the block time.

## [0.11] - 2026-10-02

How much a system can deliver, and exact answers where there were estimates.
System capacity gives the exact distribution of what a diagram can deliver
from its components' capacities (#97), with components that work at several
levels (#98), and the simulation follows the capacity delivered over a
window: the production availability (#99), which is exact over time from new
too (#124). A repairable system's availability over time and over a mission
is exact from new (#117), and so are its expected failures, outages,
downtime and cost over a window (#123), and a system's mean time to failure
(#122); all of them, and the simulation, can start from the components'
current states rather than new (#125). `analysis_routes()` says, without running anything, how each
analysis will be computed: exactly, numerically, by simulation or not at all
(#127). Availability simulations run faster, and about ten times as fast
again when compiled with numba (#119, #120), which now also runs
maintenance, inspections, repair crews, standby groups, nested RBDs and
capacities (#155), and `import repyability` is quicker (#121). Components can share a limited number of repair crews (#89),
with exact long-run values from a Markov chain when their lives and repairs
are exponential (#90); a duty unit and its spares can be a standby group,
repaired one unit at a time (#91); `spares_demand` and `spares_stock` count
the spares each component uses and the stock to hold for a lead time (#95),
block-replaced and proof-tested components' too (#147);
a component can be replaced on condition at periodic inspections (#96),
with numerical long-run values (#145), or
early at a stop of its maintenance group, sharing its set-up (opportunistic
maintenance, #108), and repaired imperfectly, by Kijima's virtual age, or
replaced at the N-th failure (#109); a simulation run can be split across
machines and merged, to the last bit (#114, #151), through any executor
(#152), and controlled by the system's exact twin, worth up to a hundred
times as many simulations (#154); and small failure probabilities are estimated by
rare-event simulation (#115), and keep their full precision where they are
exact (#148); phased missions are new, exact and simulated (#100, #101), as
are the two-terminal reliability of undirected networks (#104), both
decided by decision diagrams that keep meshed ones exact (#142, #143), and
demonstration test planning (#129). A safety function's PFDavg takes in
the terms SIL verification asks for: common-cause groups in repairable
diagrams, staggered tests, and proof tests that miss failures (#136).
Common-cause groups enter the importance measures, parameter sensitivity
and uncertainty, and redundancy allocation (#140). Hidden failures are
numerical for any life, not only a constant failure rate, when tests and
repairs take no time (#144). With repair crews, and for standby groups, the
values over time come from the same Markov chains, and the importance
measures with crews from their definitions (#146).
Meshed diagrams are decided by a binary
decision diagram, in milliseconds where their path sets took minutes (#102,
#103). Timelines are new: up/down histories, from outage logs or kept
whole from the simulations, merged as a diagram's structure into a
system's, with the component behind each of its failures (#157). Non-parametric nodes, and the fits to simulated lifetimes behind some
standby and load-sharing models, are deprecated; everything deprecated goes
in 0.12, and warns with a `FutureWarning`.

Behaviour changes: seeded repairable simulations give different numbers,
once, as each random quantity now has a stream of its own (#119);
`mean()` and `mean_time_to_failure()` are exact, the old estimate is
behind `method="simulate"`, and the exact MTTF refuses common-cause groups
(#122); the number of simulations is `mc_samples` everywhere, and its cap
`max_samples`, with the old names deprecated until 0.12 (#105);
`is_analytically_solvable()` flags only simulated nodes (#127); a
simulation's totals are rounded once from their exact sums, rather than
added in order, so seeded totals move in their last bit (#151);
cold-standby
reliabilities are more accurate, so they move slightly (#128); a cost
breakdown has a seventh category, `"setup"` (#108); perfect junction
nodes are left out of the importance measures and allocations, and a
`RegressionNode` gives a float for a scalar time (#134); a common-cause
group that splits a probability warns once its members' probability of
failing passes 0.1 (#132); hot standby, warm standby with one unit
operating, cold standby of identical units and load sharing of identical
units, and cold standby of different units with two operating, are no
longer fits to simulated lifetimes, so their reliabilities lose their
Monte-Carlo error (#135, #138, #139); `fussell_vesely` is exact, with the
rare-event sum behind `method="rare_event"` (#137); and surpyval 0.21
is required.

### Added

- **Common cause, staggered tests and test coverage in repairable
  diagrams** (#136), the three terms an IEC 61508/61511 PFDavg needs beyond
  independent channels tested together:
  - `RepairableRBD(..., ccf_groups=[CCFGroup(members, BetaFactor(β))])`
    (or `MGL`): the model splits the members' failure rate between their
    own causes and shared ones, each failing the members it names that are
    up at once. The long-run values (`mean_availability`,
    `mean_unavailability`, `system_failure_frequency`, MTBF, MUT, MDT, the
    cost rate, `capacity_distribution` and the interval choices) are exact,
    from a Markov chain of which members are down together, for
    exponential lives either tested or repaired at exponential rates: a
    tested 1oo2 pair with a β of 5% has a PFDavg of 5.29e-4, against
    1.01e-4 without. Each member's own values are unchanged. The importance
    measures take the groups in too (#140); the allocations, values over
    time from new and the simulations refuse a diagram with groups, as yet
    (#158), and say so in `analysis_routes()`.
  - An inspection's `"offset"` (the time of the first test) staggers the
    tests of redundant components: half an interval apart, a 1oo2 pair's
    PFDavg falls from about `(λτ)²/3` to `5(λτ)²/24`, and a shared cause's
    term halves.
  - An inspection's `"coverage"` (the chance that a test finds a failure)
    with a `"full_test"` interval (a whole multiple of the interval, whose
    tests find every failure): a failure a test misses stays hidden until
    a full test, about `(1 − c)λT/2` more. Exact in the long run and from
    new; simulated in Python, where a stream of its own decides whether a
    test finds each failure, and each test that misses one is charged. A
    unit whose tests can miss failures starts new (its state is not
    taken), and `optimal_inspection_intervals` does not choose its
    interval.
- **Common-cause groups in importance, sensitivity, uncertainty and
  allocation** (#140). With a `NonRepairableRBD`'s `ccf_groups`:
  - The importance measures (Birnbaum, improvement potential, RAW, RRW,
    criticality, Fussell–Vesely) are exact with the groups. A member's
    measures condition on its state through the groups' shock outcomes:
    the system's reliability given that it works and given that it has
    failed. A node outside the groups is held working and failed, as
    before, and Fussell–Vesely sums each outcome's probability that a
    minimal cut set containing the node has failed. Each is a sum of
    products, so a small probability keeps its precision, and `beta = 0`
    gives the measures without the group. A member cannot be held working
    or broken.
  - `parameter_sensitivity` reports a group's parameters once, under the
    tuple of its members: the derivative as the parameter moves for all of
    them, by differences of the exact system reliability. The parameters of
    the group's own model come with them, as `ccf_beta` (and `ccf_gamma`,
    ... for MGL).
  - The parameter-uncertainty methods (`sf_uncertainty`,
    `mean_uncertainty`, `time_to_reliability_uncertainty`,
    `bx_life_uncertainty`) work out each draw with the groups. A group's
    members are given together, in one tuple, and its own model can be
    uncertain too: `{group: {"beta": distribution}}`, or a list of models.
    `mean_uncertainty` refuses a group splitting a probability, as `mean`
    does.
  - In `allocate_redundancy` and `redundancy_front`, a copy of a
    `BetaFactor` group's member joins its group, so the shared cause fails
    it too: active copies of the member's own model, any number of them
    required. Copies of an `MGL` group's member (whose letters are for its
    group's size), and options or cold spares for a member, are refused.
    `allocate_reliability_redundancy` takes the groups in, and refuses to
    choose a member's reliability, which is the group's.
  - A `RepairableRBD`'s importance measures take its groups in, a member's
    conditioned on its state at each long-run time, then averaged over the
    times as every node's are.
- **Hidden failures with any life** (#144). A component with an
  `"inspection"` and a life that is not exponential, tested and repaired in
  no time, has numerical values, exact to rounding, where it had only
  simulated ones. A failed unit is found and renewed at the next test, so
  renewals fall on the tests: a cycle lasts `S = 1 + Σ R(mτ)` intervals,
  and the unit is up `u` after a test with probability `(R(u) + Σ R(mτ +
  u)) / S`. That gives the long-run values (`mean_availability`,
  `mean_unavailability` (the PFDavg), `system_failure_frequency`, MTBF,
  MUT, MDT, the importance measures, the cost rate and
  `capacity_distribution`) and `optimal_inspection_intervals`. Over time,
  a renewal equation over the tests gives `point_availability`,
  `mission_availability`, `expected_events`, `expected_cost` and the
  capacities, from new, from a state (its age and the time since its last
  test) or from its long run. A wearing valve's PFD over its first ten
  years is a third of its long-run PFDavg. Within an interval, the units
  renewed before the last test are interpolated from Chebyshev points, and
  summed directly where the life bends (at a threshold), so a curve costs
  little more at many times than at one, and a life of tens of thousands
  of intervals takes under a second. `analysis_routes()` reports these
  diagrams as `numerical` rather than refused. Tests or repairs that take
  time, and tests that miss failures of a life that is not exponential,
  are still simulated (#159).
- **Age and block replacement compiled** (#155). The compiled engine
  simulates scheduled preventive maintenance under the age and block
  policies, in zero time (a working unit renewed in place) or taking time
  (a planned outage, from the maintenance time's stream), with fixed or
  drawn preventive costs: the Python loop's events and arithmetic, in the
  same order, so the engines agree to the last bit (checked on maintained
  systems of every policy, with costs, ties from fixed lives on the
  schedule, held nodes, antithetic pairs, threads and tolerances, and on
  one too large for the table of states). It ran 9 to 12 times as fast as
  Python on one thread, and 18 to 29 times on four, on 3, 12 and 70
  components. Replacement on condition, opportunistic groups and a
  maintenance time that cannot be streamed stay in Python. An engine another package adds is still given only plain
  components (its interface, ``engines.API``, is unchanged): a maintained
  system runs on numba's own loop, which ``engine="auto"`` chooses.
- **Inspections compiled** (#155). The compiled engine also simulates
  hidden failures found by periodic tests: tests in zero time (the unit
  found working, or its failure found and its repair started) or taking
  time (a planned outage, from the test time's stream), staggered by an
  offset, and with a coverage below 1 (a test that misses a failure, from
  the detection stream, leaves it to the next full test), with fixed or
  drawn inspection costs, and repair and replace costs charged when a test
  finds the failure. The engines agree to the last bit (checked on
  inspected systems of every kind, with costs, ties from fixed lives that
  fail on a test, inspections beside age and block replacement, held
  nodes, antithetic pairs, threads and tolerances, and on one too large for
  the table of states). With every unit tested, it ran 17 to 27 times as
  fast as Python on one thread, and 32 to 63 times on four, on 3, 12 and 70
  components. A run from the components' states keeps a maintained or
  inspected system in Python (a state's phase shifts its calendar), as
  does a test time that cannot be streamed; and, as for maintenance, an
  engine another package adds is not given inspected systems.
- **Repair crews compiled** (#155). With fewer repair crews than
  components, a job that falls due (a repair, maintenance that takes time,
  or a test that takes time) waits for a crew in the compiled engine as in
  Python: the next crew free starts the waiting job of highest priority,
  then the one due first, then the one queued first, and the job ends as
  late as it waited (a unit off line for a test does not age meanwhile).
  The engines agree to the last bit on crewed systems with priorities and
  without, maintenance and tests waiting for crews, jobs falling due
  together, and one too large for the table of states. With one, two and
  four crews it ran 13, 9 and 5 times as fast as Python on one thread, and
  19, 25 and 8 times on four, on 3, 12 and 70 components (on 70, drawing
  the numbers and adding up the results take most of the time). An engine
  another package adds is still not given crewed systems.
- **Standby groups compiled** (#155). The compiled engine simulates
  standby groups as the Python loop does: each unit's life used up at the
  dormant rate while it waits and at rate 1 while it operates, the spare
  that has waited longest switched in (a switch that fails, from the
  group's switch stream, leaving the position empty), failed units
  repaired by the RBD's crews, and the group's next event in the heap,
  superseded ones skipped as Python skips them. The engines agree to the
  last bit on cold, warm and hot groups, switches that fail, groups sharing
  crews with each other and with maintained and tested components, events
  falling together, and a system too large for the table of states. With
  half the nodes groups of three units, it ran 6 to 8 times as fast as
  Python on one thread, and 7 to 19 times on four, on 1, 12 and 70 nodes.
  An engine another package adds is still not given them.
- **Nested RBDs compiled** (#155). The compiled engine simulates nested
  RBDs of up to 20 components (with their own maintenance, tests, crews
  and standby groups, nested as deep as they go): each is a level of its
  own, with its own heap and crews, stepped to its next change as
  ``RepairableRBD.next_event`` steps it, its levels on the way down
  waiting on a stack. The engines agree to the last bit on nested RBDs
  maintained (their planned outages planned outages of the system), tested,
  with crews inside and out, with standby groups, three deep, and with
  events falling together across levels. With every node a nested pair it
  ran 6 times as fast as Python on one thread, and 17 to 18 times on four.
  A run from the components' states keeps a system with nested RBDs in
  Python, as do a nested RBD of more than 20 components and what Python
  alone simulates anywhere inside one.
- **Capacities compiled** (#155). The compiled engine follows a system's
  capacity over time too, on systems of up to 63 components: the loop
  records each change of a component's state and the components up after
  it (as the bits of an integer), and a second compiled pass turns the
  records into the Python trace's changes of the expected capacity, time at
  each level and fraction of the demand delivered, with its arithmetic in
  its order. The engines agree to the last bit on capacities of one level
  and of several, unlimited nodes, a demand given and the design
  capacity's, maintenance, tests, crews and standby groups, a nested RBD's
  capacity, and a system too large for the table of states. It ran 10, 13
  and 6 times as fast as Python on one thread, and 11, 16 and 6 times on
  four, on 3, 12 and 62 components (on 62, working out the capacity of the
  90 000 sets of components down that 500 simulations met takes most of
  the time).
- **Faster capacity simulations** (#155). The capacity of each set of
  components down that a run meets is worked out once, and now those a
  simulation meets first (a batch of simulations, compiled) are worked out
  together when every node works at one level: each probability is then 0
  or 1, so each comes out exactly as on its own (with a node of several
  levels, each is still worked out on its own). On those 62 components the
  Python engine took 16 s rather than 11 minutes. Both engines also keep a
  run's changes of capacity in arrays rather than one entry per time: on a
  million changes, a fifth of the memory, and the result built four times
  as fast. The results are the same, to the last bit, and saved chunks are
  unchanged.
- **The compiled engine's threads are Python's** (#155). With ``n_jobs``
  the compiled loop runs on a pool of threads, releasing the GIL, rather
  than on numba's ``prange``, which compiled the whole loop a second time:
  numba now compiles it once, in about a minute and a half the first time
  ever (it took that long again the first time a run used threads), and as
  fast on four threads as before. numba's own thread count is no longer
  touched.
- **Faster compiled simulations of large systems** (#150). Above 20
  components (where the compiled loop has no table of every state), the
  compiled engine keeps whether the system works up to date as components
  fail and are repaired: each series, parallel or k-out-of-n part keeps how
  many of its members work, and a change goes up the structure only as far
  as it changes something (a core's path sets keep how many members are
  down). An event costs the depth of the structure at most, rather than
  the size of it: 70 components in redundant pairs ran 1.7 times as fast on
  one core and 1.5 times on four. The results are the same, to the last
  bit.
- **Simulation engines from other packages.** A package can register a
  compiled engine for `RepairableRBD` simulations under the
  `repyability.engines` entry point group (the interface is in
  `repyability.rbd.engines`). `availability`, `cost` and `compare` then take
  its name as `engine`, and `engine="auto"` prefers it to numba when its
  priority is higher, on what the compiled engine simulates; if it cannot
  load, `"auto"` warns and runs on the next engine. `analysis_routes()`
  reports it as the engine `"auto"` runs.
- **System capacity** (#97): how much a system can deliver, not just
  whether it works. Every RBD class takes `capacity={node: throughput}`,
  each node's throughput while it works. The system's capacity is the
  diagram's maximum flow: a series chain carries the least of its nodes'
  capacities and a parallel group the sum. A node given no capacity limits
  nothing, and a k-out-of-n node passes flow only while at least `k` of its
  inputs are reached, so the capacity is positive exactly when the system
  works. `NonRepairableRBD.capacity_distribution(x)` gives the exact
  distribution of the capacity at time/s `x` (honouring common-cause
  groups), `RepairableRBD.capacity_distribution()` in the long run, and
  `RBD.system_capacity(node_probabilities)` from given node
  probabilities. They return a `CapacityDistribution`: its `levels` and
  their `probabilities`, with `meets(demand)` (the probability of meeting a
  demand), `mean()` (the expected capacity) and `delivered_fraction(demand)`
  (the expected fraction of a demand delivered: in the long run, the
  production availability). Series-parallel parts reduce in closed form
  (the universal generating function), and the rest (e.g. a bridge) by
  conditioning on its parts' capacities, keeping each cut's running total,
  so the analysis costs about what the system reliability does. The
  capacities are saved with the RBD. A new guide page, System capacity,
  covers it.
- **Multi-state components** (#98) in the capacity analysis: a component
  can work at several levels, a pump at full or half output. A capacity can
  be a dict `{level: probability}` of the levels a node works at and the
  probability of each while it works. A `DegradingNode` runs through stages,
  each at its own capacity for a time from its own lifetime model, and has
  failed once the last ends: its stage at a time comes from the
  convolution of its stages' times, and in the long run, renewed after each
  failure, it spends its up time in each stage in proportion to the stage's
  mean. It is a `StandbyModel` of its stages (its lifetime is their sum), so
  it is a node model like any other. A nested RBD with capacities brings its
  own distribution. Binary nodes are the special case of each, and the
  distributions combine through series (least) and parallel (sum) as
  before.
- **Delivered capacity over time** (#99): when nodes have capacities,
  `RepairableRBD.availability()` also follows what the system can deliver.
  The result gains the mean capacity curve (`capacity_timeline`,
  `capacity`), the time spent at each capacity (`capacity_time`,
  `mean_capacity`), and each simulation's fraction of the demand delivered
  (`delivered`, `delivered_fraction`, `delivered_fraction_interval()`): the
  production availability over the window. `availability(demand=...)`
  sets the demand, by default the design capacity. The capacity is worked
  out after every component failure and repair, from the exact
  distribution given which components are up, so a node working at several
  levels counts at each in proportion, and the simulated failures and
  repairs are the same as without capacities. Over a long window the
  averages approach the exact long-run values.
- **Exact availability over time** (#117): `RepairableRBD.point_availability(x)`
  gives the probability that the system is up at each time `x`, every
  component new at 0, and `mission_availability(t)` its mean over `[0, t]`:
  what `availability()` estimates by simulation, exactly and in a fraction
  of a second. Each component alternates up and down periods, and its point
  availability solves the renewal equation, solved numerically on a grid of
  2,000 steps over its typical up time (an error of about 1e-7); the
  components are independent, so the system's is the exact system
  computation at theirs, at each time. Down periods that start at a known
  time (the repair of a unit dead on arrival, the first failure of an exact
  lifetime, the first age replacement, every block replacement) are kept
  out of the grid, exact however short they are, and so are the later age
  replacements of units that each reach their age, which fall at nearly
  fixed times: two units in parallel replaced at the same age are down
  together as often as they should be. Age and block replacement, nested
  RBDs and hidden failures with a constant failure rate are covered, as in
  `mean_availability`. A mission of decades costs no more than one of
  hours: once the components have settled, at their long-run values or
  repeating with their calendar, the integral is extended exactly. The
  curves settle at `mean_availability()`, and a long mission's average
  exceeds it by the start-up term renewal theory predicts; on the benchmark
  diagrams they agree with the simulation.
- **Exact expected failures, outages and cost over a window** (#123).
  `RepairableRBD.expected_failures(t)` gives the expected number of system
  failures in `[0, t)`, every component new at 0;
  `expected_events(t)` an `ExpectedEvents`: the system's expected failures,
  planned outages and downtime, and each component's failures, corrective
  and preventive actions, tests and downtime; and `expected_cost(t)` an
  `ExpectedCost`: the expected cost of the window by the same categories
  and components as `cost()`, with the acquisition cost beside it
  (`total`). They are the means `availability()` and `cost()` estimate,
  exactly and with no simulation, where `total_cost()` assumes the
  long-run rate from the start. A component's failure takes the system
  down if it is critical then, which, the components being independent,
  it is with probability its Birnbaum importance at their availabilities
  then: the system's failures are the time-dependent Birnbaum/Vesely
  formula, integrated over the window, with each component's expected
  failures from its renewal equation on the grid of its availability (to
  about 1e-7). Events at exact times are counted exactly and together: a
  replacement due at the window's end falls after it, as in the
  simulation, and components replaced at the same age or block times take
  the system down once, and make one stop of their maintenance group.
  Past the time the components settle, the counts grow at their long-run
  rates, so a window of decades costs no more than one of a few years.
  They cover what the availability over time covers (age and block
  replacement, nested RBDs, hidden failures with a constant failure rate
  and instant tests and repair) and refuse the rest with the reason, as
  `analysis_routes()` reports. Checked against the alternating renewal
  process's closed forms, the formula integrated by quadrature, tests of
  hidden failures, timelines worked by hand, the long-run rates they
  settle at, and the simulation's means.
- **Exact capacity over time from new** (#124).
  `RepairableRBD.point_capacity(x)` gives the distribution of the system's
  capacity at each time `x`, every component new at 0, and
  `mission_capacity(t)` the expected fraction of a window spent at each
  level, so its `delivered_fraction(demand)` is the production
  availability of the window, which `availability(demand=...)` estimated
  by simulation: both `CapacityDistribution`s, as `capacity_distribution()`
  gives in the long run, which they settle at. Each component is up at
  `t` with its point availability (#117), at its levels while up in
  proportion, and the system's distribution is the same exact computation
  at those. A degrading component (`DegradingNode`) is in each stage with
  the probability its first unit is, by its age, or a later unit is, by
  its renewals, solved on the grid of its availability, and a nested RBD
  with capacities brings its own distribution over time: neither of which
  the simulation follows. The window's mean is integrated as
  `mission_availability` integrates the availability. Checked against
  binomial pumps at their availabilities, a degrading pump's Markov chain,
  the long run, a nested skid drawn flat, and the simulation's delivered
  fraction.
- **Repairable analyses from the plant as it is now** (#125). The exact
  analyses over time (`point_availability`, `mission_availability`,
  `expected_failures`, `expected_events`, `expected_cost`,
  `point_capacity`, `mission_capacity`) and the simulation
  (`availability`, `cost`, `compare`, `simulate_chunk`,
  `initialize_event_queue`) take `state={node: NodeState(...)}`: a
  component up at an age, down part way through a repair or its
  maintenance, or, on a calendar (block replacement, tests), at a phase,
  and a nested RBD a dict of its components' states; a component left out
  starts new. `NodeState` gains `down_for`, `maintenance`, `phase` and
  `stationary`, and the exact methods take `state="stationary"`, every
  component in its long-run state, for a plant long in service whose
  state is not known. Exactly, only each component's first period
  changes: a first life with survival `R(a + s) / R(a)` and a replacement
  due when it reaches its age (at once if it has), or what is left of a
  repair, `G(r + s) / G(r)`, after which its units are new; under block
  replacement its own curve to the first block time, which starts the
  intervals' recursion; a unit with hidden failures last known up at its
  last test. In the simulation, the component draws what is left from one
  uniform of a stream of its own (the inverse transform of the
  conditional distribution, through the cumulative hazard), so seeded
  runs stay reproducible and runs from new are unchanged; a run from a
  state is simulated in Python, a component down at the start holds a
  repair crew, and a system that starts down shows so in the
  availability over time. Standby groups and imperfectly repaired
  components take no state, and the simulation no long-run start: both
  refuse with the reason. Checked against the memoryless exponential
  unit, one down recovering as its two-state chain, the remaining life of
  a Weibull unit, timelines worked by hand, the long run a stationary
  start stays in, and the simulation from the same states.
- **A compiled simulation engine** (#119). With numba installed, an
  optional dependency (`pip install "repyability[fast]"`),
  `RepairableRBD.availability()`, `cost()` and `compare()` can run their
  simulations compiled: about ten times as fast as in Python on one core,
  and faster still on several (`n_jobs` runs it on that many threads). A new
  argument, `engine`, chooses: `"auto"` (the default) compiles when numba is
  installed, the engine simulates the system and the run is long enough to
  repay loading it (a third of a second from numba's cache; some seconds
  the first time ever, while numba compiles it); `"numba"` asks for it, and
  `"python"` keeps to Python. The two give the same results, to the last
  bit: the compiled loop is the Python one over arrays, reading the same
  random streams, and CI checks them against each other. It simulates plain
  components (surpyval parametric models) in any structure, with nodes held
  working or broken, costs, antithetic pairs, tolerances and common random
  numbers; preventive maintenance, inspections, nested RBDs, capacities and
  other models run in Python, which `"auto"` chooses by itself.

- **Know in advance how each analysis will be computed** (#127).
  `NonRepairableRBD.analysis_routes()` and `RepairableRBD.analysis_routes()`
  say, without running anything, how each analysis of the diagram would be
  computed: exactly, numerically (deterministic, to a small stated error),
  by simulation, or not at all. Each comes with the reason and the nodes that
  decide it, in an `AnalysisRoute`.
  - A refusal's reason is the message the method would raise, found by the
    same check the method runs.
  - For a repairable simulation, it also gives the engine `engine="auto"`
    would use, and why.
  - The saving guide's table of exact and simulated analyses is tested
    against it.
  - `StandbyModel` gains `is_simulated`, as `LoadSharingModel` has.
- **The exact system MTTF** (#122). `NonRepairableRBD.mean()` and
  `mean_time_to_failure()` integrate the exact system reliability,
  `MTTF = ∫ R(t) dt`, by adaptive Gauss-Legendre quadrature to about `1e-10`,
  relative, instead of averaging 100 000 simulated lifetimes (a standard
  error of about 0.3%). The integral splits at the node models' quantiles,
  the steps of non-parametric fits and the grids of numerical curves, and
  follows heavy tails until what is left is negligible. A system that may
  never fail, or whose tail falls too slowly for a finite mean, has an
  infinite MTTF. The MTTF is as exact as the node reliabilities it is made
  of, and `analysis_routes()` says which nodes limit it. Nested RBDs and
  `RepeatedNode`s bring their exact MTTFs too (`node_mttf`, `mean`).
- **Shared repair crews** (#89). `RepairableRBD(..., repair_crews=n)`
  lets at most `n` repairs proceed at once. A component whose job (a repair
  or replacement, maintenance that takes time, or a test that takes time)
  finds every crew busy waits, down, until one is free: the next crew takes
  the waiting job of the highest `"priority"` (a new component spec key),
  and of those the one that fell due first, and stays with it until it is
  done. A test that waits keeps its component off-line, not ageing. A
  nested RBD has crews of its own. The simulations (`availability`, `cost`,
  `compare`) follow the queue, in Python, with the same streams, so seeded
  runs stay reproducible and paired; the exact methods refuse while a job
  can wait (but see #90), and `analysis_routes()` says so. With the default
  (None), or at least as many crews as components, nothing waits and every
  result is the same as before. The crews and priorities are saved with the
  RBD. Checked against the machine-repair model's closed form (identical
  exponential units in parallel or k-out-of-n, with one or more crews).
- **Exact long-run values with shared repair crews** (#90). When the
  components the crews work on have exponential lives and exponential (or
  instant) repairs, with no scheduled maintenance or inspection, the system
  is a Markov chain: its state is which components are under repair and
  which wait, in the order the crews will take them. `mean_availability`,
  `node_availability`, `system_failure_frequency`, `mean_up_time`,
  `mean_down_time`, `mean_time_between_failures`, `expected_cost_rate`,
  `total_cost` and `capacity_distribution` solve it exactly, for up to
  15,000 states (seven components first come, first served, or more with
  priorities, which fix the queue's order); a component held working or
  broken needs no crew. Beyond that, or with other lives, repairs,
  maintenance or inspections, the exact values refuse with the reason; the
  importance measures, the availability over time and the allocations,
  which assume independent components, refuse whenever a job can wait.
  `analysis_routes()` reports each. The chain is solved directly, keeping
  every state's probability to its relative precision however small.
  Checked against the machine-repair model's closed forms (to 1e-12), a
  chain worked by hand (an instant repair), and the simulation of #89 on a
  bridge with priorities.
  - **Over time and importance with crews** (#146). From new, or from the
    components' states (each up, or down in a repair, no more than there
    are crews; or all in the long run), the same chain is followed over
    time by uniformization, to about 1e-13: `point_availability`,
    `mission_availability`, `expected_failures`, `expected_events`,
    `expected_cost`, `point_capacity` and `mission_capacity` are
    numerical. A nested RBD, with crews of its own, enters the availability
    over time through its own, the system worked out for each pattern of
    the nested RBDs up and down; the expected events and the capacity over
    time refuse a nested RBD, as yet (#162). The importance measures are
    exact, from their definitions: Birnbaum's is the system's long-run
    availability with the node held working less that with it held failed,
    each from the chain solved without it, and the improvement potential,
    RAW and RRW are built on the same values; the criticality and
    Fussell-Vesely measures are probabilities over the chain's states.
    Only the allocations still refuse while a job can wait. A chain whose
    rates are too far apart to follow in reasonable time is refused, with
    the simulation to run instead. Checked against the closed forms with a
    crew for each component (to 1e-13), chains written out by hand and the
    machine-repair model over time (against the matrix exponential), the
    long-run values, and the simulation.
- **Repairable standby groups** (#91). A component spec's `"standby"` makes
  the node a group of identical units, `"k"` operating and the rest waiting
  as spares, cold, warm or hot (`"dormancy_factor"`). When an operating
  unit fails, the spare that has waited longest is switched in, with
  `"switching_probability"`; a failed switch leaves the position empty
  until a repaired unit fills it. Each failed unit is repaired on its own, a
  job for the RBD's repair crews (#89) at the group's priority, and returns
  to fill an empty position or wait as a spare. Repair costs are charged at
  each unit's failure. The simulations follow the group, in Python, with
  each unit's draws from streams of its own. With exponential units the
  group is a small Markov chain, so its long-run availability, failure
  frequency and costs are exact and enter the RBD's exact values and
  importance measures, also with a crew limit while the group's units are
  the crews' only jobs (the "one repairman" case); crews shared with other
  components are simulated. The groups are saved with the RBD. Checked against the textbook chains (two-unit cold
  and warm standby with one or two repairers, imperfect switching), a
  switch that always fails leaving a single unit, timelines worked by hand
  with a shared crew, and the simulation.
  - **Over time** (#146). With exponential units, the group's chain is
    followed over time by uniformization from every unit ready, or from its
    long-run state (`NodeState(stationary=True)`): its availability over
    time and over a mission, and its expected failures and repairs (and so
    their costs) over a window, are numerical, to about 1e-13, and enter
    the system's like any component's. Checked against the matrix
    exponential of the group's chain, a cold pair written out by hand, and
    the simulation.
- **Spares demand and stock** (#95). `RepairableRBD.spares_demand(horizon)`
  gives the distribution of each component's replacements (its failures
  and preventive replacements) over a horizon from new, for a system or a
  `fleet`: a `SparesDemand`, with `mean()`, `std()`, `covered(s)` and
  `stock(p)`, the fewest spares that last the horizon with probability
  `p`. `RepairableRBD.spares_stock(lead_time, fill_rate=...,
  stockout_probability=...)` gives the fewest to hold when each spare used
  is reordered at once and arrives a lead time later (one-for-one
  replenishment), in the long run: a `SparesStock`, with the fill rate and
  stock-out probability it achieves and the distributions of the spares on
  order, at a random time and as a demand finds them. A component's
  replacements are a renewal process (an up time, the smaller of its life
  and its replacement age, then a repair or maintenance time), counted for
  any life and repair models on a grid refined to about 1e-6, with their
  atoms (a replacement age, work in no time) exact, over `[0, horizon)` as
  the simulation counts them (a replacement at the horizon itself falls
  after it); a fleet's systems add up independently. Standby groups and
  waiting for repair crews are not renewal processes: the counts refuse
  them, and `spares_demand(method="simulate")` counts every component's
  replacements in simulations of the whole system. `analysis_routes()`
  reports both.
  - Block-replaced and proof-tested components are counted too (#147).
    Under block replacement, a unit's next replacement is at its failure
    or at the next block time, whichever comes first, and a unit down then
    skips it. That is counted block interval by block interval on a grid
    with the block times on it, repairs and replacements taking any time.
    A tested component (tests and repairs in no time, any life) is
    replaced at the tests that find it failed, and is counted there
    exactly: from new, from a random time (the next replacement `j` tests
    on with probability `R((j - 1) T) / S`), and before a replacement. So
    `spares_demand` takes both, and `spares_stock` the tested ones. A
    block-replaced component's stock (#160), and tests or repairs that take
    time (#159), still refuse. Checked against the plain renewal count of
    each block interval, binomial counts for a constant failure rate, a
    Monte Carlo of the replacements, and the RBD's simulation. Checked against Poisson closed forms (constant failure
  rates, instant replacement), a direct simulation of the renewal process
  (Weibull lives, lognormal repairs and age replacement: from new, from a
  random time and before a replacement) and the RBD's simulation. A new
  guide page, Spares, covers them.
- **Replacement on condition** (#96). A `"preventive"` schedule with
  `"policy": "condition"` inspects the unit at every multiple of its
  `"interval"`, while it is up, and replaces it only if it is then more
  likely than `"threshold"` to fail before the next inspection, given its
  age (`1 - R(a + T) / R(a)`, the conditional survival `sf_given_state`
  uses). Each inspection is charged `"inspection_cost"`, and a replacement
  takes the schedule's `"duration"` and costs its `"cost"`, as under the
  other policies. A threshold of 0 is block replacement at the interval,
  draw for draw, and a threshold of 1 is run to failure; a constant failure
  rate is replaced at every inspection or at none. The simulations
  (`availability`, `cost`, `compare`, the event-stepping API) follow it, in
  Python, deciding each unit's replacement once when it is put into
  service. The threshold and inspection cost are saved with the RBD.
  Checked against a timeline worked by hand, block replacement and run to
  failure (identical results), constant failure rates, and a direct
  simulation of the policy.
  - Its long-run values are numerical (#145). The inspections that replace
    the unit are regeneration points, and a cycle between two is followed
    one inspection interval at a time. As under block replacement, the
    units put into service in an interval are an alternating renewal
    process of lives and repairs. The units an inspection keeps on carry
    into the next interval by age, failing as their ages say. That gives
    the long-run availability, failure frequency, MTBF, MUT, MDT, the cost
    rate (each inspection charged at the unit's availability just before
    it), the importance measures, and the calendar profile that units
    inspected at the same times share. A cycle in which the unit fails
    many times before an inspection replaces it is summed as a geometric
    series once it falls at a steady rate. A threshold of 0 gives block
    replacement's values to every digit, and one no age reaches gives run
    to failure. Over time it still refuses, with the reason (#161).
- **Small failure probabilities** (#115).
  `NonRepairableRBD.unreliability_interval(x)` estimates `P(T <= x)` by
  simulation to a relative precision (`relative_tolerance`, by default
  0.1), for the diagrams whose far tail has no exact value (a simulated
  standby or load-sharing node), drawing each lifetime from a row of
  uniforms as `random` does. Its methods: plain sampling, Latin hypercube
  samples and scrambled Sobol points (randomised quasi-Monte Carlo) in
  replicates, importance sampling from a mixture of Gaussians fitted by the
  cross-entropy method, and subset simulation with adaptive conditional
  sampling (`repyability/rbd/rare_event.py`). `method="auto"` chooses by
  rules: plain sampling if a pilot sees enough failures, the cross-entropy
  method if the system fails in at most eight ways of plain components,
  subset simulation otherwise. The sample sizes of subset simulation and
  the cross-entropy method are planned from pilots, not stopped as soon as
  a skewed estimate looks precise. On a benchmark suite (to 1e-8, against
  exact values) subset simulation was worth 20 000 to 40 000 plain
  lifetimes each at 1e-8, the cross-entropy method nearly two million
  where it applies (and wrong where a system fails in more ways than its
  mixture follows, which "auto" avoids), and "auto" was within 9 % of
  every exact value; the simulation guide gives the table.
  `ConfidenceInterval` gains a `method` field, set where the method
  chooses.
- **Runs split across machines** (#114). `RepairableRBD.simulate_chunk(
  t_simulation, start, stop, seed=...)` runs simulations `start` to
  `stop - 1` of the run `availability(t_simulation, mc_samples=N,
  seed=...)` makes (each simulation draws from streams seeded by the seed
  and its position alone, so it is the same wherever it runs) and returns
  a `SimulationChunk`: their totals, which save to JSON
  (`to_json`/`from_json`, `to_dict`/`from_dict`) and merge
  (`SimulationChunk.merge`). `availability_from_chunks` turns chunks into
  the run's `AvailabilityResult`: chunks of simulations `0` to `N - 1` give
  the same per-simulation values and timeline as the run, and its totals
  to the last bit (#151). Chunks carry their run's settings and a hash of the
  system, and only chunks of one run merge. `NonRepairableRBD.
  random_block(block, seed)` draws one 10 000-lifetime block of the
  lifetimes `random(size, seed=seed, n_jobs=...)` draws. The simulation
  guide gives the engines' throughput on one machine, in a table.
  - **The curve on a grid** (#153). `availability(..., curve_points=G)`
    keeps the availability over time on `G` steps of the window instead of
    at every change of every simulation: the simulations count their
    changes in the steps, so the curve costs `G` counts however many run,
    and its values at the grid's times are the full curve's there,
    exactly. Everything else in the result is the same. `simulate_chunk`
    takes it too, and a chunk then carries the counts rather than every
    change (chunks counted on different grids are of different runs).
    The default, None, keeps the full curve. A compiled run of 40 960
    simulations whose full curve had 3 million points ran 16% faster with
    1 000 steps. `cost()`, whose result has no curve, now counts the
    changes on a one-step grid rather than keeping them.
  - **Exact totals** (#151). Every total a simulation run adds up (the
    system's and each node's up and down times, the times each node is up
    and down with the system, the costs by category and by component, the
    capacity curve and the time at each capacity) is kept exactly, as
    floats whose exact sum it is, and rounded once, correctly, when the
    result is built (`repyability/rbd/_exact.py`). Chunks merged in any
    grouping then give the run's totals to the last bit, and the totals are
    slightly more accurate than sums in order. An array of values is
    summed exactly at once by error-free extraction (Rump, Ogita and
    Oishi's AccSum), so it costs about 5% of a compiled run of a small
    system, and less in Python. Checked by sums of 5 000 values of 24
    orders of magnitude cut into 2, 7 and 64 pieces in any order, and by
    runs cut into uneven pieces (with antithetic pairs, a node held broken
    and costs, on both engines), against the whole run, field by field.
  - **Shards** (#152). `RepairableRBD.shards(t_simulation, mc_samples,
    seed=...)` cuts a run into shards: ranges of its simulations as plain
    data (JSON holding the system as `to_dict` saves it, the run's settings
    and the number its seed gives the streams), each a whole number of the
    run's widest block of draws. `repyability.run_shard(shard)` runs one
    anywhere, on any engine, and gives back its partial: its totals, as the
    bytes of a NumPy `.npz` file read without pickle
    (`SimulationChunk.to_npz` and `from_npz`); so does `python -m
    repyability.rbd.shards < shard.json > partial.npz`, for batch systems.
    `availability_from_chunks(partials, mc_samples=N)` puts them together
    in any order, and refuses a missing one. `availability(...,
    shard_map=...)` and `cost` do it all through any map
    (`concurrent.futures`, Ray, Dask, ...), in rounds for a `tolerance`,
    with the same result to the last bit. A worker refuses a shard of
    another RePyability version, and the result partials of other shards.
    A system whose models would not load back as themselves from JSON (a
    subclass of a surpyval model) is refused, as a worker would simulate
    another system, and `analysis_routes()` says so. `n_jobs`' processes now
    get the system once, when they start, and send back each block's totals
    rather than every simulation, which the parent merges: on four cores,
    `n_jobs=4` ran a 12-component system over 5 000 h 1.5 times as fast as
    before (2.6 times its one-process speed, from 2.0) and a 70-component
    one over 2 000 h 1.6 times (3.1, from 1.9), as the parent's share of a
    250-simulation block fell from 12 to 0.6 ms and from 29 to 2.2 ms.
- **Control variates from an exact twin** (#154). `availability(...,
  control_variate=True)` and `cost` simulate the system alongside its exact
  twin: the same diagram, components and models, failing and repaired
  independently, without a limit on repair crews or maintenance groups
  and, component by component, without what the exact methods over time
  do not take (a standby group's switching, its units then operating
  together; imperfect repair; replacement on condition; inspections they
  do not take). The twin's mean availability and cost over the window are
  exact (`mission_availability`, `expected_cost`), and it draws the
  system's random numbers (common random numbers, as `compare` does), so
  its error against its exact value is taken off the system's mean, with
  the coefficient that leaves the least variance: unbiased, with `1 -
  corr**2` times the variance. `mean_availability_interval` and the cost's
  `mean_interval` give the controlled estimates, a `tolerance` is judged
  on them, and the new `ControlVariate` (the results' `control_variate`)
  holds the twin's values, its exact value and the coefficient; everything
  else in the result is the simulations' own. A system that is its own
  twin gets its exact value, with a standard error of 0.
  `analysis_routes()` says, in the new `AnalysisRoute.twin`, what the twin
  leaves out, or why there is none. On 4 000 simulations over 1 000 hours
  the mean availability's variance fell 7 times with one repair crew for
  three pumps, 109 times when their failures are rare, 11 times with two
  crews for four, 81 times for a cold-standby group of three Weibull units,
  15 times for an opportunistic maintenance group, and 1.4 times with
  imperfect repair (antithetic pairs: 1.3 to 2.4 times); the cost's, 1.2 to
  189 times. Each controlled estimate was within 1.8 standard errors of a
  plain run of 100 000 simulations. It is an option, as its gain depends on
  how close the twin is (the simulation guide gives the table).
- **Timelines** (#157): up/down histories, and a system's from its
  components'.
  - `Timeline(changes, end, up=True, planned=None, name=None)` is a unit's
    history over `[0, end]`: the times it changes state, each change down
    a failure or, `planned`, maintenance. `Timeline.from_outages` takes an
    outage log, `Timeline.from_durations` the durations up and down in
    turn. Its measures: `uptime`, `downtime`, `availability`, `failures`,
    `planned_outages`, `restorations`, `first_failure`, `up_intervals`,
    `down_intervals` and `state(t)`. `Timelines` holds many histories of
    one unit (one per simulation, say) and works out each measure for all
    of them at once, with `availability_curve()` and
    `point_availability(t)`.
  - `repyability.timelines.series`, `parallel` and `k_out_of_n` (and `a &
    b`, `a | b`, `~a`) merge timelines as a diagram's structure does, in
    one sweep over their changes, every history at once. Each change of a
    merged timeline keeps its cause, the input whose change made it, so
    `failures_by_cause()` says which component took the system down each
    time. Changes at the same time are taken one after another in the
    order of the inputs, so an instant repair takes the system down and
    back up at that instant.
  - `RBD.system_timeline({node: timeline})` merges the components'
    timelines up the diagram's modules and core (path set by path set, or
    decided at each change for a large core), with repeated nodes and
    junctions: from outage logs, a what-if edit of one, or simulated
    histories.
  - `RepairableRBD.simulate_timelines(t_simulation, mc_samples, seed,
    ..., engine="auto", n_jobs=None, start=0)` keeps each simulation's
    histories whole, every component's and the system's, in a new
    `TimelineSimulation`: the simulations `availability` runs with the
    same seed, and their histories the event loop's, every change with the
    component it is credited to, whichever engine makes them (each
    simulation's uptime is `availability`'s to the last bit). So any
    measure of a history can be read off them: the first system failure,
    the longest outage. Both event loops record them as they run, numba's
    for every system it simulates (`engine` chooses as for
    `availability`), at 3% to 32% more time than `availability`'s run of
    the same simulations, 5 to 16 times as fast as the Python loop on
    systems with repair crews, standby groups and maintenance. On the
    Python engine, independent components' histories are drawn straight
    from their streams instead, a batch at once, several times as fast as
    its loop, with a simulation in which components change at the same
    instant run in the loop. `n_jobs` runs on threads (numba, the streams)
    or processes (the Python loop), with the same histories; `start` makes
    a part of a run, and `TimelineSimulation.join` joins parts made apart
    (as `Timelines([a, b])` joins a unit's). A nested RBD's history is its
    own system's. Refused, as the simulations are, with common-cause
    groups; `analysis_routes()` says which engine records a long run. The
    timelines guide shows them all.
- **Imperfect repair** (#109). A component spec's `"repair": {"model":
  "kijima1" | "kijima2", "q": q}` makes its repairs imperfect: a repair
  after the unit has operated `x` since the last takes its virtual age from
  `v` to `v + q x` (Kijima I) or `q (v + x)` (Kijima II), and each life is
  drawn given it (`H(v + X) = H(v) + E`, as surpyval's virtual-age models
  draw it; in closed form for Exponential and Weibull lives, by surpyval's
  `conditional_gaps` otherwise), from a stream of its own. `q = 0` renews
  the unit at every repair, as before, and `q = 1` is minimal repair.
  `"replace_after": N` replaces the unit, as new, at the N-th failure since
  it was renewed. A repair is charged its `"repair_cost"`; a replacement its
  `"repair_cost"` and `"replace_cost"`, and it uses a spare. A preventive
  replacement renews the unit, and age replacement counts its operating
  time since it was renewed; hidden failures are repaired imperfectly too.
  The simulations (`availability`, `cost`, `compare`, the event-stepping
  API, nested diagrams) follow it, in Python; the exact long-run values,
  the availability over time and the spares counts refuse it with the
  reason, and `analysis_routes()` says so. Saved with the RBD. Checked
  against timelines worked by hand (with replacement at the third failure,
  and age replacement), the cumulative hazard under minimal repair (the
  power law), availability falling and failures rising with `q`, `q = 0`
  and `replace_after=1` against the renewed component (identical draws,
  with age replacement and hidden failures too), and the closed-form lives
  against surpyval's.
- **Opportunistic maintenance** (#108). Components with the same `"group"`
  form a maintenance group: each failure of a member, and each scheduled
  replacement, opens a *stop* of the group, at which every other member
  that is working and at least its `"opportunity"` age (a new key of an
  age-replacement `"preventive"` schedule) is replaced too, taking its own
  maintenance time and cost. `RepairableRBD(...,
  maintenance_groups={group: {"setup_cost": c, "system_down": b}})` prices
  each stop's set-up, charged once per stop (all the work started at one
  instant), and with `system_down` makes every system outage a stop as
  well. A member due at the stop's instant keeps its own replacement, so
  units on one schedule are replaced together, on schedule. The
  simulations (`availability`, `cost`, `compare`, the event-stepping API,
  nested diagrams, repair crews) follow it, in Python; the results count
  each component's early renewals (`opportunistic_renewals`) and the costs
  have a `"setup"` category. A component that can be renewed early is
  refused by the exact long-run values, the availability over time and
  the spares counts; with none, `expected_cost_rate` charges a set-up at
  each failure and preventive replacement of a member (refusing two
  members replaced on a clock, which share stops). Groups and
  opportunities are saved with the RBD. Checked against timelines worked
  by hand, plain age replacement (an opportunity at the interval gives the
  same draws), the exact cost rate, and a two-unit train with a large
  set-up cost, whose cost rate grouping lowers.
- **Phased missions** (#100, #101). `PhasedMission([(name, duration, rbd),
  ...])` is a mission through phases (take-off, cruise, landing), each with
  its own duration and `NonRepairableRBD` over the same components: a node
  name is one component, with one model, whose life runs through the whole
  mission, so a component lost in one phase stays lost in the later ones.
  `reliability()`, `unreliability()` (to its own precision when small) and
  `phase_failure_probabilities()` (the chance of getting through the
  earlier phases and failing in each) are exact: each component's life is a
  chain of independent segments, one per phase (Esary and Ziehms), and the
  phases' structures over the segments are combined in one binary decision
  diagram (#142, after Zang, Sun and Trivedi), each phase's built from its
  diagram's modules and core with no path set listed, and a component's
  segments decided together, in phase order (`_ordered_bdd.py`, a reduced
  ordered decision diagram with a unique and a computed table). Eight
  bridges in a chain, 65,536 path sets a phase, take a fraction of a
  second, where the decomposition of the path sets (still there, as
  `phased_mission.METHOD = "paths"`) took 1.5 seconds at four and minutes
  beyond; a diagram of more than a million nodes refuses, pointing to the
  simulation.
  `method="simulate"` draws each component's life once per mission, and
  `reliability_interval` gives the simulated reliability with a confidence
  interval, to a `tolerance`, with antithetic pairs if asked. Checked
  against one phase (the diagram's `sf` at its duration), phases of one
  diagram (its `sf` at their total), series and parallel phases in closed
  form, missions worked out by enumerating the phase each component fails
  in (repeated nodes, nested RBDs, fixed probabilities and a phase of no
  duration among them), and the simulation.
  A new guide page, Phased missions, covers it.
- **A decision diagram for meshed diagrams** (#102, #103). The part of a
  diagram that the reduction to modules leaves (the core) is decided by a
  binary decision diagram built from its graph (`bdd.py`) when it may have
  more than a hundred minimal path sets (a bound found in one pass over the
  core), and by the Shannon decomposition of its path sets, as before,
  when it has fewer. The core's vertices are decided in a topological
  order chosen to keep the frontier narrow (the better of a greedy and a
  breadth-first order), and each state (the frontier's reached vertices,
  and any repeated component decided for a later appearance) is solved
  once; a component drawn in several places is decided where it first
  appears, so the diagram is ordered and, reduced, canonical. Its size
  grows with the mesh's width rather than its number of paths: six bridges
  in series take 0.06 seconds against 9, fifty bridges or a 10 × 10 grid,
  which the path sets do not finish, a fraction of a second, and a long
  mesh no longer reaches Python's recursion limit. The diagram has the
  plan's format, so the probabilities, their complements, the gradients
  behind the importance measures and the minimal cut sets replay it
  unchanged; the structure function walks it, the simulated lifetimes come
  from it, and the path sets are listed from it only when asked for.
  `repyability.rbd.modular.CORE_METHOD` (`"auto"`, `"paths"` or `"bdd"`)
  forces either route. Checked against the path-set route on 400 random
  diagrams with k-out-of-n nodes and repeated components (path and cut
  sets, relevant nodes, probabilities, gradients, structure function,
  lifetimes), through the public methods, against the closed form of forty
  bridges in series, and by the whole test suite run with each route
  forced.
- **Networks** (#104). `Network(links, source, target, nodes=None)` is
  an undirected network whose links (each a name mapped to its two nodes
  and a lifetime model or probability), and optionally nodes, fail. `sf`,
  `ff` (to its own precision) and `mean` give the reliability of the
  connection between the two terminals, exactly, by a binary decision
  diagram built from the network link by link (#143, after Hardy, Lucet and
  Limnios): after each link, what is left depends only on how the nodes
  with links still to come are joined up, and which groups hold the
  terminals, so equal states are solved once and the diagram grows with the
  network's width rather than its paths. A 6 × 6 grid (over a million
  paths) takes a tenth of a second and an 8 × 8 one about a second, where
  the paths' decomposition (still there, as `network.METHOD = "paths"`)
  did not finish a 5 × 5 one; a diagram of more than a million states
  refuses, pointing to the simulation. `cut_sets` comes from the same
  diagram, `birnbaum_importance` gives every element's at once from its
  gradient (from whichever of the reliability and unreliability is the
  smaller, so a small one keeps its precision), and `path_sets` lists the
  simple paths, up to 100,000. `method="simulate"` and `random(size)` draw each
  element's lifetime and find each sample's longest-lasting path, adding
  links longest-lived first until the terminals join. Checked against the
  bridge network's closed form, enumeration of link and node states on
  random small networks, series and parallel networks against their
  diagrams, means in closed form, and the simulation; the decision diagram
  against enumeration and the paths' decomposition on ladders, grids,
  failing nodes and terminals and parallel links, and a 7 × 7 grid against
  the simulation. A new guide page, Networks.
- **Demonstration test planning** (#129): how many units, or how long a
  test, demonstrates a reliability at a confidence level, and what a
  finished test demonstrated. `demonstration_sample_size` (the success run,
  and binomial tests that allow failures), `demonstrated_reliability` (the
  Clopper-Pearson bound, surpyval's `success_run` with no failures), and
  Weibayes plans that test each unit for several missions when the Weibull
  shape is known (`test_multiple`, `shape`, `demonstration_test_multiple`).
  For a constant failure rate, `mtbf_test_time` and `demonstrated_mtbf`
  (chi-squared). `demonstration_pass_probability` and
  `mtbf_pass_probability` give a plan's operating characteristic: the
  chance a design passes, for its consumer's and producer's risks. A new
  guide page, Demonstration testing, covers them.
- **When is a simulation needed?** The README says, by what you ask, the
  components and the maintenance, what is computed exactly, what is
  simulated, and which of those simulations must be and which could be made
  exact (with the issues that would do it). A test keeps it in line with
  `analysis_routes()`.
- **Standby and load sharing without simulation** (#135, #138, #139).
  Arrangements that were Kaplan–Meier fits to simulated lifetimes are
  worked out instead, deterministically:
  - hot standby of any units, exactly: it is k-out-of-n, the number of
    failed units a Poisson-binomial count of the units' own probabilities
    (each tail to full precision);
  - warm standby with one unit operating, of any units: a recursion over
    the spares' switch-ins on a time grid (a spare switched in at `τ` has
    aged `dormancy_factor · τ`), accurate to about 1e-5;
  - cold standby of identical units with several operating: each operating
    position runs a renewal process of the units' lives, so the failures by
    `t` are a sum of renewal counts, from the cold-standby convolution;
  - cold standby of different units with two operating: a recursion over
    the switch-ins on the time and the other operating unit's start (the
    newcomer starts new), whose steps are convolutions, accurate to about
    1e-4 (1e-3 for lives with a steep start);
  - load sharing of identical units: they age alike, so they fail in the
    order of their exposures to failure, and a recursion over the failures
    on a grid of exposure and time gives the lifetime, to about 1e-4.
  Their routes are exact or numerical, `is_simulated` is `False`, and they
  no longer warn. Cold standby takes imperfect switching with several units
  operating too (one probability for all, or one per spare), in its
  numerical reliability and its simulations alike. Still simulated (and
  deprecated): warm standby with several operating, cold standby of
  different units with three or more operating, and load sharing of
  different units. Checked against parallel and k-out-of-n units, Erlang and
  hypoexponential lives, the two-unit warm integral, the expected order
  statistics of a load-sharing group, and a million simulated lifetimes of
  each.
- **Uncertainty intervals on the MTTF and lifetimes** (#133):
  `mean_uncertainty`, `bx_life_uncertainty` and
  `time_to_reliability_uncertainty` carry the uncertainty of fitted
  component models to the MTTF, a B*X* life and the time to a reliability,
  as `sf_uncertainty` does to the reliability: the same `uncertainty`
  argument and draws (`"fit"`, parameter distributions, or lists of models,
  shared by the nodes given together), each draw's value worked out
  exactly (the area under its reliability, or by root-finding on it), and
  an `UncertaintyResult`. On fitted data the MTTF's interval is many times
  wider than the simulation error `mean_time_to_failure_interval`
  reports, which is all an interval on the MTTF said before. Checked
  against each drawn model's own mean and quantiles, the diagram rebuilt
  with each drawn model, and the known distribution of a shared rate's
  MTTF and B10.
- **Common-cause groups over a lifetime** (#132): `BetaFactor(beta,
  basis="rate")` and `MGL(..., basis="rate")` split each member's failure
  *rate* rather than its probability. The shared cause is a shock that has
  not struck by `t` with probability `R(t)^β`, and each member survives its
  own causes with `R(t)^(1 − β)`; under MGL each specific set of members
  has a cause of its own, striking independently. Every member keeps its
  own life distribution, whatever it is, and the model holds over the
  whole life, so the system's reliability falls to 0 with its members'
  (the probability split leaves a parallel pair at 0.288 for ever with
  `β = 0.2`). The exact `mean` includes such a group, and `random`,
  `mean(method="simulate")`, `mean_time_to_failure_interval`, `compare`
  and `unreliability_interval` draw its shared shocks (through the
  members' quantile function; members whose model has none are refused).
  To first order in `Q` the two splits agree, so over a mission or a
  proof-test interval they give about the same. Small probabilities and
  long-life survivals keep their precision. The basis is saved with the
  diagram (only when it is `"rate"`, so files are unchanged). Checked
  against the closed forms, enumeration of the MGL causes, and simulation.

### Changed

- **`fussell_vesely` is exact** (#137). The Fussell–Vesely importance was
  the rare-event sum of the probabilities of the minimal cut sets
  containing a node, over the system's unreliability: the only importance
  measure that was not exact, and once failures stop being rare it passes
  1 (nearly 2 on a bridge near failure). It is now the probability that
  some minimal cut set containing the node has failed, from the exact
  engine, so it is between 0 and 1 and keeps its precision however small:
  down the module tree it is a product of the modules' factors, and the
  part that is not series-parallel takes one Shannon decomposition of the
  cut sets through each of its nodes, kept for later calls. Values change
  where a node is in more than one minimal cut set (a bridge, a vote,
  parallel chains), most where failures are likely. `method="rare_event"`
  gives the sum, as many PRA tools report it, on `NonRepairableRBD`,
  `RepairableRBD` and `FaultTree`; with `fv_type="p"`, the measure takes
  the union of the minimal path sets' failures (or, with `"rare_event"`,
  their sum) as before.
- **A diagram that is not one says what is wrong** (#131). Building an RBD
  that is not a valid diagram raised `ValueError: RBD not correctly
  structured` and nothing more; the message (and the warning, with
  `on_infeasible_rbd="warn"`) now lists each finding on a line of its own:
  a node in the edges with no model, a model for a name in no edge (with
  the node it was likely meant for: "did you mean 'pump'?"), a cycle, more
  than one node with no incoming or no outgoing edges, a `k` of 0, above the
  node's inputs, or for no node. A model for a name in no edge is reported
  as such, and left out of the diagram, rather than added as an isolated
  node, which hid the input and output nodes and reported them as missing
  models; `structure_check` gains `nodes_with_no_model` and
  `nodes_in_no_edge`. `on_infeasible_rbd="ignore"` now builds a diagram
  with k-out-of-n errors (it raised that no path reached the output), so
  its `structure_check` can be read. A `RepairableRBD` component with no
  `"repairability"` raises a ValueError that says so, not a KeyError.
- **Perfect junction nodes are no components to rank or allocate** (#134).
  A `PerfectReliability` node, such as the vote of a k-out-of-n
  arrangement, is a drawing device: the importance measures (Birnbaum,
  improvement potential, risk worths, criticality, Fussell–Vesely,
  `importances_given_state` and the structural importance) leave it out,
  the structural importance takes it as always working (so the other nodes'
  values change: each pump of a 2-out-of-3 vote now has 0.5, not 0.25), and
  the allocations hold it at 1 and leave it out of their results, where
  `equal_allocation` and `simple_allocation` gave it a value below 1.
- **A common-cause group that splits a probability warns beyond its
  range** (#132). The default split is a rare-event model, and over a
  lifetime it gives impossible results: at the members' MTTF a beta-factor
  pair came out more reliable than an independent one, and 29% of systems
  never failed. A diagram now warns, once for each group, when its
  members' probability of failing passes 0.1 (where it is about 3.5% off
  the rate-based split), naming the group, the `Q` reached and the
  rate-based model to use. `MGL`'s repr shows one letter as `MGL(0.1)`
  rather than `MGL(0.1,)`, and a model's repr shows its basis when it is
  `"rate"`.
- **`RegressionNode.sf` and `ff` give a float for a scalar time** (#134),
  as every other node does (and surpyval's models), rather than a
  one-element array.
- **Versions have two parts**, major.minor: this release is 0.11, not
  0.11.0, and the next, whether it adds or fixes, will be 0.12. From 1.0,
  only a release that breaks compatibility raises the major number.
- **Cost breakdowns have a seventh category** (#108):
  `CostResult.by_category` (and so `cost()` and `availability().cost`)
  gains `"setup"`, a maintenance group's set-up costs, 0.0 without groups.
  Code that compares the whole dict needs the new key.
- **`mean()` is exact** (#122). `NonRepairableRBD.mean()` and
  `mean_time_to_failure()` return the exact MTTF (see Added), so they give
  different numbers than in 0.10, within the old estimate's sampling error.
  `method="simulate"` gives the Monte-Carlo estimate as before, with the
  same seeded numbers; a simulation option (`mc_samples`, `seed`,
  `tolerance`, ...) given without it is ignored, with a
  `DeprecationWarning`. The exact MTTF refuses common-cause groups (their
  models split a failure probability they assume is small, and over a
  whole lifetime it runs to 1), where the simulated one left them out
  without a warning; `method="simulate"` still does. `node_mttf()` no
  longer simulates: a nested RBD's and a repeated node's MTTF are exact,
  and a simulated standby or load-sharing node's is the mean of the
  lifetimes it was built from; its `mc_samples` and `seed` are ignored and
  deprecated. `RepeatedNode.mean()` is exact too, and simulates with
  `method="simulate"`.
- **One name for the number of simulations** (#105): `mc_samples`, and
  `max_samples` for its cap, everywhere. `RepairableRBD.availability()`,
  `cost()` and `compare()` took `N` and `max_N`; `StandbyModel` and
  `LoadSharingModel` took `n_sims` (now the attribute `mc_samples`); the
  node models' `mean()` took `N`; and `Repairable`'s simulated policies
  took `n_simulations`. The old names still work, with a
  `FutureWarning`, until 0.12. Saved files store `mc_samples`; files
  that store `n_sims` still load. (A result's `n_simulations`, the number
  of simulations it was made from, keeps its name.)

- **`is_analytically_solvable()` and `get_non_analytic_nodes()` flag only
  simulated nodes** (#127). They counted every standby, repeated-standby and
  load-sharing node as simulation-backed, even one with a closed form or a
  numerical convolution. They now flag a node only when its reliability is
  fitted to simulated lifetimes (see `is_simulated`), and so does
  `structure_check`. A diagram with a cold spare for one unit, say, is now
  solvable.
- **Seeded repairable simulations give new numbers, once** (#119). Every
  random quantity a `RepairableRBD` simulation draws now comes from a
  stream of its own: each component's times to failure, its repair times,
  its maintenance or test times, and each cost given as a distribution,
  named by the component's place and the quantity and seeded from the
  run's seed. The results are as correct as before, but a seeded run of
  `availability()`, `cost()` or `compare()` gives different numbers than in
  0.10, within their sampling error. In exchange:
  - a simulation is the same however the run is split up: a run with
    `n_jobs` gives the same results as one without (before, a parallel run
    differed from a serial one), a run to a tolerance that stops after `n`
    simulations is the run of `mc_samples=n`, and the first `n`
    simulations of any run are a run of `n`;
  - one component's draws never depend on another's, and each simulation's
    `k`-th draw of each stream is fixed, so antithetic pairs pair every
    quantity (costs too, which were unpaired) and `compare` matches every
    one (costs too);
  - a model whose draws cannot be streamed no longer sends every other
    component back to drawing one number at a time: it draws from numpy's
    global RNG, seeded afresh for each simulation, and the rest stream;
  - without a seed, a run takes one number from numpy's global RNG as its
    seed (so `np.random.seed(s)` beforehand gives the run `seed=s` gives)
    and otherwise leaves it as it was, where it used to consume as many
    numbers as the simulations drew.
  Stepping a system through its events by hand (`initialize_event_queue`,
  `next_event`) still draws from the global RNG, as before.
- **Faster availability simulation** (#120, #119). Even without the
  compiled engine, `RepairableRBD.availability()` and `cost()` run 3.8–5.9×
  faster per core than in 0.10.
  - The loop works the event queue's heap directly, comparing times as
    floats.
  - The structure function is evaluated only when an event could change
    the system (a repair can't take a coherent system down, nor a failure
    bring it up), and it is called directly.
  - Normal and lognormal quantiles skip scipy.stats' argument checks
    (surpyval [#469](https://github.com/derrynknife/SurPyval/issues/469)).
  - A component with no maintenance or inspection takes its next draw
    directly.
  - Each simulation adds up its components' up times, and their overlaps
    with the system's, as it goes, instead of working them out from their
    timelines at the end.
- **Faster large lifetime draws** (#114). `NonRepairableRBD.random` takes
  a large vectorised draw's uniforms about a million at a time, from the
  same stream, so its arrays stay in cache: the same lifetimes, up to
  about three times as fast on a wide diagram.
- **Faster start-up** (#121). `import repyability` no longer loads
  scipy.signal, tqdm or the process-pool machinery until a call needs them
  (about 0.2 s less here; surpyval's share is
  [surpyval #470](https://github.com/derrynknife/SurPyval/issues/470)).
  - A parallel run (`n_jobs`) under the forkserver start method (Linux's
    default from Python 3.14) has the server load RePyability once, so its
    processes start with it loaded: from the second run on, they start at
    once rather than taking a second or more each.
  - `n_jobs=-1` counts the CPUs this process may run on, which in a
    container can be fewer than the machine has.
  - CI also tests Python 3.14.
- **More accurate cold standby** (#128). The numerical convolution behind a
  `StandbyModel` (cold, one operating unit), a `RepeatedStandbyNode` and a
  `DegradingNode` now adds up the units' cumulative probabilities rather
  than their densities: the probability that the sum so far ends in each
  cell of its time grid, against the next unit's CDF half a cell back.
  - Early-life units, whose density is infinite at 0 (a Weibull or gamma
    with shape below 1), lost the probability near 0. Two Weibull(100, 0.8)
    units in cold standby had an MTTF of 227.31, not 226.60; three gamma
    units of shape 1/3 one of 108.5, not 100. They now come to 226.6007 and
    100.001, with reliabilities within about `1e-6` (`1e-5` for gamma
    shapes of 0.5 or less).
  - Units whose density is positive at 0, such as Exponential ones, were
    about `1e-4` off, and are now within `1e-7`: two exponential units with
    a 90% switch have an MTTF of 190.000002, the formula's 190, where they
    had 189.9. Units whose density starts at 0 move by less than `1e-6`.
  - It needs only each unit's CDF (or survival function), not its density.
    Building one takes about as long as before; twice as long for gamma
    units, whose CDF costs more to evaluate than their density.
- **Requires surpyval 0.21** (was 0.20), and drops the code that worked
  around surpyval 0.20 (#86). surpyval 0.21 draws the lifetimes of
  limited-failure-population and zero-inflated models, gives their mean and
  the continuous part of their density, rebuilds a model with new parameters
  keeping its offset, `p` and `f0` (`with_params`), and keeps a
  non-parametric estimate's answers in the shape of the query; RePyability
  now relies on these. Results are the same: every replaced workaround
  computed what surpyval 0.21 now does, draw for draw.
- `Repairable` passes its seed to a generalized-renewal model's simulations
  as `random_state`, surpyval 0.21's name; a model whose simulations take
  only `seed` is no longer supported.
- CI also runs the tests on the oldest surpyval `pyproject.toml` allows, so
  the declared minimum stays tested after surpyval releases.

### Deprecated

- `N` and `max_N` (`RepairableRBD.availability()`, `cost()`, `compare()`),
  `n_sims` (`StandbyModel`, `LoadSharingModel`), `N` (the node models'
  `mean()`) and `n_simulations` (`Repairable`): use `mc_samples` and
  `max_samples` (#105). They go in 0.12.
- Simulation options passed to `NonRepairableRBD.mean()` or
  `mean_time_to_failure()` without `method="simulate"`, and `node_mttf()`'s
  `mc_samples` and `seed`: they are ignored (#122).
- `RepeatedStandbyNode`'s `N` and `lower`, which it has ignored since its
  reliability became a numerical convolution: passing them now warns.
- The `fussel_vesely()` alias and `NonRepairable.find_optimal_replacement()`'s
  `options`, deprecated earlier, go in 0.12 too, with the ignored arguments
  above (#149).
- **Every deprecation now warns with a `FutureWarning`**, which Python
  always shows, rather than a `DeprecationWarning`, which it hides outside
  scripts and notebooks: each gives one minor release's notice and goes in
  the next.
- **Non-parametric nodes** (a surpyval `KaplanMeier`, `NelsonAalen` or
  other non-parametric fit as a node's life or repair time, directly or
  inside a standby, repeated or degrading node or a `NonRepairable`): an
  RBD built with one warns, with a `FutureWarning`, and 0.12 will refuse
  it (#149). Their curves end at the data, so the MTTF, B-lives and long-run
  values beyond it are artefacts, and their draws cannot be paired or
  streamed. Fit a parametric distribution in surpyval instead. The
  standalone `NonRepairable` maintenance policies keep them.
- **The fit to simulated lifetimes** that stands in for a reliability with
  no exact or numerical form: warm and hot standby, and cold standby with
  two or more units operating, of units that are not identical
  Exponentials (`StandbyModel`), and load sharing of units that are not
  identical with an Exponential baseline (`LoadSharingModel`). Such a
  model warns when built, with a `FutureWarning`. In 0.12 it will still
  draw lifetimes for simulations, but have no `sf`: the analyses that
  need one will refuse, and point to the system's simulations (#149). #135, #138
  and #139 would make most of these cases exact or numerical. Hot
  standby is *k*-out-of-*n*: its units as parallel nodes are exact now.
  `allocate_redundancy`'s scoring of cold standby that needs two copies
  working uses such a model internally, without the warning.

### Fixed

- A run left its interim state in the nested RBDs of the system it
  simulated (only the system's own was cleared), so a parallel run after
  one, of a system with a standby group inside a nested RBD, could not
  pickle the system for its processes (#155).
- `parameter_sensitivity` left a model's offset, limited-failure-population
  and zero-inflation parameters out of the unperturbed value of a one-sided
  difference (taken where one side of a parameter is not valid), so such a
  difference was of two different models (#140).
- **New users' first stumbles** (#134):
  - `repr()` of an RBD summarises it: its nodes, input and output, k-out-of-n
    nodes and, for a `NonRepairableRBD`, repeated nodes, junctions and
    common-cause groups, or for a `RepairableRBD`, its maintained, tested,
    standby and nested components and repair crews.
  - A number given as a `NonRepairableRBD` node's model raised an
    `AttributeError` at the first analysis; it is refused at construction,
    with what to give instead (`FixedEventProbability.from_params(q)`, q the
    probability of failing). A number as a `RepairableRBD` component's
    `"reliability"` or `"repairability"` says the same, with the
    exponential of that mean and the fixed time as the alternatives.
  - `remaining_life(state)`, with the state where the target goes, raised a
    `TypeError` about comparing a float and a dict; it says which argument
    is which, and checks the target is in (0, 1). A state may give a plain
    number as a node's age.
  - `to_json(fp)` writes to a path or a file and `from_json` reads one, as
    surpyval's do, for RBDs, fault trees and simulation chunks; without a
    path `to_json` returns the text, as before.
  - A `FaultTree` has an RBD's `ff` and `sf` (the latter in its own right,
    to its full precision), and `get_min_cut_sets` and `get_min_path_sets`
    (sets); an RBD has a fault tree's `minimal_cut_sets` and
    `minimal_path_sets` (lists, smallest first), so the same code runs on
    either.
  - `NonRepairable`'s `time_to_replace` defaults to None (an instant
    replacement), so `help()` no longer prints a surpyval model inside the
    signature.
  - A `RegressionNode` with a covariate vector of the wrong width says how
    many covariates the model was fitted with.
- **Risk achievement and reduction worth keep their precision** when the
  system's unreliability is small: they divide unreliabilities, which were
  computed as one less a reliability close to 1, losing digits (for a
  system failing with probability 1e-12 given a component works, all but
  four). They are now worked out as failure probabilities in their own
  right, to full precision; values move slightly, most for very reliable
  systems.
- **Small failure probabilities keep their precision** (#148).
  `NonRepairableRBD.ff` and `unreliability` were one less the reliability,
  and `RepairableRBD.mean_unavailability` one less the availability, which
  keeps only about 1e-16 of absolute precision: a 2-out-of-3 system
  failing with probability 3e-15 was 0.08 % off, and below about 1e-16 it
  got 0. They are now worked out in their own right, as sums of products
  of each component's own probability of being failed (its model's `ff`;
  `MTTR / (MTTF + MTTR)`; `1 - exp(-λu)` between tests; or its Markov
  chain's down states, with repair crews or in a standby group), honouring
  common-cause groups and working or broken nodes, to full relative
  precision. `Hf` and `df` (so `hf`) keep it too, the mean down time and
  the downtime cost rate take the precise unavailability, and
  `NonRepairable.mean_unavailability`, a series `RepeatedNode`'s `ff` and
  a `RegressionNode`'s `ff` are worked out directly (along a covariate
  schedule, a proportional-odds model's still loses it: surpyval #528).
  Ordinary values move only by rounding.
- **Importance measures and failure frequencies keep their precision**
  (#148). Birnbaum importance was `R(i works) - R(i failed)`, a difference
  of two values near 1 in a reliable system: in a parallel pair of units
  down 1e-12 of the time it was 2e-5 off, and below about 1e-16 it was 0.
  So were the measures built on it and the system failure frequency
  (`sum_i I_B(i) * omega_i`), and with it `mean_time_between_failures`,
  `mean_up_time` and `mean_down_time`, the quantities a high-demand safety
  function's failure rate (PFH) comes from. Birnbaum importance is now
  every node's derivative of the system's probability of working, from one
  pass of the decomposition: products through the modules, and in a
  meshed core a difference of the smaller conditional probabilities. Every
  importance measure (Birnbaum, improvement potential, criticality, the
  risk worths and Fussell–Vesely) takes each node's probability of failing
  from its model's `ff` (or, for a repairable component, its unavailability
  worked out in its own right), and the planned outages at block
  replacements are the rise in the system's unavailability. An
  age-replaced component's failures per cycle come from its model's `ff`.
  All keep full relative precision; ordinary values move only by
  rounding. Birnbaum importance is also faster: one pass for every node,
  not two system evaluations per node (30 times as fast for 300 nodes).
- A repairable component whose reliability is a cold `StandbyModel` with a
  unit that may never fail (a surpyval model with `p < 1`) had a long-run
  availability of NaN. It is now 1, as for any component some of whose
  units never fail: sooner or later it gets one, and is up for good.
- **Exact values that changed from call to call.** A simulated
  `StandbyModel` or `LoadSharingModel` (one with no closed form or
  convolution) drew fresh lifetimes from numpy's global RNG for its
  `mean()` at every call. So the exact long-run values of a repairable RBD
  with such a node changed slightly each time they were asked for: its
  `mean_availability`, failure frequency, costs and importance measures.
  `mean()` is now the mean of the lifetimes simulated when the node was
  built, which its Kaplan-Meier `sf` is fitted to. It is the same on every
  call, consistent with `sf`, and reproducible with the node's `seed`.
  `mean(mc_samples=..., seed=...)` still makes a fresh estimate.
- A `RepairableRBD` accepted a component of any type, so a `Repairable` (a
  model of imperfect repair, which cannot be a node) or a bare surpyval
  model failed only at the first analysis, with an `AttributeError`. The
  constructor now raises a `TypeError` that says what a component can be.

## [0.10.1] - 2026-09-29

Works with surpyval 0.21 without deprecation warnings, as well as with
0.20. No behaviour changes: results are the same under either version.

### Fixed

- **Works with surpyval 0.21 without deprecation warnings.** surpyval 0.21
  renames its simulations' `seed` to `random_state` (the old name warns
  until surpyval 0.22 removes it). `Repairable` passed `seed=` to a
  generalized-renewal model's `mcf` and `count_terminated_simulation`, so
  every simulated overhaul or failure-limit calculation warned under 0.21,
  and would fail under 0.22. The seed now goes by the name the model's method
  takes, so surpyval 0.20, 0.21 and models exposing the same methods all
  work, with the same results.

## [0.10.0] - 2026-09-28

Meet a repairable system's availability and cost targets. Choose the
age-replacement intervals of its components together
(`optimal_replacement_intervals`), the proof-test intervals that keep a
safety function's PFDavg within its target (`optimal_inspection_intervals`),
and the availability, MTTF and MTTR each component needs
(`availability_allocation`, `mttf_mttr_allocation`). Block replacement now
has exact long-run values, like age replacement and inspection, averaged
over the schedules of components maintained or tested together. Node models
are saved in surpyval's own format, and RePyability works with surpyval's
next release as well as with 0.20.

Behaviour changes: node models are saved in surpyval's format, which earlier
versions cannot load (files they saved still load); a `RegressionNode` on a
Cox model raises from `mean()` and `random()`, as documented, instead of
returning a number; one with a proportional-odds model and a covariate
schedule follows surpyval (see Fixed); and a simulated `StandbyModel` or
`LoadSharingModel` gives a float for a single time, as the others do, where
it gave a 1-element array.

### Added

- **Availability allocation** (#107):
  `RepairableRBD.availability_allocation(target, method=...)` runs a
  reliability allocation method (`"cost_based"`, `"improvement"`,
  `"minimum_effort"` or `"equal"`) on the components' long-run
  availabilities, scoring the system as `mean_availability` does, and gives
  for each component the MTTF that meets its share at the current MTTR and
  the MTTR that meets it at the current MTTF.
  `RepairableRBD.mttf_mttr_allocation(target)` chooses the cheapest MTTFs
  and MTTRs together: Mettas's cost-based allocation with both levers, each
  with its own feasibility and limit; `levers="mttr"` holds the failure
  behaviour (maintainability allocation). Only components with corrective
  repair alone are allocated; those with preventive maintenance or
  inspection keep their availability, and enter over their calendar, so the
  allocation meets the target exactly. The results are
  `AvailabilityAllocation`s.
- **Choosing proof-test intervals**:
  `RepairableRBD.optimal_inspection_intervals()` chooses the inspection
  interval of components with hidden failures for the lowest cost rate, the
  lowest that keeps the system availability to a target (for a safety
  function, a PFDavg of at most `1 - min_availability`), or the highest
  availability within a cost rate (#94). Components tested at the same times
  are down together, so the intervals are chosen from a calendar
  (`allowed`): every combination when there are at most 2000, a local search
  otherwise. One inspected component's interval can be chosen freely. A
  1oo2 pair of shutdown valves meets a PFDavg of `1e-3` with tests every two
  years, where one valve needs them monthly.
- **Choosing maintenance intervals for the system**:
  `RepairableRBD.optimal_replacement_intervals()` chooses the age-replacement
  interval of every component (or of those named) together, for the lowest
  long-run cost rate, the lowest that keeps the system availability to a
  target (`min_availability`), or the highest availability within a cost
  rate (`max_cost_rate`) (#93). The long-run values are exact, and the search
  is a gradient search from several starting points; it returns a
  `MaintenancePlan`. A component alone in the line comes out replaced later
  than the same component with a standby, whose replacements cost the plant
  nothing.
- **Exact long-run values for block replacement** (#92). A `RepairableRBD`
  with components under block replacement now has an exact
  `mean_availability`, `system_failure_frequency`, MUT, MDT, MTBF,
  `expected_cost_rate`, `total_cost`, importance measures and
  `allocate_redundancy`, which used to raise `NotImplementedError`. A
  component's renewals are the block times at which it is up; between two of
  them it is an alternating renewal process of lives and repairs, and a
  repair can run over a block time. The renewal equations are solved on a
  grid, to about one part in a million. Components replaced at the same
  block times go down together, so the system's values average over the
  block interval (over the time the schedules take to repeat together, with
  different intervals) instead of combining each component's own average: a
  pair of pumps in parallel, both replaced at the same block times, is down
  for every replacement, which the per-component average misses entirely.
  The exact values need a surpyval parametric lifetime with a density and
  repairs that always end; the simulation covers the rest.
- **Non-parametric node models can be saved** (#85): Kaplan–Meier,
  Nelson–Aalen and the other surpyval non-parametric fits, which used to
  raise `NotImplementedError`.
- An `upstream` workflow runs the tests against surpyval's development
  branch on pull requests, on pushes to dev and master, and daily, so a
  surpyval change that breaks RePyability shows before surpyval releases it.

### Changed

- A simulated `StandbyModel` or `LoadSharingModel` (a Kaplan–Meier fit to
  simulated lifetimes) answers `sf` and `ff` in the shape of the query, a
  float for a single time, as the closed forms do. It gave a 1-element array
  on surpyval 0.20 and a float on surpyval's next release, which returns
  every model's values in the shape of the query (surpyval#381); now it
  gives the same on both, and a 2-D query keeps its shape.
- **Node models are saved in surpyval's own format** (#85):
  `{"kind": "surpyval", "model": model.to_dict()}`, loaded with
  `surpyval.from_dict`, instead of RePyability's name-and-parameters format.
  Everything surpyval keeps round-trips, including a fit's covariance, so
  parameter uncertainty can still be propagated after loading. Files saved
  by earlier versions still load, but files saved by 0.10.0 do not load in
  earlier versions.
- The simulated numbers of imperfect repair in the maintenance guide and the
  `Repairable` examples are quoted as approximate: surpyval's next release
  simulates recurrent events differently, so the same seed gives slightly
  different estimates.

### Fixed

- **Works with surpyval's next release** (#106). surpyval's development
  branch makes `mean()` infinite when some units never fail (`p < 1`),
  refuses infinite observations, and fixes a Cox model's survival before its
  first event. RePyability relied on the old behaviour in three places,
  which now work on surpyval 0.20 and on its next release alike:
  - a cold standby (and a `RepeatedStandbyNode`) of units that may never fail
    gave `nan`: its grid is now sized by the mean lifetime of the units
    that fail;
  - a warm standby, or a simulated k-out-of-n standby, of such units raised
    `ValueError`: the arrangements that never fail are now right-censored in
    the Kaplan-Meier fit;
  - a `RegressionNode`'s `mean()` and `random()` on a Cox model returned a
    number instead of raising as documented: a semiparametric model is now
    recognised by its type, not by the shape of its curve.
- `NonRepairable.find_optimal_replacement()` returns `inf` at once for a
  model some of whose units never fail, instead of reaching it by a search
  that started from `log(mean())`.
- A `RegressionNode` with a proportional-odds model and a covariate
  `schedule` follows surpyval: refused where surpyval cannot evaluate it
  (0.20), and surpyval's survival along the path where it can.

## [0.9.0] - 2026-09-28

The **Design and Maintenance** milestone. Choose redundancy and component
reliability for the most reliability under several resources, or the least
cost (`allocate_redundancy`, `redundancy_front`, reliability-redundancy
allocation), and a repairable system's redundancy for the lowest total cost
of ownership. Price and simulate scheduled preventive maintenance and the
proof tests that find hidden failures. Analyse fault trees, carry the
uncertainty in fitted component models to the system reliability, and
simulate to a tolerance, in antithetic pairs, in parallel, or two designs
with common random numbers. The exact engine reduces series-parallel parts
of a diagram to closed-form modules, so large diagrams stay fast, and the
documentation is rewritten, tested, and joined by a nine-lesson Learn
course.

Behaviour changes: `criticality_importance` defaults to the failure-oriented
form (`kind="success"` gives the old values); `simple_allocation` finds the
smallest change in the node log-odds; a repeated node no longer changes a
diagram's logic, which changes results wherever joining its appearances
added paths; and limited-failure-population and zero-inflated models are
handled throughout (see Fixed).

### Added
- **Monte-Carlo precision and speed (closes #35).** The simulations of both
  kinds of RBD can now run to a tolerance, reduce their variance and run in
  parallel, and two designs can be compared with common random numbers.
  - *Simulating to a tolerance.* `RepairableRBD.availability` and `cost`,
    and `NonRepairableRBD.mean`, `mean_time_to_failure` and
    `mean_time_to_failure_interval`, take `tolerance` and `confidence`:
    after the first `N` (`mc_samples`), and each further batch of that
    size, the run stops once the half-width of the mean's confidence
    interval (the mean availability over the window, the mean cost, or the
    MTTF) is at most the tolerance, or at `max_N` (`max_samples`, by default
    100 times the first batch) with a `RuntimeWarning`. Without `n_jobs`, a
    run that stops after `n` gives exactly the result of a run of `n`.
  - *Antithetic pairs* (`antithetic=True`, on the same methods and
    `NonRepairableRBD.random`): the second of each pair uses `1 - u` for
    every uniform `u` of the first. In a `RepairableRBD` each component
    draws from a stream of its own, so its `k`-th draw is paired whatever
    the order of the events. Intervals are worked out from the pairs' means.
    On the examples, pairing cuts the standard error of an MTTF by a quarter
    and of a window's availability and cost by a fifth to a third.
  - *Parallel runs* (`n_jobs`, -1 for one process per CPU): the simulations
    run in blocks (250 availability simulations, 10 000 lifetimes), seeded
    in turn from one `SeedSequence`, so the results depend on the seed and
    not on the number of processes.
  - *Common random numbers.* `RepairableRBD.compare(other, t_simulation,
    ...)` (availability or cost) and `NonRepairableRBD.compare(other, ...)`
    (MTTF) simulate both designs with each component drawing the same
    numbers in both (by name, and by place in nested RBDs), and return a
    `ConfidenceInterval` of the difference: seven times smaller a standard
    error than two separate runs when two plants differ only in their pumps'
    repair times.
  - `AvailabilityResult` gains `uptimes` (each simulation's up time),
    `antithetic` and `mean_availability_interval(confidence)`, the interval
    of the mean availability over the window; `CostResult` gains
    `antithetic`.

  Tests (66): runs to a tolerance against runs of their size, the
  non-convergence warning, antithetic pairs sample by sample (each
  component's draws add to one) and against exact values (the transient
  availability of exponential components, averaged over the window, a
  window's expected cost, and MTTFs as integrals of the reliability),
  parallel results across numbers of processes, comparisons against the
  exact differences and sample-path dominance (a spare pump never leaves the
  system up for less time, in any simulation), and validation. Mutation
  testing catches all 32 mutations. Docs: a Simulation precision and speed
  guide page, a Lesson 6 section with an exercise, concepts, glossary and
  overview pages.
- **Parameter (epistemic) uncertainty (`NonRepairableRBD.sf_uncertainty`,
  closes #43).** A node's model is estimated from data, so its parameters
  are uncertain; `sf_uncertainty(x, uncertainty, n_draws, seed)` carries
  that to the system reliability. Each draw gives every uncertain node a
  plausible model and the system is computed exactly, all draws at once by
  the vectorised exact engine; an `UncertaintyResult` holds the draws, the
  nominal value, and their mean, median, spread, percentiles and
  equal-tailed intervals, per time for an array of times. RePyability fits
  nothing: a node's uncertainty is `"fit"` (its parameters drawn from the
  normal approximation of its surpyval maximum-likelihood fit, `hess_inv`,
  on the log or logit scale so that every draw is valid, keeping any
  offset, zero-inflation or LFP parameter), distributions over named
  parameters (anything with `qf` or `ppf`), or a list of models (e.g.
  bootstrap refits or posterior draws made in surpyval). A tuple of nodes
  (units of one population) shares one draw; drawing them independently
  would understate the uncertainty. Tests check the draws against their
  known distributions (the log-normal rate of an exponential fit, uniform
  and normal priors, a beta prior on a fixed probability, surpyval's own
  confidence bounds), every draw against the diagram rebuilt with the drawn
  models, shared against independent draws, reproducibility and
  validation; mutation testing catches all 14 mutations. Docs: a guide
  section, a Lesson 2 section on aleatory and epistemic uncertainty with an
  exercise, concepts and glossary.
- **Fault tree analysis (`FaultTree`, closes #39).** Static fault trees:
  OR, AND and VOTE (k-out-of-n) gates over basic events, each a probability
  or a lifetime model, with events and gates that feed several gates
  (repeated events). The tree is evaluated exactly by the diagrams' engine:
  every gate below which nothing is shared is a closed-form module, and what
  the repeated events tie together is solved by the Shannon decomposition,
  with no rare-event or min-cut approximation. `top_event_probability(t)`,
  `minimal_cut_sets()`, `ranked_cut_sets(t)`, `minimal_path_sets()`,
  `occurs(events)`, and the Birnbaum, criticality, Fussell–Vesely, RAW and
  RRW importance measures (the diagrams' definitions). `FaultTree.from_rbd`
  turns a `NonRepairableRBD` into the tree of its failure (series blocks OR
  gates, parallel blocks AND gates, k-out-of-n blocks VOTE gates, a bridge an
  OR over its cut sets; perfect junctions drop out), and `to_rbd` turns any
  tree into a diagram with the same logic (repeated events as repeated
  nodes, votes through perfect junctions). Trees are saved as JSON. Tests
  check 150 random trees with repeated events and shared gates against the
  enumerated definition (probability, cut and path sets, every state, every
  importance measure), conversion both ways (every random tree, random
  diagrams, a bridge round trip), validation and saving; mutation testing
  catches all 21 mutations of the logic. Docs: a Fault trees guide page, a
  Lesson 3 section with worked examples and an exercise, concepts and
  glossary.
- **Acquisition cost, total cost of ownership, and the redundancy that
  minimises it (`RepairableRBD`, closes #71).** A component spec's new
  `"acquisition_cost"` is the one-off price of the unit: `acquisition_cost`
  sums them, and `total_cost(horizon)` gives the undiscounted cost of owning
  the system, `acquisition_cost + expected_cost_rate() * horizon`. Buying is
  not a running cost, so it stays out of `has_costs`, `expected_cost_rate`
  and the simulated samples, and `CostResult` reports it separately, as
  `acquisition_cost`. `allocate_redundancy(horizon)` chooses how many
  identical, independently repaired, active copies of each component (by
  default, each with an acquisition cost) give the lowest total cost over
  the horizon, optionally with a `min_availability`: each copy costs its
  price plus its running cost over the horizon and saves system downtime
  cost, and every design is scored exactly (`n` copies down `(1 - A) ** n`
  of the time, inspected components' copies tested together), so the result
  is `total_cost`, `expected_cost_rate` and `mean_availability` of the
  design drawn out. The total is not monotone in the copies; the exact
  search is bounded because the `k+1`-th copy of a component of
  unavailability `U` saves at most `H * downtime_cost_rate * U**k * (1 - U)`,
  and a design no worse than the greedy one spends no more on copies than
  its total. Components in series with the rest (and without hidden
  failures) are allocated by #40's dynamic program, for any number of them;
  other structures by branch and bound. The result is a typed
  `TotalCostAllocation`. Tests check a hand calculation (one pump against
  two, and the break-even horizon), the exact search against every design on
  random priced bridges (with and without a minimum availability), the
  dynamic program against the branch and bound, every scored design against
  the RBD with the copies drawn out (mixing preventive maintenance and
  inspection), and the chosen design's simulated running cost against its
  closed-form rate; mutation testing confirms each bound and rule is needed.
- **Hidden failures and periodic inspection in `RepairableRBD` (closes
  #70).** A component spec's new `"inspection"` key (`interval`, and
  optional `duration` and `cost`) makes its failures *hidden*: a failure
  takes the component down, but nobody knows until an inspection (a proof
  test) at a multiple of the interval finds it, and only then does its
  repair start. A test that takes time takes a working component off-line
  (a planned outage, during which it does not age); a failure found is
  repaired once the test is done, and its repair and replace costs are
  charged then; an inspection due during a repair is skipped. Hidden time
  counts as downtime in every simulated output, and `CostResult.by_category`
  gains `"inspection"`. The exact long-run methods cover hidden failures
  with a constant failure rate, instant tests and instant repair: a
  component is up `(1 - e^(-λτ)) / (λτ)` of the time, and because
  components inspected at the same times go down together, the system's
  availability, failure frequency, MUT/MDT, cost rate and importance
  measures are averaged over one period of the inspection schedules (the
  least common multiple of the intervals), by quadrature exact to rounding.
  So `mean_unavailability` gives a safety function's PFDavg: `≈ λτ/2` for
  one channel and `(1/τ)∫(1 - e^(-λt))² dt ≈ (λτ)²/3` for 1oo2 tested
  together, not the product of the channels' averages. Other cases raise
  `NotImplementedError` and are simulated. Tests check every event on
  deterministic lifetimes, the exact values against the issue's closed
  forms (one channel, 1oo2, 2oo3, mixed intervals, the cost trade-off
  `c_i/τ + c_d·U(τ)` and its optimum near `√(2c_i/(λc_d))`) and the
  simulation against them; an inspected system joins the fixtures that
  prove batched and one-at-a-time draws identical. Inspection schedules
  are saved with the RBD.
- **Redundancy allocation (`NonRepairableRBD.allocate_redundancy`, closes
  #40).** Solves the Redundancy Allocation Problem: given a per-copy cost for
  the nodes that may be duplicated, choose how many identical, independent,
  active copies of each to fit, either to **maximise system reliability
  within a budget** or to **minimise cost while meeting a reliability
  target**. `n` copies of a node with reliability `p` are scored as
  `1 - (1 - p) ** n` inside the exact system computation, so any RBD
  structure works, not only series-of-subsystems. `method="exact"` (the
  default) returns a proven optimum — for a budget it only scores designs
  that cannot afford another copy, since adding a copy never lowers a
  coherent system's reliability — and stops with guidance if a problem is too
  large to search; `method="greedy"` (best log-reliability gain per unit
  cost) is fast at any size but not guaranteed optimal. `max_units` caps
  copies per node, the "cost" can be any additive resource (money, weight,
  volume), and the result is a typed `RedundancyAllocation` (`units`,
  `reliability`, `cost`, `method`). Tests check both forms against an
  independent brute force on a (non-series-parallel) bridge network, and the
  `1 - (1 - p) ** n` model against an RBD with the copies drawn out
  explicitly. RBDs with CCF groups are not yet supported.
- **Redundancy allocation with several resources, and by dynamic programming
  (closes #76).** What one copy of each node uses may now be a dict of
  resources (e.g. `{"cost": 4000, "weight": 30}`) with `budget` a dict of
  limits on any of them — the multi-constraint problem of Fyffe, Hines & Lee
  (1968) — and a `target` may be combined with a `budget` (the cheapest design
  that meets the target within the limits). `minimise` names the resource a
  target minimises, and the result's new `resources` field totals every
  resource (`cost` is the total of the minimised one). When every costed node
  is in series with the rest of the system, `method="exact"` now solves the
  problem by a dominance-based dynamic program over the nodes (Kettelle-style,
  with any number of resources), so a series of many subsystems — fourteen
  with two limits, far beyond the exhaustive search — is solved exactly in a
  fraction of a second; other structures keep the exhaustive search. The
  greedy heuristic measures a copy by its total share of the limits when there
  are several. Tests check the dynamic program's final front against every
  design, and each form against an independent brute force with one, two and
  three resources, on series systems and on a bridge network.
- **Redundancy allocation with a choice of component types (closes #77).** A
  node in `allocate_redundancy`'s `costs` may be given a list of
  `ComponentOption(name, reliability, cost)` — candidate component types, each
  with its own reliability (a model evaluated at the mission time, or a fixed
  probability) and cost (a number or a dict of resources). Its copies are then
  any mixture of types (`mixing=True`, the default; Coit & Smith, 1996) or all
  of one type (`mixing=False`; Fyffe, Hines & Lee, 1968), `max_units` caps
  them all together, and the result's new `mix` field gives the number of each
  type per node. Both exact methods now work on each node's list of designs,
  less those another design of the node beats (no more of any resource and at
  least as reliable), and the dynamic program is vectorised; the greedy
  heuristic can also change the type of a copy. The exact search returns the
  best published reliabilities of the classic 14-subsystem benchmark with
  mixing (0.986811, 0.986416, 0.985922 and 0.954565 at weight limits 191, 190,
  189 and 159), in about two seconds each; tests also check both forms, with
  and without mixing, against an independent brute force on series systems and
  on a bridge network.
- **Redundancy allocation with k-out-of-n nodes and standby spares (closes
  #78).** `allocate_redundancy` gains `required` (how many of a node's copies
  must work; Coit & Liu, 2000), `strategy` (`"active"`, the default; `"cold"`
  standby, where spares wait unpowered and are switched in as units fail —
  Coit, 2001; or `"choose"`, letting the optimiser pick for each node — Coit,
  2003) and `switching_probability` (for cold standby), each for every node or
  per node. Active k-out-of-n nodes are exact (binomial tails, or their
  mixed-type generalisation); cold-standby nodes are `StandbyModel`s of their
  copies (exact for identical Exponential units, numerical convolution for one
  unit required, seeded simulation otherwise). The result's new `strategy`
  field gives each node's strategy, and the documented "active copies only"
  limit is gone. Tests check the node reliabilities against binomial tails and
  Erlang and Poisson sums, and the allocations — including a choice that keeps
  some nodes active and gives others cold spares under imperfect switching —
  against an independent brute force.
- **The cost-reliability trade-off (`NonRepairableRBD.redundancy_front`,
  closes #80).** Returns every design that no other beats by using no more of
  every resource while being at least as reliable (the Pareto front of
  multi-objective redundancy allocation; Taboada et al., 2007), as a list of
  `RedundancyAllocation` by increasing cost, taking the same arguments as
  `allocate_redundancy` except `target`. Exact: from the dynamic program when
  the costed nodes are in series, by evaluating every design within the budget
  otherwise (`redundancy_allocation.exact_front`). Tests check it against the
  non-dominated set of every allocation, with one and two resources and with
  types and strategies, on series systems and bridge networks, and every point
  against `allocate_redundancy` in both forms.
- **Reliability-redundancy allocation
  (`NonRepairableRBD.allocate_reliability_redundancy`, closes #79).** Chooses
  each node's component reliability, within `bounds`, and its number of
  copies together, to maximise system reliability when what the copies use is
  a function of both (`uses[node](r, n)`, a number or a dict of resources,
  within a `budget`; Tillman, Hwang & Kuo, 1977). Any structure works. It is
  solved exactly over the copies by branch and bound — each copy vector that
  fits at the lowest reliabilities is bounded by the system reliability with
  every node at the most reliable component it could afford alone, and solved
  in decreasing order of that bound until none beats the best found — with the
  reliabilities for each vector found by SLSQP with the exact gradient (a local
  optimum; global for a series system with convex costs). The result is a new
  `ReliabilityRedundancyAllocation`. It returns the best published solutions
  of the four classic benchmarks — series (0.931682), series–parallel
  (0.99997665), bridge (0.99988964) and overspeed protection (0.99995467) —
  which the tests check, along with plain redundancy allocation when the
  reliabilities are fixed and a closed form for one node.
- **Cost of a repairable system (`expected_cost_rate`).** `RepairableRBD`
  components may now carry `repair_cost` and `replace_cost` (charged per
  corrective action) and an optional `downtime_cost` rate, alongside a
  system-level `downtime_cost_rate` for production lost while the system is
  down. `expected_cost_rate()` returns the long-run cost per unit time in
  closed form — no simulation — as
  `downtime_cost_rate·(1 − A_sys) + Σ ωᵢ·(repair + replace) + Σ (1 − Aᵢ)·downtime`,
  reusing the existing availability and failure-frequency machinery, and it
  accepts the usual `working_nodes`/`broken_nodes` conditioning. Every cost is
  optional and defaults to 0, so any subset can be priced; when nothing is
  priced `has_costs` is `False` and the method short-circuits without doing the
  work. Costs are corrective-only and undiscounted, and they persist through
  serialisation. Unknown keys in a component spec are now rejected at
  construction, so a mistyped cost key can no longer be silently priced at zero.
- **Simulated cost distribution (`RepairableRBD.cost`, closes #54).** The
  availability simulation now accumulates costs (only when some cost is
  declared): each replication yields the window's total cost, and
  `cost(t_simulation, N, seed)` — or `availability(...).cost` from the same
  replications — returns a `CostResult` with the `samples`, `mean`, `std`,
  `percentile(q)` (a P90 planning budget, which the exact mean cannot give),
  `mean_se` and `mean_interval(confidence)` (a confidence interval for the
  expected cost, to judge whether `N` was enough), a per-category breakdown
  (repair / replace / component-downtime / system-downtime) and the mean
  attributable cost per component. `result.cost_rate` converges to the exact
  `expected_cost_rate()`, and the test suite asserts that identity. With
  nothing priced, `cost()` returns `None` and no cost work is done.
- **Costs drawn from distributions.** `repair_cost` and `replace_cost` may be
  a distribution of the cost (e.g. a surpyval model fitted to past invoices)
  instead of a number: the simulation draws a fresh cost at every failure,
  and the closed form uses the mean. The draws come from their own random
  stream, so seeded runs stay reproducible and pricing never changes the
  failure/repair simulation or any availability output. A cost distribution
  must have a finite mean and no appreciable probability of a negative cost,
  and it persists through serialisation; the downtime costs stay numbers.
- **Instantly repaired components.** A `RepairableRBD` component may declare
  `"repairability": "instant"` — repaired in zero time. It still *fails*
  (failure events fire and repair/replace costs are charged) but every outage
  has zero length, so it contributes no downtime and its availability is
  exactly 1: the modelling shorthand for parts swapped much faster than the
  timescale under study, or with no repair-time data.
- **Warm and hot standby (`StandbyModel(dormancy_factor=...)`, closes #41).**
  `dormancy_factor` is the dormant-to-operating aging ratio: `0` is the
  existing cold standby (default, unchanged), values in between are **warm**
  (a dormant spare ages at that fraction of the operating rate — the
  cumulative-exposure / virtual-age model shared with `LoadSharingModel` —
  and can fail *latent*, dead before it is needed), and `1` is **hot**,
  which is exactly k-out-of-n parallel. Identical Exponential units get an
  exact hypoexponential closed form for any `dormancy_factor` (Erlang and
  the parallel order-statistic as the cold/hot endpoints); other lifetimes
  are simulated. Spares are promoted in list order; the factor persists
  through serialisation. Imperfect switching remains cold-`k=1`-only.

- **Named reliability-allocation methods.**
  `minimum_effort_allocation(target, node_probabilities)` is Albert's (1958)
  minimization-of-effort algorithm (MIL-HDBK-338B). For a series system it
  raises the least reliable nodes to one common level, the least total
  effort for any effort function meeting Albert's conditions, in closed form.
  `cost_based_allocation(target, node_probabilities, max_probabilities=None,
  feasibility=None)` is Mettas's (2000) cost-based allocation, for any
  structure: the cheapest node probabilities meeting the target, with each
  node's cost `exp((1 − f)(R − R_min)/(R_max − R))` rising from its current
  probability towards its maximum, faster for a lower feasibility `f`. It
  works on the log-odds scale from both ends of the exact engine, with exact
  gradients, so it stays exact at any size (100 nodes in series or 30 in
  parallel, say). Tests hold the first to
  a direct minimisation of two different effort functions, and the second to
  a direct minimisation of Mettas's problem and to the optimality conditions.
  The design guide now maps each allocation helper to its named method:
  `equal_allocation` is equal apportionment, `improvement_allocation`
  ARINC-style proportional apportionment, and `simple_allocation`, which is
  not a named method, the smallest change in the node log-odds (see below).
- **Failure-oriented criticality importance (closes #72).**
  `criticality_importance` on `NonRepairableRBD` and `RepairableRBD`, and
  `NonRepairableRBD.importances_given_state`, take `kind="failure"` (the
  default) or `kind="success"`. The failure-oriented form (Rausand & Høyland),
  `I_B(i) · (1 − p_i) / (1 − P_sys)`, is the probability that node *i* has
  failed and is critical given that the system has failed: its share of the
  system failures, or of the downtime at long-run availabilities. It is
  computed from the node unreliabilities through the minimal cut sets, so it
  keeps its precision for a highly reliable system where `1 − P_sys` would
  cancel. Either form is `nan` (without a warning) where it is undefined: a
  system that cannot fail, or cannot work. Tests check both forms against a
  brute-force enumeration of a bridge network (non-repairable and
  repairable), and the failure form against exact rational arithmetic on a
  bridge that fails with probability ~1e-12.

- **Scheduled preventive maintenance in `RepairableRBD` (closes #69).** A
  component's dict takes `"preventive": {"interval": T, "policy": "age" |
  "block", "duration": model | "instant", "cost": c_p}`. Under age
  replacement the unit is replaced `T` after it was last put into service as
  new (a failure restarts the clock); under block replacement at `T, 2T, …`
  whatever its age (skipped while it is down). A replacement renews the unit,
  so the failure it was heading for never happens; with a duration the unit
  is down meanwhile, a planned outage, which counts as downtime in every
  availability output but not as a failure (the result's new
  `system_planned_outages` counts them); `"instant"` renews it in place.
  `CostResult.by_category` gains `"preventive"`. The exact long-run methods
  (`expected_cost_rate`, `mean_availability`, `node_availability`, the
  frequencies, MUT/MDT and the importance measures) price an age-replaced
  component through its renewal-reward cycle; block replacement has no exact
  long-run values, so they raise `NotImplementedError` for it and the
  simulation prices it. Nested RBDs report their planned outages to their
  parent. Tests list the events of deterministic lifetimes by hand, hold
  the exact cost rate of age replacement to `NonRepairable.cost_rate(T)`
  and a long simulation to every exact value, check block replacement of an
  exponential unit against `λ·c_u + c_p/T`, that an infinite interval gives
  results identical to no maintenance, and that the batched and one-draw
  simulations still agree. `NonRepairable.avg_replacement_time` now also
  integrates models whose `sf` returns an array for a scalar age (such as
  `ExactEventTime`).

### Changed
- **Series–parallel (modular) reduction before exact evaluation, so large
  redundant diagrams stay fast (closes #83).** The exact engine used to work
  from every minimal path set, which multiply with redundancy: `n`
  duplicated stages in series have `2 ** n`. On construction the diagram is
  now reduced to modules, each with a closed form, until nothing more
  reduces: series chains, parallel groups (nodes sharing their predecessors,
  successors and `k`), k-out-of-n groups (all the inputs of a voting node,
  when they share their own inputs) and nodes that a direct edge bypasses
  (irrelevant, so left out). Only what is left, the part that is not
  series-parallel (a bridge, a cross-tie), is evaluated from its minimal
  path sets by the Shannon decomposition, over modules and nodes, and a
  series-parallel diagram reduces to a single module and never needs its
  path sets. Fourteen duplicated Weibull stages in series (16,384 path sets)
  took 19 s to build, 11 s for the first `sf` over 400 times and 2 s for
  the six importance measures at one time; each now takes a few
  milliseconds, the six measures over all 400 times about 0.03 s, and
  thirty stages (over 10^9 path sets) evaluate exactly just as fast.
  Everything built on the engine benefits: `sf`/`ff`, the repairable
  closed forms, every importance measure (still per original component),
  redundancy and reliability allocation, `random`/`mean` (a module's
  lifetime is the min, max or k-th longest of its members'), and
  `get_min_cut_sets`, now built from the modules. The structure function
  that simulations evaluate at every event is compiled to straight-line
  Python, about twice as fast on small diagrams too (a pumps-and-valve
  availability simulation runs 15% faster). Every probability is computed
  with its complement as sums of products, so both keep their full relative
  precision; `method="c"` now computes the unreliability by the same
  decomposition rather than over the minimal cut sets. `get_min_path_sets`
  expands the modules on first use instead of searching on construction, so
  a long chain in series no longer reaches Python's recursion limit.
  Diagrams with nothing to reduce (a bridge) are evaluated as before, and
  structures that are not valid RBDs are not reduced. Results match the
  unreduced engine to rounding: tests compare them with enumeration of every
  state on 600 random diagrams (bridges, k-out-of-n nodes, bypasses, direct
  edges), with the unreduced engine on nested series, parallel,
  k-out-of-n and bridge compositions and through every public method, and
  mutation testing confirms each reduction rule's conditions are needed.
- **`criticality_importance` now defaults to the failure-oriented form
  (#72).** The success-oriented form it returned, `I_B(i) · p_i / P_sys`, is
  exactly 1 for every node in series with the rest of the system, however
  unreliable, so it could not rank the nodes in series. Pass
  `kind="success"` for the old values; the `"criticality"` of
  `importances_given_state` follows the same default.
- **`simple_allocation` now finds the smallest change in the node log-odds.**
  It used to minimise the squared shortfall from the target with BFGS,
  starting every node at 0.5, and return wherever the optimiser stopped. On
  asymmetric systems other optimisers met the target with quite different
  allocations; a target out of reach (a node with weight 0 stays at 0.5) came
  back as the closest miss, without an error; and on large systems the search
  could not move at all: 30 nodes in parallel at 0.5 are within 1e-9 of
  certain success, so a target of 0.5 returned every node at 0.5. It now
  minimises the weighted change in log-odds, `sum(s_i ** 2 / w_i)`, subject
  to meeting the target: a well-defined answer, in which nodes that matter
  more and more heavily weighted nodes move further, exact at any size (it
  is solved with `trust-constr` on the log-odds scale from both ends of the
  exact engine, and tests hold it to its optimality conditions and to a
  direct minimisation). A node's weight now scales its change directly: at
  equal sensitivity, twice the weight moves it twice as far. Symmetric
  allocations are unchanged; others differ (the docstring's example moves
  from a, b = 0.911 and c = 0.998 to 0.940 and 0.994). An unreachable target
  raises `ValueError` with the reachable range, a negative weight raises
  `ValueError`, and `res.x` now holds the nodes' log-odds.
- Require **surpyval >= 0.20**, and the requirement is now **uncapped** (was
  `>=0.16,<0.17`). 0.20 adds a first-class `Hypoexponential` distribution, so
  RePyability's private `_HypoexponentialSurvival` (the closed-form group
  lifetime behind identical-Exponential `LoadSharingModel` and warm/hot
  `StandbyModel`) is deleted in favour of it — the same maths, now with
  `random`, `qf`, `var` and serialisation through `surpyval.from_dict` for
  free, per the rule that univariate distributions live in surpyval. Where
  the stage rates are not distinct (surpyval rejects them; the old private
  class silently produced nonsense there) both nodes now fall back to
  simulation. Verified against surpyval 0.19.0 and 0.20.0: the whole test
  suite, the type checks and the strict docs build pass unchanged — no
  RePyability API depended on anything that moved, and the `sf_tvc` /
  `StepSchedule` time-varying-load path is unaffected. RePyability consumes
  a small, stable surface of surpyval, and the one-minor-wide caps used
  until now meant every surpyval minor release made `pip` refuse to
  co-install the two packages until a RePyability release followed; new
  surpyval minors are now picked up without one.
- Python **3.13** is now supported and tested in CI (classifiers and test
  matrix; surpyval declares 3.11–3.13, so 3.14 waits on upstream).
- Maintenance: CI actions moved off the deprecated Node 20 runtime
  (`actions/checkout@v5`, `actions/setup-python@v6`); the docs stack is held
  on MkDocs 1.x / mkdocs-material 9.x (MkDocs 2.0 removes the plugin system
  with no migration path); black's `target-version` is pinned to the minimum
  supported Python so formatting no longer depends on the interpreter it
  runs under.
- **Performance, with identical results.** The simulations used to draw one
  surpyval sample per call (tens of microseconds of overhead each) inside
  Python loops. For plain parametric models a surpyval draw is `qf(u)` of one
  uniform from numpy's global RNG, so the samplers now take the *same*
  uniforms in one block, in the order the old loops consumed them, and apply
  `qf` to the block. Per-sample loops that do deterministic work run for all
  samples at once, the exact engine records its Shannon decomposition once per
  RBD and replays it, and the repairable simulation's event queue drops
  `queue.PriorityQueue`'s thread locking (it is the same heap). Composite
  nodes -- standby, repeated, repeated-standby, load-sharing, regression and
  nested-RBD nodes -- describe how their `random(1)` consumes the RNG, so an
  RBD containing them is batched the same way, and a `RepairableRBD` nested
  in another draws its components' failure and repair times from the outer
  simulation's block of uniforms too. Minimal cut sets are read off
  the exact engine's decomposition (a node's cut sets either spare its pivot
  component or contain it with a cut set of the failed branch) instead of
  Berge's algorithm, whose intermediate families could grow far beyond the
  answer, and they are computed once per RBD; the decomposition also runs on
  an explicit stack, so `sf()` and cut sets now work on wide systems of more
  than about a thousand components (1 100 units in parallel, say), where they
  used to hit Python's recursion limit. (Finding the path sets is still
  recursive, so a single chain of about a thousand nodes in series still
  exceeds the limit when the RBD is built.) Seeded results are unchanged,
  and unseeded runs leave the global RNG in the same state. Models whose
  sampling cannot be reproduced exactly this way (fixed-probability,
  limited-failure-population or zero-inflated models) keep their original
  code path. Measured at default settings:

  | Workload | Before | After |
  |---|---|---|
  | `NonRepairableRBD.mean()`, 5-node bridge (100k samples) | 26.4 s | 24 ms |
  | `NonRepairableRBD.mean()`, 12-node system | 66.2 s | 125 ms |
  | `mean()`, RBD with a standby / repeated / nested-RBD node | 18–32 s | 16–38 ms |
  | `RepairableRBD.availability(t=1000, N=300)`, 4 components | 1.81 s | 0.16 s |
  | `RepairableRBD.availability(t=1000, N=200)`, with nested `RepairableRBD`s | 0.92–2.4 s | 97–175 ms |
  | `StandbyModel` cold, k=2 of 4 Weibull units (build) | 1.81 s | 6 ms |
  | `StandbyModel` warm, 4 Weibull units (build) | 0.52 s | 13 ms |
  | `LoadSharingModel`, 3 Weibull-AFT units (build) | 0.29 s | 7 ms |
  | Six importance measures, 12-node system | 126 ms | 20 ms |
  | `get_min_cut_sets()`, 6 stages of 3 in parallel (729 path sets) | 6.8 s | 52 ms |
  | `fussell_vesely()`, same system | 6.9 s | 140 ms |

  The one trade-off is flat systems, whose cut sets Berge found almost
  instantly: a first `get_min_cut_sets()` on 500 components in plain series
  or parallel now takes tens of milliseconds instead of a few (later calls
  are cached).

  `test_performance_equivalence.py` holds each fast path to the code it
  replaced (kept as the fallback, or for the exact engine and the cut sets a
  verbatim copy of the original algorithm): same samples, same simulation
  outputs, same final RNG state, exactly the same exact-engine probabilities,
  and the same minimal cut sets.

### Deprecated
- The `options` argument of `NonRepairable.find_optimal_replacement()` was
  never used. Passing it now issues a `DeprecationWarning`; it will be
  removed in a future release.

### Fixed
- **Limited-failure-population and zero-inflated node models.** A surpyval
  model with `p < 1` (a fraction `1 - p` of units never fail) or `f0 > 0` (a
  fraction dead on arrival) was mishandled in several places:
  - every simulation crashed on a limited-failure-population model
    (`NonRepairableRBD.random`/`mean`, `RepairableRBD.availability`/`cost`,
    and the standby, repeated and load-sharing nodes' own draws), because
    surpyval's `random` returns survival data for it, not lifetimes. Its
    lifetimes (and a zero-inflated model's) are now drawn by its quantile
    function, one uniform each: infinite for a unit that never fails, 0 for
    one dead on arrival. They are drawn in blocks like any parametric
    model's, so they also support antithetic pairs and `compare`;
  - `NonRepairable.mean_availability` and `failure_frequency`, and through
    them `RepairableRBD`'s long-run values, used surpyval's *defective*
    mean (0.9 of the failing units' mean, for `p = 0.9`) as the MTTF: a
    unit that may never fail got an availability of 0.988 instead of 1 (it
    ends up with a replacement that never fails) and a failure frequency of
    0.012 instead of 0. With replacements that may never finish too, the
    availability is now the probability of ending up for good;
  - `node_mttf` reported the defective mean (79.8 for a Weibull(100, 2)
    with `p = 0.9`) instead of an infinite MTTF;
  - a cold-standby arrangement of such units (`StandbyModel`,
    `RepeatedStandbyNode`) had `sf = 1.0` everywhere: the numerical
    convolution's grid search never ended, as the survival never falls
    below `1 - p`. The convolution now carries each unit's mass at 0 and at
    infinity. Identical exponential units with an offset, `p` or `f0` no
    longer take the plain exponentials' closed form either (whose rate,
    `1 / mean`, was wrong for them);
  - warm standby of such units produced NaN lifetimes (`inf - inf`);
  - saving a diagram dropped the offset, `p` and `f0` (the model reloaded as
    a plain one), and parameter sensitivity rebuilt the model without them.

  Seeded simulations of zero-inflated models draw different (equally
  valid) numbers than before. Tests (37) hold the draws to the fractions
  that never fail or are dead on arrival, cold-standby sums of such
  exponentials to their closed forms (with imperfect switching too), the
  never-failing probabilities of warm, k-out-of-n and repeated nodes, a
  repaired unit's geometric number of failures and its absorption
  probabilities, and saving and sensitivity to the models themselves.
- **A repeated node could change a diagram's logic.** `NonRepairableRBD`
  joined a repeated node (one component drawn in several places) into the
  node it repeats, redirecting its edges. That can add paths the diagram does
  not have: with `X` drawn before `A` and again after `Y` on the way to `B`,
  the joined node gave the path `{X, B}`, leaving `Y` out, and a reliability
  of 0.891 instead of 0.8829, silently. Joining could also create a loop and
  reject a valid diagram, or drop a repeated voting node's own `k`. A repeated
  node now stays where it is drawn, and every calculation treats its
  appearances as the one component: the exact engine keeps them out of the
  closed-form modules and solves them together in the core, and the
  structure function, path and cut sets, importance measures and simulated
  lifetimes follow. Diagrams whose repeats joining left unchanged give the
  same results; seeded simulations of diagrams with repeated nodes may draw
  in a different order. Tests check 200 random diagrams with repeated nodes
  against the enumerated structure function (probability, path and cut
  sets, every state, Birnbaum importance, irrelevant components) and the
  simulated lifetimes against the exact reliability; mutation testing
  confirms each part of the fix is needed.
- **A deserialised `ExactEventTime` can be saved again.** Its parameter
  came back as `[[T]]`, so saving an RBD read from a file failed on it.
- **`find_optimal_replacement` returned a spurious finite age (closes #68).**
  For a lifetime without wear-out that the quick check does not recognise — a
  Weibull of shape 1 or less with an offset, zero-inflation or a limited
  failure population, or a Gamma with shape below 1 — the search ran along a
  cost rate that only falls towards the run-to-failure rate, and returned
  wherever it stopped (745,244 hours for a unit with a mean life of 2,001).
  The best age found must now beat running to failure, or `inf` is returned.
  Running to failure costs nothing in the long run when some units never fail
  (a limited failure population), and `optimal_replacement_policy` now reports
  that rate as 0 rather than `cu` over the failing units' mean life. With an
  offset, replacing at the offset itself — the end of the failure-free period,
  where the cost rate `cp / t` is lowest — is also considered; the search used
  to stop short of it (986 against 1000 in the test). surpyval's spurious
  `RuntimeWarning` when evaluating an offset model below its offset is
  silenced in the cycle-length integral.
- **Nested `RepairableRBD` simulations put the nested RBD's state changes at
  the wrong times.** A nested RBD's `next_event()` returns the time *of* its
  next state change, but the outer simulation added it to the current time as
  if it were the time *to* it (as a component's is). Every nested failure and
  restoration after the first therefore landed late, by more the longer the
  run, so `availability()` and `cost()` of any RBD containing a nested
  `RepairableRBD` were wrong: a single unit with MTTF 10 and MTTR 1, nested,
  simulated at 0.51 availability instead of 0.91. The outer simulation now
  takes a nested RBD's times as they are, and a nested RBD simulates exactly
  as it does on its own. The exact methods (`mean_availability()` and the
  failure-frequency family) were not affected; seeded simulations of RBDs
  without nested `RepairableRBD`s are unchanged.
- **A `NonRepairable` object given for several nodes shared one simulation
  state.** The object records whether it fails or is repaired next, so when
  the same object was given for two nodes of a `RepairableRBD` (or for a node
  and a node of a nested RBD, or nodes of two nested RBDs), one node's failure
  could be followed by another failure instead of a repair, and the simulated
  availability and costs were wrong. Each node now gets its own copy of the
  object, so one object can stand for several identical parts: it simulates
  exactly as separate objects would. (`rbd.components[node]` is that copy.
  Models given as `{"reliability": ..., "repairability": ...}` dicts always
  made a new object per node and are unchanged.)
- **`random()` hung on a diagram with an edge straight from the input to the
  output node.** Such a system never fails. The batched sampler returned
  `inf` for it, but the one-at-a-time sampler (used when a node's model
  cannot be sampled in blocks, e.g. a zero-inflated one) kept waiting for a
  system failure after every node had failed, so `random()`, `mean()` and the
  MTTF methods never returned. It now returns `inf` too.
- **`NonRepairable` failed with a `StandbyModel` lifetime.** A closed-form
  standby arrangement (identical Exponential units, or cold standby with
  `k = 1`) made the constructor raise `AttributeError`, and for any
  `StandbyModel` the cost methods (`avg_replacement_time`, `cost_rate`,
  `find_optimal_replacement`, `optimal_replacement_policy`) raised
  `AttributeError` too. Every form (closed form, convolution or simulation)
  now works through the arrangement's survival function. The cycle length is
  integrated by the trapezoidal rule on a fine grid (a simulated
  arrangement's survival is a step function), and the optimal age is found
  on a grid and refined; an arrangement that may never fail is never
  replaced (`inf`).
- **An `input_node` or `output_node` that is not the diagram's source or
  sink was accepted.** Naming, say, a node in the middle of a series chain
  silently analysed a different system (the nodes before it dropped out).
  Both RBD classes now raise `ValueError` unless the named node has no
  incoming (input) or no outgoing (output) edges.
- **`improvement_allocation` could return invalid probabilities or miss its
  target silently.** A target below the current system probability pushed
  node probabilities below 0, and a target the free nodes could not reach
  (because `fixed` nodes cap the system) returned the closest miss with no
  error. The failure probabilities are now capped at 1, so a lower target
  gives the lowest valid probabilities that meet it; the common factor is
  found by a bracketed root search, which meets every reachable target
  (including a target of 1, which `equal_allocation` used to meet only
  approximately); an unreachable target raises `ValueError` giving the
  reachable range; and each node probability must be a single value in
  [0, 1]. `rbd.res` still holds the common exponent in `x`.
- **Tuple node names did not survive JSON.** JSON writes a tuple as a list,
  and loading used the list as the node name, which is not hashable, so
  `from_json(to_json())` failed for an RBD with tuple node names. Loading
  now turns lists in node names (edges, models, `k`, repeated nodes,
  common-cause groups, input and output nodes) back into tuples.
- **The simulated overhaul search returned its horizon as a false optimum
  for near-minimal repair.** For a `Repairable` with imperfect repair and `q`
  close to 1, the unit's virtual age reaches ages where the baseline
  survival underflows; surpyval then ends those simulated histories early,
  the estimated `E[N(t)]` stops growing, and the cost rate appeared to keep
  falling to the end of the search. A Weibull(100, 2) with minimal repair
  and costs 10 and 50 returned 1329 (the default horizon) instead of the
  closed-form 224. When the simulation is cut short, the search now halves
  its horizon until it is not, down to no less than the age at which the
  baseline survival is 1e-10, and that example finds 219 (seed 0, the
  default 1000 simulations). The warnings of
  the discarded simulations are dropped, and an optimum on the horizon now
  warns that the cost rate is still falling there (and whether raising
  `max_interval` could help), where the horizon used to be returned in
  silence.
- **`NonRepairable` got the cycle length and survival of a non-parametric
  lifetime wrong.** `avg_replacement_time` integrated from the estimate's
  first time point instead of age 0 and stopped at the last time point below
  `t`, and the survival function was linearly extrapolated beyond the data,
  where it went negative. For a Kaplan-Meier fit to failures at 10, 20, 30
  and 40, the cycle length to age 25 came out as 5 instead of 17.19, and the
  search returned 21.6 with a policy cost rate of 0.63, where the optimum is
  20 at 0.2. The survival function is now linear between the estimate's time
  points, from 1 at age 0, and held at its last value beyond them (as
  surpyval's own step function is); the cycle length is its exact integral,
  and the search uses the same exact integral from age 0, so the policy's
  cost rate agrees with it.
- **Missing costs raised different errors in `NonRepairable`.** `cost_rate`
  and `find_optimal_replacement()` raised `AttributeError` and
  `optimal_replacement_policy()` `ValueError`; all three now raise
  `ValueError` ("costs not set"). `find_optimal_replacement()` still returns
  `inf` without costs when preventive replacement never pays.
- **Negative and non-finite costs were accepted.**
  `NonRepairable.set_costs_planned_and_unplanned` accepted a negative planned
  cost and NaN or infinite costs, and `Repairable.set_repair_and_overhaul_costs`
  NaN or infinite ones; both now raise `ValueError`. (`RepairableRBD`
  already rejected them.)
- **`system_probability` took any `method` other than `"p"` as cut sets.** It
  now raises `ValueError` unless `method` is `"p"` or `"c"`, as
  `is_system_working` does, and so do the methods that pass their `method`
  on to it (`sf`, `mean_availability`, ...).
- **An invalid `on_infeasible_rbd` passed unnoticed on a valid diagram.**
  `RBD` and `RepairableRBD` checked the value only when the structure was
  invalid; every RBD class now checks it on construction. The structure
  warning's "Strucutral" typo is corrected.
- **`find_optimal_replacement()` ran its search twice.** A helper meant to
  minimise the log of the cost rate duplicated the plain one, so for a
  parametric lifetime the same search ran twice and the second result could
  never be chosen. It now runs once, with bit-identical results (checked on
  200 lifetime and cost combinations). `PerfectUnreliability.random`'s first
  parameter is renamed `cls`, as it is a class method.
- `test_non_parametric_optimal_replacement` drew its data from the unseeded
  global RNG and failed about one run in 150; it is now seeded.
- `test_weibull_no_optimal_replacement` no longer asserts that a warning is
  emitted for an offset (3-parameter) Weibull. The warning was an incidental
  numerical `RuntimeWarning` raised inside surpyval while evaluating the
  model's mean, not a contract of `find_optimal_replacement()`; surpyval 0.19
  evaluates that mean cleanly, so the assertion no longer held. The behaviour
  under test — a finite (non-`inf`) replacement age for an offset Weibull — is
  unchanged and still asserted.

### Documentation
- **Hidden failures and proof tests.** A new Costs section covers
  `"inspection"`: a shutdown valve's PFDavg alone and in 1oo2, the test
  interval, simultaneous testing and choosing an interval by cost. Lesson 8
  of the Learn course derives `λτ/2`, the 1oo2 result and the best test
  interval by hand, with a new exercise, and the glossary defines hidden
  failure, proof test and PFDavg.
- **How the exact engine reduces a diagram.** Lesson 3 of the Learn course
  now explains the two stages of the engine (reduce the series, parallel and
  k-out-of-n parts, then pivot on what is left) with a thirty-stage plant
  that has over a billion path sets yet is evaluated instantly, and Lesson 2
  no longer says the library does not reduce diagrams. The concepts page,
  the performance notes (the recursion limit now applies only to the part of
  a diagram that does not reduce), the building and guide index pages and
  the glossary describe the reduction.
- **Learn: a short course in system reliability engineering.** A new
  section of nine lessons teaches the ideas behind the library from first
  principles: lifetimes (reliability, hazard, MTTF, the exponential and the
  Weibull, the bathtub curve); series, parallel and k-out-of-n systems; path
  and cut sets and how the exact engine works; importance measures, built
  from the idea of a critical component, including both forms of criticality
  importance; standby, load sharing and common-cause failures; repair and
  availability; costs; preventive maintenance (age and block replacement,
  when replacing early cannot pay, and why the best interval changes inside
  a system); and reliability and redundancy allocation. Each lesson poses a
  concrete question, works it out by hand with small numbers, repeats it with
  RePyability, lists the usual pitfalls, and ends with a summary and
  exercises with worked answers. Every code block runs in the documentation
  tests and every number quoted in them is checked. The site now renders
  formulas (MathJax) and diagrams and charts (Mermaid), and has collapsible
  answers and tabs; each user-guide page links to the lessons behind it, and
  a glossary defines every term with the lesson that teaches it.
- **Reliability allocation has a full guide section.** The design guide now
  covers every allocation method: a table of what each needs and how it
  picks among the allocations that meet a target, how to choose between
  them, what they share (one mission time, independent nodes, a node as a
  block, requirements rather than designs), and each method's uses and
  limits, with the methods compared side by side on a series system and on a
  redundant one. It explains why `simple_allocation` is a structural
  allocation: it uses no component data, and with every node at 0.5 the
  Birnbaum importance is the structural importance, so small changes follow
  each node's weight times its structural importance.
- **Long chains in series are a documented known limit.** Finding the path
  sets is recursive, so building an RBD with a chain of about a thousand or
  more nodes in series exceeds Python's default recursion limit; the
  performance section now says so, with the workaround
  (`sys.setrecursionlimit(10_000)` handles chains of several thousand nodes).
- **The documentation is rewritten for complete coverage.**
  - **The user guide** is now eleven pages, one per task: building an RBD,
    reliability, importance measures, condition-based evaluation, redundancy
    models, common-cause failures, repairable systems, costs, design and
    allocation, maintenance policies, and saving/reproducibility/performance.
    Every public capability is covered, with the arguments it takes, what it
    returns and its limits, including ones the old guide never mentioned:
    - repeated components (one component drawn in several places);
    - the structure check, irrelevant nodes and all path sets;
    - density, hazard and per-node values;
    - the simulated criticality measures;
    - nested repairable RBDs, and stepping a simulation by hand;
    - the reliability-allocation helpers.
  - **Concepts** gains the theory of lifetimes by simulation, standby and
    repeated nodes, availability (renewal, the frequency formula, MUT/MDT,
    the simulation and its criticality measures), costs, allocation, and the
    maintenance models.
  - **The tutorial** now runs end to end, including the fits, and adds steps
    on pricing redundancy and on the cost of the repairable skid. The home
    page maps every capability to its guide page.
  - **The API reference** now includes `FailureLimitPolicy` and
    `minimal_repair_time_to_nth_failure`.
- **Every public class, method and property has a complete numpy-style
  docstring**, with parameters, returns, errors and examples, run as
  doctests.
- **The docs are now tested.** `test_docs_examples.py` runs every code block
  on every page and checks each number quoted in a `# -> value` comment
  against the code. `test_api_docstrings.py` requires every public member to
  have a docstring that documents each of its parameters. The strict docs
  build now also fails on broken links and anchors.
- **Stale statements are corrected.** The tutorial's field-data snippets did
  not run, and several guide examples used undefined names. The
  performance entry above overstated which systems no longer hit the
  recursion limit.
- The common-cause (CCF) docs now state the models' assumption: they are the
  PRA basic-event models, which split each member's failure *probability*, and
  hold while that probability is small (a mission or proof-test interval, not
  a whole life). They also say that `random()`, `mean()` and the MTTF interval
  sample members independently and do not include CCF, since an MTTF spans the
  whole life. No behaviour changed.

## [0.8.0] - 2026-07-22

The **Dependent Failures** milestone: model redundant components that fail
*together* or *drive each other's aging*, rather than independently — load-
sharing groups whose survivors carry more and age faster as siblings fail
(`LoadSharingModel`), common-cause failures through the beta-factor and Multiple
Greek Letter models (`CCFGroup`), and covariate loads that vary over a
component's life (`RegressionNode` schedules, built on surpyval 0.16's
`sf_tvc`).

### Added
- **Time-varying load for `RegressionNode` (load-dependent aging, #37).**
  `RegressionNode` now accepts a `schedule=` (a surpyval `StepSchedule`) as an
  alternative to a fixed covariate vector: the load the component runs under
  varies over its life, and reliability is the exact survival along that
  covariate path (`model.sf_tvc`). Conditioning on `age` gives the go-forward
  reliability from the component's current life under the schedule — the
  digital-twin / load-dependent-aging node — with no change to the
  condition-based layer, since `sf_tvc(age+x)/sf_tvc(age)` is exactly surpyval's
  `sf_tvc(..., given=age)`. Works for accelerated-failure-time and
  proportional-/additive-hazards families (not proportional-odds). The schedule
  persists through serialisation. (Uses surpyval's `sf_tvc`, which is why the
  minimum surpyval is now 0.16 — see below.)
- **`LoadSharingModel`: load-sharing dynamic node (dependent failure).** A
  sibling to `StandbyModel` where *n* coupled units share a total load and the
  survivors carry more (and so age faster) as siblings fail — the group works
  while ≥ `k` survive. Each unit is a fitted AFT model, and a cumulative-exposure
  event loop advances every unit's baseline clock at rate `phi(L / survivors)`.
  Identical Exponential-baseline units get an exact hypoexponential closed form;
  otherwise the survival function is a Kaplan-Meier fit to simulated lifetimes
  (`is_simulated` reports which). With no load effect (`phi == 1`) the group
  reduces exactly to the ordinary k-out-of-n parallel result. Plugs into an RBD
  as a single simulation-backed node and serialises with it. (#38)
- **Common-cause failures (CCF) — beta-factor and Multiple Greek Letter.**
  `NonRepairableRBD` gains a `ccf_groups` argument taking
  `CCFGroup(members, model)`, where the model couples a symmetric redundant
  group through a shared failure cause. `BetaFactor(beta)` is the all-or-nothing
  model (a fraction `beta` of failures fail the whole group at once, the rest
  are independent); `MGL(beta, gamma, ...)` is the Multiple Greek Letter model,
  which also captures *partial* common causes (a cause failing some but not all
  of the group) — `MGL(beta)` on two members is exactly `BetaFactor(beta)`.
  System reliability is computed **exactly** by conditioning on each group's
  mutually-exclusive shock outcomes and reusing the ordinary independent engine
  per branch, so `beta = 0` reproduces the independent result. Honoured by
  `sf()`/`ff()` (and quantities derived from them) and persisted through
  serialisation; the probability-dependent importance/sensitivity and
  condition-based methods raise a clear error on a CCF RBD for now
  (`structural_importance`, being probability-free, is unaffected).
  Alpha-factor is a planned extension. (#44)

### Changed
- Require **surpyval >= 0.16** (was >= 0.15): the time-varying-load
  `RegressionNode` schedule mode is built on surpyval's `sf_tvc`, added in 0.16.

### Documentation
- Concepts gains theory sections for covariate/time-varying reliability, load
  sharing, and common-cause failures; the tutorial gains a worked
  dependent-failures step exercising all three. (#37, #38, #44)

## [0.7.0] - 2026-07-20

The **Maintenance & Covariates** milestone: price imperfect-repair (generalized
renewal / Kijima) and replace-at-N-th-failure maintenance policies on
`Repairable`, and drive component reliability from operating covariates with
fitted surpyval regression models — alongside a harmonised RBD constructor
signature.

### Added
- **Imperfect repair (generalized renewal / Kijima) in `Repairable`.** The
  `Repairable` component now spans the whole repair-effectiveness spectrum
  rather than only minimal repair: it sources the expected number of failures
  `E[N(t)]` from the model it is given — analytic `cif` for minimal repair
  (unchanged; e.g. a surpyval Crow-AMSAA) or a seeded `mcf` for imperfect
  repair (a fitted `GeneralizedRenewal`, Kijima I/II). Minimal repair is the
  `q = 1` special case; perfect repair (`q = 0`) is the `NonRepairable`
  boundary. The overhaul/replacement-interval policy is unchanged in form
  `(cr·E[N(t)] + co)/t`; for a simulation-backed model `optimal_overhaul_policy`
  / `find_optimal_overhaul_interval` take `seed`, `n_simulations` and
  `max_interval` for a reproducible, bounded search, and `is_simulated` reports
  which path a `Repairable` uses.
- **Replace-at-N-th-failure policy on `Repairable`.** As an alternative to
  renewing at a fixed age, `optimal_failure_limit_policy` /
  `find_optimal_replacement_failure_count` price repairing on each failure and
  replacing at the N-th, minimising `(cr·(n-1) + co) / E[T_n]`, and return a new
  typed `FailureLimitPolicy(failure_count, cost_rate)`. `E[T_n]` comes from a
  seeded simulation (`expected_time_to_nth_failure`); for the minimal-repair
  (power-law) limit the exact closed form is exposed as
  `minimal_repair_time_to_nth_failure(alpha, beta, n)`.
- **`RegressionNode`: covariate-dependent components in an RBD.** A new node
  wraps a fitted surpyval **regression** survival model (accelerated-failure-
  time, proportional-hazards/Cox, proportional-odds, ...) together with a fixed
  covariate vector `Z` — the component's operating conditions — so its
  reliability is `model.sf(x, Z)`. It is an ordinary univariate node: system
  reliability, importance, MTTF and the condition-based (`age`) layer all work
  with no special handling, because at fixed covariates
  `R(x | age) = sf(age + x, Z) / sf(age, Z)` holds for every regression family.
  The fitted model serialises with the RBD via `surpyval.from_dict`. Simulation-
  based `mean`/`random` require a proper parametric lifetime (AFT/PO/parametric);
  a semiparametric Cox baseline has no defined MTTF and says so clearly. (#37)

### Changed
- **Bumped the surpyval pin to `>=0.15,<0.16`** (from `>=0.13,<0.14`), for the
  regression-model serialisation (`to_dict`/`surpyval.from_dict`) that
  `RegressionNode` relies on. No change to existing behaviour.
- **Harmonised the RBD constructor signatures.** The base ``RBD`` now takes
  ``(edges, nodes, k, ...)`` instead of ``(edges, k, nodes, ...)``, so all
  three classes share the same positional order: the structure first (``edges``
  then the node definition -- ``nodes``/``reliabilities``/``components``), then
  the optional ``k`` (k-out-of-n) modifier. ``NonRepairableRBD`` and
  ``RepairableRBD`` are unchanged. This only affects code that constructed the
  base ``RBD`` with a *positional* ``k`` (``RBD(edges, k_dict)``); pass ``k`` by
  keyword (``RBD(edges, k=k_dict)``) or as the third argument.

## [0.6.0] - 2026-07-19

The **Condition-Based Reliability** milestone: evaluate a system from the
current, sensor-known state of each component (a "digital twin"), and round out
the importance suite with design-time and data-targeting measures.

### Added
- **`structural_importance()`** on every RBD: the Birnbaum importance with all
  node reliabilities at 1/2 — a probability-free, design-time ranking of where
  each node is pivotal in the structure. It depends only on the diagram, so it
  is identical for a `NonRepairableRBD` and a `RepairableRBD` on the same
  layout, and honours the usual `working_nodes`/`broken_nodes` conditioning.
- **`parameter_sensitivity(t)`** on `NonRepairableRBD`: the sensitivity of
  system reliability to each node's distribution parameters,
  `dR_sys/d_theta = birnbaum_importance(node) * d sf_node/d_theta`. The
  parameter derivative is computed numerically (via surpyval's `from_params`),
  so it applies to any parametric model without per-distribution formulae.
  Composite and non-parametric nodes are omitted; forced nodes report zero.
- **Condition-based ("digital twin") evaluation** on `NonRepairableRBD`, driven
  by a new public `NodeState(age, alive)` per node. Each component conditions on
  its own current life `R_i(x | X_i) = R_i(X_i + x) / R_i(X_i)` and that
  propagates exactly through the system:
  - `sf_given_state(x, state)` — system reliability a further `x` from now given
    each node's state (the conditional generalisation of `sf`; an empty state
    reproduces `sf`).
  - `remaining_life(target, state)` — remaining useful life (RUL): the further
    time until system reliability falls to `target`.
  - `importances_given_state(x, state)` — the Birnbaum and criticality
    importances re-evaluated at the conditioned reliabilities.

  Supports ordinary distribution components; dynamic (standby) and composite
  nodes raise if given a state. State is transient input — the structure still
  persists via serialisation.

### Documentation
- **Doctest-backed examples** on the primary public methods (`sf`/`ff`, the
  importance measures, `time_to_reliability`/`bx_life`, the condition-based
  methods, `mean_availability`, and serialisation). They run in CI via
  `--doctest-modules`, so the documented examples cannot silently rot. Examples
  use deterministic (non-simulated) quantities so they need no seeding.
- **Expanded documentation site**: a start-to-finish
  [Tutorial](https://derrynknife.github.io/RePyability/tutorial/) and a
  [Concepts](https://derrynknife.github.io/RePyability/concepts/) reference
  (path/cut sets, exact computation, choosing an importance measure, and
  condition-based conditioning).

## [0.5.1] - 2026-07-18

### Fixed
- **Clean installs of 0.5.0 could not `import repyability`.** Importing
  `surpyval` (which RePyability does at load time) failed with
  `ModuleNotFoundError: No module named 'joblib'`: surpyval 0.11 imported
  joblib unconditionally without declaring it as a dependency, and
  RePyability listed joblib only in its `dev` extra, so a plain
  `pip install repyability` never pulled it in. Fixed by requiring
  `surpyval >= 0.13`, which declares `joblib` as a dependency — a plain
  install now imports cleanly.

### Changed
- Bumped the `surpyval` requirement from `>=0.11.1,<0.12` to `>=0.13,<0.14`.
  surpyval 0.13 relocated its recurrent-event models
  (`surpyval.CrowAMSAA` → `surpyval.recurrent.CrowAMSAA`, and likewise
  `Duane`); `Repairable` is unaffected (it only needs an object exposing
  `cif`), and the docs and tests were updated to the new import path.
- Dropped the now-redundant `joblib` entry from the `dev` extra (surpyval
  provides it transitively).

## [0.5.0] - 2026-07-18

This release focuses on packaging, tooling, correctness and API-consistency
foundations (Phases 0 and 1 of the project's development plan; forward-looking
work is tracked as issues in the GitHub repository).

### Added
- **Installable packaging.** `pyproject.toml` now declares a PEP 621
  `[project]` table and a `[build-system]`, so the package builds and installs
  from source (`pip install .`) with correct name, version and dependencies.
  Version is sourced from `repyability/_version.py`.
- **`scipy` dependency** is now declared (it was used directly but undeclared).
  `joblib` is declared in the `dev` extra only — it is a testing requirement
  (the suite imports surpyval's RandomSurvivalForest), not a runtime
  dependency.
- **Optional dependency groups**: `.[dev]` (pinned tooling) and `.[docs]`.
- **Seedable Monte-Carlo.** Simulation entry points accept a `seed=`
  argument (`NonRepairableRBD.random/mean/mean_time_to_failure/node_mttf`,
  `RepairableRBD.availability`, `StandbyModel`, `RepeatedNode`,
  `RepeatedStandbyNode`). Seeding is reproducible and restores the caller's
  global RNG state afterwards.
- **System-level `df` (failure density), `hf` (hazard rate) and `Hf`
  (cumulative hazard)** on `NonRepairableRBD`.
- **Curated public API.** Common classes are re-exported from the top-level
  package (`from repyability import NonRepairableRBD, StandbyModel, ...`) and
  listed in `__all__`. `repyability.__version__` is available.
- **`NonRepairableRBD.is_fixed`** property.
- CI now runs a lint/type gate and a test matrix (Python 3.11–3.12) with a
  coverage `fail_under` gate; a release workflow publishes to PyPI via trusted
  publishing.
- `CHANGELOG.md`, `CONTRIBUTING.md`, and issue/PR templates.
- **Typed result objects** for `RepairableRBD.availability()`
  (`AvailabilityResult`, `Criticalities`, `UpDownImportance`,
  `FailureCriticalityIndex`, `RestorationCriticalityIndex`), exported from the
  top-level package. They give documented, IDE-discoverable attribute access
  (`result.criticalities.iou.up`) while remaining backwards compatible with the
  old nested-dict API (subscript, `keys`/`items`/`values`, `in`, `dict()`).
- **Documentation site** (MkDocs + mkdocstrings): overview/quickstart, a user
  guide, and an auto-generated API reference. A CI job builds it with
  `--strict`, and it is published to GitHub Pages
  (https://derrynknife.github.io/RePyability/) on every push to `master`.
- **Serialisation.** RBDs round-trip to/from a JSON-friendly structure via
  `to_dict`/`from_dict`/`to_json`/`from_json`, so a diagram can be saved,
  loaded, shared and version-controlled. The structure (edges, k-out-of-n,
  repeated nodes, nested RBDs) and node models (surpyval parametric
  distributions, and the standby/repeated/`NonRepairable` wrappers) are
  reconstructed faithfully; integer and string node names both survive JSON.
  `RBD.from_dict`/`from_json` dispatch on the document's declared type. Fitted
  non-parametric models raise a clear `NotImplementedError` (no reconstruction
  API upstream).
- **Inverse-reliability queries** on `NonRepairableRBD`:
  `time_to_reliability(target)` (solves ``R(t) = target``) and `bx_life(x)`
  (the B\ :sub:`X` life, e.g. `bx_life(10)` is the B10 life). Both accept the
  usual `working_nodes`/`broken_nodes`/`method` arguments.
- **Exact steady-state repairable metrics** on `RepairableRBD` via the
  Birnbaum/Vesely frequency formula: `system_failure_frequency()`,
  `mean_up_time()` (MUT), `mean_down_time()` (MDT) and
  `mean_time_between_failures()` (MTBF), all supporting
  `working_nodes`/`broken_nodes` conditioning; `NonRepairable` gains
  `failure_frequency()`. `AvailabilityResult` gains `system_downtime`,
  `system_failures`, `system_restorations` and `n_simulations` fields plus
  simulation-estimate properties `mean_up_time`, `mean_down_time` and
  `failure_frequency` to cross-check against the exact values.
- Fixed `NonRepairable.mean_availability()` crashing with the default
  (instant) replacement time — surpyval's `ExactEventTime.mean()` raises
  `AttributeError`; a workaround computes its mean from its parameter.
- **Simulation uncertainty quantification and validation.**
  `AvailabilityResult` gains `availability_se` (pointwise standard error) and
  `availability_interval(confidence)` (pointwise Wilson confidence band);
  `NonRepairableRBD` gains `mean_time_to_failure_interval()` returning a new
  `ConfidenceInterval` result type (estimate, bounds, standard error). The
  simulator's transient availability is now validated in the test suite
  against exact Markov closed forms (single-component, series and parallel
  exponential systems) within Monte-Carlo sampling error, and the
  finite-window censoring bias of the simulation MUT/MDT estimates is
  documented.
- **Component maintenance-policy layer.** The vestigial `Repairable` class
  is now a real minimal-repair ("as bad as old") economics tool: it takes a
  surpyval recurrence model (anything exposing `cif`, e.g. Crow-AMSAA,
  Duane) and finds the optimal overhaul interval minimising
  `(cr*Lambda(t) + co)/t` (the Barlow-Hunter policy), returning `inf` when
  the unit does not wear out (HPP-like, `beta <= 1`) — validated against
  the power-law closed form. New typed result `MaintenancePolicy`
  (`interval`, `cost_rate`), returned by
  `Repairable.optimal_overhaul_policy()` and the new
  `NonRepairable.optimal_replacement_policy()`.
  `NonRepairable.find_optimal_replacement()` now returns `inf` for an
  exponential lifetime (memoryless — preventive replacement never pays),
  matching the existing Weibull shape <= 1 behaviour. A user-guide section
  explains renewal ("as good as new", `NonRepairable` — also the component
  representation inside `RepairableRBD`) vs minimal repair (`Repairable` —
  standalone; not a valid RBD node model, since RBD repairs are assumed to
  renew).

### Changed
- **API standardised across `RBD`/`NonRepairableRBD`/`RepairableRBD`**
  (breaking):
  - **Numpy-style return contract**: scalar time in → `float` out; array in →
    `numpy.ndarray` out. Applies to `sf`/`ff`/`reliability`/`unreliability`/
    `df`/`hf`/`Hf`/`cs`, the per-node accessors, and the importance measures
    (dicts of floats for scalar input). Previously everything returned
    length-1 arrays for scalar input. `mean_availability` and friends return
    plain `float`.
  - `x=None` on a time-varying RBD now raises a clear `ValueError` (it
    previously failed cryptically); fixed-probability RBDs broadcast over an
    array `x` instead of collapsing it to one value.
  - **Renames**: `sf_by_node`/`ff_by_node` → `node_sf`/`node_ff`;
    `get_nodes_names()` → `node_names()`; `time_varying_rbd()` → property
    `is_time_varying`; `initialize_event_queue(working_components,
    broken_components)` → `(working_nodes, broken_nodes)`;
    `AvailabilityResult.components_uptime`/`components_downtime` →
    `node_uptime`/`node_downtime`.
  - **Constructor parity**: `RepairableRBD` now accepts `input_node`,
    `output_node` and `on_infeasible_rbd` like `NonRepairableRBD`, validates
    its node list, and raises a clear error for graph nodes with no component
    definition (previously a `KeyError` mid-simulation).
  - **Importance measures unified**: explicit signatures on both classes,
    typed returns, and all measures now accept `working_nodes`/`broken_nodes`
    to condition the analysis. `RepairableRBD` importances return floats
    (steady state has no time dimension).
  - The `check_x`/`check_probability` decorators now use `functools.wraps`,
    so `help()`, IDE hovers and `inspect` see the real names, docstrings and
    signatures (they previously reported `wrap` with no docstring).
- **`fussel_vesely` → `fussell_vesely`** (corrected Fussell-Vesely spelling).
  The old name remains as a deprecated alias that emits a `DeprecationWarning`.
- Reliability-model dispatch no longer branches on surpyval `dist.name` string
  literals scattered across modules; it goes through capability helpers
  (`repyability/rbd/_model_utils.py`), covered by a surpyval-compatibility
  test.
- `requirements.txt` / `requirements_dev.txt` now install the package and its
  (pinned) extras from `pyproject.toml`, removing the previous mismatch where
  dev pulled surpyval from git HEAD while runtime pinned a release.
- Coverage now measures the library only (tests omitted) and enforces a
  `fail_under` threshold.
- `RepairableRBD.availability()` now returns a typed `AvailabilityResult`
  instead of a plain `dict`. It is a `collections.abc.Mapping` (not a `dict`
  subclass), so dict-style access is preserved but `isinstance(result, dict)`
  is now False (use `isinstance(result, Mapping)`).
- Requires Python >= 3.11 (raised from 3.10: `surpyval` >= 0.11.1, the core
  dependency, itself requires Python >= 3.11, so the declared 3.10 support could
  not actually be installed).

### Fixed
- Removed a stray debug `print()` in `RBD.improvement_allocation`.
- Replaced `type(x) == Cls` checks with `isinstance` in `NonRepairable`.
- Replaced mutable default arguments (`{}`, `[]`) with `None` sentinels.
- Removed cross-instance access to a name-mangled attribute (`__fixed_probs`).
- **`working_nodes`/`broken_nodes` ("force node working/failed") logic:**
  - `RepairableRBD.availability()` no longer crashes with `ZeroDivisionError`
    (and the operational/IOU criticality indices no longer return `NaN`) when
    a simulation records zero system failures/restorations — e.g. when a
    redundant node is forced working. Zero-exposure measures return 0.
  - A component forced broken in `availability()` is now accounted as **down**
    from t=0 (its uptime and the initial system state were previously assumed
    "up", so a broken component reported full uptime and a system it kept down
    was over-counted as available at the start).
  - `working_nodes`/`broken_nodes` now **raise** on invalid input instead of
    silently ignoring it: unknown node names, the input/output node, or the
    same node in both sets. Previously a typo silently returned a
    plausible-but-wrong result.
  - Setting a repeated node broken now raises the same error as setting it
    working (previously it was silently ignored — an asymmetry).
- **`RepairableRBD.availability()` component downtime accounting.**
  `components_downtime` was accumulated against the *running cumulative*
  component uptime instead of the current simulation's uptime, giving wrong
  (often negative) totals after the first simulation. Each component's
  uptime + downtime now correctly sums to `N * t_simulation`. (`sf`/`ff` and
  the derived `reliability`/`unreliability`/`cs` paths were audited for the
  working/broken overrides and found correct; regression tests added.)
- `Repairable.find_optimal_overhaul_interval()` returned a raw
  `scipy.optimize` result object instead of the interval, performed an
  unbounded search from an arbitrary starting point, and the class
  docstring described the wrong class ("stores the non-repairable
  information"). All fixed in the rebuild; the previously commented-out
  `test_repairable.py` is resurrected with closed-form validation.

### Removed
- Dead/unused `repyability/rbd/rbd_args_check.py` module and the never-
  implemented `working_components`/`broken_components` parameters that its
  (commented-out) check referenced in `sf()`'s docstring.

### Deprecated
- `NonRepairableRBD.fussel_vesely` / `RepairableRBD.fussel_vesely` (use
  `fussell_vesely`).
