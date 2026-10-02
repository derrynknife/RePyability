# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions have two parts, major.minor (until 0.10.1 they had three): from
1.0, a release that breaks compatibility raises the major number, and any
other release, fixes included, the minor.

## [Unreleased]

## [0.11] - 2026-09-30

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
again when compiled with numba (#119, #120), and `import repyability` is
quicker (#121). Components can share a limited number of repair crews (#89),
with exact long-run values from a Markov chain when their lives and repairs
are exponential (#90); a duty unit and its spares can be a standby group,
repaired one unit at a time (#91); `spares_demand` and `spares_stock` count
the spares each component uses and the stock to hold for a lead time (#95);
a component can be replaced on condition at periodic inspections (#96), or
early at a stop of its maintenance group, sharing its set-up (opportunistic
maintenance, #108), and repaired imperfectly, by Kijima's virtual age, or
replaced at the N-th failure (#109); a simulation run can be split across
machines and merged (#114); and small failure probabilities are estimated by
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
repairs take no time (#144).
Meshed diagrams are decided by a binary
decision diagram, in milliseconds where their path sets took minutes (#102,
#103). Non-parametric nodes, and the fits to simulated lifetimes behind some
standby and load-sharing models, are deprecated; everything deprecated goes
in 0.12, and warns with a `FutureWarning`.

Behaviour changes: seeded repairable simulations give different numbers,
once, as each random quantity now has a stream of its own (#119);
`mean()` and `mean_time_to_failure()` are exact, the old estimate is
behind `method="simulate"`, and the exact MTTF refuses common-cause groups
(#122); the number of simulations is `mc_samples` everywhere, and its cap
`max_samples`, with the old names deprecated until 0.12 (#105);
`is_analytically_solvable()` flags only simulated nodes (#127); cold-standby
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
  components, and the availability over time, are simulated. The groups are
  saved with the RBD. Checked against the textbook chains (two-unit cold
  and warm standby with one or two repairers, imperfect switching), a
  switch that always fails leaving a single unit, timelines worked by hand
  with a shared crew, and the simulation.
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
  after it); a fleet's systems add up independently. Block replacement, hidden failures, standby groups and
  waiting for repair crews are not renewal processes: the counts refuse
  them, and `spares_demand(method="simulate")` counts every component's
  replacements in simulations of the whole system. `analysis_routes()`
  reports both. Checked against Poisson closed forms (constant failure
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
  service; the exact long-run values and the availability over time refuse
  it with the reason, and `analysis_routes()` says so. The threshold and
  inspection cost are saved with the RBD. Checked against a timeline
  worked by hand, block replacement and run to failure (identical results),
  constant failure rates, and a direct simulation of the policy.
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
  to the last digits. Chunks carry their run's settings and a hash of the
  system, and only chunks of one run merge. `NonRepairableRBD.
  random_block(block, seed)` draws one 10 000-lifetime block of the
  lifetimes `random(size, seed=seed, n_jobs=...)` draws. The simulation
  guide gives the engines' throughput on one machine, in a table.
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
