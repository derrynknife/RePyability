# RePyability — project notes for Claude

## Architecture & scope

- **Distribution/lifetime fitting stays in surpyval, not RePyability.** surpyval
  (same maintainer) owns fitting failure/event data to distributions.
  RePyability *consumes* already-fitted surpyval models (and equivalents) as
  RBD node inputs; it does not implement its own data-fitting APIs. When a
  "from data to system reliability" workflow is wanted, do the fitting in
  surpyval and pass the resulting models in — do not add fitting logic here.

- **Visualization stays out of RePyability.** The Reliafy app (separate,
  open source) handles plotting/dashboards/reporting. RePyability is the
  computational reliability engine; keep it free of plotting dependencies.

- **Model behaviour belongs in surpyval too.** When a surpyval model
  misbehaves (its draws, means, densities or save format), raise an issue in
  derrynknife/surpyval rather than only working around it here. Keep any
  RePyability workaround small and list it below with its issue, so it can go
  once RePyability's minimum surpyval version (`pyproject.toml`) has the fix.

- **surpyval releases reach RePyability users at once.** surpyval is
  required with no upper bound, so its next release is what a fresh install
  gets. No automated run tests surpyval's `develop` (the maintainer wants
  one automated test run, below): before surpyval releases, run
  RePyability's tests against its `develop` by hand (`pip install
  "surpyval @ git+https://github.com/derrynknife/SurPyval@develop"`), and
  when they fail, fix RePyability (working with both the released surpyval
  and `develop`) and release that first. CI's `test (minimum dependencies)`
  job tests the oldest surpyval `pyproject.toml` allows (and every other
  dependency at its floor); raise that minimum,
  rather than keep code for older versions, once RePyability needs what a
  newer surpyval does.

## Code

- **Clean, simple implementations (#204).** Optimise through algorithms
  and data structures (complexity, exact methods, work not done, numpy,
  numba compiling the same plain algorithm), not machine-level tricks: no
  logic written out twice for speed, no second design of a computation
  that must agree with the first, no fast path that hands rare cases
  back, no speed constant that changes results. A compiled version
  follows the plain one's algorithm and shape; a better algorithm goes
  into both. If code needs a "keep in sync with X" note, simplify it
  instead.

- **`RepairableRBD`'s code lives in modules by analysis.** The class
  (`repairable_rbd.py`) keeps its constructor, its public methods (their
  signatures, decorators and docstrings, each calling a function of the
  same name), the hooks `RBD` calls, and the methods a test overrides on
  one diagram (`_crews_couple`, `_conditional_modules`). The work is done
  by functions taking the diagram, in `repyability/rbd/`: `_event_loop`
  (the Python loop and its streams), `_runs` (availability and cost runs,
  the twin, conditional runs, chunks and shards), `_curves` (each node's
  curve, the start states, `_settling`), `_windows` (values over time),
  `_long_run`, `_costs`, `_repairable_capacity`, `_repairable_importance`,
  `_ccf_groups`, `_crews`, `_intervals`, `_repairable_allocation`,
  `_spares`, `_repairable_uncertainty`, `_sensitivity`,
  `_repairable_routes`, `_requirements` (the `_require_*` checks) and
  `_common` (small helpers they share). A new analysis goes into the
  module of its kind, and its public method onto the class. Modules low in
  the imports (`_compiled`, `_tally`, `_timeline_runs`, ...) import these
  inside the functions that use them, as they import the modules above
  them.

## Simulation engines and seeded results

- **A `RepairableRBD` simulation has two engines that must agree to the last
  bit**: the Python event loop (`_event_loop._replicate`) and the compiled
  one (`repyability/rbd/_kernel.py`, numba, the optional `fast` extra). A
  change to the loop's events, arithmetic or order goes into both;
  `test_simulation_engines.py` checks them against each other (run by CI's
  `test (with numba, ...)` jobs) and the Python loop against a reference
  written from the streams' definition. What the compiled engine does not
  simulate, `_compiled.unsupported` sends to Python: numba's own loop takes
  `numba=True`, as it also runs maintenance, inspections, repair crews,
  standby groups, nested RBDs and capacities (#155), while engines from
  other packages keep the plain-components contract. Inside `_kernel`, one
  loop (`_simulate`) takes every level's events, the system's and its
  nested RBDs' (#206), and keeps whether each level works up to date in
  one structure over every level (`_compiled._kept`; no table of states,
  #255). Per-event helpers there take few arrays: each array
  a call binds costs reference counting that numba cannot prune in a
  function this size, and passing the run's tuples to a function per
  simulation cost several times the loop (measured in #206).
- **Simulations take turns across threads (#216).** The event loop keeps
  a run's state on the diagram (`_run_state`, an `_events._Run`) and draws that cannot be
  streamed come from numpy's global RNG, so a run holds
  `repyability.utils.wrappers.SIMULATIONS`, a process-wide `RLock`:
  `_event_loop._run` (but a sharded run's parent, whose shards take it
  where they run, so a `shard_map` on threads cannot wait on it)
  and `numpy_seed`. New code that runs the loop or seeds the global RNG
  goes through one of these; threads that work for a run (numba's) must
  not take it.
  `test_threads.py` checks seeded calls on threads give their serial
  results.
- **CI's plain test jobs have no numba.** A test that asks for
  `engine="numba"` skips without it (`pytest.importorskip("numba")`, or
  `needs_numba`), unless what it checks comes before numba is needed: a
  compiled engine refuses what it does not simulate first. A machine with
  numba installed runs neither way, so run such tests with numba hidden
  too, `importlib.util.find_spec("numba")` returning None and `import
  numba` raising `ModuleNotFoundError` (which `importorskip` needs).
- **`simulate_timelines`' histories are the event loop's, on every
  engine.** Both loops record them as they run (`_event_loop._replicate` with
  `_Context.history`; `_kernel._simulate` when given room to record, the
  system's own level only): each top-level
  component's changes, and each of the system's with the component that
  made it (#205: there is no other path to them). A change to what the
  loops record goes into both: `test_timelines.py` checks every engine's
  histories against each other and against `availability`.
- **Capacity states are worked out in batches**
  (`_tally._CapacityRecorder.evaluate`): a simulation's in Python, a batch's
  compiled, so a state's capacity must not depend on what is worked out
  with it. It does not while every node works at one level (each
  probability is then 0 or 1, and every sum exact); with several levels
  each state is worked out on its own. Keep to that if the batching
  changes: `test_states_worked_out_together_are_each_on_its_own` checks it.
- **Engines from other packages** (`repyability/rbd/engines.py`) run what
  `_compiled.unsupported` allows and are handed the run's own objects (the
  `_compiled.Runner` arguments), so they may build on `_compiled`'s
  `_System`, `_Store` and `_structure` and on `_streams`' blocks. Keep those
  compatible, or raise `engines.API` (with a CHANGELOG entry) when an engine
  would have to change with them.
- **A conditional run (#189, `repyability/rbd/_conditional.py`, and
  `RepairableRBD._conditional_modules` and `_runs._modules_rbd`) simulates
  the modules alone** (`_conditional_modules`: the nodes
  `_curves._node_over_time` refuses, with their maintenance groups), as a diagram
  of their own (`_runs._modules_rbd`) whose streams are named as in the system,
  so its simulations are a plain run's; its tally keeps each simulation's
  cost beside its histories, on both engines. Given each joint state of
  the modules, the rest comes from `_windows._window` and `point_availability`
  with the modules held. A change to what makes a node need simulating
  goes into `_curves._node_over_time`, and anything that ties a module to the
  other nodes (crews, a group's `system_down`) must refuse in
  `_conditional_modules`. Changes at one instant are ordered as the loop
  orders them (`_conditional.paths`: failures, the stops they open, then
  restorations; a module's before the other nodes'): a change to that
  order in the loop goes into `paths` too. `test_conditional.py` checks
  each simulation's values against closed forms from its module's
  history, and the estimates against plain runs.
- **A run's means are exact or conditional by default (#187, #189).**
  `availability()` and `cost()` take the exact methods' expected values
  where they work them out (`_runs._exact_means`, as the controls of the system
  itself), and otherwise, where a conditional run applies, each
  simulation's expected values given its modules' histories
  (`_runs._conditioned_run`, the modules simulated again from the run's
  entropy); `availability_from_chunks` gives merged chunks the same
  (`_runs._default_means`). The simulations are a plain run's either way. A test
  that checks the simulation against the exact methods must run plainly
  (`control_variate=False`), or it compares the exact values with
  themselves.
- **A run's changes are put in time order by numpy alone (#201, #208)**
  in `repyability/rbd/_time_order.py`: `_net_by_time` nets the +1 and -1
  changes of state with one sort of integers, each time's bits above the change's
  sign (a time is never negative); `_by_time` sorts the capacity's (time,
  change) pairs, and the changes in how many systems can carry an
  unlimited amount, as complex numbers; `_capacity_totals` merges the two
  the first way. What is
  made of a time's changes is exact (whole counts, exact sums), so their
  order within a time does not matter. `test_time_order.py` checks each
  against its definition written out plainly.
- **A core's decision diagram has one search and one replay** (#202,
  #207): `bdd.search` and `shannon.replay`/`replay_gradient`, which numba
  compiles as written (`_bdd_kernel`) where it is installed and a core is
  large. Keep them to what numba compiles. Their plans and values are
  recorded (`tests/bdd_records.json`, by `tests/_bdd_cases.py`), and
  `test_bdd_records.py` checks both modes against the records; re-record
  only after a deliberate change of the plans.
- **The random streams (`repyability/rbd/_streams.py`) define every seeded
  result** (#209): uniform `k` of simulation `r` is numpy's Philox's with
  the stream's key (from the entropy and the stream's name) and the counter
  `(0, r, 0, 0)` (`_philox`). Changing a stream's name, key or that
  definition changes seeded results: a behaviour change for the CHANGELOG,
  with `seeded_event_loop.json` re-recorded and the docs' quoted numbers
  updated. The layout (`first_rows`, `block_width`, `BLOCK_DRAWS`,
  `MAX_WIDTH`, a chunk's rows) only affects speed: `test_streams.py`
  checks it changes nothing.
- **The repair crews' Markov chain (`repyability/rbd/_crew_chain.py`)
  copies the simulation's queue (`_events._Crews`)**: which waiting job a free crew
  takes, and how instant jobs pass through. A change to one goes into the
  other; `test_crew_chain.py` checks the chain's exact values against the
  simulation. Likewise a standby group's chain (`_standby_chain.py`) copies
  `_events._StandbyGroup`'s rules (switching, spares, repairs), checked by
  `test_repairable_standby.py`.
- **A common-cause group's chains (`repyability/rbd/_ccf_chain.py`) copy
  the simulation's causes** (`_events._Cause`, `_event_loop._strike`, #158): each cause, a
  member's own or a shared one, strikes at its share of the failure rate
  and fails the members it names that are up, at once; with tests that can
  miss, one coin for all the failures it makes. A change to one goes into
  the other: `test_ccf_over_time.py` checks the simulation against the
  chains over time. The allocations' chains count a member's copies
  (`_Counted`) rather than tell them apart, which holds for a
  `BetaFactor`'s one shared cause: `test_ccf_allocations.py` checks them
  against the chain of every copy.
- **A common-cause group's chain with tests and repairs that take time
  (`_ccf_chain._Timed`, #220) copies the simulation's inspections** too
  (`_event_loop._inspected_follow_up`, `_event_loop._inspected_next`, `_event_loop._strike`): a working member
  is off line for its test, unaged and not struck; a failure found is
  repaired once the test is over; a test in a member's own test or repair
  is not done. A change to one goes into the other: `test_ccf_timed.py`
  checks the chain against the simulation, and against the members' own
  model (`_hidden_tests`) where no cause is shared.
- **A tested unit's numerical model (`repyability/rbd/_hidden_tests.py`,
  #159) copies the simulation's inspections** (`_event_loop._inspected_follow_up`,
  `_event_loop._inspected_next`): a test takes a working unit off line without ageing
  it, a failure is repaired once its test is over, the tests in a repair
  are not done, and a failure a test misses waits for the next full test.
  A change to one goes into the other; `test_hidden_failures_timed.py`
  checks the model against the simulation.

## How each analysis is computed

- **A `RepairableRBD`'s junctions are folded out of its structure.** A node
  given `PerfectReliability`, or a spec whose life never fails
  (`_model_utils.always_works`: that, or a probability of failing of 0), is
  no component: `RBD._decomposition()` folds
  it in as always working (`modular.fold`), so whatever evaluates the
  structure (the curves, the long-run values, both simulation engines, the
  timelines, the path and cut sets) never sees it. Evaluate the structure
  through `_decomposition()`, not the graph, so that this holds; only the
  capacity analysis's reduced diagram (`flow`) keeps the junctions, and
  `_capacity_arrays` passes them as working. `test_junctions.py` checks
  every public method against the same system drawn without a junction.

- **Identical components share one curve** (`_curves._availability_curves`,
  `_curves._curve_twin`): a plain unit's curve follows from its life and repair
  models and its state at 0 alone. Anything new that makes a unit's curve
  depend on more (a schedule, a group, a coupling) must leave it out in
  `_curves._plain_unit`, or two units would be given one curve that is only one's:
  `test_identical_components_share_one_curve` checks the shared curves
  against a curve each.

- **`analysis_routes()` (both RBD classes) must agree with the methods.** It
  says, without running anything, whether each public analysis is exact,
  numerical, simulated or refused. Refusals go through checks the report
  calls too (`_requirements._require_*` helpers, `_requirements._inspected_rate`, ...), so its reasons
  are the methods' own messages. A refusal is a `NotImplementedError` (what
  an analysis cannot work out) or a `routes.Refused` (a `ValueError`: the
  diagram has nothing the analysis works on); `routes.refusal` raises any
  other error, as a fault. When a method is added, or gains a refusal
  or changes how it computes, update `analysis_routes`:
  `test_analysis_routes.py` checks that it covers every public method, that
  each method does what it says on diagrams of every kind, and that the
  saving guide's table agrees with it. It calls every method the report
  lists (a new one needs its call in `NONREPAIRABLE_CALLS` or
  `REPAIRABLE_CALLS`) on every diagram of `tests/catalogue.py`, the one
  registry of diagram kinds, which the engines' agreement and the
  exact-against-simulated checks (`test_catalogue.py`) share. A new node
  class, spec key, policy or diagram option needs a diagram there:
  `test_the_catalogue_has_every_kind` names what no diagram uses.
- **The README's "When is a simulation needed?" table follows the routes.**
  It says, by what is asked, the components and the maintenance, what is
  simulated and whether it must be (or could be exact, with the issue).
  `test_the_readme_says_what_is_simulated` checks each row against
  `analysis_routes()`: when an analysis is made exact, update its row.
- **A diagram too meshed to work out keeps only its graph (#172).** When a
  core's decision diagram passes `bdd.STEP_LIMIT`, `RBD._decomposition()`
  is a `modular.GraphStructure`, which works out a state and a lifetime
  from the graph, for the simulations, and refuses the rest with the
  reason. Both classes' `analysis_routes` end with `_meshed_routes(out,
  free, drawn)`, which refuses every exact or numerical analysis but those
  in `free`, which need no structure, and the simulated ones in `drawn`,
  which work the structure out for each draw of parameters: a new method
  that needs none (a node's own values) goes in `free`. The compiled
  engine (`_compiled.unsupported`) leaves such a diagram to the Python
  loop. `test_meshed_structures.py` checks the
  stand-in against the structure worked out, and `test_analysis_routes.py`
  the routes of diagrams too meshed.
- **A non-repairable structure's common-cause groups are worked out
  module by module (#219, `repyability/rbd/_ccf_modules.py`)**: each group
  conditioned on within the smallest module holding its members, and where
  a module's groups would multiply past `COMBINATIONS` outcomes, their
  causes written out as shock events (repeated nodes of the diagram,
  repeated events of the tree), exclusive causes through their independent
  equivalent (`ccf._as_independent`) where it exists. Either way the values
  must be those of conditioning on every combination of every group's
  outcomes (`ccf.shock_outcomes`), which `test_ccf_modules.py` checks them
  against. A change to a model's outcomes (`_split`) goes into its causes
  (`_causes`, `_fired`) too.
  A `RepairableRBD`'s groups are conditioned on module by module too
  (#218, `_ccf_modules.Tabled`), on their chains' combinations of the
  members up or down, an owner's laid out as a last axis of the arrays;
  `test_ccf_repairable_modules.py` checks every long-run value, measure
  and the groups' system over time against the times split by every
  combination (`_ccf_groups._with_ccf_groups`, which the capacity distribution and the
  allocations still take, behind `_ccf_chain.check_split`).
- **The integrals over a window are summed on coarse pieces**
  (`repyability/rbd/_quadrature.py`, #164): the curves' breaks, cut to a
  few steps of the finest grid still changing, and halved until their
  quadrature agrees with their halves'. A curve class that is linear on a
  grid says so with `grids()`, and where else it bends with `breaks()`;
  one that has neither has every knot taken as a break, which is right but
  makes every knot a piece. Never clip a piece's integral: a count that
  dips integrates out, where clipping made the total depend on the pieces.
  `test_quadrature.py` checks the pieces against summing between every
  knot.
- **Unavailability over time is worked out as itself (#237)**, not as
  one less the availability, which rounds to 0 below about 1e-16:
  `_windows._system_at(..., down=True)` takes each component's
  probability of being down from its closed form where it has one
  (`_curves._closed_form_down`: an exponential life and repair, nothing else
  about it), from its curve otherwise (`_curves._curves_down_at`), and from the
  crews' and common-cause groups' chains (`GroupsCurve.down_at`); the
  mission's integral is refined to its own size (`_integrated(...,
  relative=)`). A change to how `point_availability` works out a kind of
  component goes into the down side too: `test_unavailability.py` checks
  both against closed forms and each other.
- **An MTTF is integrated on pieces that start at its models' kinks**
  (`repyability/rbd/_mean_lifetime.py`, #229): every kink
  (`model_kinks`: a numerical curve's grid, where a support starts)
  starts a piece, the other knots (quantiles) are thinned to
  `_PER_DECADE` a decade, and each piece is integrated by the
  Gauss-Kronrod (7, 15) rule and halved until `|K15 - G7|` is within
  `RTOL`. A new numerical model class gives its grid to `_collect`'s
  kinks: without them its bends fall inside pieces, which then take many
  halvings (right, but slow). `test_evaluation_costs.py` checks the
  points an MTTF takes and its accuracy.

## API conventions

- **One name for the number of simulations**: `mc_samples`, and `max_samples`
  for its cap in a run to a `tolerance`, in every method and constructor
  that simulates; `seed` seeds it (#105). The old names (`N`, `max_N`,
  `n_sims`, `n_simulations`) went in 0.12, and are refused naming their
  replacement (#232): `deprecation.refuse_removed_names` wraps every
  public function of the classes that simulate (an `RBD` subclass gets it
  from `RBD.__init_subclass__`); a new class whose methods take
  `mc_samples` gets `@refuse_removed_names`. A seed goes through
  `checks.seed` where it is taken.
- **Inputs are checked where they are given (#233)**: numbers through
  `checks.is_number`/`number_or_nan` (not `bool`, not text), times through
  `checks.real_array`, a choice of option through `checks.one_of`, and
  models through `checks.no_distribution`, which refuses surpyval's
  distribution itself (`surv.Weibull`) for a model of it. A new argument
  that takes a number, a time, an option or a model uses them.
- **A model is probed with `_model_utils.MODEL_ERRORS`** (and saved with
  `SAVE_ERRORS`): code that asks a model what it can do (a quantile, a
  mean, a saved form) catches what a model that cannot do it raises, never
  `Exception`, so a fault in a model or in RePyability is raised.
- **Warnings point outside the package (#232)**: `warnings.warn(...,
  stacklevel=outside_level())` (`repyability/utils/wrappers.py`), never a
  counted level, which goes stale as calls are wrapped. A message never
  puts `'s` after a quoted name (`{node!r}'s` reads `'a''s`): write "the
  life of component {node!r}"; `test_messages.py` scans for it.
- **Docstrings serve the web pages and `help()` (#238).** Cross-reference
  as ``[`name`][repyability.Class.name]``: mkdocstrings links it from the
  source, and `repyability/__init__.py` rewrites it at import, for the
  classes and functions in `__all__`, to ````name```` (`utils.docs`), so a
  new public class goes in `__all__`. Long explanations belong in the
  guide, linked, not in a parameter's description; `test_help.py` checks
  `help()`.
- **Every result is a `results._ResultMapping` dataclass (#235)**, so it
  has `to_dict()` (through `results.plain`, ready for `json.dumps`); a
  result holding arrays of a run's length gets a summary `__repr__`.
  `test_result_objects.py` sends each kind through JSON.
- **A diagram's levers are listed in one place (#244)**:
  `_sensitivity._levers` (repairable) and `_nonrepairable_levers`, each
  lever with the range its constructor checks. `RepairableRBD`'s
  `parameter_sensitivity`, both classes' `levers()` and `with_levers`
  take them from there (`NonRepairableRBD.parameter_sensitivity` walks its
  nodes itself). A new lever (a spec option, a kind of model) goes there
  with its bounds: `test_levers.py` checks, on every catalogue diagram,
  that `levers()` lists what `parameter_sensitivity` reports, in its
  order, that each value lies in its range, and that `with_levers`
  refuses a value past it.
- **A deprecation gives one minor release's notice.** It warns in one
  minor release and the next removes it, with a `FutureWarning` (always
  shown) through `repyability/utils/deprecation.py`. What 0.11 deprecated
  went in 0.12 (#149), and `test_removed_in_0_12.py` keeps it gone: the
  old names and ignored arguments raise `TypeError`, a diagram refuses a
  non-parametric node, and a standby or load-sharing model with no exact
  or numerical reliability refuses one (`is_simulated`), where a fit to
  simulated lifetimes stood in, and is left to the simulations. What 0.12
  deprecated went in 0.13, and `test_removed_in_0_13.py` keeps it gone:
  calling `SparesDemand.mean()`/`std()` (#184) and those models'
  `mc_samples`, `lower` and `seed` raise `TypeError`, and `StandbyModel`'s
  options after them are given by name. What 0.13 deprecates goes in 0.14
  (`NEXT_REMOVAL`): `optimal_inspection_intervals(offsets=)`, renamed
  `offset_shares=` (#222, through `deprecation.renamed`); calling
  `CapacityDistribution.mean()`, now a property (#235, through
  `deprecation.called`); and `mc_samples` and `seed` given to the exact
  `mean` of a `StandbyModel`, `LoadSharingModel` or `DegradingNode`,
  which ignores them (#233, through `deprecation.ignored` in
  `standby_node.drawn_mean`), all of which
  `test_what_0_13_deprecates_goes_in_0_14` holds to: it fails once the
  version reaches `NEXT_REMOVAL`.
- **No result depends on the unit of time (or of a covariate).** A step,
  grid or probe is sized from the models (a typical life,
  `failure_time_scale`), never in absolute time, and a proportional
  change of a lever is of its scale (`_sensitivity._proportional_scale`:
  1 for a log-scale parameter, none for a covariate). `test_units.py`
  checks results scale with the unit, from 1e-6 to 1e6. A model's unit
  is surpyval's to carry (SurPyval#845); until it does, every node is
  taken to be in one unit, unchecked.
- **Exact by default, simulation on request.** Where an analysis can be
  computed exactly or numerically, that is the default, and the Monte-Carlo
  estimate is a `method="simulate"` away (as for `NonRepairableRBD.mean`).

## Performance

- **In a loop, a dot product of long vectors goes through
  `repyability.utils.vectors.dot`, not `@`.** OpenBLAS threads one of more
  than about 10,000 terms, at milliseconds a call; a loop of thousands of
  them (a renewal sum, a band of a grid) then runs hundreds of times
  slower. Matrix products are left to BLAS.

## Testing

- **One automated test run: the full suite on the pull request from dev
  into master.** `actions.yml` runs on nothing else: not on pushes, not on
  pull requests into dev, not on a schedule, and there is no other test
  workflow. Work going into dev is checked locally by the tests of what it
  changes (the files that exercise the code touched, and new tests for
  it), and is merged without waiting for CI. Do not start, re-run or add
  other automated test runs; run the full suite locally only when the
  maintainer asks.

## Releasing

Releases are cut from master by `.github/workflows/release.yml`, which this
session can run: it cannot push tags or create GitHub Releases itself. Every
merge and release needs the maintainer's go-ahead.

Versions have two parts, major.minor, from 0.11 (they had three until
0.10.1): from 1.0, a release that breaks compatibility raises the major
number, and any other release, fixes included, the minor. There are no
patch releases, and `release.yml` refuses a version that isn't
major.minor.

1. On the working branch, bump `repyability/_version.py`. Roll the CHANGELOG
   `[Unreleased]` section into `## [X.Y] - YYYY-MM-DD`, opening with a
   summary paragraph and the behaviour changes (they open the release notes),
   and update the version in `docs/guide/saving.md`. PR to dev, then dev to
   master, listing "Closes #N" for each finished issue: commit messages'
   "(#N)" close nothing.
2. When the full suite has passed on the pull request from dev into
   master, merge it (a merge commit) and run the workflow with
   `actions_run_trigger`: `run_workflow`, workflow `release.yml`, ref
   `master`, inputs `{"version": "X.Y", "dry_run": "true"}`. If that
   passes, run it again with `"dry_run": "false"`. Then check the run, the
   tag, the GitHub Release and `https://pypi.org/pypi/repyability/X.Y/json`.

## surpyval workarounds to remove

surpyval 0.24 (the minimum) gives a mixture a precise tail (`sf`, #671) and
has `conditional_gaps` read a limited failure population's `lfp_p`, not a
mixture's `p` (#626); 0.23 pickles its fits (#573), names the
limited-failure proportion `lfp_p` (#608) and takes `success_run`'s
`alpha_ci` (#580). RePyability uses these directly. What remains:

- **A mixture's quantile** (surpyval #821). surpyval 0.24's `MixtureModel`
  has a `qf` (#651), but it inverts `F` in the upper tail, losing the long
  lives' precision (about 1e-4), and takes some 200 times as long as
  RePyability's. `_sampling.mixture_quantile` inverts the distribution
  (with its components' `sf` summed above the median), which
  `stream_sampler` gives a `RepairableRBD`'s streams (a change to it
  changes seeded results). Take the mixture's own `qf` once the minimum
  surpyval's is as precise and fast.
- **A mixture's life left at a virtual age** (surpyval #857, #295:
  `conditional_gaps` inverts `H` past its quantile's limit in 12–15
  evaluations where Newton's steps take 3–4, and `MixtureModel.Hf` costs
  about 100 µs a call). `MixtureLife.aged`
  draws it for `_aged_life` by Newton's steps on the mixture's `H`,
  worked out from its components, from a table of `H` (a change to it
  changes seeded results in the last bits). Take `conditional_gaps`
  again, and drop `MixtureLife`, once the minimum surpyval has both.
- **A degradation process's level given survival** (surpyval #836).
  Replacement on condition by level (#271) needs the level at the next
  inspection of a unit that has not failed by then, which surpyval's
  `WienerProcessModel` and `GammaProcessModel` do not give.
  `_degradation.level_after` works it out from the fitted parameters (the
  method of images for a Wiener path; a gamma increment truncated at the
  threshold), and `_condition_next` draws it from the `LEVEL` stream (a
  change to it changes seeded results). Take surpyval's once the minimum
  surpyval has it.

- **Rescaling a model's time for a duty cycle** (surpyval #845).
  A spec's `"duty"` puts a life in operating time on the calendar,
  `R(d t)`, by `_duty._SCALED`, one rule per surpyval distribution (a new
  one needs its rule there, which `test_units` checks against `R(d t)`).
  Take surpyval's rescaling once the minimum surpyval has it.

List each new workaround here with its surpyval issue and where it lives,
so it can go once the minimum surpyval in `pyproject.toml` includes the
fix.
