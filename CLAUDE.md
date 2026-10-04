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
  gets. `.github/workflows/upstream.yml` runs the tests against surpyval's
  `develop`; when it fails, fix RePyability (working with both the released
  surpyval and `develop`) and release that before surpyval releases. CI's
  `test (minimum surpyval)` job tests the oldest surpyval `pyproject.toml`
  allows; raise that minimum, rather than keep code for older versions,
  once RePyability needs what a newer surpyval does.

## Simulation engines and seeded results

- **A `RepairableRBD` simulation has two engines that must agree to the last
  bit**: the Python event loop (`RepairableRBD._replicate`) and the compiled
  one (`repyability/rbd/_kernel.py`, numba, the optional `fast` extra). A
  change to the loop's events, arithmetic or order goes into both;
  `test_simulation_engines.py` checks them against each other (run by CI's
  `test (with numba, ...)` jobs) and the Python loop against a reference
  written from the streams' definition. What the compiled engine does not
  simulate, `_compiled.unsupported` sends to Python: numba's own loop takes
  `numba=True`, as it also runs maintenance, inspections, repair crews,
  standby groups, nested RBDs and capacities (#155), while engines from
  other packages keep the plain-components contract. Inside `_kernel`, the
  system's own events (`_simulate`) and a nested RBD's (`_advance`, which
  copies `RepairableRBD.next_event`) are written out separately, for speed:
  a change to one goes into the other too.
- **Simulations take turns across threads (#216).** The event loop keeps
  a run's state on the diagram (`_RUN_STATE`) and draws that cannot be
  streamed come from numpy's global RNG, so a run holds
  `repyability.utils.wrappers.SIMULATIONS`, a process-wide `RLock`:
  `RepairableRBD._run` (but a sharded run's parent, whose shards take it
  where they run, so a `shard_map` on threads cannot wait on it),
  `numpy_seed`, and `_timeline_runs._looped`. New code that runs the loop
  or seeds the global RNG goes through one of these; threads that work
  for a run (numba's, the timelines' stream draws) must not take it.
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
  engine.** Both loops record them as they run (`_replicate` with
  `_Context.history`; `_kernel._simulate` when given room to record, the
  system's own level only, so `_advance` records nothing): each top-level
  component's changes, and each of the system's with the component that
  made it. On the Python engine, plain units' histories are drawn from
  their streams instead (`repyability/rbd/_timeline_runs.py`), added up
  as the loop adds them, and a simulation with changes of different
  components at one instant is run in the loop. A change to what the loops
  record goes into both; one to how the loop draws or adds up a plain
  unit's lives and repairs goes into `_timeline_runs._unit` too, and
  anything new that couples components into `independent`:
  `test_timelines.py` checks every engine's histories against each other
  and against `availability`.
- **Capacity states are worked out in batches**
  (`_CapacityRecorder.evaluate`): a simulation's in Python, a batch's
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
- **A conditional run (#189, `repyability/rbd/_conditional.py`) simulates
  the modules alone** (`_conditional_modules`: the nodes
  `_node_over_time` refuses, with their maintenance groups), as a diagram
  of their own (`_modules_rbd`) whose streams are named as in the system,
  so its simulations are a plain run's; its tally keeps each simulation's
  cost beside its histories, on both engines. Given each joint state of
  the modules, the rest comes from `_window` and `point_availability`
  with the modules held. A change to what makes a node need simulating
  goes into `_node_over_time`, and anything that ties a module to the
  other nodes (crews, a group's `system_down`) must refuse in
  `_conditional_modules`. Changes at one instant are ordered as the loop
  orders them (`_conditional.paths`: failures, the stops they open, then
  restorations; a module's before the other nodes'): a change to that
  order in the loop goes into `paths` too. `test_conditional.py` checks
  each simulation's values against closed forms from its module's
  history, and the estimates against plain runs.
- **A run's means are exact or conditional by default (#187, #189).**
  `availability()` and `cost()` take the exact methods' expected values
  where they work them out (`_exact_means`, as the controls of the system
  itself), and otherwise, where a conditional run applies, each
  simulation's expected values given its modules' histories
  (`_conditioned_run`, the modules simulated again from the run's
  entropy); `availability_from_chunks` gives merged chunks the same
  (`_default_means`). The simulations are a plain run's either way. A test
  that checks the simulation against the exact methods must run plainly
  (`control_variate=False`), or it compares the exact values with
  themselves.
- **A run's changes are put in time order on two paths that must agree to
  the last bit** (#201): numpy's (`_by_time`, `_group_totals`,
  `_capacity_totals` and `_working_over_time` in `repairable_rbd.py`) and
  the compiled one (`repyability/rbd/_time_order.py`: a stable radix sort
  of the times' bits, the groups and the capacity's merge in one pass
  each), taken where numba is installed and a run has `_COMPILED_ORDER`
  changes or more. A change to how either orders, groups or adds up the
  changes goes into both: `test_time_order.py` checks them against each
  other, with the sort's blocks of every size.
- **A core's decision diagram is built and replayed on two paths that
  must agree step for step** (#202): `bdd._build` and, where numba is
  installed and the core's search may be long, `_bdd_kernel.build`, whose
  states `bdd._compiled_build` packs into integers; and a plan's
  probabilities and gradient by `modular.Decomposition._core_value` and
  `_core_gradient` or `_bdd_kernel.replay` and `value_and_gradient`. A
  change to the search (its states, its order, the steps it counts
  against `STEP_LIMIT`) or to the replay's arithmetic goes into both:
  `test_bdd_compiled.py` checks them against each other.
- **The random streams (`repyability/rbd/_streams.py`) define every seeded
  result.** Changing how a stream is named, seeded or laid out (its width,
  `BLOCK_DRAWS`, `MAX_WIDTH`, `first_rows`, the expected draws in
  `_expected_draws`) changes seeded results: that is a behaviour change for
  the CHANGELOG, `seeded_event_loop.json` must be re-recorded, and the docs'
  quoted numbers updated. The rows of a chunk only affect speed.
- **The repair crews' Markov chain (`repyability/rbd/_crew_chain.py`)
  copies the simulation's queue (`_Crews`)**: which waiting job a free crew
  takes, and how instant jobs pass through. A change to one goes into the
  other; `test_crew_chain.py` checks the chain's exact values against the
  simulation. Likewise a standby group's chain (`_standby_chain.py`) copies
  `_StandbyGroup`'s rules (switching, spares, repairs), checked by
  `test_repairable_standby.py`.
- **A common-cause group's chains (`repyability/rbd/_ccf_chain.py`) copy
  the simulation's causes** (`_Cause`, `_strike`, #158): each cause, a
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
  (`_inspected_follow_up`, `_inspected_next`, `_strike`): a working member
  is off line for its test, unaged and not struck; a failure found is
  repaired once the test is over; a test in a member's own test or repair
  is not done. A change to one goes into the other: `test_ccf_timed.py`
  checks the chain against the simulation, and against the members' own
  model (`_hidden_tests`) where no cause is shared.
- **A tested unit's numerical model (`repyability/rbd/_hidden_tests.py`,
  #159) copies the simulation's inspections** (`_inspected_follow_up`,
  `_inspected_next`): a test takes a working unit off line without ageing
  it, a failure is repaired once its test is over, the tests in a repair
  are not done, and a failure a test misses waits for the next full test.
  A change to one goes into the other; `test_hidden_failures_timed.py`
  checks the model against the simulation.

## How each analysis is computed

- **A `RepairableRBD`'s junctions are folded out of its structure.** A node
  given `PerfectReliability` is no component: `RBD._decomposition()` folds
  it in as always working (`modular.fold`), so whatever evaluates the
  structure (the curves, the long-run values, both simulation engines, the
  timelines, the path and cut sets) never sees it. Evaluate the structure
  through `_decomposition()`, not the graph, so that this holds; only the
  capacity analysis's reduced diagram (`flow`) keeps the junctions, and
  `_capacity_arrays` passes them as working. `test_junctions.py` checks
  every public method against the same system drawn without a junction.

- **`analysis_routes()` (both RBD classes) must agree with the methods.** It
  says, without running anything, whether each public analysis is exact,
  numerical, simulated or refused. Refusals go through checks the report
  calls too (`_require_*` helpers, `_inspected_rate`, ...), so its reasons
  are the methods' own messages. When a method is added, or gains a refusal
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
  that needs none (a node's own values) goes in `free`. The compiled engine and the timelines'
  streams (`_compiled.unsupported`, `_timeline_runs.independent`) leave such
  a diagram to the Python loop. `test_meshed_structures.py` checks the
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
  combination (`_with_ccf_groups`, which the capacity distribution and the
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

## API conventions

- **One name for the number of simulations**: `mc_samples`, and `max_samples`
  for its cap in a run to a `tolerance`, in every method and constructor
  that simulates; `seed` seeds it (#105). The old names (`N`, `max_N`,
  `n_sims`, `n_simulations`) went in 0.12. Use these names in new code.
- **A deprecation gives one minor release's notice.** It warns in one
  minor release and the next removes it, with a `FutureWarning` (always
  shown) through `repyability/utils/deprecation.py`. What 0.11 deprecated
  went in 0.12 (#149), and `test_removed_in_0_12.py` keeps it gone: the
  old names and ignored arguments raise `TypeError`, a diagram refuses a
  non-parametric node, and a standby or load-sharing model with no exact
  or numerical reliability refuses one (`is_simulated`), where a fit to
  simulated lifetimes stood in, and is left to the simulations. What 0.12
  deprecates goes in 0.13 (`NEXT_REMOVAL`): calling
  `SparesDemand.mean()`/`std()`, now properties (#184, through
  `deprecation.called`), and those models' `mc_samples`, `lower` and
  `seed`, which set the fit (through `deprecation.ignored`).
  `test_the_calls_go_in_the_release_after_next` fails once the version
  reaches it. What 0.13 deprecates goes in 0.14 (`REMOVAL_AFTER_NEXT`):
  `optimal_inspection_intervals(offsets=)`, renamed `offset_shares=`
  (#222, through `deprecation.renamed`), which
  `test_what_0_13_deprecates_goes_in_0_14` holds to.
- **Exact by default, simulation on request.** Where an analysis can be
  computed exactly or numerically, that is the default, and the Monte-Carlo
  estimate is a `method="simulate"` away (as for `NonRepairableRBD.mean`).

## Performance

- **In a loop, a dot product of long vectors goes through
  `repyability.utils.vectors.dot`, not `@`.** OpenBLAS threads one of more
  than about 10,000 terms, at milliseconds a call; a loop of thousands of
  them (a renewal sum, a band of a grid) then runs hundreds of times
  slower. Matrix products are left to BLAS.

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
2. When CI has passed on master's merge commit, run the workflow with
   `actions_run_trigger`: `run_workflow`, workflow `release.yml`, ref
   `master`, inputs `{"version": "X.Y", "dry_run": "true"}`. If that
   passes, run it again with `"dry_run": "false"`. Then check the run, the
   tag, the GitHub Release and `https://pypi.org/pypi/repyability/X.Y/json`.

## surpyval workarounds to remove

- **Fitted models in worker processes** (surpyval #573). A surpyval fit
  holds a closure, so pickle cannot take it. `_montecarlo.dumps`, which
  pickles a run for `n_jobs`' worker processes, sends a surpyval model that
  pickle refuses in its saved form (`to_dict`), rebuilt in the worker;
  `test_fitted_models_in_parallel.py` checks the results are a single
  process's. Remove the override once the minimum surpyval's fits pickle
  (keep `dumps`' message for what still cannot be sent).

- **`success_run`'s `alpha_ci`** (surpyval #580). surpyval 0.23 names
  the bound's level `alpha_ci = 1 - confidence` and deprecates
  `confidence`, which 0.22 needs: `test_demonstration.py`'s `_success_run`
  passes the name the installed surpyval takes. Pass `alpha_ci` directly
  once the minimum surpyval is 0.23.

- **The limited-failure proportion's name** (surpyval #608). surpyval
  0.23 renames `p` (the `from_params` argument, the attribute and the
  `extras` key) to `lfp_p`, which 0.22 does not know: `_model_utils.lfp_p`
  reads either, `lfp_extras` gives `from_params`' keyword (the tests and
  files saved before 0.10 use it), `_PLAIN` holds both keys, and
  `NonRepairable.mean_availability`'s example is skipped. Read and pass
  `lfp_p` directly, and run the example, once the minimum surpyval is 0.23.

- **A mixture's quantile, `p` and tail** (surpyval #651, #626, #671).
  surpyval's `MixtureModel` has no `qf` (#651), keeps its EM
  responsibilities as `p` (#626), which surpyval's `conditional_gaps`
  takes for a limited failure population's, and has an `sf` of `1 - ff`
  (#671), imprecise in the tail. `_sampling.mixture_quantile` inverts its
  distribution (with its components' `sf` summed above the median), which
  `stream_sampler` gives a `RepairableRBD`'s streams (a change to it
  changes seeded results), and `MixtureLife` hands `conditional_gaps` its
  `Hf` and that `qf` (`_aged_life`). Take the mixture's own `qf`, and drop
  `MixtureLife`, once the minimum surpyval has them.

List each new workaround here with its surpyval issue and where it lives,
so it can go once the minimum surpyval in `pyproject.toml` includes the
fix.
