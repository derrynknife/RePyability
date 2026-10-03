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
  saving guide's table agrees with it.
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
  free)`, which refuses every exact or numerical analysis but those in
  `free`, which need no structure: a new method that needs none (a node's
  own values) goes in `free`. The compiled engine and the timelines'
  streams (`_compiled.unsupported`, `_timeline_runs.independent`) leave such
  a diagram to the Python loop. `test_meshed_structures.py` checks the
  stand-in against the structure worked out, and `test_analysis_routes.py`
  the routes of diagrams too meshed.
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
  reaches it.
- **Exact by default, simulation on request.** Where an analysis can be
  computed exactly or numerically, that is the default, and the Monte-Carlo
  estimate is a `method="simulate"` away (as for `NonRepairableRBD.mean`).

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

List each new workaround here with its surpyval issue and where it lives,
so it can go once the minimum surpyval in `pyproject.toml` includes the
fix.
