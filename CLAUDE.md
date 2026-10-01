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
  simulate, `_compiled.unsupported` sends to Python.
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

- **`analysis_routes()` (both RBD classes) must agree with the methods.** It
  says, without running anything, whether each public analysis is exact,
  numerical, simulated or refused. Refusals go through checks the report
  calls too (`_require_*` helpers, `_inspected_rate`, ...), so its reasons
  are the methods' own messages. When a method is added, or gains a refusal
  or changes how it computes, update `analysis_routes`:
  `test_analysis_routes.py` checks that it covers every public method, that
  each method does what it says on diagrams of every kind, and that the
  saving guide's table agrees with it.

## API conventions

- **One name for the number of simulations**: `mc_samples`, and `max_samples`
  for its cap in a run to a `tolerance`, in every method and constructor
  that simulates; `seed` seeds it (#105). The old names (`N`, `max_N`,
  `n_sims`, `n_simulations`) warn until 1.0, through
  `repyability/utils/deprecation.py`. Use these names in new code.
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

- **Normal and LogNormal quantiles** (surpyval #469). surpyval computes
  them through `scipy.stats.norm.ppf`, whose argument checks cost four
  times the maths. `_DIRECT_QF` in `repyability/rbd/_sampling.py` computes
  the same values through `scipy.special.ndtri` for the simulations;
  `test_direct_quantiles_are_surpyvals` checks that they stay identical.
  Remove it once the minimum surpyval computes them directly.

List each new workaround here with its surpyval issue and where it lives,
so it can go once the minimum surpyval in `pyproject.toml` includes the
fix.
