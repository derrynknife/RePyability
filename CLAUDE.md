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

- **A `RepairableRBD` simulation has three engines that must agree to the
  last bit**: the Python event loop (`RepairableRBD._replicate`) and two
  compiled ones, numba (`repyability/rbd/_kernel.py`, the optional `fast`
  extra) and Mojo (`repyability/rbd/_mojo_kernel/kernel.mojo`, driven by
  `_mojo.py`, the optional `mojo` extra; `engine="auto"` prefers it). A
  change to the loop's events, arithmetic or order goes into all three;
  `test_simulation_engines.py` checks the compiled ones against Python (run
  by CI's `test (with numba, ...)` and `test (with Mojo, ...)` jobs) and the
  Python loop against a reference written from the streams' definition.
  What the compiled engines do not simulate, `_compiled.unsupported` sends
  to Python.
- **The Mojo kernel reads its arrays through two tables of addresses and
  sizes**, laid out by `_mojo.ADDRESSES` and `_mojo.SIZES`; the kernel's
  `A_`/`S_` constants must match them (a test checks). It keeps whether the
  system works up to date as components change (per-term counts of working
  members, and per-path-set counts of members down for a core) rather than
  evaluating the structure at each event, and reads draws in place from the
  streams' blocks. It is compiled from source on first use and cached by a
  hash of the source.
- **Mojo releases reach users at once, like surpyval's**: the `mojo` extra
  has no upper bound and Mojo's syntax still changes between releases.
  `upstream.yml` compiles the kernel with the newest Mojo daily; when it
  fails, make the kernel compile with both the minimum and the newest Mojo,
  or raise the minimum in `pyproject.toml`. Meanwhile `engine="auto"` falls
  back to numba or Python with a warning.
- **The random streams (`repyability/rbd/_streams.py`) define every seeded
  result.** Changing how a stream is named, seeded or laid out (its width,
  `BLOCK_DRAWS`, `MAX_WIDTH`, `first_rows`, the expected draws in
  `_expected_draws`) changes seeded results: that is a behaviour change for
  the CHANGELOG, `seeded_event_loop.json` must be re-recorded, and the docs'
  quoted numbers updated. The rows of a chunk only affect speed.

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

## Releasing

Releases are cut from master by `.github/workflows/release.yml`, which this
session can run: it cannot push tags or create GitHub Releases itself. Every
merge and release needs the maintainer's go-ahead.

1. On the working branch, bump `repyability/_version.py`. Roll the CHANGELOG
   `[Unreleased]` section into `## [X.Y.Z] - YYYY-MM-DD`, opening with a
   summary paragraph and the behaviour changes (they open the release notes),
   and update the version in `docs/guide/saving.md`. PR to dev, then dev to
   master, listing "Closes #N" for each finished issue: commit messages'
   "(#N)" close nothing.
2. When CI has passed on master's merge commit, run the workflow with
   `actions_run_trigger`: `run_workflow`, workflow `release.yml`, ref
   `master`, inputs `{"version": "X.Y.Z", "dry_run": "true"}`. If that
   passes, run it again with `"dry_run": "false"`. Then check the run, the
   tag, the GitHub Release and `https://pypi.org/pypi/repyability/X.Y.Z/json`.

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
