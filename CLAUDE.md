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
  surpyval and `develop`) and release that before surpyval releases.

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

## surpyval workarounds to remove (tracked in #86)

Remove each once the pinned minimum surpyval includes its fix; then
`repyability/tests/test_limited_failure_population.py` must still pass
unchanged.

- surpyval#403 (`random()` gives survival data for `p < 1`; zero-inflated
  draws go through a binomial): `rbd/_sampling.py`, `draw`, `_defective` and
  the `qf` branch of `inverse_sampler`.
- surpyval#404 (`mean()` is the defective mean for `p < 1`):
  `rbd/_model_utils.py`, `model_mean` returning inf.
  (`NonRepairable._long_run`, the probability of ending up for good, stays.)
- surpyval#405 (zero-inflated `df(0)` is the point mass):
  `rbd/numerical_convolution.py`, `_continuous_part`.
- surpyval#406 (no rebuild with new parameters keeping `gamma`, `p`, `f0`):
  `rbd/_model_utils.py`, `model_extras` (used by sensitivity and
  uncertainty).
- surpyval#407 (zero-inflated `ff`/`sf` nonzero before time 0): no
  workaround; only a diagram's `sf(t)` at `t < 0` is affected.
- surpyval 0.21's `seed` -> `random_state` rename (surpyval#422): the
  `seed_keyword` shim in `utils/wrappers.py`, used by `Repairable` for
  `mcf` and `count_terminated_simulation`. Pass `random_state=` directly
  once the minimum surpyval is 0.21.
- surpyval#381 (a non-parametric estimate gives a 1-element array for a
  single time, and spreads a 2-D query): `rbd/_model_utils.py`, `shaped`,
  used by the simulated `sf` and `ff` of `StandbyModel` and
  `LoadSharingModel`. The `shape` tests in `test_warm_standby.py` and
  `test_load_sharing.py` must still pass.
