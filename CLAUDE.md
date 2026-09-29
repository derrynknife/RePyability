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

None at present: surpyval 0.21 fixed every one listed before (#86), and
RePyability requires it. When a new one is needed, list it here with its
surpyval issue and where it lives, so it can go once the minimum surpyval
in `pyproject.toml` includes the fix.
