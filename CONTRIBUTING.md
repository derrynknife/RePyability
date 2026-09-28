# Contributing to RePyability

Thanks for your interest in improving RePyability! This guide covers setting up
a development environment and the checks your change needs to pass.

## Development setup

Use a virtual environment, then install the package with its dev tooling
(pinned in `pyproject.toml`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

This installs RePyability in editable mode plus `black`, `isort`, `flake8`,
`mypy`, `pytest`, `coverage`, and `pre-commit`.

## Pre-commit hooks

Install the git hooks so the formatters and linters run automatically on
commit:

```bash
pre-commit install
pre-commit run --all-files   # optional: run against the whole tree once
```

## Running the checks locally

CI runs the same checks; running them before pushing avoids round-trips:

```bash
black --check repyability      # formatting (drop --check to auto-format)
isort --check-only repyability # import ordering (drop --check-only to fix)
flake8 repyability             # linting
mypy repyability               # static type checks
coverage run -m pytest         # tests
coverage report                # enforces the coverage fail_under gate
```

## Tests

- Every behavioural change should come with a test. Prefer asserting against a
  **closed-form or textbook reference value** (as the existing suite does)
  rather than only self-consistency.
- Monte-Carlo tests must be deterministic — pass a `seed=` to the simulation
  entry points.
- Keep the library coverage above the `fail_under` gate in `pyproject.toml`.

## Pull requests

1. Branch off `master`.
2. Keep changes focused; update `CHANGELOG.md` under `[Unreleased]`.
3. Make sure all checks above pass.
4. Open a PR describing the change and its motivation.

## Releases (maintainers)

Versioning follows [SemVer](https://semver.org/). To release:

1. On a branch, bump `repyability/_version.py` and roll the `CHANGELOG.md`
   `[Unreleased]` section into a dated `## [X.Y.Z] - YYYY-MM-DD` section. Its
   opening paragraphs become the summary at the top of the release notes.
   Update any version shown in the docs, then merge to `dev`, and `dev` to
   `master`.
2. Once CI has passed on master, run the `release` workflow on master with
   the version: Actions → release → Run workflow, or
   `gh workflow run release.yml --ref master -f version=X.Y.Z`. Add
   `-f dry_run=true` to check and build without publishing. The workflow
   checks the version, the CHANGELOG section, that the tag is new, that CI
   passed and that PyPI does not have the version. Then it builds, publishes
   to PyPI via trusted publishing, and creates the tag `vX.Y.Z` and a GitHub
   Release.

Publishing a GitHub Release by hand for a new tag also works: the workflow
then builds that tag and publishes it to PyPI.
