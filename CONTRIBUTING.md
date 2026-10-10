# Contributing to RePyability

Thanks for your interest in improving RePyability! This guide covers setting
up, the checks a change needs, and the rules the code keeps to: several
parts of the package must agree with each other, and the tests that hold
them together say which.

## Development setup

Use a virtual environment, then install the package with its dev tooling
(pinned in `pyproject.toml`) and, to run the compiled simulation engine, the
`fast` extra (numba):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .[dev]              # add ,fast for numba, ,docs for mkdocs
pre-commit install                 # run the formatters and linters on commit
```

RePyability needs Python 3.11 or later and surpyval 0.24 or later.

## Branches and the one automated test run

Work branches off **`dev`** and goes back into `dev` by pull request.
Releases go from `dev` into `master`.

There is **one automated test run**: the full suite on the pull request from
`dev` into `master` (`.github/workflows/actions.yml`). Nothing runs on pushes
or on pull requests into `dev`, so check your change locally before it goes
in:

```bash
scripts/check.sh                                    # black, isort, flake8, mypy, as CI runs them
scripts/check.sh repyability/tests/test_units.py    # the same, then those tests on every core
```

Run the tests of what you changed: the files that exercise the code you
touched, and new tests for it. The whole suite has over 6,000 tests and takes
a long time even on every core (`python -m pytest -n auto`); it runs on the
release pull request.

## Tests

- Every behavioural change comes with a test. Assert against a
  **closed-form or textbook reference value** where there is one, not only
  self-consistency.
- Monte-Carlo tests are deterministic: pass a `seed=`.
- Keep the library's coverage above the `fail_under` gate in
  `pyproject.toml`.
- A test that asks for the compiled engine (`engine="numba"`) skips without
  numba (`pytest.importorskip("numba")`). With numba installed, also check
  such tests with it hidden, as CI's plain jobs have none.
- The documentation's code is run too: every ```` ```python ```` block in
  `docs/` and the README runs in order, and a value quoted after `# ->` is
  checked against what the line returns (`test_docs_examples.py`).

## What must agree

Several computations exist twice, by design, and a test compares them. A
change to one goes into the other:

- **The two simulation engines**: the Python event loop
  (`RepairableRBD._replicate`) and the compiled one (`rbd/_kernel.py`) agree
  to the last bit (`test_simulation_engines.py`, run with numba).
- **The Markov chains that copy the simulation's rules**: repair crews
  (`_crew_chain.py`), standby groups (`_standby_chain.py`), common-cause
  groups (`_ccf_chain.py`) and tested units (`_hidden_tests.py`), each
  checked against the simulation by its own test.
- **`analysis_routes()`**, which says without running anything whether each
  public method is exact, numerical, simulated or refused, agrees with the
  methods on every diagram of `tests/catalogue.py`
  (`test_analysis_routes.py`). A new method, refusal, node class, spec key
  or option needs its route, and a catalogue diagram that uses it
  (`test_the_catalogue_has_every_kind` names what no diagram uses). The
  README's "When is a simulation needed?" table follows the routes too.
- **Seeded results** are defined by the random streams (`rbd/_streams.py`).
  Changing a stream's name or definition changes them: re-record
  `tests/seeded_event_loop.json`, update the docs' quoted numbers, and say
  so in the CHANGELOG.

The maintainer's notes in `CLAUDE.md` list these and the other conventions
(input checks, warnings, result objects, levers) in full.

## surpyval

RePyability consumes fitted surpyval models; fitting stays in surpyval. When
a surpyval model misbehaves, raise an issue in
[derrynknife/SurPyval](https://github.com/derrynknife/SurPyval), keep any
workaround here small, and list it with its issue in `CLAUDE.md`, so it can
go once the minimum surpyval version has the fix.

## Pull requests

1. Branch off `dev`.
2. Keep changes focused, and update `CHANGELOG.md` under `[Unreleased]`.
3. Run `scripts/check.sh` with the tests of what you changed.
4. Open a pull request into `dev` describing the change and its motivation.

## Deprecations

A deprecation warns (a `FutureWarning`, through
`repyability/utils/deprecation.py`) for one minor release, and the next
removes it. `test_what_0_13_deprecates_goes_in_0_14` and its kind fail once
the version reaches the removal, as a reminder.

## Releases (maintainers)

Versions have two parts, major.minor (e.g. 0.11): from 1.0, a release
that breaks compatibility raises the major number, and any other
release, fixes included, the minor. There are no patch releases. To
release:

1. On a branch, bump `repyability/_version.py` and roll the `CHANGELOG.md`
   `[Unreleased]` section into a dated `## [X.Y] - YYYY-MM-DD` section. Its
   opening paragraphs become the summary at the top of the release notes.
   Update the version in `docs/guide/saving.md`. Merge to `dev`, then open
   the pull request from `dev` into `master`, which runs the full suite.
2. Once it has passed, merge it and run the `release` workflow on master
   with the version: Actions → release → Run workflow, or
   `gh workflow run release.yml --ref master -f version=X.Y`. Add
   `-f dry_run=true` to check and build without publishing. The workflow
   checks the version, the CHANGELOG section, that the tag is new, that CI
   passed and that PyPI does not have the version. Then it builds, publishes
   to PyPI via trusted publishing, and creates the tag `vX.Y` and a GitHub
   Release.
