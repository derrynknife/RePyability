# RePyability

[![actions](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml/badge.svg)](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml)
[![docs](https://github.com/derrynknife/RePyability/actions/workflows/docs.yml/badge.svg)](https://derrynknife.github.io/RePyability/)

Reliability Engineering Tools

This is a series of tools created to make an open source set of methods to be used by reliability engineers to make it more accessible for students right through to practicing professionals.

## Install
RePyability can be installed via pip using the PyPI [repository](https://pypi.org/project/repyability/)

```bash
pip install repyability
```

## Documentation
The full documentation — overview, user guide, and API reference — is hosted at
**[derrynknife.github.io/RePyability](https://derrynknife.github.io/RePyability/)**.

It is built with [MkDocs](https://www.mkdocs.org/) from the sources in `docs/`
and `mkdocs.yml`, and published to GitHub Pages on every push to `master`. To
build and serve it locally:

```bash
pip install -e .[docs]
mkdocs serve            # then open http://127.0.0.1:8000
```

The source lives in `docs/` and `mkdocs.yml`; start with `docs/index.md`.

## Testing
Run the testing suite by simply executing:
```bash
pytest
```
or use coverage to get a coverage report:
```bash
coverage run -m pytest  # Run pytest under coverage's watch
coverage report         # Print coverage report
coverage html           # Make a html coverage report (really useful), open htmlcov/index.html
```

## Pre-commit
### TL;DR
- Pip install `pre-commit` (it's in `requirements_dev.txt` anyways)
- Run `pre-commit install` which sets up the git hook scripts
- If you'd like, run `pre-commit run --all-files` to run the hooks on all files
- When you go to commit, it will only proceed after all the hooks succeed

### Why?
To ensure the good code quality and consistency it is recommended that when contributing to this
repository to use the provided `.pre-commit-config.yaml` configuration for the Python package
`pre-commit` (https://pre-commit.com). Upon making a commit, it checks that imports
and requirements are sorted, syntax is up-to-date, code is formatted, linted, and statically type-checked,
all with the same tools and configurations as one another.
