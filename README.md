# RePyability

[![actions](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml/badge.svg)](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml)
[![docs](https://github.com/derrynknife/RePyability/actions/workflows/docs.yml/badge.svg)](https://derrynknife.github.io/RePyability/)

Reliability Engineering Tools

This is a series of tools created to make an open source set of methods to be used by reliability engineers to make it more accessible for students right through to practicing professionals.

RePyability builds and analyses systems as reliability block diagrams (RBDs),
taking already-fitted lifetime models (from
[surpyval](https://github.com/derrynknife/SurPyval) or anything exposing
`sf`/`ff`) as its components:

- **Reliability**: exact system reliability, hazard and conditional survival;
  MTTF with confidence intervals; B*X* life.
- **Importance**: Birnbaum, improvement potential, RAW, RRW, criticality,
  Fussell–Vesely, structural importance and parameter sensitivity.
- **Live state**: reliability, remaining life and importance given each
  component's current age, and covariate-dependent components.
- **Redundancy and dependence**: cold, warm and hot standby; repeated nodes;
  load sharing; beta-factor and MGL common-cause groups.
- **Repairable systems**: exact long-run availability, failure frequency and
  MUT/MDT/MTBF; simulated availability over time with criticality measures.
- **Cost, design and maintenance**: exact and simulated running costs,
  optimal redundancy allocation, and age-replacement and overhaul policies.

```python
import surpyval as surv
from repyability import NonRepairableRBD

rbd = NonRepairableRBD(
    [("s", "pump1"), ("s", "pump2"), ("pump1", "valve"), ("pump2", "valve"), ("valve", "t")],
    {
        "pump1": surv.Weibull.from_params([100, 2]),
        "pump2": surv.Weibull.from_params([100, 2]),
        "valve": surv.Weibull.from_params([200, 1.5]),
    },
)
rbd.sf(50)                     # 0.839: system reliability at t = 50
rbd.birnbaum_importance(50)    # which component matters most
```

## Install
RePyability can be installed via pip using the PyPI [repository](https://pypi.org/project/repyability/)

```bash
pip install repyability
```

## Documentation
The full documentation — tutorial, user guide, concepts and API reference — is
hosted at
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
