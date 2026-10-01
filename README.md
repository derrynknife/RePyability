# RePyability

[![actions](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml/badge.svg)](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml)
[![docs](https://github.com/derrynknife/RePyability/actions/workflows/docs.yml/badge.svg)](https://derrynknife.github.io/RePyability/)

Reliability Engineering Tools

This is a series of tools created to make an open source set of methods to be used by reliability engineers to make it more accessible for students right through to practicing professionals.

RePyability builds and analyses systems as reliability block diagrams (RBDs),
taking already-fitted lifetime models (from
[surpyval](https://github.com/derrynknife/SurPyval) or anything exposing
`sf`/`ff`) as its components:

- **Fault trees**: static fault trees (OR, AND and VOTE gates, repeated
  events) evaluated exactly, with cut sets, importance measures and
  conversion to and from block diagrams.
- **Reliability**: exact system reliability, hazard and conditional survival;
  the exact MTTF (or simulated, with confidence intervals); B*X* life;
  uncertainty intervals from uncertain (fitted) component models.
- **Networks**: undirected networks whose links fail: the exact
  reliability of the connection between two terminals.
- **Phased missions**: missions through phases (take-off, cruise,
  landing), each with its own diagram over the same components: the exact
  mission reliability and the chance of failing in each phase.
- **Testing**: demonstration test plans (the units, or the test time, that
  demonstrate a reliability or an MTBF), what a test demonstrated, and the
  chance a design passes.
- **Importance**: Birnbaum, improvement potential, RAW, RRW, criticality,
  Fussell–Vesely, structural importance and parameter sensitivity.
- **Live state**: reliability, remaining life and importance given each
  component's current age, and covariate-dependent components.
- **Redundancy and dependence**: cold, warm and hot standby; repeated nodes;
  load sharing; beta-factor and MGL common-cause groups.
- **Repairable systems**: exact long-run availability, failure frequency and
  MUT/MDT/MTBF; exact availability over time and over a mission; simulated
  histories with criticality measures; shared repair crews, exact in the
  long run for exponential components; repairable standby groups (a duty
  unit and its spares, repaired one at a time); and imperfect repair
  (Kijima's virtual age), with replacement at the N-th failure.
- **Capacity**: how much a system delivers, from its components'
  capacities (with several levels, or degrading through stages): the exact
  distribution of its capacity at a time or in the long run, the
  probability of meeting a demand, and the production availability.
- **Spares**: how many spares each component uses over a horizon, for a
  system or a fleet, and the stock that meets a fill rate or a stock-out
  target for a replenishment lead time.
- **Simulation**: seeded Monte-Carlo run to a tolerance, antithetic pairs,
  parallel runs, runs split across machines and merged, comparisons of
  designs with common random numbers, and small failure probabilities by
  rare-event simulation (subset simulation, cross-entropy importance
  sampling); repairable systems simulated compiled, with numba installed.
- **Cost, design and maintenance**: exact and simulated running costs,
  including scheduled (age or block) preventive replacement at system level,
  replacement on condition at periodic inspections and opportunistic
  maintenance of grouped components at each other's stops,
  with its intervals chosen for a cost or availability target, hidden
  failures found by periodic inspection, with the test intervals chosen for a
  PFDavg target, the total cost of ownership, optimal redundancy allocation
  (for the lowest total cost of a repairable system, too), reliability
  allocation by the classic named methods (equal and ARINC-style
  apportionment, minimum effort, cost-based), availability allocation to
  MTTF and MTTR targets, and age-replacement and overhaul policies.

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

New to reliability engineering? The documentation includes
[Learn](https://derrynknife.github.io/RePyability/learn/), a short course
that teaches system reliability from a single part's lifetime to designing
and maintaining whole systems, working every idea out by hand and then with
RePyability, with exercises and worked answers.

## Install
RePyability can be installed via pip using the PyPI [repository](https://pypi.org/project/repyability/)

```bash
pip install repyability
```

Repairable systems simulate about ten times as fast with the optional
compiled engine, which needs [numba](https://numba.pydata.org) (available for
64-bit Linux, macOS on Apple silicon and Windows):

```bash
pip install "repyability[fast]"
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
