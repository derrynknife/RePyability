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
  uncertainty intervals on the reliability, MTTF, B*X* life and time to a
  reliability from uncertain (fitted) component models.
- **Networks**: undirected networks whose links fail: the exact
  reliability of the connection between two terminals, by a decision
  diagram that keeps meshed networks fast.
- **Phased missions**: missions through phases (take-off, cruise,
  landing), each with its own diagram over the same components: the exact
  mission reliability and the chance of failing in each phase.
- **Testing**: demonstration test plans (the units, or the test time, that
  demonstrate a reliability or an MTBF), what a test demonstrated, and the
  chance a design passes.
- **Importance**: Birnbaum, improvement potential, RAW, RRW, criticality,
  Fussell–Vesely, structural importance and parameter sensitivity, on
  repairable systems in the long run or over time.
- **Sensitivity, the Greeks**: how far the system moves with each
  component (delta) and with each lever, a life, a repair, the maintenance
  or a crew; the shares of a change, which add up (differential
  importance); whether two improvements are complements or substitutes
  (gamma); which component is moving the availability now, and which
  caused the failures (theta, Barlow–Proschan); and whose parameter
  uncertainty widens the answer (vega).
- **Live state**: reliability, remaining life and importance given each
  component's current age, and covariate-dependent components.
- **Redundancy and dependence**: cold, warm and hot standby; repeated nodes;
  load sharing; beta-factor and MGL common-cause groups.
- **Repairable systems**: exact long-run availability, failure frequency and
  MUT/MDT/MTBF; exact availability over time and over a mission; simulated
  histories with criticality measures; shared repair crews, exact for
  exponential components in the long run and numerical over time, with
  their importance; repairable standby groups (a duty unit and its spares,
  repaired one at a time); and imperfect repair
  (Kijima's virtual age), with replacement at the N-th failure.
- **Capacity**: how much a system delivers, from its components'
  capacities (with several levels, or degrading through stages): the exact
  distribution of its capacity at a time or in the long run, the
  probability of meeting a demand, and the production availability.
- **Spares**: how many spares each component uses over a horizon, for a
  system or a fleet, and the stock that meets a fill rate or a stock-out
  target for a replenishment lead time.
- **Timelines**: up/down histories, from outage logs or simulated, with
  their measures (time up, failures and who caused them, first failure);
  merged as a diagram's structure, so a system's history follows from its
  components', with the component behind each of its outages.
- **Simulation**: seeded Monte-Carlo run to a tolerance, antithetic pairs,
  control variates from the system's exact twin, parallel runs, runs split
  across machines and merged, or sharded through any executor (Ray, Dask,
  a batch system), comparisons of
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

## When is a simulation needed?

RePyability computes exactly by default: by closed forms, or by
deterministic numerical methods (quadrature, convolution, renewal equations)
that give the same answer every run, to a small stated error. It simulates
only when

- the answer is itself random: sampled lifetimes or histories, or the
  spread of an outcome over a window (a cost's percentiles, how many
  failures);
- components depend on each other in a way no exact method solves in
  general; or
- an exact method could exist but has not been written yet.

**Exact or numerical, in any architecture** (series, parallel,
*k*-out-of-*n*, bridges and other meshed diagrams, components in several
places, nested diagrams):

- **components:** surpyval lifetime distributions and fixed
  probabilities (a diagram refuses a non-parametric fit, such as
  Kaplan–Meier: fit a parametric distribution in surpyval); repeated
  nodes; cold standby with one or
  two units operating (any units) or more (identical units); warm standby
  with one unit operating, and hot standby (any units); load sharing of
  identical units; common-cause groups, for the system's reliability,
  importance, sensitivity and redundancy allocation (and, splitting the
  failure rate, its MTTF);
- **non-repairable questions:** reliability, hazard, MTTF, B*X* life and
  importance measures at any time, also given each component's current age;
  the distribution of the system's capacity;
- **repairable questions,** for independent components repaired when they
  fail, replaced on age or block schedules or on condition at inspections,
  or tested for hidden failures (any life, the tests and repairs instant or
  taking time, the tests finding every failure or missing some): the
  long-run availability,
  failure frequency, MUT/MDT/MTBF and cost rate; the availability over time
  and over a mission, and the expected failures, outages, downtime and cost
  over a window, from new or from the components' current states (their
  ages, the repairs going on, where they are in their calendars, or the
  long run); the capacity distribution in the long run and over time, with
  the production availability of a window; and the spares used over a
  horizon, and the stock to hold for a lead time (under block replacement,
  with repairs and block replacements in no time). Common-cause groups of
  members with exponential lives, tested or repaired, take nothing away:
  the long-run values, importance and allocations stay exact, and the
  values over time from new numerical.

**Simulated**, or refused with the simulation to run instead:

| Situation | Today | Must it be simulated? |
|---|---|---|
| **What you ask** | | |
| Sampled lifetimes or histories, and distributions or percentiles of an outcome over a window | Simulated (from new, or from the components' current states); each simulation's histories, the system's and its components', kept whole as timelines by `simulate_timelines` (from new) | Yes: the answer is a sample. Its mean over a window (failures, outages, downtime, cost, the capacity delivered) is exact, from new or from a state: `expected_events`, `expected_cost`, `mission_capacity`. |
| Comparing two designs (`compare`) | Simulated, with common random numbers | No, where both are exact: compare their exact values. |
| The uncertainty from fitted component parameters (`sf_uncertainty`, `mean_uncertainty`, `bx_life_uncertainty`, `time_to_reliability_uncertainty`) | Sampled over the parameters, each draw exact | Sampling is the method. |
| Small failure probabilities, with a node only simulations take | Rare-event simulation (`unreliability_interval`); `ff` refused | Only while the node has no reliability of its own: an exact diagram gives `ff` directly, to full precision however small (a numerical node, such as a cold-standby group of non-exponential units, to its own accuracy, about 1e-6). |
| **Components** | | |
| Warm standby with two or more units operating, of non-exponential units | Simulated in the system's simulations; the analyses that need the group's reliability refused | Yes, in general: which spare is switched in where, and how much each has aged, branch with the order of the failures. With one unit operating it is numerical, and hot standby (k-out-of-*n*) exact. |
| Cold standby with three or more different units operating, and load sharing of different units | Simulated in the system's simulations; the analyses that need the group's reliability refused | Yes, in general: which spare goes where, and how old the others are, branch with the order of the failures. Identical units are numerical in both, and so are two different units operating. |
| Anything such a node is part of | Simulated in the system's simulations; the analyses that need the node's reliability refused | Only while the node has no reliability of its own. |
| Common-cause groups: analyses given ages, and the MTTF of a group splitting a failure probability | Refused (a simulated MTTF leaves a probability split out) | No: the analyses given ages need a model of members of different ages. A group splitting the failure rate (`basis="rate"`) has an exact MTTF, and the simulations draw its shared shocks. Importance, parameter sensitivity and redundancy allocation are exact with groups (a beta-factor member's copies join its group), and parameter uncertainty is sampled with them. |
| **Architecture and maintenance** | | |
| Phased missions and networks too large for their decision diagrams | Refused, pointing to `method="simulate"` | Only in practice: the diagrams grow with the phases' and the network's width rather than their paths, so meshed missions and networks are exact (a network grid of 100 nodes in a second and a half); one of 121 nodes passes the limit, `repyability.network.MAX_STATES`, which can be raised. |
| Block diagrams too meshed for their decision diagrams | Simulated (lifetimes, availability, cost and timelines, in Python); the exact and numerical analyses refused | Only in practice: the diagram grows with how wide the mesh is rather than with its paths, so most meshes are exact (a 10 × 10 grid in 0.04 seconds, a random mesh of 60 nodes and 345 links in 2); one of 70 nodes and 485 links passes the limit, `repyability.rbd.bdd.STEP_LIMIT`, which can be raised. |
| Common-cause groups in a repairable diagram | For exponential lives, tested or repaired: the long run, importance and the allocations exact (a beta-factor member's copies join its group, and the availability allocations keep the members' availability), the values over time from new numerical (the groups' Markov chains), and the simulations draw the shared causes, in Python. Members of other lives refused, and members held or started from a current state | Other lives: a shared cause has no one rate for members of different ages, so they need a model first. From a current state, no: the chains and the simulations could start from the members' states. |
| Shared repair crews | For exponential lives and repairs, the long run and importance exact, and the values over time numerical (the same Markov chain, followed by uniformization), but the allocations; other lives simulated. The maintenance and test intervals are chosen as if every repair started at once on request (`assume_unlimited_crews=True`), for a plan to simulate with the crews | Other lives: yes, in general. |
| Standby groups (a duty unit and its spares, repaired) | For exponential units, the long run and importance exact, and the values over time numerical (the units' Markov chain, followed by uniformization); other units simulated | Other units: yes, in general. |
| Opportunistic maintenance (renewals at a group's stops) | Simulated | Yes: each member's renewals depend on the others' ages. |
| Imperfect repair (Kijima), with or without replacement at the *N*-th failure | Simulated; but minimal repair (`q = 1`) in no time is numerical over a window from new (it fails `H(t)` times by `t`), its long run refused | Yes, in general: a repair does not renew the unit. |
| The stock of block-replaced components whose repairs or block replacements take time | Refused; their demand over a horizon is numerical (and simulated with `spares_demand(method="simulate")`). The stock of block-replaced ones repaired and replaced in no time is numerical, and so are tested components' spares, whatever their tests and repairs take and whether their tests miss failures | No: the work carried over a block time could be numerical (#160). With crews, standby groups, opportunistic maintenance or imperfect repair: yes. |

Where a system needs simulating for a few of its nodes (a standby group of
other lives, a nested RBD sharing a crew, a unit repaired imperfectly),
`availability(conditional=True)` and `cost(conditional=True)` simulate only
those, and take every other node exactly given their states: the window's
expected values, with a fraction of the error of a plain run.

For your own diagram, `analysis_routes()` says how each analysis will be
computed (exact, numerical, simulated or refused) and why, without running
anything; a refusal's message names the simulation to run instead. The
[guide](https://derrynknife.github.io/RePyability/guide/saving/#what-is-exact-and-what-is-simulated)
lists every method's route.

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
