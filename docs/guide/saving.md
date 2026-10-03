# Saving, reproducibility and performance

## Saving and loading

An RBD round-trips to a plain, JSON-friendly structure, so it can be saved,
shared, version-controlled, and loaded wherever the analysis runs:

```python
import surpyval as surv
from surpyval import FixedEventProbability
from repyability import RBD, NonRepairableRBD

rbd = NonRepairableRBD(
    [("s", 1), (1, 2), (2, "t"), ("s", 3), (3, "t")],
    {1: surv.Weibull.from_params([100, 2]),
     2: FixedEventProbability.from_params(0.01),
     3: surv.Weibull.from_params([80, 2])},
)
data = rbd.to_dict()            # a JSON-friendly dict
text = rbd.to_json(indent=2)    # a JSON string (keyword arguments go to json.dumps)

clone = NonRepairableRBD.from_json(text)
clone.sf(30) == rbd.sf(30)      # True
```

As surpyval's models do, `to_json` writes to a file given a path (or a file
opened for writing), and `from_json` reads a path as well as the text:

```python
import tempfile
from pathlib import Path

path = Path(tempfile.mkdtemp()) / "plant.json"
rbd.to_json(path)                       # written; returns None
NonRepairableRBD.from_json(path).sf(30) == rbd.sf(30)   # True
type(RBD.from_dict(data)).__name__   # 'NonRepairableRBD': the base class dispatches on type
data["type"], data["repyability_version"]   # ('NonRepairableRBD', '0.11')
```

What is saved:

- the structure: edges, `k`, input and output nodes, `on_infeasible_rbd`,
  repeated components and nested RBDs (of either kind);
- common-cause groups, and for a `RepairableRBD` every cost, including cost
  distributions and acquisition costs, preventive and inspection schedules,
  maintenance groups and their set-up costs, `"instant"` repairs and
  `NonRepairable` components;
- the node models: surpyval's parametric models (including
  `FixedEventProbability`), in surpyval's own format (`model.to_dict()`, loaded with `surpyval.from_dict`),
  so everything surpyval keeps round-trips: an offset, a
  limited-failure-population or zero-inflation parameter, and a fit's
  covariance, so parameter uncertainty can still be propagated after
  loading; `PerfectReliability` and `PerfectUnreliability`; and the standby,
  repeated, repeated-standby, load-sharing, regression and `NonRepairable`
  wrappers recursively. Files saved before 0.10.0, which stored a
  parametric model by name and parameters, still load.

String, integer and tuple node names all survive JSON (JSON turns a tuple
into a list, and loading turns it back). Loading with the wrong class
(`NonRepairableRBD.from_dict` on a `RepairableRBD` document) raises
`ValueError`; `RBD.from_dict` and `RBD.from_json` always pick the right one.

!!! note "One limit"
    **Models surpyval does not know** (your own class with an `sf`) cannot
    be saved: `to_dict` raises `NotImplementedError`. Standby and
    load-sharing nodes are saved by their inputs (units, `k`,
    `dormancy_factor`, switching, load) and reload exactly, those only a
    simulation works out included: they hold no fitted curve.

Condition-based state (`NodeState`) is not part of the RBD and is not saved.

## Reproducibility

Every Monte-Carlo method takes a `seed`:

| Where | Methods |
|---|---|
| `NonRepairableRBD` | `random`, `mean` and `mean_time_to_failure` with `method="simulate"`, `mean_time_to_failure_interval`, `compare` |
| `RepairableRBD` | `availability`, `cost`, `compare`, `spares_demand` with `method="simulate"` |
| Node models | `RepeatedNode.random`, `RepeatedNode.mean` with `method="simulate"`, `RepeatedStandbyNode.random`, `StandbyModel.random`, `LoadSharingModel.random`, and the `mean(mc_samples=..., seed=...)` of a simulated `StandbyModel` or `LoadSharingModel` |
| `PhasedMission` | `reliability`, `unreliability` and `phase_failure_probabilities` with `method="simulate"`, `reliability_interval` |
| `Network` | `sf`, `ff` and `mean` with `method="simulate"`, `random` |
| `Repairable` | every simulation-backed method |

surpyval samples from numpy's **global** random number generator, so a seed
is applied to it for the duration of the call and the caller's generator
state is restored afterwards. Seeded calls are reproducible without
disturbing the surrounding program; unseeded calls draw from wherever the
global generator is. A `RepairableRBD`'s simulations draw from random
streams of their own instead, one for each component and quantity, seeded
from `seed` (see [Random streams](simulation.md#random-streams)); an
unseeded run takes its seed from the global generator.

```python
(rbd.random(1_000, seed=0) == rbd.random(1_000, seed=0)).all()   # True
```

A seed reproduces a result on the same platform. numpy's mathematical
functions can differ in the last bit between operating systems and
processors. So on another machine, the same seed can give times that differ
in their last digits, and rarely a different count, where two events
nearly coincide.

The simulations draw the same random numbers in the same order however they
are computed internally (in blocks for speed, or one at a time), so seeded
results do not depend on which internal path a model takes. A parallel run of a `NonRepairableRBD` (`n_jobs`)
seeds each block of simulations in turn from `seed`, so its results do not
depend on the number of processes; they differ from a run without
`n_jobs`. A `RepairableRBD`'s results are the same with `n_jobs` or
without, and with either engine (see [Parallel
runs](simulation.md#parallel-runs)).

## What is exact and what is simulated

Each analysis is computed one of four ways:

- **exact**: closed forms, or the exact structure function over exact node
  values;
- **numerical**: deterministic numerical methods, which give the same result
  every time, to a small, stated error;
- **simulated**: Monte Carlo, reproducible with a seed;
- **refused**: the method raises, and says why.

The [README](https://github.com/derrynknife/RePyability#when-is-a-simulation-needed)
sums this up by what you ask, the components and the maintenance, and says
which of the simulated analyses must be simulated and which could be made
exact.

On a diagram of plain components (surpyval distributions, with no preventive
maintenance or hidden failures):

| Quantity | Route | How it is computed |
|---|---|---|
| `sf`, `ff`, `Hf`, `cs`, `birnbaum_importance` and the other importance measures (`differential_importance` among them), `sf_given_state`, `structural_importance` | exact | From the node reliabilities. |
| `df`, `hf` | numerical | The exact reliability, differentiated numerically. |
| `time_to_reliability`, `bx_life`, `remaining_life` | numerical | The exact reliability, inverted by root-finding. |
| `parameter_sensitivity` | numerical | The exact Birnbaum importance times a numerical parameter derivative; a repairable diagram's, central differences of its long-run values in each lever. |
| `reliability_rate` | exact | Each node's Birnbaum importance times its own failure density: the system's density, split by node. |
| `NonRepairableRBD.barlow_proschan_importance` | numerical | Each node's part of the system's failure density, integrated over time by adaptive quadrature (to about `1e-9`). |
| `mean`, `mean_time_to_failure` | numerical | The exact reliability, integrated over time by quadrature (to about `1e-10`). |
| `random`, `mean_time_to_failure_interval` | simulated | Monte Carlo. |
| `mean_availability`, `system_failure_frequency`, `mean_up_time`, `mean_down_time`, `mean_time_between_failures`, `expected_cost_rate`, `total_cost`, and the repairable importance measures (`RepairableRBD.barlow_proschan_importance` among them, from the failure frequency's terms) | exact | From the long-run node availabilities. |
| `capacity_distribution`, `system_capacity` | exact | From the node reliabilities (at a time) or long-run availabilities. |
| `point_capacity`, `mission_capacity` | numerical | The capacity distribution at each component's availability over time from new or from its state (each solved on a grid, to about `1e-7`), and its mean over a window. |
| `point_availability`, `mission_availability` | numerical | Each component's renewal equation, from new or from its state, solved numerically (to about `1e-7`), and the system at its components' availabilities at each time; with repair crews, and for a standby group, a Markov chain followed by uniformization (to about `1e-13`). |
| `availability_rate` | numerical | Each component's point availability differentiated on the grid it is solved on, times its Birnbaum importance; the jumps at scheduled events split along the path between the values either side. |
| `expected_failures`, `expected_events` | numerical | Each component's expected events from its renewal equation, on the grid of its availability (to about `1e-7`), and the system's failures by the time-dependent Birnbaum/Vesely formula; `expected_cost` prices them, numerically when anything is priced. |
| `availability` (with the capacity over time and the delivered fraction), `cost`, `compare` | simulated | Discrete-event simulation. |
| `spares_demand`, `spares_stock` | numerical | Each component's replacements, a renewal process, counted on a grid (to about `1e-6`); `spares_demand(method="simulate")` counts them in simulations instead. |
| `allocate_redundancy` (both kinds of RBD) | exact | Exact scoring: `method="exact"` is a proven optimum, `"greedy"` a heuristic. Cold standby (`strategy="cold"` or `"choose"`) that needs two or more units working, of units that are not identical Exponentials, is scored from 10 000 seeded simulated lifetimes. |

The nodes can change a route:

- **Standby and load-sharing nodes.** Their reliability is exact or numerical
  where a closed form or convolution applies, and is otherwise fitted to
  simulated lifetimes (see [Redundancy
  models](redundancy-models.md#how-the-survival-function-is-obtained)). The
  analyses built on such a node are then simulated too.
- **Maintenance.** Preventive maintenance makes the long-run values numerical,
  and the values over time too, replacement on condition (`"policy":
  "condition"`) among it (#161). The exact
  values refuse a component renewed early at its maintenance group's stops
  (an `"opportunity"`). A group's set-up cost is
  exact when no member is renewed early, unless two members are replaced on
  a clock (block replacement, or never failing before an instant age
  replacement): their replacements can share a stop, and `cost()` counts
  that.
- **Hidden failures.** Their values are exact in closed form for a
  constant failure rate and tests and repairs in no time, and numerical
  otherwise: summed over the test intervals for any other life, and, with
  tests or repairs that take time or tests that miss failures, the cycle
  from one test that finds a failure to the next followed test by test on
  a grid (#159). A test that can last as long as its interval makes the
  exact methods refuse.
- **Standby groups.** A group's long-run values are exact from its own
  Markov chain when its units' lives and repair times are exponential, and
  refused otherwise; its values over time are numerical, from the same
  chain followed by uniformization.
- **Repair crews.** Fewer `repair_crews` than components make components wait
  for each other. With exponential lives and repairs, the long-run values
  and the importance measures are then exact from a Markov chain of the
  components' states and the repair queue (up to 15,000 states), and the
  values over time numerical, the chain followed by uniformization, around
  nested RBDs too (#162); otherwise they refuse. The allocations refuse,
  and the simulations follow the queue, in Python.
- **Spares.** The spares counts need each component's replacements to be a
  renewal process, or to fall on a calendar: under block replacement they
  are counted block interval by block interval (the stock too, with repairs
  and block replacements in no time, #160),
  and with hidden failures on the tests that find them, whatever the tests
  and repairs take (with tests that can miss a failure, over where each
  cycle starts between the full tests). Standby groups, renewals at a
  maintenance group's stops and waiting for repair crews make them refuse,
  and `spares_demand(method="simulate")` counts them instead.
- **Imperfect repair.** `Repairable` policies are analytic for a power-law
  process and simulated for imperfect repair.

**For your own diagram, ask it.** `analysis_routes()` gives each analysis's
route, the reason and the nodes that decide it, without running any of them.
For a refusal, the reason is the message the method would raise. For a
repairable simulation, it also gives the engine `engine="auto"` would use:

```python
import surpyval as surv
from repyability import RepairableRBD

tested = RepairableRBD(
    [("s", "pump"), ("pump", "t")],
    {
        "pump": {
            "reliability": surv.Weibull.from_params([500, 1.5]),
            "repairability": "instant",
            "inspection": {
                "interval": 720,
                "duration": surv.Exponential.from_params([0.001]),  # 1,000 h on average
            },
        }
    },
)
routes = tested.analysis_routes()
routes["mean_availability"].route  # 'refused': a test can outlast its interval
routes["mean_availability"].nodes  # ('pump',)
routes["availability"].route       # 'simulated'
routes["availability"].engine      # 'numba' with numba installed, else 'python'
```

## Performance

- **The exact engine is fast and cached.** On construction it reduces the
  diagram's series, parallel and *k*-out-of-*n* parts to modules with closed
  forms, and decomposes whatever is left once; every evaluation replays the
  result, so importance measures, allocation searches and array-valued times
  are cheap. Minimal path and cut sets are derived once per RBD, on first
  use.
- **Structure size.** Series-parallel diagrams stay fast at any size: they
  reduce to a single module, and their path sets are never listed. Thirty
  duplicated stages in series have `2**30` minimal path sets, yet `sf` over
  400 times takes milliseconds. The cost is in the part that does not reduce
  (bridges, cross-ties, shared nodes), the *core*: a core with few minimal
  path sets is pivoted on them, and one with more is decided by a decision
  diagram (below). Its minimal path and cut sets multiply with meshing, and
  only `get_min_path_sets()`, `path_set_probabilities` and
  `fussell_vesely(fv_type="p")` list every path set, and
  `get_min_cut_sets()` and `fussell_vesely(method="rare_event")` every cut
  set. They are read off the core's plan, one lookup each, so the 40,000
  minimal cut sets of a ladder of a hundred bridges take under a second.
  `get_all_path_sets()` lists every simple path: avoid it on large
  redundant diagrams. The exact `fussell_vesely` lists no cut sets: on
  first use it builds from the core's plan, for each node, the decision
  diagram of some minimal cut set through the node having failed (kept for
  later calls), so a ladder of thirty bridges, with 3,600 minimal cut sets,
  takes a fifth of a second, and a 4 × 8 grid of nodes about as long as the
  rare-event sum.
- **Meshed diagrams: a decision diagram.** A part that does not reduce
  and may have more than a hundred minimal path sets is decided by a
  binary decision diagram built from its graph instead, without listing
  them: its nodes are decided one at a time, in an order that keeps the
  mesh's frontier narrow, and each sub-problem is known by how many
  working inputs each node still to come already has. Its size grows with
  how wide the mesh is, not with how many paths it has. Six bridges in
  series (4,096 minimal path sets) take 1.5 seconds by their path sets and
  0.003 seconds this way, and seven 25 seconds by path sets, while fifty
  bridges take 0.02 seconds, a 10 × 10 grid of nodes 0.04, and a random
  mesh of 60 nodes and 345 links 2. Every result is the same to rounding
  either way (the whole test suite passes with either forced). To force
  one, set `repyability.rbd.modular.CORE_METHOD` to `"paths"` or `"bdd"`
  (by default `"auto"`) before building the RBD.
- **Too meshed to work out.** A core whose decision diagram would take
  more than `repyability.rbd.bdd.STEP_LIMIT` steps to build (25 million,
  about five seconds: a random mesh of 70 nodes and 485 links needs more)
  is not worked out. The RBD is still built, with
  `structure_check["is_too_meshed"]` set, and its simulations follow the
  graph itself, a node working while it has not failed and enough of its
  inputs work: a `NonRepairableRBD`'s `random`, `mean(method="simulate")`
  and `unreliability_interval`, and a `RepairableRBD`'s `availability`,
  `cost` and `simulate_timelines`, on the Python engine. The exact and
  numerical analyses refuse, saying why (`analysis_routes()` lists them).
  Raise the limit to try harder: the mesh of 70 nodes takes 45 seconds.
- **Simulations** are vectorised where the models allow it: `mean()` of a
  system of parametric components draws 100 000 lifetimes in well under a
  second. Availability simulations step through events, so their cost grows
  with `mc_samples` times the number of failures and repairs in the window; with
  numba installed (`pip install "repyability[fast]"`) they run compiled,
  about ten times as fast (see [The compiled
  engine](simulation.md#the-compiled-engine)).
- **Monte-Carlo error** shrinks like `1/√N`: use the confidence intervals
  (`mean_time_to_failure_interval`, `mean_availability_interval`,
  `availability_interval`, `CostResult.mean_interval`) to judge `mc_samples`, or pass
  a `tolerance` to simulate until they are narrow enough. Antithetic pairs
  and common random numbers (`compare`) get more precision from each
  simulation, and `n_jobs` spreads the simulations over several CPUs: see
  [Simulation precision and speed](simulation.md).

!!! note "Known limit: long paths through a mesh"
    The minimal path sets of the part of a diagram that does not reduce are
    found recursively, one level per node along its longest path, so a
    non-series-parallel part with a path of about a thousand or more nodes
    exceeds Python's default recursion limit of 1,000 and raises
    `RecursionError`. Raise the limit before building such an RBD:
    `sys.setrecursionlimit(10_000)` handles paths of several thousand nodes.
    A mesh with more than a hundred minimal path sets is decided by its
    decision diagram (above), which builds without recursion: four hundred
    bridges in series, 2,000 nodes, take 0.16 seconds. Series chains and
    parallel groups are reduced first, so long chains and wide systems are
    not affected.
