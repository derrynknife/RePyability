# Simulation precision and speed

!!! tip "Learning this for the first time?"
    This page is the reference. Why a simulation has an error, and how
    these options shrink it, is taught in [Lesson
    6](../learn/availability.md#precise-enough-sooner).

Some quantities are simulated: a `NonRepairableRBD`'s lifetimes, and a
`RepairableRBD`'s histories over a window, with their costs. (A
`NonRepairableRBD`'s mean time to failure is exact, and can be simulated
too; a `RepairableRBD`'s availability over time is also exact, with
`point_availability`: see
[what is exact and what is simulated](saving.md#what-is-exact-and-what-is-simulated).)
A simulated estimate has a sampling error, which shrinks like `1/√N`:
halving it takes four times as many simulations. This page covers how a
repairable system's simulations draw their random numbers, and the options
that choose `mc_samples` for you, that get the same precision from fewer
simulations, that spread the simulations over several CPUs, that compare
two designs far more precisely than two separate runs can, and that run the
simulations compiled.

## Exact or simulated?

A repairable system's expected values over a window need no simulation
where its exact methods take it: only their spread does.

| Question | Exact, no simulation | Simulated (`availability`, `cost`) |
|---|---|---|
| The mean availability over a window | `mission_availability(t)` | `mean_availability_interval()` |
| The availability at given times | `point_availability(times)` | the result's curve, `availability` |
| The expected failures, planned outages and down time | `expected_failures(t)`, `expected_events(t)` | the totals over the simulations (`system_failures`, ...) |
| The expected cost, by category and component | `expected_cost(t)` | `cost.mean_interval()` |
| The long run | `mean_availability()`, `system_failure_frequency()`, ... | |
| How a window varies: each simulation's up time and cost, percentiles, the chance of no failure | | `uptimes`, `cost.samples`, `cost.percentile(q)` |

The exact methods take independent components, with their maintenance and
tests, and repair crews and standby groups where their Markov chains do;
`analysis_routes()` says which a system's take, and its `availability`
route says whether its expected values need a simulation. Where they take
it, `availability` and `cost` take them too: by default a run's mean
intervals (`mean_availability_interval`, the cost's `mean_interval`) are
the exact values, with no error, and a run to a `tolerance` stops at once;
the simulations give the rest. Where a system has a few nodes the exact
methods do not take, its means are by default taken given their histories,
the rest exact (see [Conditional runs](#conditional-runs)).
`control_variate=False` keeps the simulations' own means, for a quick run
that should not wait for the exact part (a few tenths of a second, more
than a few hundred simulations of a small system take).

| Option | Where | What it does |
|---|---|---|
| `tolerance`, `confidence`, `max_samples` | `availability`, `cost`; `mean` and `mean_time_to_failure` with `method="simulate"`, `mean_time_to_failure_interval` | Simulate until the estimate is known to within `tolerance`. |
| `antithetic=True` | the same, and `NonRepairableRBD.random` | Simulate in antithetic pairs. |
| `n_jobs` | the same, and `RepairableRBD.compare` | Run the simulations on several CPUs. |
| `control_variate=True` | `RepairableRBD`'s `availability` and `cost` | Control the estimate by the system's exact twin (by default, the system itself where the exact methods take it). |
| `conditional=True` | `RepairableRBD`'s `availability` and `cost` | Simulate only the nodes the exact methods do not take, the rest exact given their states (by default, a whole run's means are taken given their histories). |
| `compare(other, ...)` | `RepairableRBD`, `NonRepairableRBD` | The difference between two designs, simulated with common random numbers. |
| `engine` | `RepairableRBD`'s `availability`, `cost` and `compare` | Run the simulations compiled, with numba. |
| `shard_map`; `shards`, `run_shard` | `RepairableRBD`'s `availability` and `cost` | Run the simulations as shards, in other processes or on other machines, through any map. |

The examples use two units in parallel and the plant of
[Repairable systems](repairable.md):

```python
import numpy as np
import surpyval as surv
from repyability import NonRepairableRBD, RepairableRBD

unit = surv.Weibull.from_params([100, 2])

def parallel(n):
    """n units in parallel."""
    names = [f"u{i}" for i in range(n)]
    edges = [("s", u) for u in names] + [(u, "t") for u in names]
    return NonRepairableRBD(edges, {u: unit for u in names})

pair = parallel(2)

def repairable(failure_rate, repair_rate):
    return {
        "reliability": surv.Exponential.from_params([failure_rate]),
        "repairability": surv.Exponential.from_params([repair_rate]),
    }

#   s -> (A | B) -> C -> t        MTTF 10 / MTTR 1 pumps, MTTF 50 / MTTR 2 valve
edges = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]
plant = RepairableRBD(
    edges,
    {"A": repairable(0.1, 1.0), "B": repairable(0.1, 1.0), "C": repairable(0.02, 0.5)},
)
```

## Random streams

A `RepairableRBD` simulation draws each random quantity from a stream of its
own: each component's times to failure, its repair times, its maintenance or
test times, each cost given as a distribution, and, for a component started
from a [state](repairable.md#from-the-plant-as-it-is-now), what is left of
its life or repair (one draw a simulation, so a run from new draws what it
always did). A stream is named by the
component's place (its node name, and the names down through nested RBDs)
and the quantity, and seeded from the run's `seed`, and simulation `r` takes
the same numbers from it whatever the rest of the run does. So:

- a simulation does not depend on how many the run has, on the order they
  run in, on how many processes or threads run them, or on which engine: the
  first 100 simulations of a run of 1 000 are a run of 100;
- one component's draws never depend on another's: a component modelled
  the same way fails and is repaired at the same times whatever the rest of
  the system is, which is what [`compare`](#comparing-two-designs) builds
  on;
- without a `seed`, the run takes one number from numpy's global RNG as its
  seed, so `np.random.seed(s)` beforehand makes it the run `seed=s` gives;
  the global RNG is otherwise left as it was.

```python
short = plant.availability(t_simulation=100.0, mc_samples=100, seed=0)
long = plant.availability(t_simulation=100.0, mc_samples=1_000, seed=0)
bool((short.uptimes == long.uptimes[:100]).all())   # True
```

A stream's numbers are laid out in blocks of simulations, fewer to a block
the more draws a simulation is expected to take from the stream, so which
numbers a simulation gets depends on the component's models and on the
window too: a run over another window, or a component in another place,
draws other numbers. A
model whose draws cannot be streamed (other than a surpyval parametric one,
or a subclass of the component classes, which may draw its events its own
way) draws from numpy's global RNG, seeded afresh for each simulation from
the run's seed: the run stays reproducible, but that component cannot take
part in antithetic pairs or common random numbers.

## Simulating to a tolerance

`tolerance` says how precise the estimate must be: within `tolerance` either
side, at `confidence` (0.95 by default). After the first `mc_samples`
lifetimes, and after each further `mc_samples`, the run checks the
half-width of the confidence interval of the mean, and stops once it is at
most `tolerance`:

```python
ci = pair.mean_time_to_failure_interval(mc_samples=10_000, seed=1, tolerance=0.5)
ci.n_samples               # -> 30000
ci.upper - ci.estimate     # -> 0.49     at most the tolerance
ci.estimate                # -> 114.56   the exact MTTF is 114.58
```

With `method="simulate"`, `mean` and `mean_time_to_failure` take the same
arguments and return the estimate alone (by default they are exact). For a `RepairableRBD`, `availability(tolerance=...)` judges
the mean availability over the window, the fraction of it the system is up
(`result.mean_availability_interval()`), and `cost(tolerance=...)` the mean
cost of a window (`result.mean_interval()`), checking after every `mc_samples`
simulations:

```python
result = plant.availability(t_simulation=100.0, mc_samples=1_000, seed=0,
                            tolerance=0.001, control_variate=False)
result.n_simulations                  # -> 6000
window = result.mean_availability_interval()
window.estimate                       # -> 0.9547   the exact value is 0.9544
window.upper - window.estimate        # -> 0.00094
```

That run simulates to the tolerance, as `control_variate=False` asks. This
plant's components fail and are repaired independently, so its expected
values over the window are exact, and by default a run takes them, with no
error: one to a tolerance stops at once (the simulations are the same as
without it; only the means are exact):

```python
exact = plant.availability(t_simulation=100.0, mc_samples=1_000, seed=0,
                           tolerance=0.001)
exact.n_simulations                            # -> 1000
exact.mean_availability_interval().estimate    # -> 0.95444   mission_availability(100.0)
```

`analysis_routes()["availability"]` says, for a system, whether its
expected values over the window need a simulation at all.

If the tolerance is not reached within `max_samples` simulations (or
lifetimes), by default 100 times `mc_samples`, the run stops there with a
`RuntimeWarning`, and the result is from those. A run that stops after `n`
simulations is the run of `N = n` from the start: the further simulations
continue the same random numbers. (`mean_availability_interval()` is the
interval of the mean over the window; the pointwise band
`availability_interval()` is the curve's, at each time: see
[Repairable systems](repairable.md#availability-over-time-simulated).)

## Antithetic pairs

With `antithetic=True` the simulations come in pairs, and in the second of
each pair every component draws `1 − u` for each uniform random number `u`
it drew in the first, in the same order: where the first drew an early
failure (or a short repair), the second draws a correspondingly late one.
Each simulation is still a correct one, so the estimate is unbiased, but the
two of a pair tend to err in opposite directions, so their mean varies less
than two independent simulations':

```python
plain = pair.mean_time_to_failure_interval(mc_samples=10_000, seed=1)
paired = pair.mean_time_to_failure_interval(mc_samples=10_000, seed=1, antithetic=True)
plain.standard_error      # -> 0.43
paired.standard_error     # -> 0.32   worth 1.8 times as many independent lifetimes
```

A coherent system's lifetime rises with every component's lifetime, which is
what makes the pairing work so well for it. A window's availability and cost
depend on the draws in a less simple way, and gain less, but still gain:

```python
single = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0,
                            control_variate=False)
pairs = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0,
                           antithetic=True, control_variate=False)
single.mean_availability_interval().standard_error   # -> 0.00082
pairs.mean_availability_interval().standard_error    # -> 0.00063
```

(The plant's mean over the window is exact, so by default the intervals
would be that value; `control_variate=False` keeps the simulations' own, to
compare them.)

The pairs, not the simulations, are independent, so the intervals are worked
out from the pairs' means (`result.antithetic` records it), and
`mc_samples` must be even. `NonRepairableRBD.random(size,
antithetic=True)` returns the pairs themselves: lifetimes `2i` and `2i + 1`.
In a `RepairableRBD` every [stream](#random-streams) is paired, so an
antithetic run of `mc_samples` simulations is the first half of the run without
pairs, each followed by its mirror image.

Every component's draws must be replayable from uniform random numbers:
surpyval parametric distributions, and the composite nodes built from them.
Otherwise `antithetic=True` raises `NotImplementedError`.

## An exact twin

A repairable system that needs a simulation is usually close to one that
does not: the same components, failing and repaired independently, whose
availability and cost over a window are exact. `control_variate=True`
simulates that *exact twin* alongside the system, with common random
numbers (each of its components draws what the system's draws, as in
[`compare`](#comparing-two-designs)), and takes the twin's error against its
exact value off the system's mean:

    estimate = mean(x) - b * (mean(twin) - exact)

with the coefficient `b` that leaves the least variance. The estimate is
unbiased (but for `b` coming from the same run, an error of order `1/n`),
and its variance is `1 - corr**2` times the plain mean's, `corr` being the
correlation of the system's simulations with the twin's. Three pumps with
Weibull lives, sharing one repair crew:

```python
pump = {
    "reliability": surv.Weibull.from_params([100, 1.5]),
    "repairability": surv.LogNormal.from_params([1.5, 0.6]),
}
crewed = RepairableRBD(edges, {n: pump for n in "ABC"}, repair_crews=1)
crewed.analysis_routes()["availability"].twin
# 'the system without the limit on repair crews.'
plain = crewed.availability(t_simulation=1000.0, mc_samples=2_000, seed=1)
twin = crewed.availability(t_simulation=1000.0, mc_samples=2_000, seed=1,
                           control_variate=True)
plain.mean_availability_interval().standard_error   # -> 0.000399
twin.mean_availability_interval().standard_error    # -> 0.000145
twin.control_variate.variance_reduction             # -> 7.6
```

The controlled run is worth 7.6 times as many plain simulations.
`mean_availability_interval` and the cost's `mean_interval` give the
controlled estimates; everything else in the result (`uptimes`, the curve,
the totals) is the simulations' own, and `result.control_variate` holds the
twin's values, its exact value and `b`. A run to a `tolerance` is judged on
the controlled estimate, so it stops sooner: here 1 000 simulations instead
of 5 000 for `tolerance=0.0005`.

- **What the twin leaves out.** What ties the components together: a limit
  on repair crews, and maintenance groups. And, component by component,
  what the exact methods over time do not take: a standby group's switching
  (unless its units are exponential, when its own chain follows it), whose
  units then operate together; imperfect repair (but minimal repair in no
  time, which the exact methods take); and inspections they do not take
  (tests that can last as long as their interval), whose failures are then
  revealed. It
  keeps the rest, age and block replacement and replacement on condition
  among it: the closer the twin, the more it gains.
  `analysis_routes()["availability"].twin` says what it leaves out, or why
  there is none (a model that is a probability, or a simulated life).
- **A system that is its own twin** (plain components, nothing left out)
  gets its exact value, with a standard error of 0: its simulations are the
  twin's, so they run once, and the run costs its simulations and the exact
  values. Its means need no simulation at all (`mission_availability`,
  `expected_cost`); the run adds the simulations' spread. That is the
  default (`control_variate=None`) for every system whose expected values
  the exact methods work out, crews and standby groups their chains follow
  among them: its twin is then the system itself.
- **What it costs.** The twin's exact values take a fraction of a second,
  and its simulations run compiled where they can. Every draw must come
  from a stream (surpyval parametric models); the streams are laid out as
  `compare` lays them, so a seeded run's simulations can differ from a run's
  without it; and it runs with `shard_map` only as the system itself.

What it gains, on 4 000 simulations over 1 000 hours (variances of the mean
availability, and of the mean cost, plain over controlled; *per second*
counts the twin's time; antithetic pairs on the same system for
comparison):

| System | Availability | ÷ variance | per second | ÷ cost variance | antithetic ÷ |
|---|---|---|---|---|---|
| One crew, three pumps (MTTF ~90 h) | 0.937 | 7.1 | 1.9 | 11.1 | 2.2 |
| One crew, three pumps (MTTF ~900 h) | 0.996 | 109 | 46 | 189 | 2.0 |
| Two crews, four pumps | 0.970 | 10.9 | 3.5 | 27.6 | 1.3 |
| One crew, with age replacement | 0.980 | 3.5 | 1.3 | 5.5 | 1.6 |
| Cold standby of two units (Weibull) | 0.987 | 4.6 | 1.1 | 4.0 | 2.3 |
| Cold standby of three units (Weibull) | 0.987 | 81 | 18 | 6.3 | 2.4 |
| Inspections that take time | 0.958 | 2.8 | 2.5 | 5.0 | 1.8 |
| Imperfect repair (Kijima I, q = 0.5) | 0.946 | 1.4 | 1.0 | 1.2 | 1.5 |
| An opportunistic maintenance group | 0.987 | 14.9 | 5.9 | 8.8 | 2.4 |

It gains most where the system is a small step from its twin: a crew
rarely busy, a spare rarely needed. Each controlled estimate was within 1.8
standard errors of a plain run of 100 000 simulations. It is not the
default: where the twin is far from the system (imperfect repair, above) it
gains nothing for the time it takes. (The default takes a system's modules
instead, see [Conditional runs](#conditional-runs).)

## Conditional runs

A system needs simulating when some of it does: a standby group of other
lives, a nested RBD sharing a repair crew, a unit repaired imperfectly. The
rest is usually independent components, whose point availabilities and
expected events are exact, and whose randomness is most of a plain run's
error. `conditional=True` simulates only the *modules*, the nodes the exact
methods over time do not take, and takes every other node exactly given
their states: given the modules' states at `t`, the system is up with the
probability its availability with them held so gives, and each simulation
contributes its expected values given its modules' histories
(Rao-Blackwell). Twelve units in a line and a cold-standby pair of Weibull
pumps:

```python
line = {
    f"unit {i}": {
        "reliability": surv.Weibull.from_params([4000 + 500 * i, 1.8]),
        "repairability": surv.LogNormal.from_params([2.0, 0.5]),
    }
    for i in range(12)
}
pumps = {
    "reliability": surv.Weibull.from_params([800, 2.0]),
    "repairability": surv.Weibull.from_params([40, 1.5]),
    "standby": {"units": 2},
}
names = list(line) + ["pumps"]
line_plant = RepairableRBD(
    [("s", names[0])] + list(zip(names, names[1:])) + [(names[-1], "t")],
    {**line, "pumps": pumps},
)
line_plain = line_plant.availability(t_simulation=5000.0, mc_samples=2_000, seed=1,
                                     conditional=False)
line_run = line_plant.availability(t_simulation=5000.0, mc_samples=2_000, seed=1,
                                     conditional=True)
line_run.conditional.modules                               # ('pumps',)
line_plain.mean_availability_interval().standard_error    # -> 9.52e-05
line_run.mean_availability_interval().standard_error      # -> 1.45e-05
line_run.mean_availability_interval().estimate            # -> 0.98864
```

The conditional run is worth 43 times as many plain simulations, in about
the same time: it simulates the pumps alone, and the line's part of the
error is gone (by the law of total variance, what is left is the pumps').
`analysis_routes()["availability"]` names the modules a conditional run
would simulate.

By default a run of such a system takes these means too, from its own
simulations: it simulates the whole system, as a plain run does, then its
modules again alone, drawing what they drew, and its mean intervals are
those of each simulation's expected values given its modules' histories.
Everything else in its result (each simulation's values, the curve, the
totals, the percentiles, the criticalities) is its simulations' own:

```python
line_default = line_plant.availability(t_simulation=5000.0, mc_samples=2_000, seed=1)
line_default.mean_availability_interval().standard_error   # -> 1.45e-05   the conditional run's
line_default.conditional.whole                             # True
```

It costs a plain run, the modules' simulations and the exact part: here a
quarter more time than a plain run, for a standard error 6.6 times smaller.
`conditional=False` keeps the simulations' own means; `conditional=True`
leaves the rest of the simulation out, for the same means in less time,
without the spread.

- **Expected values only.** Each simulation's values are its *expected*
  values given its modules' histories: its up time (`uptimes`), the totals
  and counts (floats: the expected failures, restorations, planned
  outages and down times), the curve (on a grid of `curve_points` steps, by
  default 1 000, with its own `availability_se`) and the cost's `samples`.
  Their means, and their intervals, estimate the window's expected values
  without bias; but their spread is less than a window's own, so a cost's
  `percentile` and `std` refuse, and `criticalities` is None. For a
  window's spread, or its histories, simulate the whole system (the
  default).
- **The same simulations.** The modules draw from the streams they draw
  from in a plain run with the seed, by either engine, on `n_jobs`, in
  antithetic pairs if asked, so a conditional run's modules are a plain
  run's, and their own costs too. A run to a `tolerance` is judged on the
  conditional means, in rounds of `mc_samples`.
- **The exact part.** Each joint state of the modules the simulations meet
  (each up or down) is worked out once: the system's expected up time,
  failures and planned outages with the modules held so, on a grid of
  some 1 300 times (finer near 0, and at the curves' breaks), read between
  them to about 1e-8 of the window. It costs about what
  `expected_events` does, per state met, whatever the number of
  simulations.
- **What it needs.** The other nodes must be independent of the modules,
  and exact. Limited repair crews make every node they serve a module,
  simulated with the crews (a nested RBD, with crews of its own, stays
  exact); a system of crew-served components alone leaves none to take
  exactly, and is refused, as is a maintenance group that stops at every
  outage of the system (`"system_down"`), which ties its members to every
  component.
- **States, capacities, shards and a control variate.** From the
  components' `state`, the modules start from theirs and the rest is taken
  exactly from theirs. With capacities, each simulation's expected time at
  each level and its delivered fraction (against `demand`, as in a plain
  run) come from the exact capacity over time with the modules held, a
  module up at its levels by their probabilities, as a plain run weighs
  them. `shard_map` runs the modules' simulations as shards wherever it
  sends them, to the last bit the run's. `control_variate=True` simulates
  the exact twin's stand-ins for the modules alongside them, with common
  random numbers: each simulation's expected values given the stand-ins'
  histories, whose mean is the twin's exact one, control the means (by as
  much as the stand-ins follow the modules: little for a unit repaired
  imperfectly, whose stand-in, repaired as new, drifts from it).

What it gains depends on how much of the error the independent part makes.
On the line above, most: 43 times. On 150 units in a line with a standby
pair and a nested two-out-of-three group sharing a crew, whose failures
make most of the error, 2.8 times, worth 4 times the work at 40 000
simulations against the compiled engine; on a few units, where the exact
part's fixed cost is not repaid, nothing.

## Parallel runs

`n_jobs=k` runs the simulations on `k` CPUs (`-1`: all of them). A
`RepairableRBD`'s simulations each draw from their own
[streams](#random-streams), so the result is the same to the last bit for
any `n_jobs`, and without it:

```python
fast = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0, n_jobs=2)
same = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
bool((fast.uptimes == same.uptimes).all())   # True
```

The [compiled engine](#the-compiled-engine) runs them on `k` threads. The
Python engine gives each of `k` processes a copy of the diagram, once, sends
them blocks of 250 simulations, and merges the totals each block sends back,
in order, so it pays off for long simulations (many failures and repairs in
a window). To spread a run over machines, or over more processes than one
machine has, [shard](#shards) it.

A `NonRepairableRBD`'s lifetimes are split over the processes in blocks of
10 000, seeded in turn from `seed`, so the result depends on the seed and
not on the number of processes, but it is not the result of a run without
`n_jobs`, which draws every lifetime from one stream. It pays off for
lifetimes that must be simulated event by event, not for a few thousand
that are drawn in one vectorised step. Both work with `tolerance` (checked
after each batch of `mc_samples`) and `antithetic`.

A process needs RePyability, surpyval and scipy loaded before it can
simulate, and how long that takes depends on how the platform starts
processes:

- **Linux, Python up to 3.13 (fork):** processes start with everything
  already loaded.
- **Linux, Python 3.14 (forkserver):** RePyability has the server load them
  once. The first parallel run waits for that; later runs start their
  processes at once.
- **macOS and Windows (spawn):** every process loads them itself, which
  takes a second or two, so `n_jobs` pays off only for runs longer than
  that.

`n_jobs=-1` starts one process per CPU this process may run on (its CPU
affinity, where the platform reports one). A container limited by a CPU
quota rather than by affinity can report more CPUs than it may use: set
`n_jobs` explicitly there.

## A large run's curve

`availability()`'s curve has a point at every time a simulated system
changed state: hundreds a simulation for a busy system, millions for a
large run, which take memory and time to keep and to sort, and are most of
what a chunk (below) carries. `curve_points=G` keeps the curve on a grid of
`G` steps instead, `t_simulation * k / G` for `k` from 0: the simulations
count their changes in the grid's steps, so the curve costs `G` counts
however many run. Its value at each grid time is the full curve's value
there, exactly; between grid times it is not followed. Everything else in
the result (the up times, the mean availability and its interval, the
counts, the costs and the criticalities) is the same either way:

```python
grid = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0,
                          curve_points=100)
len(grid.timeline)       # -> 101
```

A compiled run of 40 960 simulations of a nine-component system over
5 000 hours kept 3 million points without it, and ran 16% faster with
`curve_points=1000`. With capacities, the capacity curve
(`capacity_timeline`, `capacity`) follows the grid too, each step's changes
summed exactly, so it equals the full curve at the grid's times: three
pumps with capacities over 2 000 hours, 5 000 simulations compiled, kept
5.5 million points without it and ran in a third of the time with
`curve_points=200`.

## Splitting a run across machines

A run's simulations are numbered, and simulation `i` draws from streams
seeded by the run's seed and `i` alone, so it comes out the same wherever it
runs. `simulate_chunk(t_simulation, start, stop, seed=...)` runs
simulations `start` to `stop - 1` of a run and returns a
[`SimulationChunk`][repyability.SimulationChunk]: their totals, which save to
JSON. `availability_from_chunks` merges chunks into the run's result, so a
large run can be spread over machines or preemptible workers, each running
its chunk and sending it back:

```python
from repyability import SimulationChunk

first = plant.simulate_chunk(100.0, 0, 1_200, seed=0)       # on one machine
rest = plant.simulate_chunk(100.0, 1_200, 2_000, seed=0)    # on another
sent = [SimulationChunk.from_json(c.to_json()) for c in (rest, first)]
merged = plant.availability_from_chunks(sent)
whole = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
bool((merged.uptimes == whole.uptimes).all())        # True: the same simulations
bool((merged.availability == whole.availability).all())   # True
merged.system_uptime == whole.system_uptime          # True: the totals too
```

- **The same run.** Chunks of simulations `0` to `N - 1` give the result of
  `availability(..., mc_samples=N)`: the same per-simulation values
  (`uptimes`, the cost `samples`) and timeline, and the same totals, to
  the last bit: every total is kept exactly and rounded once, so it does
  not depend on how the run is cut. With costs, the result's `cost` is the
  cost distribution. The chunks must hold simulations `0` to `N - 1`, none
  missing, so a chunk that never came back is not taken for a smaller run;
  `allow_gaps=True` takes the result of whichever simulations they hold.
  (Only `mc_samples=N` can say that the last chunk is missing.)
- **Checked.** Chunks merge only with chunks of the same run: the same
  system (a hash of it saved as JSON, which a chunk carries, so a worker
  can rebuild the system with `RepairableRBD.from_json`), window, seed,
  nodes held working or broken, `method`, `antithetic` and `demand`, and
  different simulations, in order. `SimulationChunk.merge` merges chunks
  into one, to merge in stages.
- **Any engine.** A chunk is the same from the Python engine or the
  compiled one, and with any `n_jobs`. With `antithetic`, a chunk's ends
  are even, so that it holds whole pairs.

A `NonRepairableRBD`'s lifetimes split the same way:
`random_block(block, seed)` draws block `block` of the 10 000-lifetime
blocks that `random(size, seed=seed, n_jobs=...)` draws.

### Shards

A *shard* is a range of a run's simulations as plain data: JSON holding the
system (as `to_dict` saves it), the run's settings, the number its seed
gives the [streams](#random-streams), and the range. `shards(t_simulation,
mc_samples, seed=...)` cuts a run into shards, and `run_shard(shard)` runs
one anywhere RePyability (the same version) is installed, giving back its
*partial*: the simulations' totals, as the bytes of a NumPy `.npz` file,
read without pickle. `availability_from_chunks(partials, mc_samples)` puts
the partials together, in any order, into the run's result, and refuses a
missing one:

```python
from repyability import run_shard

shards = plant.shards(100.0, 4_000, seed=0)
len(shards)                                          # -> 4
partials = [run_shard(shard) for shard in reversed(shards)]   # anywhere
merged = plant.availability_from_chunks(partials, mc_samples=4_000)
whole = plant.availability(t_simulation=100.0, mc_samples=4_000, seed=0)
merged.system_uptime == whole.system_uptime          # True
```

`availability` and `cost` do it all with `shard_map`, a map that runs the
shards wherever it sends them: `shard_map(run_shard, shards)` must give back
each shard's partial, as `map` does. A run to a `tolerance` maps a round of
shards at a time. The result is the same to the last bit:

```python
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(2) as pool:
    sharded = plant.availability(t_simulation=100.0, mc_samples=4_000, seed=0,
                                 shard_map=pool.map)
sharded.system_uptime == whole.system_uptime         # True
```

- **Any executor.** Ray's: `shard_map=lambda f, shards:
  ray.get([ray.remote(f).remote(s) for s in shards])`; Dask's:
  `shard_map=lambda f, shards: client.gather(client.map(f, shards))`; a
  batch system's, a job a shard: `python -m repyability.rbd.shards <
  shard.json > partial.npz` (or with the two files' names).
- **Seconds a shard.** A worker takes a second or so to import RePyability,
  numpy and surpyval, so make shards that run for seconds (`shard_size`, or
  `shards`' `size`; by default 1024 simulations), and keep workers alive, as
  a pool does. A shard's size is rounded up to a whole number of the run's
  widest block of draws, so that no two shards draw the same block.
- **The engine where it runs.** Each shard is simulated by `engine` on the
  worker, `"auto"` choosing there. `n_jobs` is for one machine: give the
  map's workers the CPUs instead.
- **Checked.** A worker refuses a shard of another RePyability version,
  which could simulate it differently, and the result refuses partials of
  other shards than those sent. A system that cannot travel as JSON (a
  model that is not a surpyval one, or a subclass of one, which would load
  without what it changes) cannot be sharded: run it with `n_jobs`.
- **Small partials.** A partial carries each simulation's up time (and
  cost) and every change of the system's state, which the full curve
  needs; with [`curve_points`](#a-large-runs-curve), only the grid's
  counts.

## Small failure probabilities

A highly reliable system rarely fails, so plain sampling sees few
failures: estimating a probability `p` to ±10 % takes about `384 / p`
lifetimes, some 4e10 for `p = 1e-8`. Where the unreliability is exact,
`ff(x)` gives it at once, to full precision however small it is (it is
worked out from the components' own failure probabilities, not as one less
the reliability); through a numerical node, such as a cold-standby group of
non-exponential units, only to that node's accuracy, about 1e-6. Where a
node has no exact or numerical reliability (warm standby with two Weibull
pumps operating, say), `ff` refuses, and
`unreliability_interval(x)` estimates `P(T <= x)` by simulation, to a
relative precision, with methods that find rare failures:

```python
from repyability import StandbyModel

pump = surv.Weibull.from_params([1000, 1.5])
pumps = StandbyModel([pump] * 4, k=2, dormancy_factor=0.3)
station = NonRepairableRBD([("s", "pumps"), ("pumps", "t")], {"pumps": pumps})
tail = station.unreliability_interval(50.0, seed=1)
tail.estimate                            # ~> 1.59e-06
tail.method, tail.n_samples              # ('subset', 570000)
```

Each draws the system's lifetime from a row of uniforms, as `random` does
(the standby group by its own logic), and runs until the interval's
half-width is `relative_tolerance` (by default 0.1) of the estimate, or
`max_samples` lifetimes are spent:

- **`"plain"`**: independent samples.
- **`"latin_hypercube"`, `"sobol"`**: Latin hypercube samples, and scrambled
  Sobol points (randomised quasi-Monte Carlo), in replicates whose spread
  gives the error.
- **`"cross_entropy"`**: importance sampling. The uniforms are mapped to
  standard normal variables, and drawn from a mixture of up to eight
  Gaussians shifted towards failure, fitted level by level by the
  cross-entropy method to the samples that fail soonest; each sample is
  weighted by its likelihood ratio.
- **`"subset"`**: subset simulation. `p` is a product of conditional
  probabilities of failing by ever earlier times, each about 0.1, the
  samples at each level drawn by Markov chains (adaptive conditional
  sampling) from those that failed by the level before. Independent runs
  give the error.
- **`"auto"`** (the default): plain sampling if a pilot of 20 000
  lifetimes sees at least 50 failures; else the cross-entropy method if
  the system fails in at most eight ways (minimal cut sets), each of
  components that are distributions; else subset simulation.

Measured on systems of Weibull units whose exact unreliability is known
(a cold-standby group's by convolution), as how many plain lifetimes one
lifetime of each method is worth at the same precision:

| System (uniforms) | `p` | Sobol | Cross-entropy | Subset |
|---|---|---|---|---|
| 2-out-of-3 (3) | 1e-2 | 51 | 8 | 3 |
| | 1e-4 | 1 | 370 | 19 |
| | 1e-8 | — | 1 900 000 | 29 000 |
| Bridge (5) | 1e-2 | 11 | 6 | 2 |
| | 1e-4 | 1 | 350 | 13 |
| | 1e-8 | — | 1 750 000 | 29 000 |
| Cold-standby group of 3, in series with a pair (5) | 1e-2 | 51 | 9 | 3 |
| | 1e-4 | 2 | 390 | 22 |
| | 1e-8 | — | 1 860 000 | 23 000 |
| 10 pairs in series (20) | 1e-2 | 36 | wrong | 1.5 |
| | 1e-4 | 1 | wrong | 22 |
| | 1e-8 | — | wrong | 40 000 |
| 35 pairs in series (70) | 1e-2 | 1 | wrong | 1.4 |
| | 1e-4 | 1 | wrong | 17 |
| | 1e-8 | — | wrong | 26 000 |

At `p = 1e-8` subset simulation took 1.4 to 2.4 million lifetimes (one to
twelve seconds) and the cross-entropy method about 20 000, against 4e10
for plain sampling. Latin hypercube samples gained nothing on these
indicators of failure (a factor of about 1), and Sobol points gained only
while `p` was not small and the uniforms few. Every estimate was within
2.8 standard errors of the exact value, but for the cross-entropy method
on the pairs in series: they fail in ten and 35 ways, more than its
mixture follows, and it estimated a fifth of the probability, or almost
none, with intervals that did not show it. `"auto"` chose plain sampling,
the cross-entropy method or subset simulation as above, and was within 9 %
of the exact value in every case.

## Comparing two designs

Two designs are best compared with **common random numbers**: simulate both
with the same random numbers, so that the differences between their results
come from the designs rather than from chance. `a.compare(b, ...)` does that
component by component: a component with the same name in both draws the
same random numbers in both. It returns a
[`ConfidenceInterval`][repyability.ConfidenceInterval] of the mean
difference, `a`'s result minus `b`'s, which can be negative.

For a `RepairableRBD`, a component gets the same failures and repairs in
both where it is modelled the same way, and matching ones (the same
quantiles of its own models) where it is not; components of nested RBDs are
matched by their place. The difference is in the fraction of the window the
system is up (`quantity="availability"`, the default) or in its cost
(`quantity="cost"`):

```python
faster = RepairableRBD(
    edges,
    {"A": repairable(0.1, 2.0), "B": repairable(0.1, 2.0), "C": repairable(0.02, 0.5)},
)
gain = faster.compare(plant, t_simulation=100.0, mc_samples=2_000, seed=0)
gain.estimate          # -> 0.0058    the exact difference is 0.00568
gain.standard_error    # -> 0.00017
```

Two separate runs of 2 000 simulations would estimate the difference with a
standard error of about 0.0012, seven times as large: it would take fifty
times the simulations to match. The valve fails and is repaired alike in
both designs, so its outages, most of the plant's downtime, cancel; the
pumps have the same up times in both, and only their repairs differ.

For a `NonRepairableRBD`, `compare` estimates the difference in mean time to
failure:

```python
gain = parallel(3).compare(pair, mc_samples=20_000, seed=0)
gain.estimate          # -> 14.8    a third unit adds about 15 hours
gain.standard_error    # -> 0.21    two separate estimates: about 0.42
```

The exact difference, the integral of the difference in reliability, is
14.46. The reliabilities themselves need no simulation: compare `sf`.

Both kinds need every component's draws to be replayable, as antithetic
pairs do. A model whose draws do not follow numpy's uniforms (a class of your
own that draws its own random numbers, say), which the two systems would not
share, raises `NotImplementedError`.

## The compiled engine

A `RepairableRBD`'s simulations run in Python. With
[numba](https://numba.pydata.org), an optional dependency, installed, they
can run compiled instead, about ten times as fast on one core, and faster
still on several:

```bash
pip install "repyability[fast]"
```

`engine="auto"`, the default of `availability`, `cost` and `compare`, then
compiles a run when the compiled engine simulates the system and the run is
long enough to repay loading it: about a third of a second from numba's
cache (a minute or two the first time ever, while numba compiles it).
`engine="numba"` asks for it outright, and raises an error if numba is not
installed or the system is one it does not simulate; `engine="python"` keeps
to Python. The two engines give the same results, to the last bit: the
compiled loop is the Python one over arrays, reading the same
[streams](#random-streams).

```python
fast = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
slow = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0, engine="python")
bool((fast.uptimes == slow.uptimes).all())   # True, with numba or without
```

The compiled engine simulates plain components (`reliability` and
`repairability` specs, or `NonRepairable` objects, with surpyval parametric
models) in any structure, with nodes held working or broken, costs,
antithetic pairs, tolerances and common random numbers; age and block
replacement, in zero time or taking time; hidden failures found by
periodic tests, in zero time or taking time, staggered, and with a coverage
below 1 (a failure a test misses waits for a full test); fewer repair
crews than components, the jobs waiting by priority; standby groups,
cold, warm or hot, with switches that can fail; nested RBDs of up to 20
components, each stepped to its next change as in Python; and the system's
capacity over time, on systems of up to 63 components. Replacement on
condition, maintenance groups, imperfect repair, common-cause groups (whose
shared causes the Python loop draws, see [repairable
systems](common-cause.md#repairable-systems)), a run from the components'
states with maintenance, tests or nested RBDs, and other models run in
Python, which `"auto"` chooses by itself. With `n_jobs` it
runs on that many threads, which start at once. Under age replacement (half
the units maintained in zero time, half taking time), the compiled engine
ran 9 to 12 times as fast as Python on one thread, and 18 to 29 times on
four, on the three systems below; with every unit tested (half
the tests in zero time, staggered, half taking time and missing one failure
in ten), 17 to 27 times on one thread, and 32 to 63 on four; with one,
two and four crews, 13, 9 and 5 times on one thread, and 19, 25 and 8 on
four (on the 70 components, drawing the numbers and adding up the results
take most of the time); with half the nodes standby groups of three
units, 6 to 8 times on one thread, and 7 to 19 on four; with every node a
nested pair, 6 times on one thread and 17 to 18 on four; and following
capacities, 10, 13 and 6 times on one thread, and 11, 16 and 6 on four, on
3, 12 and 62 components (on 62, working out the capacity of each set of
components down that the simulations meet takes most of the time).

On a four-core 2.8 GHz Xeon, in millions of events (failures and repairs)
a second:

| System | Window | Python | Compiled, one thread | Compiled, four threads |
|---|---|---|---|---|
| The plant above, 3 components | 1 000 h | 0.75 | 8.3 | 16.3 |
| A bridge feeding a 2-out-of-3 vote, 12 components | 5 000 h | 0.63 | 10.0 | 25.0 |
| 35 redundant pairs in series, 70 components | 2 000 h | 0.51 | 7.1 | 17.3 |

Above 20 components the compiled loop has no table of every state to look
the system up in. It keeps whether the system is up up to date as
components fail and are repaired instead, following each change up the
diagram's structure only as far as it changes anything: on the 70
components, 1.7 times as fast on one thread as working the system out at
each event, and 1.5 times on four.

A `NonRepairableRBD`'s lifetimes are drawn vectorised, a block of samples
at once through the diagram's modules; on the same machine, a million of
them:

| Diagram | Lifetimes a second | With `n_jobs=4` |
|---|---|---|
| 2 units in parallel | 12.6 million | 11.6 million |
| 10 bridges in series, 60 components | 0.51 million | 1.6 million |
| 35 redundant pairs in series, 105 components | 0.34 million | 1.1 million |

### Engines from other packages

Another package can add a compiled engine of its own, registered under the
`repyability.engines` entry point group (see `repyability.rbd.engines` for
the interface). `engine=` then takes its name, and `engine="auto"` runs it
in preference to numba when it says so, on the same systems, with the same
results to the last bit; if it cannot load, `"auto"` warns and runs on the
next engine. `analysis_routes()` reports which engine `"auto"` would run.

## Which to use

- **A precision to meet:** give a `tolerance`, rather than guessing `mc_samples`.
- **A choice between designs:** `compare` them, rather than comparing two
  separate estimates, whose errors add up.
- **A quantity that rises with the components' lifetimes:** try
  `antithetic=True`, and check that the standard error falls.
- **A system a small step from its exact twin** (a crew rarely busy, a
  spare rarely needed): `control_variate=True`.
- **A system mostly of independent components around a few that need
  simulating:** the default takes its expected values given those few's
  histories; `conditional=True` simulates only those, for its expected
  values in a fraction of the time.
- **A system the exact methods take:** its expected values need no
  simulation; the default gives them, and the simulations their spread.
- **Long simulations:** install numba (`pip install "repyability[fast]"`)
  for the compiled engine, and spread them over the cores with `n_jobs`.
- **More than one machine can run in time:** [shard](#shards) the run.
