# Timelines

A *timeline* is a unit's up/down history over a window: whether it is up at
the start, and when it goes down and comes back. RePyability builds them
from records (an outage log), merges components' timelines into a
system's through its diagram, simulates them, and reads any measure off
them: the time up, the failures and who caused them, the first failure, the
longest outage.

## A unit's timeline

[`Timeline`][repyability.Timeline] takes the times a unit changes state, in
order: down, up, down, ... (up first if it starts down). An outage log is
easier to give as outages, `(start, end)`; an end of `None` runs to the end
of the window. A pump's year, one outage of which was planned maintenance:

```python
from repyability import Timeline

pump = Timeline.from_outages(
    [(1200, 1236), (4100, 4108), (7000, 7072)],
    end=8760,
    planned=[False, True, False],
    name="pump",
)
pump.availability       # -> 0.9868   (8,760 - 116 hours down) / 8,760
pump.failures           # -> 2        the planned outage is counted apart
pump.planned_outages    # -> 1
pump.first_failure      # -> 1200
pump.down_intervals     # array([[1200., 1236.], [4100., 4108.], [7000., 7072.]])
pump.state([1000, 1220])   # array([ True, False])
```

`Timeline.from_durations(up_times, down_times, end)` takes the durations up
and down in turn instead, as a simulation draws lives and repairs. A change
at the window's end or after it is outside the window. Two changes at one
time are a change and its undoing at once, such as a failure repaired in no
time: they count as a failure and a restoration, with an outage of length
0.

## Merging timelines

Timelines combine as a diagram's structure does:
[`series`][repyability.timelines.series] is up while all its inputs are
(`a & b` for short), [`parallel`][repyability.timelines.parallel] while any
is (`a | b`), and [`k_out_of_n`][repyability.timelines.k_out_of_n] while at
least `k` are. `~a` is up while `a` is down. Each change of a merged
timeline keeps its *cause*, the input whose change made it, and whether it
was planned. Two pumps in parallel feeding a valve, from their logs:

```python
from repyability.timelines import k_out_of_n, parallel, series

a = Timeline.from_outages([(100, 300)], end=1000, name="a")
b = Timeline.from_outages([(250, 260), (600, 700)], end=1000, name="b")
v = Timeline.from_outages([(900, 905)], end=1000, name="v")
plant = (a | b) & v
plant.down_intervals    # array([[250., 260.], [900., 905.]])
plant.causes            # ['b', 'b', 'v', 'v']
plant.failures_by_cause()   # {'a': 0, 'b': 1, 'v': 1}
plant.availability      # -> 0.985
```

Inputs given one by one keep their own causes (a timeline's own changes
have its `name`, a merged one's the causes it was merged from), so `plant`
credits its outage at 250 to pump `b`, whose failure took the pair down.
Given as one mapping, each input is a cause of its own:
`series({"pumps": a | b, "valve": v})` credits that outage to `"pumps"`.

Changes at the same time are taken one after another, in the order of the
inputs and each input's own changes in their order: a unit that fails and
is repaired at once takes the system down and back up at that instant, as a
simulation does.

## A system's timeline from its diagram

[`RBD.system_timeline`][repyability.RBD.system_timeline] merges every
component's timeline up a diagram, `{node: timeline}`, through its modules
(series, parallel and k-out-of-n) and what is left of it (a bridge, say,
up while one of its minimal path sets is all up). Every block diagram has
it (`RBD`, `NonRepairableRBD` and `RepairableRBD`), and it takes any
timelines: logs, a what-if edit of one, or simulated histories. The plant
again, as a diagram:

```python
from repyability import RBD

rbd = RBD([("s", "a"), ("s", "b"), ("a", "v"), ("b", "v"), ("v", "t")])
rbd.system_timeline({"a": a, "b": b, "v": v}).downtime    # -> 15
```

Had pump `b`'s first outage taken 5 hours rather than 10:

```python
shorter = Timeline.from_outages([(250, 255), (600, 700)], end=1000, name="b")
rbd.system_timeline({"a": a, "b": shorter, "v": v}).downtime   # -> 10
```

The system's changes are credited to its components, and changes at the
same time are taken in the order of the diagram's components. A repeated
node takes the timeline of the node it repeats; a component no path set
needs, and a drawing junction (a perfect node), need none.

## Simulated timelines

[`RepairableRBD.simulate_timelines`][repyability.RepairableRBD.simulate_timelines]
simulates a repairable system and keeps each simulation's histories whole:
every component's and the system's, as
[`Timelines`][repyability.Timelines], one history per simulation, whose
measures are arrays with a value per simulation. They are the simulations
[`availability`][repyability.RepairableRBD.availability] runs with the same
seed, so any measure of a history can be had, not only the averages
`availability` keeps. The two pumps (Weibull lives, MTTF about 1,780 hours)
and the valve (MTTF 20,000 hours) over a year:

```python
import numpy as np
import surpyval as surv
from repyability import RepairableRBD

pump = {"reliability": surv.Weibull.from_params([2000.0, 1.8]),
        "repairability": surv.LogNormal.from_params([3.0, 0.6])}   # about 24 h
valve = {"reliability": surv.Exponential.from_params([1 / 20000.0]),
         "repairability": surv.LogNormal.from_params([2.0, 0.4])}  # about 8 h
plant = RepairableRBD(
    [("s", "a"), ("s", "b"), ("a", "v"), ("b", "v"), ("v", "t")],
    {"a": pump, "b": pump, "v": valve},
)
runs = plant.simulate_timelines(8760.0, mc_samples=2000, seed=1)
year = runs.system
np.mean(year.failures == 0)     # -> 0.5875   no plant failure in the year
year.failures.mean()            # -> 0.5465
year.failures_by_cause()["v"].mean()   # -> 0.4325   most of them the valve's
runs.components["a"].failures.mean()   # -> 4.53      a pump fails often
```

The pumps fail four or five times a year each, but rarely together: most of
the plant's failures are the valve's. The longest outage of each year,
from each simulation's own history:

```python
longest = np.array(
    [np.max(np.diff(h.down_intervals, axis=1), initial=0.0) for h in year]
)
np.percentile(longest, 95)      # -> 15.04
```

The result is a [`TimelineSimulation`][repyability.TimelineSimulation]:
`system`, `components` (by node), and how they were made (`method`):

- **Independent components** (plain units whose models can be streamed, and
  nested RBDs of them) have their histories drawn straight from their
  random streams, a batch of simulations at once, and added up as the event
  loop adds them: `method == "streams"`. This is several times faster than
  the Python event loop.
- **Components that depend on each other or on a schedule** (repair crews
  a job can wait for, standby groups, maintenance, tests, imperfect repair,
  maintenance groups) have theirs recorded from the Python event loop as it
  runs: `method == "event loop"`.

Either way the simulations are `availability`'s with the same seed (and
`working_nodes`, `broken_nodes` and `antithetic`): each one's time up is
the same, to the last bit.

```python
result = plant.availability(8760.0, mc_samples=2000, seed=1)
bool(np.array_equal(year.uptime, result.uptimes))   # True
```

The system's history is merged from its components' with
`system_timeline`, so the simulation's own order of changes at the same
time (from different components) may differ from the merge's, which can
only change which component a failure at such an instant is credited to.
Each history keeps every change, about 25 bytes of memory each: for a
summary of a long run, `availability` (with numba's compiled engine,
faster still) keeps less. A diagram with common-cause groups is refused, as
the simulations refuse it, and the components start new.

## Many histories at once

[`Timelines`][repyability.Timelines] holds many histories of one unit over
one window (one per simulation, or one per site from logs) and works out
each measure for all of them at once. Indexing gives one history's
`Timeline` (a slice, a `Timelines`), and `Timelines` merge as `Timeline`
objects do, history by history; a single `Timeline` among them stands for
each history.

```python
from repyability import Timelines

sites = Timelines(
    [
        Timeline.from_outages([(10, 12)], end=100),
        Timeline.from_outages([(50, 80)], end=100),
        Timeline.from_outages([], end=100),
    ]
)
sites.availability              # array([0.98, 0.7 , 1.  ])
sites.point_availability(60)    # -> 0.6667   two of the three up at 60
time, fraction = sites.availability_curve()
```

`availability_curve()` gives the fraction of histories up after each time
any of them changes, as `availability`'s result gives it.
