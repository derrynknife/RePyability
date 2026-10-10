# Repairable systems

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson
    6](../learn/availability.md) (repair and availability).

A [`RepairableRBD`][repyability.RepairableRBD] models a system whose
components are repaired when they fail. The question changes from "has it
failed yet?" to "is it up?": **availability**. Long-run quantities have exact
closed forms, and the availability over time is exact too, from new or from
the components' [current states](#from-the-plant-as-it-is-now), and so are
the expected failures, downtime and cost over a window; the histories
behind them (how much those counts and costs vary) and a family of
criticality measures come from a discrete-event simulation.
Theory: [Concepts](../concepts.md#availability).

## Components

Each component needs a reliability (time-to-failure) model and a
repairability (time-to-repair) model. A life is a surpyval distribution,
fitted or built with `from_params`, or a surpyval `MixtureModel` (a
population of two or more modes, such as infant mortality and wear-out),
whose lives the simulations draw by inverting its distribution function:

```python
import numpy as np
import surpyval as surv
from repyability import NonRepairable, RepairableRBD

def unit(failure_rate, repair_rate):
    return {
        "reliability": surv.Exponential.from_params([failure_rate]),
        "repairability": surv.Exponential.from_params([repair_rate]),
    }

#   s -> (A | B) -> C -> t        MTTF 10 / MTTR 1 pumps, MTTF 50 / MTTR 2 valve
edges = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]
plant = RepairableRBD(edges, {"A": unit(0.1, 1.0), "B": unit(0.1, 1.0), "C": unit(0.02, 0.5)})
```

A component can be given as:

- a dict with `"reliability"` and `"repairability"`, plus optional costs
  (`"repair_cost"`, `"replace_cost"`, `"downtime_cost"`, and the one-off
  `"acquisition_cost"`; see [Costs](costs.md)),
  scheduled preventive replacement (`"preventive"`; see
  [Costs](costs.md#preventive-maintenance)), for a component whose
  failures are hidden until a proof test finds them, periodic inspection
  (`"inspection"`; see [Costs](costs.md#hidden-failures-and-inspection)),
  its place in the queue for a repair crew (`"priority"`; see
  [below](#repair-crews)), imperfect repair (`"repair"` and
  `"replace_after"`; see [below](#imperfect-repair)), and the share of the
  time it operates (`"duty"`; see [below](#operating-part-of-the-time)).
  Any other key raises
  `ValueError`, so a mistyped cost key is never silently priced at zero;
- `"repairability": "instant"` for a component repaired in zero time (see
  [below](#instantly-repaired-components));
- a [`NonRepairable`][repyability.NonRepairable]`(reliability,
  time_to_replace)` object. The RBD keeps its own copy for each node, so one
  object can stand for several identical parts;
- another `RepairableRBD`, nested as a subsystem (see
  [below](#nested-repairable-rbds));
- [`PerfectReliability`][repyability.PerfectReliability] itself, for a
  *junction*: a node that never fails, such as the point where two of three
  trains must deliver (see [below](#junctions)). A spec whose
  `"reliability"` is `PerfectReliability` is one too: a what-if of a part
  that never fails.

The constructor also takes `k`, `input_node`, `output_node` and
`on_infeasible_rbd` exactly as for a
[`NonRepairableRBD`](building.md), `downtime_cost_rate` (see
[Costs](costs.md)) and `repair_crews` (see [below](#repair-crews)). Every
time is in the models' unit (see [Units](building.md#units)).

Every repair restores a component to as good as new (unless it is repaired
imperfectly), components fail and are repaired independently of each other
(unless they wait for a repair crew), and a component keeps its own
failure/repair cycle whether or not the system is up.

### Operating part of the time

A life fitted to operating hours is the wrong clock for a component that
runs only part of the time (a duty pump, a standby generator's monthly
runs): the diagram runs on the calendar, and the component ages only while
it operates. `"duty"`, the fraction `d` of the time it operates (in
`(0, 1]`), puts its life on the calendar: its life there is its operating
life over `d`, `R(d t)`, the same kind of distribution with its scale
moved, which every exact method and simulation then takes. Its repairs,
maintenance and tests stay on the calendar, and its levers and saved spec
keep the life as given, in operating time.

```python
pump = {"reliability": surv.Weibull.from_params([1000, 2]),       # operating hours
        "repairability": surv.Exponential.from_params([1 / 8])}
always = RepairableRBD([("s", "p"), ("p", "t")], {"p": pump})
part = RepairableRBD([("s", "p"), ("p", "t")], {"p": {**pump, "duty": 0.4}})
always.mean_availability()                   # -> 0.99105
part.mean_availability()                     # -> 0.9964
part.system_failure_frequency() * 8760       # -> 3.94   failures a year, against 9.796
```

The life must be a surpyval parametric distribution of a time; a
probability per demand, a mixture, a degradation process or a nested
diagram is refused (give its life on the calendar instead). The duty is a
fixed share of the time: a component that runs only while the system runs,
or ages while idle too, needs that modelled in its life.

### Junctions

A k-out-of-n vote needs a node to vote at, and that node is often no part
at all, only the point where the trains meet. Give it
`PerfectReliability`: it always works, so it passes on whatever reaches
it, and `k` says how many of its inputs it needs. Votes can then sit
anywhere, here two 2-of-3 stages in series:

```python
from repyability import PerfectReliability

#   s -> 3 trains (x) -> h1 (2 of 3) -> 3 trains (y) -> h2 (2 of 3) -> t
xs, ys = ["x0", "x1", "x2"], ["y0", "y1", "y2"]
station = RepairableRBD(
    [("s", x) for x in xs] + [(x, "h1") for x in xs]
    + [("h1", y) for y in ys] + [(y, "h2") for y in ys] + [("h2", "t")],
    {x: unit(0.1, 1.0) for x in xs}
    | {y: unit(0.02, 0.5) for y in ys}
    | {"h1": PerfectReliability, "h2": PerfectReliability},
    k={"h1": 2, "h2": 2},
)
station.mean_availability()   # -> 0.9725   (3p² − 2p³ for each stage, multiplied)
```

A junction is no component: every analysis leaves it out (it is never in a
path or cut set, has no importance, and cannot be held working or broken),
the simulations draw nothing for it, and the capacity analysis lets it pass
whatever reaches it, up to a capacity if it is given one. It may take a
repair model, never used, but no costs or maintenance.

## Long-run availability and frequencies (exact)

```python
plant.mean_availability()           # -> 0.9536   long-run fraction of time up
plant.mean_unavailability()         # -> 0.04641
plant.node_availability()           # {'A': 0.9091, 'B': 0.9091, 'C': 0.9615, 's': 1.0, 't': 1.0}
plant.system_failure_frequency()    # -> 0.03497  system failures per unit time
plant.mean_up_time()                # -> 27.27    MUT: mean length of an up period
plant.mean_down_time()              # -> 1.327    MDT: mean length of an outage
plant.mean_time_between_failures()  # -> 28.6     MTBF = MUT + MDT = 1 / frequency
```

A component's availability is `MTTF / (MTTF + MTTR)`; the system's is the
exact system computation at those availabilities. A component some of whose
units never fail (a surpyval model with `p < 1`) sooner or later gets one of
them, and is then up for good: its long-run availability is 1 and its
failure frequency 0 (simulate `availability()` for the years before). The failure frequency is
the Birnbaum/Vesely formula: each node's Birnbaum importance times its own
failure frequency `1 / (MTTF + MTTR)`, summed. All of these accept
`working_nodes`/`broken_nodes`, and `mean_availability` accepts
`method="p"/"c"`:

```python
plant.mean_availability(working_nodes=["A"])        # -> 0.9615   only C can fail now
plant.system_failure_frequency(working_nodes=["A"]) # -> 0.01923
```

A forced node never changes state, so it contributes no failures. The
importance measures on a repairable system are covered on
[Importance measures](importance.md#on-a-repairable-system).

## Availability over time (exact)

With every component new at time 0, the probability that the system is up
at a time `t`, its **point availability** `A(t)`, is exact, and so is its
mean over a mission `[0, t]`:

```python
plant.point_availability([0.0, 1.0, 5.0, 50.0])   # array([1.    , 0.9808, 0.9565, 0.9536])
plant.point_availability(1.0)                      # -> 0.9808
plant.mission_availability(100.0)                  # -> 0.9544   mean over [0, 100]
plant.mission_availability([10.0, 100.0, 1000.0]) # array([0.962 , 0.9544, 0.9537])
```

Each component's `A(t)` follows from the distributions of its up and down
times by the renewal equation, solved numerically to about `1e-7`; the
components fail and are repaired independently, so the system's is the
exact system computation at theirs, at each time (see
[Concepts](../concepts.md#availability)). The curve starts at 1 (less any
units dead on arrival) and settles at `mean_availability()`, and a mission
average differs from the long-run value by about `b / t`, for a constant `b`
of the components' up and down times. `b` is positive unless the lives vary
more than exponential ones do, and largest for components that wear out,
which fail less early on. A mission of decades costs no more than one of
hours: past the time the components have settled, the integral is extended
exactly.

Both take `working_nodes`, `broken_nodes` and `method` as
`mean_availability` does, and cover what it covers: age and block
replacement, nested RBDs, and hidden failures, for any life, tested and
repaired in no time or not, with tests that find every failure or miss
some (see [a life that wears out](costs.md#a-life-that-wears-out) and
[tests and repairs that take
time](costs.md#tests-and-repairs-that-take-time)). A test that can last as
long as its interval raises `NotImplementedError`; simulate it. Each component's curve is
computed on a grid of 1,000 steps over its typical up time: within one step
of a time at which its units start or stop on a schedule (at 0, and at its
scheduled replacements), what happens faster than a step, such as a short
repair, is smoothed over it, so a point value there can be off by up to
about the probability that the component is under repair; mission averages
are not affected.

A small unavailability, a redundant safety function's `1e-16`, say, is lost
below one less a number next to 1: `1 - point_availability(t)` cannot hold
it. `point_unavailability` and `mission_unavailability` (#237) work it out
in its own right, as `mean_unavailability` does in the long run, each
component down with one less its availability, or in closed form for one
with an exponential life and repair (and no maintenance or tests), new at 0:

```python
valve = {
    "reliability": surv.Exponential.from_params([1e-9]),
    "repairability": surv.Exponential.from_params([1 / 8]),
}
pair = RepairableRBD(
    [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")], {"a": valve, "b": valve}
)
1 - pair.point_availability(1000.0)   # -> 1.11e-16   the floor of a float
pair.point_unavailability(1000.0)     # -> 6.4e-17    (8e-9 squared)
pair.mission_unavailability(8760.0)   # -> 6.391e-17  from new, over a year
```

!!! note "A window's length, by its name"
    The methods name a window's length for what it is to them: `t`, the
    times (or missions' lengths) of the exact values over time
    (`point_availability`, `mission_availability`, `expected_events`,
    `expected_cost`); `t_simulation`, the window a simulation runs over
    (`availability`, `cost`, `compare`, `simulate_timelines`); `horizon`,
    the time a count or a cost is totalled over (`spares_demand`,
    `total_cost`, `allocate_redundancy`); and `window`, the window an
    importance measure is averaged over (#235). Each runs from 0, every
    component new, unless a `state` is given.

## Expected events over a window (exact)

From the same curves, the expected number of system failures in a window
`[0, t)` from new is exact, and so is everything else the simulation counts
on average:

```python
plant.expected_failures([10.0, 100.0, 1000.0])   # array([ 0.3384,  3.4853, 34.9538])
window = plant.expected_events(100.0)
window.system_failures    # -> 3.485   the simulation below finds 3.508
window.system_downtime    # -> 4.556   = 100 * (1 - plant.mission_availability(100))
window.node_failures      # {'A': 9.099, 'B': 9.099, 'C': 1.925}
window.node_downtime      # {'A': 9.008, 'B': 9.008, 'C': 3.772}
```

A component's failure takes the system down if the component is critical
then, which, the components being independent, it is with probability its
Birnbaum importance at their availabilities at that time. So the system's
expected failures are the time-dependent form of the Birnbaum/Vesely
formula,

```text
E[system failures in [0, t)] = ∫₀ᵗ Σᵢ I_B,i(s) dMᵢ(s)
```

with `Mᵢ(s)` component *i*'s expected failures by `s`, which its renewal
equation gives on the grid of its point availability (to about `1e-7`). In
the long run they come at `system_failure_frequency()`: `3.497` per 100
here. [`ExpectedEvents`][repyability.ExpectedEvents] holds the system's
expected failures, planned outages and downtime, and each component's
failures, corrective actions (at which its repair and replace costs are
charged: its failures, or for hidden failures those a test finds),
preventive replacements, tests and downtime.

- **Events at exact times are counted exactly.** An event at `t` itself
  falls after the window, as in the simulation, so windows one after
  another add up; components replaced at the same age or block times, or
  dead on arrival together, take the system down once.
- **They cover what `point_availability` covers** (age and block
  replacement, nested RBDs, hidden failures),
  take `working_nodes`, `broken_nodes` and
  `method`, and take an array of windows as well as one. A window of decades
  costs no more than a few years: past the time the components settle, the
  counts grow at their long-run rates.
- **Only the means are exact.** How much the counts vary, and the chance of
  no failure in the window, come from the simulation.

`expected_cost` prices the same events (see
[Costs](costs.md#the-expected-cost-of-a-window-exact)).

## From the plant as it is now

The analyses over time above start with every component new. Given the
components' current states instead, the same methods answer the
condition-based question: *given the plant as it is today, what are its
availability, failures, cost and capacity over the next month?* Pass
`state={node: NodeState(...)}` (see [`NodeState`][repyability.NodeState]):

- **up, at an age:** `NodeState(age=420.0)`, the time since the unit was
  put into service as new. What is left of its life has the survival
  function `R(a + s) / R(a)`, and under age replacement it is replaced when
  it reaches the age, `T - a` from now (at once if it already has);
- **down:** `NodeState(alive=False, down_for=6.0)`, how long it has been
  down so far, in a repair or (`maintenance=True`) in its preventive
  maintenance. What is left of that has the survival function
  `G(r + s) / G(r)`, and the unit is then new;
- **on a calendar:** `phase`, the time since its last block replacement or
  test, so that the next falls `interval - phase` from now. A unit with
  hidden failures is known to have been up only at its last test (or when
  put into service, if that was since): it may have failed since, unseen,
  and the next test finds it;
- **repaired imperfectly, at a virtual age:** `NodeState(age=a,
  virtual_age=v)`, its virtual age `v` at its last repair and its operating
  time `a` since (#269); its life left is drawn from virtual age `v + a`,
  and its next repair takes the virtual age on from `v` by the whole `a`
  plus that life. One down is at virtual age `v` once its repair is over.
  For a unit of a fitted surpyval `GeneralizedRenewal`, `unit_states()`
  gives its `virtual_age` now and `since_failure`: give
  `age=since_failure` and `virtual_age=virtual_age - since_failure`. The
  simulations take it (not the exact methods), for a component with no
  `replace_after`, maintenance or tests;
- **a nested RBD:** a dict of its own components' states;
- **long in service, state unknown:** `state="stationary"` puts every
  component in its long-run state (one on a calendar at its phase; for one
  component, `NodeState(stationary=True, phase=...)`).

A component left out starts new, and every component's later units are new,
as from new.

```python
from repyability import NodeState

pump = {
    "reliability": surv.Weibull.from_params([1000.0, 2.5]),
    "repairability": surv.LogNormal.from_params([3.0, 0.5]),
    "preventive": {"interval": 500.0, "duration": surv.Weibull.from_params([8.0, 3.0])},
}
pumps = RepairableRBD([("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")], {"a": pump, "b": pump})
now = {"a": NodeState(age=420.0), "b": NodeState(alive=False, down_for=6.0)}

pumps.point_availability([0.0, 12.0, 24.0], state=now)  # array([1.    , 0.9952, 0.9972])
month = pumps.expected_events(720.0, state=now)
month.system_failures                 # -> 0.0195   a failing while b is repaired
month.system_planned_outages          # -> 0.0188
month.node_preventive                 # {'a': 1.781, 'b': 0.8497}   a's due in 80 h
pumps.mission_availability(720.0, state=now)           # -> 0.9996
pumps.mission_availability(720.0)                      # -> 0.9942   from new
pumps.expected_events(720.0).system_planned_outages    # -> 0.7304   both reach 500 h together
pumps.mission_availability(720.0, state="stationary")  # -> 0.9996   = mean_availability()
```

From new, both pumps reach their replacement age at once, and most of
those months' outages are the two being maintained together; as the plant
is, they are out of step, and the month's risk is the old pump failing
before the other is back.

- **The exact methods** (`point_availability`, `mission_availability`,
  `expected_failures`, `expected_events`, `expected_cost`,
  `point_capacity` and `mission_capacity`) take `state=` wherever they work
  from new. Under block replacement, the unit's own curve runs to its first
  block time, and the interval-by-interval solution from there. A unit
  whose hidden failures are repaired at once cannot be down in them.
- **The simulation** (`availability`, `cost`, `compare`, `simulate_chunk`,
  `shards` and `initialize_event_queue`) takes the same states. A component started
  from one draws what is left of its life, repair or maintenance from one
  uniform of a stream of its own, by the inverse transform of its
  conditional distribution, so a seeded run is reproducible and a run from
  new is unchanged; it is simulated in Python. A simulation does not take
  the long-run start: give each component's state. With repair crews, a
  component down at the start holds a crew, so no more can be down than
  there are crews.
- **With repair crews**, the exact methods start the crews' Markov chain
  (see [below](#repair-crews)) from the components' states: each up, or
  down in a repair (how old it is, and how long it has been down, do not
  matter, as its life and repair are exponential), so no more can be down
  than there are crews; or every component in its long-run state
  (`state="stationary"`), not one alone, as the queue ties them together.
- **Not taken:** the state of a standby group (but its long-run state,
  `NodeState(stationary=True)`), and, by the exact methods, that of an
  imperfectly repaired component; leave them out (new).

## Uncertain component models

The components' models are estimated from data, so the availability worked
out from them is uncertain too: *epistemic* uncertainty, about what the
models are, as opposed to the variability they describe.
`mean_availability_uncertainty`, `point_availability_uncertainty(x)`,
`mission_availability_uncertainty(t)` and `expected_cost_rate_uncertainty`
carry it to the system, as `sf_uncertainty` does for a [system that is not
repaired](reliability.md#uncertainty-in-the-component-models): each draw
gives the uncertain models plausible parameters, and the diagram, rebuilt
with them, its value, worked out as the diagram's own is. Here the pumps'
repairs are fitted to 20 repair times, and the valve's life to 15 failures:

```python
E = surv.Exponential.from_params
pump_repair = surv.LogNormal.fit(np.exp(np.linspace(-0.8, 0.8, 20)))   # fitted in surpyval
valve_life = surv.Weibull.fit(surv.Weibull.from_params([55.0, 1.5]).qf(np.linspace(0.05, 0.95, 15)))
fitted_plant = RepairableRBD(
    edges,
    {
        "A": {"reliability": E([0.1]), "repairability": pump_repair},
        "B": {"reliability": E([0.1]), "repairability": pump_repair},
        "C": {"reliability": valve_life, "repairability": E([0.5])},
    },
)
spread = fitted_plant.mean_availability_uncertainty(n_draws=1000, seed=0)
spread.nominal         # -> 0.9505   with the fitted models
spread.interval(0.9)   # (0.9396, 0.9596)
```

By default every model that is a surpyval fit with a parameter covariance
is drawn, from the normal approximation of its fit, and the nodes that hold
the same fitted object share its draws: here the pumps' repair, one input
keyed `("A", "B")`, and the valve's life. A component has several models,
its roles: its life (`"reliability"`), its repair (`"repairability"`), and
the durations of its preventive maintenance and of its tests
(`"preventive.duration"`, `"inspection.duration"`), named as
`parameter_sensitivity`'s levers. `uncertainty=` says which are uncertain,
and how, by node, by a tuple of the nodes of one population, or by
common-cause group (its model's parameters):

```python
import scipy.stats as st

known = {
    ("A", "B"): {
        "reliability": {"failure_rate": st.uniform(0.08, 0.04)},   # known to within 20%
        "repairability": "fit",
    },
    "C": "fit",   # every fitted model of the valve's
}
fitted_plant.mean_availability_uncertainty(known, n_draws=1000, seed=0).interval(0.9)
# (0.9385, 0.9600)
```

A role's uncertainty is given as a non-repairable node's is: `"fit"`,
distributions over its parameters, or a list of models (refits to bootstrap
resamples, say); one given without a role is the life's. Every draw of a
population, or of a common-cause group's members, gives them all the same
models.

**Whose uncertainty widens the interval.** `uncertainty_importance` splits
the variance among the inputs: by the delta method by default
(`parameter_sensitivity`'s derivatives, with the parameters' covariance),
whose shares add up to 1, or by Sobol indices estimated from draws
(`method="sobol"`):

```python
vega = fitted_plant.uncertainty_importance(uncertainty=known)
vega.first_order["C"]          # -> 0.88
vega.first_order[("A", "B")]   # -> 0.12
```

The valve's life is most of it, so more valve failures would narrow the
interval most. `of=` picks the quantity: the long-run availability (the
default), the availability at the times `x` (`"point_availability"`) or
over missions of the lengths `x` (`"mission_availability"`), from new or
from `state=`, or the cost rate (`"expected_cost_rate"`). See
[Sensitivities](greeks.md#vega-whose-uncertainty-widens-the-answer) for a
station with maintenance.

- **One evaluation a draw.** In the long run a draw takes a fraction of a
  millisecond; over time, the components' curves, tens of milliseconds for
  one that wears out.
- **Quasi-random draws.** `sampling="sobol"` takes the draws from a
  scrambled Sobol sequence rather than from random numbers: points that
  fill the parameters' range evenly, so the mean and the percentiles settle
  with fewer draws. Here 256 such draws put the mean availability within
  about `1e-5` of where 20,000 put it, and 256 random draws `3e-4` away.
  The non-repairable methods take it too.
- **What the diagram refuses, the draws do.** Each draw's value is worked
  out as the diagram's own, so the uncertainty is refused where the value
  is, and `analysis_routes()` says so.
- **More draws do not narrow the interval**, they only place its ends more
  precisely. More failure data, refitted in surpyval, narrows it.

## Availability over time (simulated)

`availability(t_simulation, ...)` runs `mc_samples` independent simulations of the
system from time 0 (everything new, except any `broken_nodes`, or from the
components' [current states](#from-the-plant-as-it-is-now) with `state=`) to
`t_simulation`, and averages them: the same curve as `point_availability`,
with the histories behind it, which also give the failure counts, downtime,
costs and criticality measures over the window:

```python
result = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
result.timeline[:3]       # array([0.    , 0.0577, 0.0624])  times the mean availability changes
result.availability[:3]   # array([1.    , 0.9995, 0.999 ])  mean availability at those times
result.availability[-1]   # -> 0.9600   at t = 100
np.interp(50, result.timeline, result.availability)   # -> 0.9553   at t = 50
```

| Argument | Meaning |
|---|---|
| `t_simulation` | The length of each simulated history. |
| `mc_samples` | The number of histories `N` (default 10 000). Error shrinks like `1/√N`. |
| `seed` | Seeds the run for reproducibility: each component draws from random streams of its own (see [Random streams](simulation.md#random-streams)). numpy's global RNG is left as it was. |
| `working_nodes`, `broken_nodes` | Components that never fail, or that are down throughout. |
| `method` | `"p"` or `"c"`, for deciding whether the system is up; same result. |
| `verbose` | Show a progress bar. |
| `tolerance`, `confidence`, `max_samples` | Simulate until the mean availability over the window is known to within `tolerance` (see [Simulation precision and speed](simulation.md#simulating-to-a-tolerance)). |
| `antithetic` | Simulate in antithetic pairs, for a more precise mean from the same `mc_samples` (see [Antithetic pairs](simulation.md#antithetic-pairs)). |
| `n_jobs` | Run the simulations on several CPUs, with the same result as on one (see [Parallel runs](simulation.md#parallel-runs)). |
| `engine` | `"python"`, `"numba"` (compiled) or `"auto"`, the default: the same results, faster compiled (see [The compiled engine](simulation.md#the-compiled-engine)). |
| `demand` | With node capacities, the demand the delivered fraction is measured against (see [System capacity](capacity.md#over-a-window-simulated)). |
| `control_variate`, `conditional` | How the mean over the window is estimated: by default exactly where the exact methods take the system, and given the histories of the few nodes they do not take otherwise (see [Exact or simulated?](simulation.md#exact-or-simulated)); `control_variate=False` keeps the simulations' own mean. |

The curve starts at 1 and settles towards the long-run availability
(`0.9536` here). Its sampling error is available pointwise:

```python
result.availability_se[-1]                            # -> 0.004382   standard error at t = 100
lower, upper = result.availability_interval(confidence=0.95)   # Wilson band
lower[-1], upper[-1]                                  # (0.950, 0.9677)
```

`lower`/`upper` align with `result.timeline`, ready to draw as a band. The
mean availability over the whole window, the fraction of it the system was
up, has an interval of its own. This plant's components fail and are
repaired independently, so its mean over the window is exact
(`plant.mission_availability(100.0)`), and by default the interval is that
value, with no error; `control_variate=False` keeps the simulations' own:

```python
result.mean_availability_interval().estimate    # -> 0.9544   exact
own = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0,
                         control_variate=False)
window = own.mean_availability_interval(confidence=0.95)
window.estimate                   # -> 0.9543   simulated
window.lower, window.upper        # (0.9526, 0.9559)
```

To compare two designs, `faster.compare(plant, t_simulation)` gives how
much more of the window one is up than the other: exactly where both
designs' mission availabilities are worked out, and otherwise by simulating
them with common random numbers, far more precisely than two separate runs
(see [Comparing two designs](simulation.md#comparing-two-designs)).

### What the result holds

`availability()` returns an [`AvailabilityResult`][repyability.AvailabilityResult]:

| Attribute | Meaning |
|---|---|
| `timeline`, `availability` | The mean availability curve. |
| `availability_se`, `availability_interval(confidence)` | Its sampling error. |
| `uptimes`, `mean_availability_interval(confidence)` | Each history's up time, and the interval of the mean availability over the window. |
| `antithetic` | Whether the histories ran in antithetic pairs. |
| `system_uptime`, `system_downtime` | Total up and down time over all histories. |
| `system_failures`, `system_restorations` | Counts over all histories. |
| `system_planned_outages` | The times preventive maintenance or a test took the system down (not failures). |
| `node_uptime`, `node_downtime` | Per-node totals. |
| `mean_up_time`, `mean_down_time`, `failure_frequency` | Simulation estimates of the exact MUT, MDT and frequency above. |
| `n_simulations`, `time_simulated_to` | `mc_samples` and `t_simulation`. |
| `criticalities` | The criticality measures (below). |
| `cost` | The simulated costs, or `None` when nothing is priced (see [Costs](costs.md#the-simulated-cost-distribution)). |
| `capacity_timeline`, `capacity`, `capacity_time`, `mean_capacity`, `demand`, `delivered`, `delivered_fraction`, `delivered_fraction_interval(confidence)` | With node capacities: the mean capacity over time, the time at each capacity, and the fraction of the demand delivered (see [System capacity](capacity.md#over-a-window-simulated)). `None` without capacities. |

```python
result.mean_up_time      # -> 28.08    against the exact 27.27
result.failure_frequency # -> 0.033985 against the exact 0.03497
```

Its values are attributes and properties; what takes a confidence level,
an interval, is a method. The result also behaves as a read-only mapping
(`result["availability"]`, `result.keys()`, `dict(result)`), so dict-style
code keeps working, and `result.to_dict()` gives it as plain data, ready for
`json.dumps` (#235).

### Criticality measures

`result.criticalities` holds four measures computed from the simulated
histories. Each has an "up" and a "down" (or a "system" and a "component")
view:

```python
c = result.criticalities
c.operational_criticality_index.down   # {'A': 0.2407, 'B': 0.2405, 'C': 0.8334}
c.failure_criticality_index.per_system_failure   # {'A': 0.2247, 'B': 0.2238, 'C': 0.5516}
c.failure_criticality_index.per_system_failure["C"]   # -> 0.5516
```

| Measure | `up` / `by_system` / `per_system_failure` | `down` / `by_component` / `per_component_failure` |
|---|---|---|
| `operational_criticality_index` | Time node and system were both up, over system uptime. | Time node and system were both down, over system downtime. |
| `iou` | Intersection over union of the node's and the system's up time. | Same for down time. |
| `failure_criticality_index` | Fraction of system failures this node's failure caused. | Fraction of this node's failures that failed the system. |
| `restoration_criticality_index` | Fraction of system restorations this node's repair caused. | Fraction of this node's repairs that restored the system. |

Here the valve caused 55% of system failures although it fails a fifth as
often as a pump, and 99% of its failures took the system down: the pumps are
redundant and it is not. A node's failure "causes" a system failure when it
is the event that takes the system from up to down.

To keep each simulation's histories whole rather than these totals, every
component's and the system's with the component behind each system
failure, use `simulate_timelines` (see [Timelines](timelines.md)).

## Repair crews

By default every failed component is repaired at once, as though each had a
crew of its own. `repair_crews` limits how many repairs can proceed at once.
A plant with one technician repairs one pump while the next waits, so it is
down more often, and for longer:

```python
pump = {"reliability": surv.Exponential.from_params([0.1]),     # MTTF 10 h
        "repairability": surv.Exponential.from_params([0.5])}   # MTTR 2 h
three = [("s", p) for p in "xyz"] + [(p, "t") for p in "xyz"]
one_crew = RepairableRBD(three, {p: dict(pump) for p in "xyz"}, repair_crews=1)
one_crew.mean_availability()    # -> 0.9746
result = one_crew.availability(20_000.0, mc_samples=40, seed=1)
result.system_uptime / (40 * 20_000.0)          # -> 0.9750   simulated
RepairableRBD(three, {p: dict(pump) for p in "xyz"}).mean_availability()   # -> 0.9954   a crew each
```

(With one crew the number of pumps down is the machine-repair model's
birth-death chain, and the system is down when all three are.)

- **What needs a crew.** Every job that brings a component back up: a repair
  or replacement, preventive maintenance that takes time, and a test that
  takes time (with the repair of any failure it finds). The component is
  down from when the job falls due until it is done. Maintenance or a test
  in no time needs no crew.
- **The queue.** A job that finds every crew busy waits. The next crew to
  finish takes the waiting job of the highest `"priority"` (a component
  spec key, by default 0), and of those the one that fell due first, and
  stays with it until it is done. A test that waits keeps the component
  off-line, and it does not age. A job with `"instant"` repair takes a crew
  for no time: it waits only when every crew is busy.
- **Nested RBDs** have crews of their own: a nested `RepairableRBD`'s
  components are repaired by its `repair_crews`, not by its parent's.
- **Exact long-run values.** With fewer crews than components, components
  wait for each other, so they no longer fail and recover independently.
  When the components the crews work on all have exponential lives and
  exponential (or instant) repairs, with no scheduled maintenance or
  inspection, the system is a Markov chain: its state is which components
  are under repair and which are waiting, in the order the crews will take
  them. `mean_availability`, `node_availability`,
  `system_failure_frequency`, `mean_up_time`, `mean_down_time`,
  `mean_time_between_failures`, `expected_cost_rate`, `total_cost` and
  `capacity_distribution` solve it exactly, for up to 15,000 states. A
  component held working or broken (`working_nodes`, `broken_nodes`) needs
  no crew, and the others share them.
- **Over time.** From new, or from the components' states, the same chain
  is followed over time by uniformization: its state after a Poisson
  number of steps of a discrete chain, taken until it has settled at its
  long run, to about 1e-13. `point_availability`, `mission_availability`,
  `expected_failures`, `expected_events`, `expected_cost`, `point_capacity`
  and `mission_capacity` come from it. A nested RBD, with crews of its own,
  is independent of the chain: the availability over time is worked out for
  each pattern of the nested RBDs up and down, weighted by their own
  availabilities. Their failures count as another node's would, each at
  its importance over the chain and the other nested RBDs' patterns, and
  the capacity is worked out for each combination of their levels, the
  integrals by quadrature (#162).
- **Importance.** Under dependence the textbook formulas, products of the
  components' availabilities, no longer hold, so the measures are taken
  from their definitions: Birnbaum's is the system's long-run availability
  with the component held working less that with it held failed, each from
  the chain solved without it (held, it needs no crew), and the improvement
  potential, RAW and RRW are built on the same values; the criticality and
  Fussell-Vesely measures are probabilities over the chain's states. Here a
  pump held down leaves the other two to share the crew, so it matters
  about twice as much as with a crew each. With a crew for each component
  they are the independent ones.
- **What is simulated.** Other lives or repair times, scheduled maintenance
  and inspections, and larger chains make the exact values refuse, with the
  reason. The allocations assume independent components, so they refuse
  whenever a job can wait. The simulations (`availability`, `cost`,
  `compare`) follow the queue whatever the components, in Python. With at
  least as many crews as components, nothing waits, and every result is as
  without crews.

```python
one_crew.point_availability(5.0)       # -> 0.9865   five hours in
one_crew.mission_availability(24.0)    # -> 0.9805   over the first day
one_crew.birnbaum_importance()["x"]    # -> 0.0541   (0.0278 with a crew each)
routes = one_crew.analysis_routes()
routes["mean_availability"].route     # 'exact'
routes["point_availability"].route    # 'numerical'
routes["birnbaum_importance"].route   # 'exact'
```

The chain's size is set by the queue. First come, first served, every order
in which the waiting components can have failed is a state of its own: seven
components with one crew make 13,700 states, and eight with two crews make
54,805, too many. Priorities fix much of the order, so a priority each lets
eight components with two crews through in 1,801 states. The chain is solved
in well under a second for most diagrams, and in a few seconds near the
limit.

## Standby groups

A duty pump with a standby is not two pumps in parallel: the standby waits,
unused, until the duty pump fails, and may fail to start. Give a component
spec a `"standby"` dict and the node becomes a group of identical units,
`"k"` of them operating and the rest waiting as spares:

```python
pump = {"reliability": surv.Exponential.from_params([0.01]),     # MTTF 100 h
        "repairability": surv.Exponential.from_params([0.1]),    # MTTR 10 h
        "standby": {"units": 2, "switching_probability": 0.98}}
pumps = RepairableRBD([("s", "pumps"), ("pumps", "t")], {"pumps": pump},
                      repair_crews=1)
pumps.mean_availability()    # -> 0.9894
pumps.mean_down_time()       # -> 10.0   hours: until the first repair ends
pumps.point_availability(10.0)    # -> 0.9963   ten hours from new
result = pumps.availability(50_000.0, mc_samples=40, seed=1)
result.system_uptime / (40 * 50_000.0)          # -> 0.9889   simulated
```

(One pump alone is up 0.9091 of the time; with a switch that always works,
the pair is up 0.9910.)

- **The units.** `"units"` (by default 2) identical units, `"k"` (by default
  1) of which must operate for the group to be up. Each fails and is
  repaired as the spec's `"reliability"` and `"repairability"` say, and
  comes back as new.
- **Spares.** A spare ages at `"dormancy_factor"` of an operating unit's
  rate: 0 (the default) for cold standby, 1 for hot, anything between for
  warm. A spare that fails in standby is found at once, and repaired.
- **Switching.** When an operating unit fails, the spare that has waited
  longest is switched in, which works with `"switching_probability"` (by
  default 1). A failed switch leaves the position empty until a repaired
  unit fills it; the spare waits on. With a probability of 0, a cold
  standby group is a single unit.
- **Repairs.** Each failed unit is repaired on its own: a job for the
  RBD's repair crews (see [above](#repair-crews)) at the group's
  `"priority"`, so the group's units and the other components wait for the
  same crews. A repaired unit fills an empty position, or joins the spares.
- **Costs.** `"repair_cost"` and `"replace_cost"` are charged at each unit's
  failure, and `"downtime_cost"` while the group is down.
- **Exact values.** When the units' lives and repair times are exponential,
  the group is a small Markov chain (how many units operate, wait and are
  under repair), and its long-run availability, failure frequency and costs
  are exact. They enter the RBD's exact long-run values, and the importance
  measures, like any component's. They stay exact with limited crews while
  the group's units are the crews' only jobs (the textbook "one repairman"
  case, as here); crews shared with other components tie the group to them,
  and the exact values refuse. Over time, from new (every unit ready) or
  from its long-run state, the same chain is followed by uniformization (see
  [above](#repair-crews)): the availability over time and over a mission,
  and the expected failures and repairs, are numerical.
- **Simulation.** `availability`, `cost` and `compare` simulate groups
  whatever their units' models, in Python.

## Imperfect repair

A patched pump is still an old pump. By default every repair renews a
component, as good as new; a spec's `"repair"` makes its repairs imperfect,
by Kijima's virtual-age models (as surpyval's `GeneralizedRenewal` and
[`Repairable`](maintenance.md#imperfect-repair-repairable) do):

| Key | Meaning |
|---|---|
| `"repair"` | `{"model": "kijima1" or "kijima2", "q": q}`. A repair after the unit has operated `x` since the last one takes its virtual age from `v` to `v + q x` (Kijima I: the repair undoes a fraction `1 - q` of the age added since the last) or `q (v + x)` (Kijima II: of all of it). `q = 0` renews it (the default), `q = 1` is minimal repair, "as bad as old". |
| `"replace_after"` | `N`: the `N`-th failure since the unit was renewed replaces it, as new, instead of repairing it. |

Each life is drawn given the unit's virtual age `v`, `H(v + X) = H(v) +
E` with `E` exponential, as surpyval draws it. A pump that wears out,
repaired in about 23 h, with lost production at 500 an hour, over five
years from new:

```python
def pump(**repair):
    return {"reliability": surv.Weibull.from_params([1000, 2.5]),     # MTTF 887 h
            "repairability": surv.LogNormal.from_params([3.0, 0.5]),  # 23 h
            "repair_cost": 500.0, "replace_cost": 5000.0, **repair}

def line(**repair):
    return RepairableRBD([("s", "p"), ("p", "t")], {"p": pump(**repair)},
                         downtime_cost_rate=500.0)

five_years = 43_800.0
renewed = line().availability(five_years, mc_samples=20, seed=1)
patched = line(repair={"model": "kijima1", "q": 0.5}).availability(
    five_years, mc_samples=20, seed=1)
renewed.system_uptime / (20 * five_years)         # -> 0.9757
patched.system_uptime / (20 * five_years)         # -> 0.5264
```

| Repair | Up | Failures a year | Cost an hour |
|---|---|---|---|
| As new (`q = 0`) | 0.976 | 9.4 | 17.9 |
| Kijima I, `q = 0.25` | 0.683 | 122 | 166 |
| Kijima I, `q = 0.5` | 0.530 | 181 | 246 |
| Minimal (`q = 1`) | 0.389 | 235 | 319 |
| Kijima II, `q = 0.5` | 0.957 | 16.6 | 22.5 |
| Kijima II, `q = 0.9` | 0.890 | 42.5 | 57.5 |
| Kijima I, `q = 0.5`, replaced at the 2nd failure | 0.970 | 11.8 | 19.1 |
| Kijima I, `q = 0.5`, replaced at the 4th failure | 0.961 | 15.3 | 22.8 |

Under Kijima I the virtual age only grows, so a unit that is never renewed
fails ever more often; under Kijima II it settles. Replacing the unit
every few failures (or on a preventive schedule) bounds it.

**A model fitted to the unit's history.** A spec's `"reliability"` may be
what surpyval fits to a repairable unit's failure times (#269), and the
component is then the life and repair that model is:

- a Poisson process, `CrowAMSAA`, `Duane` or `HPP`, is minimal repair of the
  life whose cumulative hazard is the process's cumulative intensity
  (Crow-AMSAA's `(t / alpha) ** beta` is a Weibull life, and a homogeneous
  process an exponential one), so with repairs in no time its expected
  failures are the process's own;
- a `GeneralizedRenewal` (Kijima I or II) is its life distribution with the
  `"repair"` of its Kijima model and restoration factor `q`.

```python
from surpyval import CrowAMSAA

gearbox = CrowAMSAA.from_params([1500, 0.9])
rbd = RepairableRBD([("s", "g"), ("g", "t")],
                    {"g": {"reliability": gearbox, "repairability": "instant"}})
rbd.expected_failures(8760.0)   # -> 4.8952, gearbox.cif(8760.0)
```

Such a spec takes no `"repair"` of its own, and is saved as the life and
repair it is. surpyval's other renewal models (ARA, ARI, G1) are refused:
their repairs are not Kijima's.

- **Costs and spares.** A repair is charged its `"repair_cost"`; a
  replacement, at the `N`-th failure or at any failure of a unit renewed by
  its repairs, its `"repair_cost"` and `"replace_cost"`, and uses a spare
  (see [Spares](spares.md)).
- **Maintenance.** A preventive replacement renews the unit; under age
  replacement its age is its operating time since it was renewed, which
  repairs do not reset. Proof tests find its hidden failures as for any
  component, and a found failure is repaired imperfectly too. A standby
  group, or replacement on condition, cannot be repaired imperfectly.
- **Exact or simulated.** A repair does not renew the unit, so the exact
  long-run values, the availability over time and the spares counts refuse
  it, with the reason; `availability`, `cost` and `compare` simulate it, in
  Python, with its own streams (seeds, antithetic pairs and common random
  numbers work as for any component). `q = 0`, and `replace_after=1`
  whatever `q`, are the component renewed at every failure, draw for draw.
- **Minimal repair in no time is exact over a window.** Repaired minimally
  (`q = 1`) and instantly, with no `"replace_after"`, preventive maintenance
  or tests, the unit is up throughout, and its failures are a Poisson
  process whose rate is its life's hazard at its age: it fails `H(t)` times
  by `t` on average, `H` the life's cumulative hazard, exactly. The values
  over a window from new (`point_availability`, `mission_availability`,
  `expected_failures`, `expected_events`, `expected_cost`) take it in, as
  numerical as any component's; its long-run values still refuse, as its
  rate of failures need not settle.

```python
patched = RepairableRBD(
    [("s", "p"), ("p", "t")],
    {"p": {"reliability": surv.Weibull.from_params([1000, 2.5]),
           "repairability": "instant",
           "repair": {"model": "kijima1", "q": 1.0},
           "repair_cost": 500.0}},
)
patched.expected_failures(8760.0)     # -> 227.12   = (8760 / 1000) ** 2.5
patched.expected_cost(8760.0).mean    # -> 113561   500 a repair
```

## Instantly repaired components

When repairs are much faster than the time scale of interest, or there is no
repair-time data, give `"repairability": "instant"`. The component still
fails (and any repair or replacement cost is charged), but every outage has
zero length, so its availability is exactly 1:

```python
fuse = RepairableRBD(
    [("s", "f"), ("f", "t")],
    {"f": {"reliability": surv.Exponential.from_params([0.1]),
           "repairability": "instant"}},
)
fuse.mean_availability()                                     # -> 1.0
fuse.availability(50.0, mc_samples=100, seed=0).availability.min()    # -> 1.0
```

Invisible to availability, visible to cost: see [Costs](costs.md).

## Nested repairable RBDs

A `RepairableRBD` can be a component of another. The nested RBD runs its own
simulation on the same clock, and the outer system sees it change state when
its own system state changes. The exact quantities recurse:

```python
pair = RepairableRBD(
    [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
    {"A": unit(0.1, 1.0), "B": unit(0.1, 1.0)},
)
nested = RepairableRBD(
    [("s", "pumps"), ("pumps", "C"), ("C", "t")],
    {"pumps": pair, "C": unit(0.02, 0.5)},
)
nested.mean_availability()         # -> 0.9536   the same system as `plant`
nested.system_failure_frequency()  # -> 0.03497
nested.availability(t_simulation=100.0, mc_samples=2_000, seed=0).availability[-1]   # -> 0.959
```

Use one `RepairableRBD` object per place it appears: the same object used for
two nodes shares one simulation state and raises `ValueError` during
`availability()`. (One `NonRepairable` object *can* be reused; each node gets
its own copy.)

```python
pump = NonRepairable(surv.Weibull.from_params([30, 1.5]), surv.Exponential.from_params([0.5]))
twin_pumps = RepairableRBD(
    [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")], {"a": pump, "b": pump}
)
twin_pumps.components["a"] is twin_pumps.components["b"]   # False: separate copies
twin_pumps.mean_availability()                             # -> 0.9953
```

## Stepping a simulation by hand

`initialize_event_queue(t_simulation)` sets up one history, and each call to
`next_event()` advances it to the next change in the *system* state,
returning `(time, system_is_up)`. At the end of the window it returns
`(t_simulation, state)`; calling it again raises `ValueError` until the queue
is initialised again.

```python
np.random.seed(3)
plant.initialize_event_queue(100.0)
events = [plant.next_event()]
while events[-1][0] < 100.0:
    events.append(plant.next_event())
events[0]   # (17.18..., False): the plant first went down at t = 17.19
```

`initialize_event_queue(t_simulation, state=...)` starts the history from
the components' [current states](#from-the-plant-as-it-is-now), as
`availability(state=...)` does.

This is the interface a nested RBD presents to its parent; `availability()`
drives the same machinery, with each component drawing from its own random
streams. Stepped by hand, the components draw from numpy's global RNG as it
is (there is no `seed` argument), so a history stepped by hand is not one of
`availability()`'s.
