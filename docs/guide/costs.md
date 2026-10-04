# Costs

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson 7](../learn/costs.md)
    (what it costs) and [Lesson 8](../learn/maintenance.md) (preventive
    maintenance).

Availability says how often a system is up; the next question is usually
what running it costs. Price the components, and the production lost while
the system is down, and a [`RepairableRBD`][repyability.RepairableRBD] gives
the long-run cost rate exactly and the distribution of the cost over a window
by simulation. Theory: [Concepts](../concepts.md#costs).

## Pricing a system

Costs are optional keys of a component's dict, plus one system-level rate:

| Where | Key | Charged |
|---|---|---|
| Component | `repair_cost` | Per failure (labour, a callout). A number or a distribution. |
| Component | `replace_cost` | Per failure (the spare part); for a component repaired imperfectly, only per replacement (see [Imperfect repair](repairable.md#imperfect-repair)). A number or a distribution. |
| Component | `downtime_cost` | Per unit time *this component* is down, even if the system is up (a degraded-mode or per-leg penalty). A number. |
| Component | `acquisition_cost` | Once, to buy the unit. A number. Not a running cost: see [the total cost of ownership](#the-total-cost-of-ownership). |
| System | `downtime_cost_rate=` | Per unit time the *system* is down (lost production). A number. |

```python
import numpy as np
import surpyval as surv
from repyability import BetaFactor, CCFGroup, RepairableRBD

def unit(failure_rate, repair_rate, **costs):
    return {
        "reliability": surv.Exponential.from_params([failure_rate]),
        "repairability": surv.Exponential.from_params([repair_rate]),
        **costs,
    }

edges = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]
plant = RepairableRBD(
    edges,
    {
        "A": unit(0.1, 1.0, repair_cost=200.0),
        "B": unit(0.1, 1.0, repair_cost=200.0),
        "C": unit(0.02, 0.5, repair_cost=500.0, replace_cost=1500.0),
    },
    downtime_cost_rate=1000.0,
)
plant.has_costs   # True
```

Every cost defaults to 0, so any subset can be priced. Costs are
undiscounted (but for the [total cost](#discounting)'s optional discount
rate), and every failure is repaired at the same price (preventive
replacement is [below](#preventive-maintenance)). A cost must be finite and
non-negative; an unknown key (such as `repair_costs`) raises `ValueError`
rather than being priced at zero.

## The long-run cost rate (exact)

```python
plant.expected_cost_rate()   # -> 121.23   cost per unit time, in the long run
```

It is the sum of three exact terms:

```
cost rate = downtime_cost_rate · (1 − A_sys)              lost production
          + Σ ω_i · (repair_cost_i + replace_cost_i)       corrective actions
          + Σ (1 − A_i) · downtime_cost_i                  per-component downtime
```

where `A` is availability and `ω_i = 1 / (MTTF_i + MTTR_i)` is component *i*'s
long-run failure frequency. Here that is `46.41` of lost production, `18.18`
for each pump's repairs, and `38.46` for the valve's. A cost given as a
distribution enters through its mean. `expected_cost_rate` accepts
`working_nodes`/`broken_nodes`; a forced component never fails, so it incurs
no corrective cost:

```python
plant.expected_cost_rate(working_nodes=["A"])   # -> 95.1
```

With nothing priced, `has_costs` is `False` and the rate is `0.0` without any
work being done.

## The simulated cost distribution

The exact rate is a mean. Budgeting usually needs the spread: what could a
bad year cost? `cost(t_simulation, ...)` runs the availability simulation with
cost accumulation and returns a [`CostResult`][repyability.CostResult]: one
total cost per simulated window.

```python
costs = plant.cost(t_simulation=1000.0, mc_samples=500, seed=0)
costs.mean              # the expected cost of a window: exact here
costs.cost_rate         # mean / t_simulation: tends to expected_cost_rate()
costs.percentile(90)    # a planning budget: 9 windows in 10 cost less
costs.std               # how much a window's cost varies
costs.by_category       # mean split: repair, replace, preventive, inspection, component_downtime, system_downtime, setup
costs.by_component      # the part of mean each costed component accounts for
```

`cost()` takes the same arguments as `availability()` (`working_nodes`,
`broken_nodes`, `method`, `mc_samples`, `verbose`, `seed`, and `tolerance`,
`antithetic`, `n_jobs` and `engine`: see
[Simulation precision and speed](simulation.md)). The same result comes with
`availability(...)` as `result.cost`, so one simulation gives both answers.
With nothing priced, `cost()` returns `None` and `result.cost` is `None`.

### One expected value

`mean` is the run's estimate of a window's expected cost, the one
`mean_interval()` gives an interval for, and `cost_rate` and the breakdowns
follow it. By default it is exact where the exact methods work it out, as
here (`expected_cost`, below), and otherwise taken given the histories of
the system's dependent modules, or the simulations' own (see
[exact or simulated?](simulation.md#exact-or-simulated)). The
simulations' own mean, the average of `samples`, is `sample_mean`; the repr
shows both:

```python
costs.mean_interval().method              # 'exact'
costs.mean == plant.expected_cost(1000.0).mean   # True
round(costs.mean, 1)                      # -> 121155.1
round(costs.sample_mean, 1)               # -> 121702.6   the 500 windows' own
```

### Two different uncertainties

- `std` and `percentile` describe how much a window's cost **varies**. That
  is a property of the system; more simulations will not shrink it.
- `mean_se` and `mean_interval(confidence)` describe how precisely the
  **expected** cost has been estimated: exactly, here, with no error; for a
  mean that is simulated they shrink like `1/√N`. Check them before quoting
  the mean, or pass `tolerance` to `cost()` to simulate until the interval
  is narrow enough. With `control_variate=False` (and `conditional=False`),
  the mean and its interval are the simulations' own:

```python
own = plant.cost(t_simulation=1000.0, mc_samples=500, seed=0, control_variate=False)
interval = own.mean_interval(confidence=0.95)
interval.method                                                       # 'simulated'
round(interval.standard_error, 1)                                     # -> 850.0
interval.lower < plant.expected_cost(1000.0).mean < interval.upper    # True
```

`by_category` sums to `mean`; `by_component` covers each component's repair,
replace, preventive, inspection and own downtime cost (lost production is a
system cost, and a maintenance group's set-up cost the group's: neither is
attributed to a component). Under `control_variate=True`, whose twin
controls only the total, the breakdowns are the simulations' own, and sum
to `sample_mean`.

## The expected cost of a window (exact)

The long-run rate times a window is the window's expected cost only once the
components have settled: from new, the first stretch costs less (or more).
`expected_cost(t)` gives the expected cost of `[0, t)` from new exactly (or,
with `state=`, from the components' [current
states](repairable.md#from-the-plant-as-it-is-now): a repair or maintenance
going on at 0 was charged when it began, before the window), by the same
categories as the simulation:

```python
window = plant.expected_cost(1000.0)
window.mean           # -> 121155   cost() above estimates 121703 ± 1666
window.by_category    # repair 45983.07, replace 28848.37, system_downtime 46323.67, the rest 0
window.by_component   # {'A': 18183.47, 'B': 18183.47, 'C': 38464.5}
window.cost_rate      # -> 121.16   per unit time, against 121.23 in the long run
plant.expected_cost([10.0, 100.0]).cost_rate   # array([113.45, 120.45])
```

Each category is its events' expected number over the window (see
[Repairable systems](repairable.md#expected-events-over-a-window-exact))
times its mean cost: the repair and replace costs at each corrective action,
the preventive cost at each preventive replacement, the inspection cost at
each test, the downtime rates over the expected downtimes, and a maintenance
group's set-up once per stop (replacements due at one instant are one
stop). It returns an [`ExpectedCost`][repyability.ExpectedCost], with
`mean`, `by_category`, `by_component`, `acquisition_cost`, `total` (the two
added) and `cost_rate`. It covers what the availability over time covers,
and refuses the rest with the reason; with nothing priced, every category is
0. For the spread of a window's cost, simulate it with `cost()`.

## Costs drawn from distributions

`repair_cost` and `replace_cost` can be a distribution of the cost instead of
a number, such as a surpyval model fitted to past invoices. The exact rate
uses its mean; the simulation draws a fresh cost at every failure:

```python
invoices = surv.LogNormal.from_params([5.0, 0.5])   # mean ≈ 168
variable = RepairableRBD(
    edges,
    {
        "A": unit(0.1, 1.0, repair_cost=invoices),
        "B": unit(0.1, 1.0, repair_cost=invoices),
        "C": unit(0.02, 0.5),
    },
)
variable.expected_cost_rate()   # -> 30.58   = 2 × 168.17 / 11
```

- The distribution must have a finite mean and no appreciable probability of
  a negative cost.
- The cost draws come from random streams of their own (seeded from the
  run's seed), so pricing never changes the failure and repair histories: a
  seeded `availability()` gives the same availability with or without costs.
- The downtime costs must be numbers: they are rates, and the outage
  durations already make them random.

## Instantly repaired components

A component with `"repairability": "instant"` has no downtime but still
fails, so its per-failure costs are charged at its failure rate:

```python
fuse = RepairableRBD(
    [("s", "f"), ("f", "t")],
    {"f": {"reliability": surv.Exponential.from_params([0.1]),
           "repairability": "instant",
           "replace_cost": 40.0}},
)
fuse.expected_cost_rate()   # -> 4.0   = 40 × 0.1 failures per unit time
fuse.mean_availability()    # -> 1.0
```

## Preventive maintenance

A component's dict can also schedule preventive replacement, under the key
`"preventive"`, so that the cost and availability outputs price the trade
between preventive and corrective maintenance:

| Key | Meaning |
|---|---|
| `interval` | Required: the replacement interval `T` (`inf`: never). |
| `policy` | `"age"` (the default): `T` after the unit was last put into service as new, so a failure restarts the clock. `"block"`: at `T, 2T, 3T, …` whatever the unit's age, skipped while it is down. `"condition"`: inspected at `T, 2T, 3T, …`, and replaced if likely to fail before the next inspection (see [below](#replacement-on-condition)). |
| `duration` | `"instant"` (the default): renewed in place, never down. Or a time-to-maintain model: the unit is down meanwhile, a *planned outage*. |
| `cost` | Charged at each preventive replacement: a number or a distribution. |

A replacement renews the unit, so the failure it was heading for never
happens. A failure due at the same time as a replacement comes first.

`NonRepairable.find_optimal_replacement()` finds a component's best
replacement age on its own ([Maintenance policies](maintenance.md)). At
system level the answer changes, because a planned stop of a unit with a
standby costs almost no production, while the same stop of a single point of
failure halts it. Take a pump that wears out, is repaired in about 23 h and
replaced preventively in about 7 h, alone or with a standby, with lost
production at 500 per hour:

```python
def pump(interval, policy="age"):
    return {
        "reliability": surv.Weibull.from_params([1000, 2.5]),        # MTTF 887 h
        "repairability": surv.LogNormal.from_params([3.0, 0.5]),     # 23 h
        "replace_cost": 5000.0,
        "preventive": {
            "interval": interval,
            "policy": policy,
            "duration": surv.Weibull.from_params([8, 3]),            # 7 h
            "cost": 1000.0,
        },
    }

def alone(interval, policy="age"):
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": pump(interval, policy)},
        downtime_cost_rate=500.0,
    )

def with_standby(interval, policy="age"):
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": pump(interval, policy), "b": pump(interval, policy)},
        downtime_cost_rate=500.0,
    )

alone(float("inf")).expected_cost_rate()          # -> 18.0    run to failure
alone(580).expected_cost_rate()                   # -> 13.14   the best interval
with_standby(float("inf")).expected_cost_rate()   # -> 11.3
with_standby(500).expected_cost_rate()            # -> 6.985   the best interval
```

| Interval (h) | 300 | 400 | 500 | 600 | 800 | never |
|---|---|---|---|---|---|---|
| Alone | 16.92 | 14.36 | 13.35 | 13.14 | 13.84 | 18.00 |
| With a standby | 8.19 | 7.21 | 6.99 | 7.15 | 8.01 | 11.30 |

Replacement pays in both, but not at the same interval: the lone pump's
planned stops cost production, so it is best replaced less often, at about
580 h; replacing a pump with a standby costs almost no production, so the
pair is best replaced at about 500 h. Judged on its own, as `NonRepairable`
judges it (`cp=1000`, `cu=5000`), the pump is best replaced at 493 h.

### Choosing the intervals

`optimal_replacement_intervals` finds the best interval of every component
under age replacement at once, from the exact long-run values, rather than
sweeping them one at a time. The components are chosen together: the one
alone in the line is replaced later than the ones with a standby, because
its own replacements stop the plant:

```python
alone(1000).optimal_replacement_intervals().intervals["p"]   # -> 589.6
plan = with_standby(1000).optimal_replacement_intervals()
plan.intervals["a"]     # -> 497.1   and the same for "b"
plan.cost_rate          # -> 6.985
```

It can also keep the system's long-run availability to a target at the least
cost (`min_availability`), or give the most availability within a cost rate
(`max_cost_rate`):

```python
plan = alone(1000).optimal_replacement_intervals(min_availability=0.9807)
plan.intervals["p"]     # -> 604.8   a little later than the cheapest
plan.cost_rate          # -> 13.139  against 13.134
```

It returns a [`MaintenancePlan`][repyability.MaintenancePlan]: the intervals
(`inf` for a component better never replaced), and the system's cost rate
and availability with them. The search starts from the intervals given, and
from others around each component's mean life, and keeps the best; a target
no intervals can meet raises `ValueError`, with the best they can do. The
cost rate is usually flat near its minimum, so an interval some way from the
one found costs almost the same.

Replacements are often made on a calendar. `allowed` gives the intervals to
choose from (#230), one list for every component or a dict of a list each
(`inf` among them for never), and every combination is tried, as
`optimal_inspection_intervals` chooses tests':

```python
weeks = [336.0, 672.0, 1008.0, 1344.0]    # every 2, 4, 6 or 8 weeks
plan = with_standby(1000).optimal_replacement_intervals(allowed=weeks)
plan.intervals          # {'a': 672.0, 'b': 672.0}
plan.cost_rate          # -> 7.405   against 6.985 at 497 h
```

The plan is the best for the long run. A new plant's units all start new,
so the first replacements of redundant units fall due together, and where
they take the system down together its first years cost more than the
plan's rate: `expected_cost` gives the cost from new, and from units of
other ages (`state`) the cost of a staggered start.

With limited `repair_crews` a component can wait for a crew, and the exact
long-run values the search uses no longer hold, so the choice is refused.
`assume_unlimited_crews=True` (#184) chooses the intervals as if every repair
started at once, and `with_intervals(plan)` gives the diagram with them, to
simulate with its crews and see what the waiting costs:

```python
crewed = RepairableRBD(
    [("s", "a"), ("a", "b1"), ("a", "b2"), ("b1", "t"), ("b2", "t")],
    {n: pump(1000) for n in ("a", "b1", "b2")},
    downtime_cost_rate=500.0, repair_crews=1,
)
plan = crewed.optimal_replacement_intervals(assume_unlimited_crews=True)
planned = crewed.with_intervals(plan)    # the same crew, the plan's intervals
planned.cost(100_000.0, mc_samples=100, seed=1).mean / 100_000.0   # ~> 20.6   simulated
plan.cost_rate          # -> 20.12   with a crew for every job
```

`with_intervals` takes a plan or a dict of intervals (and of test offsets),
and builds the copy as the diagram was built; `compare()` then sets the plan
against the schedules the diagram has.

`expected_cost_rate` prices an age-replaced component through its renewal
cycle: it ends at a failure or a preventive replacement, whichever comes
first, with mean length `C = ∫₀ᵀ R + F(T)·MTTR + R(T)·MTTP` (`MTTP` the mean
maintenance time), so it fails `F(T) / C` times and is replaced `R(T) / C`
times per unit time, and is up `∫₀ᵀ R / C` of the time:

```
cost rate = … + Σ R_i(T_i) / C_i · preventive cost_i     preventive actions
```

With instant repair and maintenance a component's own cost rate is
`NonRepairable.cost_rate(T)`. `mean_availability`, `node_availability`, the
frequencies, MUT, MDT and the importance measures account for it the same
way.

Under block replacement the renewals are the block times at which the unit is
up: it is replaced there, and a replacement due while it is down is skipped.
Between two of them it fails and is repaired as usual, and a repair can run
over a block time. `expected_cost_rate` and the other long-run methods compute
its mean up time, failures and length between renewals numerically, from the
renewal equations of its lives and repairs within an interval, to about one
part in a million. With instant repair and replacement the cost rate is
`(c_p + c_u · M(T)) / T`, `M` the renewal function of the lives. Components
replaced at the same block times go down together, so the system's long-run
values average over the block interval (over the time the schedules take to
repeat together, for different intervals), rather than combining each
component's own average:

```python
alone(580, "block").expected_cost_rate()          # -> 14.06   against 13.14 for age
with_standby(580, "block").mean_availability()    # -> 0.9901  both replaced at once
with_standby(580).mean_availability()             # -> 0.9996
```

The exact block-replacement values need a surpyval parametric lifetime with a
density (no units dead on arrival), and repairs that always end; otherwise
they raise `NotImplementedError`, and the simulation still applies.

The simulation prices both policies. A replacement's cost is in
`by_category["preventive"]` (the expected cost, exact here, as the cost's
`mean` is), and a planned outage counts as downtime, but not as a failure:
`system_planned_outages` counts the times one took the system down in the
simulations.

```python
year = alone(580).availability(t_simulation=8760.0, mc_samples=500, seed=0)
year.system_failures / year.n_simulations          # -> 3.518
year.system_planned_outages / year.n_simulations   # -> 11.92
year.cost.by_category["preventive"]                # -> 11886.8   1000 each, expected
```

### Replacement on condition

`"policy": "condition"` inspects the unit at every multiple of `interval`,
while it is up, and replaces it only if it is then more likely than
`"threshold"` to fail before the next inspection, given its age `a`:
`1 − R(a + T) / R(a)`, the conditional survival that `NodeState` and
`sf_given_state` use. A unit renewed a while ago is left alone, and a worn
one is replaced at the last inspection before it is likely to fail.

| Key | Meaning |
|---|---|
| `threshold` | Required with `"condition"`: the probability, in `[0, 1]`, above which an inspection replaces the unit. |
| `inspection_cost` | Charged at each inspection: a number or a distribution. |

`duration` and `cost` are the replacement's, as under the other policies.
An inspection takes no time, and one due while the unit is down is skipped.
Inspected weekly and replaced when more than 20% likely to fail before the
next inspection, the lone pump above costs about what the best age
replacement does:

```python
weekly = {"interval": 168.0, "policy": "condition", "threshold": 0.2,
          "duration": surv.Weibull.from_params([8, 3]), "cost": 1000.0,
          "inspection_cost": 20.0}
inspected = RepairableRBD([("s", "p"), ("p", "t")],
                          {"p": dict(pump(580), preventive=weekly)},
                          downtime_cost_rate=500.0)
inspected.expected_cost_rate()              # -> 13.37   per hour
run = inspected.cost(200_000.0, mc_samples=20, seed=1)
run.cost_rate                               # -> 13.36   exact over the window
run.sample_mean / 200_000.0                 # -> 13.28   the 20 simulations', ± 0.2
```

| Threshold | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 |
|---|---|---|---|---|---|---|
| Cost per hour | 22.10 | 15.98 | 13.47 | 13.37 | 14.02 | 14.34 |

Judged on its age alone, a unit replaced on condition is replaced much as
under age replacement (13.14 at the best age), but on the inspection
calendar. A low threshold replaces it too young; a high one lets it fail.

- **A threshold of 0** replaces the unit at every inspection: block
  replacement at the interval. **A threshold of 1** never does: run to
  failure, with inspections.
- **A constant failure rate** gives the same probability at every age,
  `1 − exp(−λT)`: the unit is replaced at every inspection or at none, and
  it fails as often either way.
- **Numerical in the long run** (#145). The inspections that replace the
  unit are regeneration points, and a cycle between two is followed one
  inspection interval at a time: the units put into service in an interval
  as an alternating renewal process, as under block replacement, and the
  units an inspection keeps on into the next interval by their ages. So the
  long-run values, the cost rate (the inspections charged at the unit's
  availability just before them) and the importance measures are
  numerical, to about `1e-7`. A cycle in which the unit fails many times
  before an inspection replaces it is summed to its end as a geometric
  series once it falls at a steady rate.
- **Numerical over time** (#161). The same recursion, followed from new
  over all the intervals rather than over one cycle, gives the availability
  at any time and the expected events and cost of any window. Each
  interval starts with what the inspection at its start replaced, the
  repairs carried into it and the units it kept, by age, and after some
  intervals it repeats from one to the next: its long-run cycle. From a
  state, the unit in service at the start is decided on at its own age at
  each inspection, however that falls on the grid.

A year of the weekly-inspected pump from new, and the next 30 days of one
already 600 hours old, 100 hours after its last inspection:

```python
from repyability import NodeState

inspected.expected_cost(8760.0).total     # -> 114491   new: less than a year at the rate, 117139
worn = {"p": NodeState(age=600.0, phase=100.0)}
inspected.expected_cost(720.0, state=worn).total   # -> 10265   against 8517 new
```

### Opportunistic maintenance

Plants group their work: once a unit is down, for a failure or its planned
replacement, the crew services its neighbours too, as the line is stopped
anyway. Components that share such stops form a *maintenance group*, named
by their `"group"` key. Each failure of a member, and each scheduled
replacement, opens a *stop* of its group, at which every other member that
is working and at least its `"opportunity"` age is replaced as well, as
its own scheduled replacement would replace it:

| Key | Meaning |
|---|---|
| `"group"` (component) | The component's maintenance group: any hashable name. |
| `opportunity` (`"preventive"`, age policy) | The age, from 0 to `interval`, from which the unit is replaced early at a stop of its group. Left out, it never is. |
| `setup_cost` (`maintenance_groups`) | Charged once per stop, however many members it renews. |
| `system_down` (`maintenance_groups`) | `True`: every outage of the system is a stop of the group as well. |

A compressor and its motor in series wear out, and each stop of the train
costs 3,000 to set up (isolation, permits, scaffolding) besides the units'
own costs. Replaced separately at 600 h, each failure or replacement is a
stop of its own; from age 300 h, a unit is replaced at the other's stop:

```python
def unit(scale, opportunity=None):
    preventive = {"interval": 600.0,
                  "duration": surv.Weibull.from_params([8, 3]),   # 7 h
                  "cost": 1000.0}
    if opportunity is not None:
        preventive["opportunity"] = opportunity
    return {
        "reliability": surv.Weibull.from_params([scale, 2.5]),
        "repairability": surv.LogNormal.from_params([3.0, 0.5]),  # 23 h
        "replace_cost": 5000.0,
        "preventive": preventive,
        "group": "train",
    }

def train(opportunity=None):
    return RepairableRBD(
        [("s", "compressor"), ("compressor", "motor"), ("motor", "t")],
        {"compressor": unit(1000, opportunity),
         "motor": unit(1500, opportunity)},
        downtime_cost_rate=500.0,
        maintenance_groups={"train": {"setup_cost": 3000.0}},
    )

train().expected_cost_rate()          # -> 33.00   separate stops, exact
run = train(300).availability(200_000.0, mc_samples=20, seed=1)
run.cost.cost_rate                    # -> 23.45   grouped, simulated
run.opportunistic_renewals            # -> {'compressor': 3355, 'motor': 3481}
```

| Opportunity age (h) | none | 500 | 400 | 300 | 200 | 100 |
|---|---|---|---|---|---|---|
| Cost per hour (± 0.15) | 33.00 | 29.98 | 26.76 | 23.45 | 23.17 | 23.16 |

Grouping pays twice: the stops, and their set-ups, are fewer (5.63 an hour
of set-ups against 10.33), and the units' outages overlap, so the train is
down less (12.0 an hour of lost production against 16.7).

- **A stop is an instant.** Work started at one instant shares one set-up,
  and a member whose own failure or replacement falls due then keeps it, as
  part of the stop: two units on the same schedule are replaced together,
  on schedule, at one set-up.
- **Durations overlap.** Each member renewed at a stop takes its own
  maintenance time, from the stop, so the set-up time each one's
  maintenance includes is spent once in a series train.
- **Results.** `opportunistic_renewals` counts each component's early
  renewals, summed over the simulations; the set-ups are under
  `by_category["setup"]`, and each early renewal is charged the member's
  preventive cost and uses a spare.
- **Exact or simulated.** A component that can be renewed early depends on
  the others, so the exact long-run values, the availability over time and
  the spares counts refuse it; `availability`, `cost` and `compare`
  simulate it, in Python. With no `opportunity` below an interval, the
  set-up cost is exact: `expected_cost_rate` charges a set-up at each
  failure and each preventive replacement of a member (so `opportunity`
  equal to the interval is plain age replacement, plus set-ups). It
  refuses two members replaced on a clock (block replacement, or never
  failing before an instant age replacement), whose replacements share
  stops.
- Members cannot have hidden failures or be standby groups.

## Hidden failures and inspection

Some failures announce themselves: a running pump stops, and the operators
know. Others do not: a relief valve that has seized, a standby pump that will
not start or a trip that no longer trips looks just like a working one until
something tests it. Such a failure is *hidden* (or *unrevealed*): the
component is down, nobody knows, and it stays down until a periodic
*inspection* (a proof test) finds it. The key `"inspection"` in a
component's dict makes its failures hidden:

| Key | Meaning |
|---|---|
| `interval` | Required: the time `τ` between inspections, positive and finite. The component is inspected at `τ, 2τ, 3τ, …`, or from its `offset`. |
| `duration` | `"instant"` (the default): the test takes no time. Or a time-to-test model: the component is off-line while it is tested (a planned outage), and does not age meanwhile. |
| `cost` | Charged at each inspection: a number or a distribution. |
| `offset` | The time of the first test, from 0 to less than `τ`: tests at `offset, offset + τ, …`, so that redundant components can be tested apart (staggered). By default 0, no offset: the first test at `τ`, none at the start. A positive offset puts one there, so an offset just above 0 adds a test (and its outage and cost) at the start; one within a billionth of `τ` of 0 is taken as 0. |
| `coverage` | The chance that a test finds a failure, from 0 to 1 (by default 1): its *proof-test coverage*. A failure a test misses stays hidden until a full test. |
| `full_test` | With a coverage below 1, required: the time between full tests, which find every failure, a whole multiple of `τ` (the tests at `offset` and every `full_test` after it). Often the mission time, after which the component is renewed. |

A failure found by an inspection is repaired once the test is done (taking a
time drawn from the component's `"repairability"`), and its repair and
replace costs are charged when it is found. An inspection due while the
component is being repaired is skipped. The time a failure lies hidden counts
as downtime in every output, and each inspection's cost is in
`by_category["inspection"]`. A component can have an inspection or a
preventive schedule, but not both.

### The average probability of failure on demand

A protective function whose failure is hidden does not act when it is
needed, so its long-run unavailability is its *average probability of
failure on demand*, the PFDavg that safety-integrity (SIL) verification asks
for (IEC 61508 and 61511). With a constant rate `λ` of such failures,
inspected every `τ`, with instant tests and repairs, a single channel is
down, after a failure, until the next test: half an interval on average, so
`PFDavg = 1 − (1 − e^(−λτ)) / (λτ) ≈ λτ/2`. Take a shutdown valve with
dangerous undetected failures at `2 × 10⁻⁶` per hour, proof-tested yearly:

```python
def valve(interval):
    return {
        "reliability": surv.Exponential.from_params([2e-6]),   # dangerous, undetected
        "repairability": "instant",
        "inspection": {"interval": interval},
    }

one = RepairableRBD([("s", "v"), ("v", "t")], {"v": valve(8760.0)})
one.mean_unavailability()      # -> 0.008709   about λτ/2 = 0.00876
pair = RepairableRBD(
    [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
    {"v1": valve(8760.0), "v2": valve(8760.0)},
)
pair.mean_unavailability()     # -> 1.0098e-4   about (λτ)²/3
```

Two valves in parallel (1oo2), tested together, are down only when both
have failed since the last test: `PFDavg = (1/τ) ∫₀^τ (1 − e^(−λt))² dt ≈
(λτ)²/3`. That is a third more than the `(λτ/2)²` of two channels that fail
independently in time, because both have gone untested for the same time.
The diagram supplies the voting, so 2oo3 and larger architectures follow in
the same way (see [k-out-of-n nodes](building.md#k-out-of-n-nodes)). Halving
the interval halves a single channel's PFDavg but quarters the pair's:

| Test interval | 6 months | 1 year | 2 years |
|---|---|---|---|
| One valve | 0.00437 | 0.00871 | 0.0173 |
| Two valves (1oo2) | 2.54 × 10⁻⁵ | 1.01 × 10⁻⁴ | 3.99 × 10⁻⁴ |

These long-run values are exact. Components inspected at the same times go
down together, so `mean_availability` and the other long-run methods average
the system's availability over one period of the inspection schedules (the
least common multiple of the intervals: components on different intervals
can be mixed), and the importance measures are ratios of those averages.
With instant tests and repairs they are in closed form for a constant
failure rate, and numerical for any other life (see [a life that wears
out](#a-life-that-wears-out)); tests and repairs that take time are
numerical too (see [tests and repairs that take
time](#tests-and-repairs-that-take-time)). Testing both valves at once, for
example, takes the whole function off-line during the test, which here
costs far more than the hidden failures:

```python
def tested(interval, offset=0.0):
    return {
        "reliability": surv.Exponential.from_params([2e-6]),
        "repairability": surv.LogNormal.from_params([np.log(24), 0.5]),   # about a day
        "inspection": {
            "interval": interval,
            "duration": surv.Weibull.from_params([4, 3]),                  # about 3.6 h
            "offset": offset,
        },
    }

pair = [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")]
both = RepairableRBD(pair, {"v1": tested(8760.0), "v2": tested(8760.0)})
both.mean_unavailability()                          # -> 4.28e-4
1 - both.mission_availability(10 * 8760.0)          # -> 3.95e-4   its first ten years
both.expected_events(10 * 8760.0).system_planned_outages   # -> 9.0   one per test
apart = RepairableRBD(pair, {"v1": tested(8760.0), "v2": tested(8760.0, 4380.0)})
apart.mean_unavailability()                         # -> 7.12e-5
```

Tested half a year apart, a test takes one valve off-line while the other
holds the function, and the PFDavg falls six-fold.

### A life that wears out

The closed forms above need a constant failure rate. With any other life,
and instant tests and repairs, the values are numerical, exact to rounding.
A failed unit is found and renewed at the next test, so every renewal falls
on a test, and a unit renewed `m` intervals before a test is still up with
probability `R(mτ)`. In the long run a cycle lasts `S = 1 + Σ R(mτ)`
intervals (over `m ≥ 1`), and the unit is up a time `u` after a test with
probability `(R(u) + Σ R(mτ + u)) / S`. From new, the chance of a renewal
at each test follows a renewal equation over the tests, and settles to
`1/S`. A valve that wears out, with a mean life of about 15 years:

```python
worn = surv.Weibull.from_params([150_000.0, 2.5])   # wears out; MTTF 133,090 h
wearing = RepairableRBD(
    [("s", "v"), ("v", "t")],
    {"v": {
        "reliability": worn,
        "repairability": "instant",
        "inspection": {"interval": 8760.0},
    }},
)
wearing.mean_unavailability()               # -> 0.03186   about τ / (2 MTTF) = 0.0329
1 - wearing.mission_availability(87600.0)   # -> 0.01126   its first ten years
years = np.array([1, 5, 10, 20]) * 8760.0 - 1.0   # just before a test
1 - wearing.point_availability(years)
# -> [0.00082 0.01908 0.04983 0.06844]
```

For a short interval the long-run PFDavg is about `τ / (2 MTTF)`, whatever
the life's shape. What the wear changes is the PFD over time. A new valve
rarely fails in its first years, so its first ten years' PFD is a third of
the long run's, and the PFD just before each test climbs with its age
before it settles. The values over a window, from new or from the valve's
current state (its age, and the time since its last test), are exact too:
`expected_events`, `expected_cost` and the other [analyses over
time](repairable.md#availability-over-time-exact).

The sums run over the intervals a unit can last. Within an interval, every
unit but the one renewed at the last test ages smoothly, so their sum is
worked out once, at Chebyshev points across the interval, and interpolated
between them, to rounding. A term that bends there (a life with a
threshold, say) is summed directly instead. A life that lasts more than
200,000 intervals (the most summed) refuses.

### Tests and repairs that take time

A test of a working unit takes it off-line (a planned outage) for the
test's time, during which it does not age; a failure found by a test is
repaired once the test is over, and the tests that fall in the repair are
not done. Then renewals no longer fall on the tests, and a test that can
miss a failure (see [test coverage](#common-cause-staggered-tests-and-test-coverage))
leaves the failure for a later one. The values are still numerical, for
any life (#159). The tests that find failures are the unit's regeneration
points: from one, a new unit is put into service the test's time plus the
repair's later, and its age at its later tests is a random walk (each test
holds it back by the test's time), followed test by test on a grid of
2,000 steps or more an interval. The long run follows by renewal-reward
over that cycle (over the cycles from each place in the full tests'
period, with tests that miss), and from new, or from a state, the tests
that find failures are a renewal process on the tests. When a test is
over, and when a unit is back in service, are kept out of the grid, exact
at any time; the rest is exact to the square of the grid's step, about
1e-8. An exponential life needs no ages: its chain of states from test to
test (working, failed, missed, in a repair) is followed instead.

```python
def valve(coverage=1.0):
    inspection = {
        "interval": 8760.0,
        "duration": surv.ExactEventTime.from_params([8.0]),     # 8 h off-line
    }
    if coverage < 1.0:
        inspection.update(coverage=coverage, full_test=5 * 8760.0)
    return RepairableRBD(
        [("s", "v"), ("v", "t")],
        {"v": {
            "reliability": surv.Weibull.from_params([150_000.0, 2.5]),
            "repairability": surv.ExactEventTime.from_params([24.0]),   # a day
            "inspection": inspection,
        }},
    )

valve().mean_unavailability()       # -> 0.03289   0.03186 with tests and repairs in no time
valve(0.7).mean_unavailability()    # -> 0.06845   70% coverage, a full test every 5 years
valve(0.7).expected_events(10 * 8760.0).system_planned_outages   # -> 8.750
```

`spares_demand` and `spares_stock` count such a valve's spares on the
tests that find its failures (see [how spares are
counted](spares.md#how-it-is-computed)). Only a test that can last as long
as its interval stays simulated. A common-cause group's members' tests and
repairs may take time too, of a fixed length or an exponential one (see
[below](#common-cause-staggered-tests-and-test-coverage)).

### Common cause, staggered tests and test coverage

Three more terms decide a real safety function's PFDavg, and each has its
place in the diagram:

- **Common cause.** Redundant valves of one design, in one service, fail
  together more often than chance allows. Give the diagram `ccf_groups`, as
  for a [non-repairable one](common-cause.md): a `BetaFactor(β)` makes a
  share `β` of each valve's failures a shared cause that fails both at
  once. For a redundant function the shared term usually dominates: about
  `βλτ/2`, five times the pair's independent `(λτ)²/3` here.
- **Staggered tests.** An `"offset"` tests one valve half an interval after
  the other: a shared failure is then found by whichever test comes first,
  and the independent term falls from `(λτ)²/3` to about `5(λτ)²/24`.
  Each valve is restored by its own test alone: the first test restores
  its valve, and with it the pair, and the other stays down until its own
  test (where practice has the first test find and restore both, the
  value is a little less, the pair then not left on one valve until the
  second test).
- **Test coverage.** A proof test that finds only a share `c` of the
  failures (a `"coverage"`) leaves the rest hidden until a full test
  (`"full_test"`), say the ten-year overhaul: about `(1 − c)λT/2` more,
  `T` the full test's interval.

```python
def proof_tested(**inspection):
    return {
        "reliability": surv.Exponential.from_params([2e-6]),
        "repairability": "instant",
        "inspection": {"interval": 8760.0, **inspection},
    }

redundant_edges = [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")]
common = [CCFGroup(["v1", "v2"], BetaFactor(0.05))]
shared = RepairableRBD(
    redundant_edges,
    {"v1": proof_tested(), "v2": proof_tested()},
    ccf_groups=common,
)
shared.mean_unavailability()        # -> 5.290e-4   (λτ)²/3 + βλτ/2, about
staggered = RepairableRBD(
    redundant_edges,
    {"v1": proof_tested(), "v2": proof_tested(offset=4380.0)},
    ccf_groups=common,
)
staggered.mean_unavailability()     # -> 2.779e-4   the shared term halved
partial = RepairableRBD(
    [("s", "v"), ("v", "t")],
    {"v": proof_tested(coverage=0.9, full_test=87600.0)},
)
partial.mean_unavailability()       # -> 0.01642    0.9λτ/2 + 0.1λT/2, about
```

All three are exact with constant failure rates and instant tests and
repairs, and staggered tests and test coverage are numerical with any life
too, and with tests and repairs that take time (see [tests and repairs that
take time](#tests-and-repairs-that-take-time)). With common causes, a
group's members are a Markov chain of which of them are down, with each
member found by its own tests, and a shared failure found alike by every
test (the coverage is the group's), each member it struck restored by its
own test, as in the simulations (#237). Their tests and repairs may take no
time, a fixed time or an exponential one (#220): a working member is off
line while it is tested, and neither ages nor fails meanwhile, a failure a
test finds is repaired once the test is over, and a test that falls in a
member's own test or repair is not done, as in the simulations. IEC
61508-6's 1oo2 with an eight-hour mean repair time:

```python
repaired = {**proof_tested(), "repairability": surv.ExactEventTime.from_params([8.0])}
mrt = RepairableRBD(
    redundant_edges, {"v1": repaired, "v2": repaired}, ccf_groups=common
)
mrt.mean_unavailability()           # -> 5.300e-4   IEC 61508-6: 5.316e-4
```

A test of an exponential length followed by a repair of a fixed one, or
fixed lengths that run into the member's next test, are refused, saying
so: estimate those by simulation.
The importance measures take the groups in, a member's conditioned on its
state at each time, and so do the allocations, the values over time from
new (the chain followed through the tests from every member up) and the
simulations, which draw the shared causes; see [repairable
systems](common-cause.md#repairable-systems) and `analysis_routes()`.

### Choosing the interval

Each inspection costs `c_i`, and each unit of time the component lies failed
costs `c_d` (its `"downtime_cost"`, or the system's `downtime_cost_rate`).
Frequent tests cost more, and rare ones leave failures hidden for longer, so
the cost rate `c_i/τ + c_d·U(τ)` is least in between, near
`τ* = √(2c_i / (λc_d))`. `expected_cost_rate` prices it exactly:

```python
def pump(interval):
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {
            "reliability": surv.Exponential.from_params([1e-4]),
            "repairability": "instant",
            "downtime_cost": 500.0,                               # per hour failed
            "inspection": {"interval": interval, "cost": 2000.0},
        }},
    )

pump(100.0).expected_cost_rate()     # -> 22.49   testing too often
pump(283.0).expected_cost_rate()     # -> 14.08   near √(2 × 2000 / (1e-4 × 500)) = 283
pump(1000.0).expected_cost_rate()    # -> 26.19   failures hidden too long
```

`optimal_inspection_intervals` finds it, and with a target, the cheapest
intervals that meet it. Tests are made on a calendar, and components tested
at the same times are down together, so with several components the
intervals are chosen from those allowed (`allowed`): every combination when
there are at most 2000, a local search otherwise. With one inspected
component, any interval can be chosen:

```python
pump(100.0).optimal_inspection_intervals().intervals["p"]   # -> 285.5   the formula gives 283
```

For a safety function, `min_availability=1 - PFDavg target` gives the
cheapest tests that meet the target. With the valves above, each test costing
500, and tests monthly, quarterly, half-yearly, yearly or every two years:

```python
def priced_valve():
    return {
        "reliability": surv.Exponential.from_params([2e-6]),
        "repairability": "instant",
        "inspection": {"interval": 8760.0, "cost": 500.0},
    }

calendar = [730.0, 2190.0, 4380.0, 8760.0, 17520.0]   # hours
single = RepairableRBD([("s", "v"), ("v", "t")], {"v": priced_valve()})
single.optimal_inspection_intervals(
    allowed=calendar, min_availability=1 - 1e-3
).intervals["v"]                                        # -> 730.0   monthly
redundant = RepairableRBD(
    [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
    {"v1": priced_valve(), "v2": priced_valve()},
)
redundant.optimal_inspection_intervals(
    allowed=calendar, min_availability=1 - 1e-3
).intervals["v1"]                                       # -> 17520.0 every two years
```

Redundancy relaxes the tests: one valve needs them monthly to keep the
PFDavg at most `10⁻³`, two in parallel every two years. The result is a
[`MaintenancePlan`][repyability.MaintenancePlan], as for
[age-replacement intervals](#choosing-the-intervals).

When the valves are tested matters as well. Tested together, both are down
for as long as a common-cause failure stays hidden; tested half an interval
apart, it is found twice as soon (and the valve it found restored, the
other waiting for its own test). `offsets` chooses the times of the first
tests with the intervals (#184), as shares of each interval in [0, 1): a
list for every component, a dict of one per component, or `"stagger"`, the
shares `0, 1/n, …, (n − 1)/n` of `n` components, among which are tests of
one interval spread evenly over it. With a 10% common cause and a PFDavg of
at most `5 × 10⁻⁴`:

```python
paired = RepairableRBD(
    [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
    {"v1": priced_valve(), "v2": priced_valve()},
    ccf_groups=[CCFGroup(["v1", "v2"], BetaFactor(0.1))],
)
together = paired.optimal_inspection_intervals(
    allowed=calendar, min_availability=1 - 5e-4)
together.intervals        # {'v1': 4380.0, 'v2': 8760.0}
together.cost_rate        # -> 0.1712
apart = paired.optimal_inspection_intervals(
    allowed=calendar, min_availability=1 - 5e-4, offset_shares="stagger")
apart.intervals           # {'v1': 8760.0, 'v2': 8760.0}
apart.offsets             # {'v1': 0.0, 'v2': 4380.0}   six months apart
apart.cost_rate           # -> 0.1142   a third less
```

Shifting every test by one time changes nothing in the long run, so the
first component's tests stay from 0, unless other tested components keep
their schedules (then its offset is chosen too). Offsets change no cost, so
of plans that cost the same the most available is chosen. The
`offset_shares` searched are shares of the interval; the result's
`offsets` are the times of the first tests, as `with_intervals(offsets=)`
takes them, and a plan has them whether they were searched or not.

## The total cost of ownership

The costs so far are running costs. Buying the system is a one-off cost:
give each component an `"acquisition_cost"`, and `total_cost(horizon)` adds
the purchase to the running cost over the time the system is owned,

```text
total_cost(H) = acquisition_cost + expected_cost_rate() · H
```

(undiscounted by default: see [Discounting](#discounting)). The
acquisition cost is not a running cost, so it is left out of `has_costs`,
`expected_cost_rate` and the simulated samples; a `CostResult` reports it
beside them, as `acquisition_cost`.

```python
pump = {
    "reliability": surv.Exponential.from_params([1e-3]),   # MTTF 1000 h
    "repairability": surv.Exponential.from_params([0.1]),  # MTTR 10 h
    "repair_cost": 500.0,
    "acquisition_cost": 20000.0,
}
line = RepairableRBD([("s", "pump"), ("pump", "t")], {"pump": pump},
                     downtime_cost_rate=100.0)
line.acquisition_cost          # -> 20000.0
line.expected_cost_rate()      # -> 1.4851   (500 + 100 × 10) / 1010 per hour
line.total_cost(87600.0)       # -> 150099.0   ten years
```

`total_cost` runs the system at its long-run rate from the start. The
exact expected cost of owning it from new is `expected_cost(H).total`: the
pump is new at the start, so it is down a little less early on than in the
long run.

```python
line.expected_cost(87600.0).total   # -> 150089
```

### Discounting

Money spent in ten years is worth less than money spent now. `discount_rate`
gives `total_cost` (and `allocate_redundancy`) the present value (#184): the
components are bought at the start, and the running costs, spent at a steady
rate, are discounted continuously at `r` per unit time, so the horizon counts
as `(1 − e^(−r·H)) / r`. A rate is per unit time of the models: 7% a year,
with models in hours, is `math.log(1.07) / 8760`.

```python
import math
seven = math.log(1.07) / 8760
line.total_cost(87600.0, discount_rate=seven)   # -> 114538   ten years count as 63,656 hours
line.total_cost(math.inf, discount_rate=seven)  # -> 212287   owned for ever: 1 / r hours
line.total_cost([8760.0, 87600.0], discount_rate=seven)   # one total for each horizon
```

A rate per year given with the models in hours (`0.07`) discounts every cost
after the first few hours away: a warning says so, and how to convert it
(#231).

`total_cost` spends the long-run cost rate from the start. The exact present
value from new discounts each cost when it falls, which matters most in the
early years, where the two differ: `expected_cost` takes a `discount_rate`
too (#231), and is then a present value, `acquisition_cost` (paid at the
start) beside it:

```python
line.expected_cost(87600.0, discount_rate=seven).total   # -> 114528   from new
```

Discounting favours what is cheaper to buy and dearer to run, as the running
costs it saves come later: it can change which design wins (see the
[fourth pump train](#whole-trains) below, which no longer pays at 15% a
year). The cost rates and the simulated costs stay undiscounted.

### Buying redundancy

A redundant copy is bought once and then runs: it fails, is repaired and
costs money for as long as it is owned, while it saves the lost production
of the outages it covers. `allocate_redundancy(horizon)` chooses how many
identical, independently repaired, active copies of each component give the
lowest total cost of ownership:

```python
best = line.allocate_redundancy(87600.0)
best.units               # {'pump': 2}
best.total_cost          # -> 127591.4   a second pump saves 22,508 over ten years
best.acquisition_cost    # -> 40000.0
best.availability        # -> 0.999902
line.allocate_redundancy(8760.0).units   # {'pump': 1}   over one year it does not pay
```

The second pump costs `20000 + 0.495·H` (its price and its repairs) and saves
`100 · (U − U²) · H = 0.980·H` of lost production, with `U = 10/1010` the
fraction of the time one pump is down; it pays once `H` exceeds about 41,200
hours. A third would save at most `100 · U² · (1 − U) = 0.0097` per hour,
less than the 0.495 per hour its repairs cost, so it never pays, whatever it
costs to buy.

The result is a [`TotalCostAllocation`][repyability.TotalCostAllocation]:
`units`, `total_cost`, `acquisition_cost`, `cost_rate` (the running cost
per unit time) and `availability` of the design, the `horizon` and the
`method`. Each design is scored exactly: `n` copies of a component of
long-run availability `A` are all down `(1 − A)ⁿ` of the time, each copy has
the running cost of the original, and the system's availability comes from
the exact engine, so the result is what `total_cost`, `expected_cost_rate`
and `mean_availability` give for the system with the copies drawn out as
separate nodes. The arguments:

| Argument | Meaning |
|---|---|
| `nodes` | The components that may be given copies; by default every one with an `"acquisition_cost"` (none, given `trains`). The others are counted once. |
| `trains` | Chains of components that may be given copies as a whole, `{name: [nodes]}`: see [Whole trains](#whole-trains). |
| `min_availability` | Only designs at least this available, in (0, 1): e.g. a contractual availability, met at the lowest total cost. |
| `max_units` | The most copies of every node (an int) or of some (a dict). A node whose copies cost nothing needs one. |
| `method` | `"exact"` (the default): a proven optimum. `"greedy"`: adds or removes one copy at a time while that lowers the total; fast, not guaranteed optimal. |

```python
line.allocate_redundancy(87600.0, min_availability=0.99999).units   # {'pump': 3}
```

Unlike the reliability of a non-repairable design, the total cost is not
monotone in the copies: each copy costs as much as the one before and saves
less. The exact search is bounded by two facts. The `k+1`-th copy of a
component down a fraction `U` of the time can save at most
`H · downtime_cost_rate · Uᵏ · (1 − U)` (all it could ever save, were
everything else perfect), so no copy beyond the point where that falls below
a copy's cost can pay (without a `min_availability`, which may need copies
that do not pay). And a design no worse than the best found spends no more
on copies than that design's total. When every node considered lies on every
path (in series with the rest of the system) and has no hidden failures, the
system's availability is theirs times the rest's, and a dynamic program over
the nodes finds the optimum for any number of them; on other structures a
branch and bound over the designs does, which suits a handful of nodes.

The model and its limits: copies are active and repaired independently of
each other (as many repair crews as failed copies), and fail independently
unless the component is in a common-cause group: then its copies join the
group, a `BetaFactor` group's shared cause failing them all (see [repairable
systems](common-cause.md#repairable-systems); an `MGL` group's member is
refused). Copies of a component with hidden failures are inspected together. A nested
`RepairableRBD` cannot be given copies. Copies of a component under block
replacement are replaced together, at the same block times. Costs are not
discounted unless `discount_rate` is given (see [Discounting](#discounting)).
For non-repairable systems, redundancy allocation within a budget or to a
reliability target is in
[Design and allocation](design.md#redundancy-allocation).

### Whole trains

Redundancy often comes a train at a time: a station that needs two of its
three pump trains, each a pump and then its motor, asks whether a fourth
train pays. `trains={name: [nodes]}` names chains of components that may be
given copies as a whole (#184). A copy of a train is another path alongside
it, fed as its first node is and feeding the node its last one feeds, which
keeps its `k`: copies of a train of a 2-out-of-3 vote make it 2-out-of-4.

```python
motor = {
    "reliability": surv.Exponential.from_params([2e-4]),   # MTTF 5000 h
    "repairability": surv.Exponential.from_params([0.05]), # MTTR 20 h
    "acquisition_cost": 10000.0,
}
trains = (1, 2, 3)
station = RepairableRBD(
    [("s", f"pump {i}") for i in trains]
    + [(f"pump {i}", f"motor {i}") for i in trains]
    + [(f"motor {i}", "t") for i in trains],
    {**{f"pump {i}": pump for i in trains}, **{f"motor {i}": motor for i in trains}},
    k={"t": 2}, downtime_cost_rate=2000.0,
)
station.total_cost(87600.0)    # -> 319927   three trains
fourth = station.allocate_redundancy(87600.0, trains={"train 1": ["pump 1", "motor 1"]})
fourth.units                   # {'train 1': 2}   train 1 and a copy: four trains
fourth.total_cost              # -> 295306
station.allocate_redundancy(87600.0, nodes=["pump 1", "motor 1"]).total_cost   # -> 311130
fifteen = math.log(1.15) / 8760   # 15% a year
station.allocate_redundancy(87600.0, trains={"train 1": ["pump 1", "motor 1"]},
                            discount_rate=fifteen).units   # {'train 1': 1}
```

Copies of train 1's own nodes, each in parallel with its own, stand in for
train 1 alone, where a fourth train stands in for whichever train is down:
the fourth train is the cheaper design. Discounted at 15% a year it no
longer pays: bought now, it saves running costs and downtime that come
later. Name one of identical trains.
`units` counts it and its copies under its name, which is not a node's, and
the result's `trains` lists each train's nodes.

A train is a chain: its first node may be fed by several nodes (its copies
are fed by them all), each of the others by the one before it alone, and its
last node feeds one node alone, where the copies join. A node is in one
train at most, and not in `nodes` as well; given `trains`, `nodes` is by
default none, and a component may be both copied alone and in a train only
in separate searches. Each design is scored exactly, the copies drawn out as
trains of their own. The search's bound on what a copy can save is that the
`n + 1`-th train changes the system only when at most `k − 1` of the `n`
there work: it saves at most `H · downtime_cost_rate · A · P(at most k − 1
of n work)`, `A` the train's availability.

Costs, including cost distributions and acquisition costs, and maintenance
and inspection schedules are saved with the RBD.
