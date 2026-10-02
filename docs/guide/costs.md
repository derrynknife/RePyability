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
undiscounted, and every failure is repaired at the same price (preventive
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
costs.mean              # mean total cost of a window
costs.cost_rate         # mean / t_simulation: converges to expected_cost_rate()
costs.percentile(90)    # a planning budget: 9 windows in 10 cost less
costs.std               # how much a window's cost varies
costs.by_category       # mean repair, replace, preventive, inspection, component_downtime, system_downtime, setup
costs.by_component      # mean cost attributable to each costed component
```

`cost()` takes the same arguments as `availability()` (`working_nodes`,
`broken_nodes`, `method`, `mc_samples`, `verbose`, `seed`, and `tolerance`,
`antithetic`, `n_jobs` and `engine`: see
[Simulation precision and speed](simulation.md)). The same result comes with
`availability(...)` as `result.cost`, so one simulation gives both answers.
With nothing priced, `cost()` returns `None` and `result.cost` is `None`.

### Two different uncertainties

- `std` and `percentile` describe how much a window's cost **varies**. That
  is a property of the system; more simulations will not shrink it.
- `mean_se` and `mean_interval(confidence)` describe how precisely the
  **expected** cost has been estimated. They shrink like `1/√N`; check them
  before quoting the mean, or pass `tolerance` to `cost()` to simulate until
  the interval is narrow enough.

```python
interval = costs.mean_interval(confidence=0.95)
interval.lower < plant.expected_cost_rate() * 1000.0 < interval.upper   # True
```

`by_category` sums to `mean`; `by_component` covers each component's repair,
replace, preventive and own downtime cost (lost production is a system cost
and is not attributed to components).

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
`by_category["preventive"]`, and a planned outage counts as downtime, but not
as a failure: `system_planned_outages` counts the times one took the system
down.

```python
year = alone(580).availability(t_simulation=8760.0, mc_samples=500, seed=0)
year.system_failures / year.n_simulations          # -> 3.518
year.system_planned_outages / year.n_simulations   # -> 11.92
year.cost.by_category["preventive"]                # -> 11918.0   1000 each
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
run = inspected.cost(200_000.0, mc_samples=20, seed=1)
run.mean_interval().estimate / 200_000.0    # -> 13.28   per hour, simulated
```

| Threshold | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 |
|---|---|---|---|---|---|---|
| Cost per hour (± 0.2) | 22.12 | 16.09 | 13.62 | 13.28 | 14.03 | 14.42 |

Judged on its age alone, a unit replaced on condition is replaced much as
under age replacement (13.14 at the best age), but on the inspection
calendar. A low threshold replaces it too young; a high one lets it fail.

- **A threshold of 0** replaces the unit at every inspection: block
  replacement at the interval. **A threshold of 1** never does: run to
  failure, with inspections.
- **A constant failure rate** gives the same probability at every age,
  `1 − exp(−λT)`: the unit is replaced at every inspection or at none, and
  it fails as often either way.
- **Simulated.** The exact long-run values and the availability over time
  refuse a component replaced on condition, with the reason; `availability`,
  `cost` and `compare` simulate it, in Python.

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
| `offset` | The time of the first test, from 0 to less than `τ` (by default 0): tests at `offset, offset + τ, …`, so that redundant components can be tested apart (staggered). |
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
They need a constant failure rate, instant tests and instant repair, and
raise `NotImplementedError` otherwise: then simulate. Testing both valves at
once, for example, takes the whole function off-line during the test, which
here costs far more than the hidden failures:

```python
def tested(interval):
    return {
        "reliability": surv.Exponential.from_params([2e-6]),
        "repairability": surv.LogNormal.from_params([np.log(24), 0.5]),   # about a day
        "inspection": {
            "interval": interval,
            "duration": surv.Weibull.from_params([4, 3]),                  # about 3.6 h
        },
    }

both = RepairableRBD(
    [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
    {"v1": tested(8760.0), "v2": tested(8760.0)},
)
decade = both.availability(t_simulation=10 * 8760.0, mc_samples=20000, seed=0)
1 - decade.system_uptime / (decade.n_simulations * decade.time_simulated_to)
# -> 3.9e-4
decade.system_planned_outages / decade.n_simulations   # -> 9.0   one per test
```

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

All three are exact, with the constant failure rates and instant tests and
repairs the long-run values need: a group's members are a Markov chain of
which of them are down, with each member found by its own tests, and a
shared failure found alike by every test (the coverage is the group's).
The importance measures take the groups in, a member's conditioned on its
state at each time; the allocations, the values over time from new and the
simulations do not take a common-cause group in yet, and refuse it (#158);
see `analysis_routes()`.

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

## The total cost of ownership

The costs so far are running costs. Buying the system is a one-off cost:
give each component an `"acquisition_cost"`, and `total_cost(horizon)` adds
the purchase to the running cost over the time the system is owned,

```text
total_cost(H) = acquisition_cost + expected_cost_rate() · H
```

(undiscounted). The acquisition cost is not a running cost, so it is left
out of `has_costs`, `expected_cost_rate` and the simulated samples; a
`CostResult` reports it beside them, as `acquisition_cost`.

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
| `nodes` | The components that may be given copies; by default every one with an `"acquisition_cost"`. The others are counted once. |
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
each other (as many repair crews as failed copies), with no common-cause
failures between them (see [Common-cause failures](common-cause.md)).
Copies of a component with hidden failures are inspected together. A nested
`RepairableRBD` cannot be given copies. Copies of a component under block
replacement are replaced together, at the same block times. Costs are not
discounted. For non-repairable systems, redundancy allocation within a
budget or to a reliability target is in
[Design and allocation](design.md#redundancy-allocation).

Costs, including cost distributions and acquisition costs, and maintenance
and inspection schedules are saved with the RBD.
