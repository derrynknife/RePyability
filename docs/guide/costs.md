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
replacement is [below](system-maintenance.md#preventive-maintenance)). A cost must be finite and
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
round(costs.sample_mean, 1)               # -> 120634.4   the 500 windows' own
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
round(interval.standard_error, 1)                                     # -> 862.0
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
