# Costs

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
| Component | `replace_cost` | Per failure (the spare part). A number or a distribution. |
| Component | `downtime_cost` | Per unit time *this component* is down, even if the system is up (a degraded-mode or per-leg penalty). A number. |
| System | `downtime_cost_rate=` | Per unit time the *system* is down (lost production). A number. |

```python
import numpy as np
import surpyval as surv
from repyability import RepairableRBD

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
costs = plant.cost(t_simulation=1000.0, N=500, seed=0)
costs.mean              # mean total cost of a window
costs.cost_rate         # mean / t_simulation: converges to expected_cost_rate()
costs.percentile(90)    # a planning budget: 9 windows in 10 cost less
costs.std               # how much a window's cost varies
costs.by_category       # mean repair, replace, preventive, component_downtime, system_downtime
costs.by_component      # mean cost attributable to each costed component
```

`cost()` takes the same arguments as `availability()` (`working_nodes`,
`broken_nodes`, `method`, `N`, `verbose`, `seed`). The same result comes with
`availability(...)` as `result.cost`, so one simulation gives both answers.
With nothing priced, `cost()` returns `None` and `result.cost` is `None`.

### Two different uncertainties

- `std` and `percentile` describe how much a window's cost **varies**. That
  is a property of the system; more simulations will not shrink it.
- `mean_se` and `mean_interval(confidence)` describe how precisely the
  **expected** cost has been estimated. They shrink like `1/√N`; check them
  before quoting the mean.

```python
interval = costs.mean_interval(confidence=0.95)
interval.lower < plant.expected_cost_rate() * 1000.0 < interval.upper   # True
```

`by_category` sums to `mean`; `by_component` covers each component's repair,
replace, preventive and own downtime cost (lost production is a system cost
and is not attributed to components).

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
- The cost draws come from their own random stream (seeded from the run's
  seed), so pricing never changes the failure and repair histories: a seeded
  `availability()` gives the same availability with or without costs.
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
| `policy` | `"age"` (the default): `T` after the unit was last put into service as new, so a failure restarts the clock. `"block"`: at `T, 2T, 3T, …` whatever the unit's age, skipped while it is down. |
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
def pump(interval):
    return {
        "reliability": surv.Weibull.from_params([1000, 2.5]),        # MTTF 887 h
        "repairability": surv.LogNormal.from_params([3.0, 0.5]),     # 23 h
        "replace_cost": 5000.0,
        "preventive": {
            "interval": interval,
            "duration": surv.Weibull.from_params([8, 3]),            # 7 h
            "cost": 1000.0,
        },
    }

def alone(interval):
    return RepairableRBD(
        [("s", "p"), ("p", "t")], {"p": pump(interval)}, downtime_cost_rate=500.0
    )

def with_standby(interval):
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": pump(interval), "b": pump(interval)},
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
Sweeping the interval like this is the way to choose one.

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
way. Block replacement has no exact long-run values (those methods raise
`NotImplementedError`): simulate it.

The simulation prices both policies. A replacement's cost is in
`by_category["preventive"]`, and a planned outage counts as downtime, but not
as a failure: `system_planned_outages` counts the times one took the system
down.

```python
year = alone(580).availability(t_simulation=8760.0, N=500, seed=0)
year.system_failures / year.n_simulations          # -> 3.666
year.system_planned_outages / year.n_simulations   # -> 11.82
year.cost.by_category["preventive"]                # -> 11824.0   1000 each
```

Costs, including cost distributions, and maintenance schedules are saved
with the RBD.
