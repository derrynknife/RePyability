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

Every cost defaults to 0, so any subset can be priced. Costs are corrective
only (every failure is repaired at the same price) and undiscounted. A cost
must be finite and non-negative; an unknown key (such as `repair_costs`)
raises `ValueError` rather than being priced at zero.

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
costs.by_category       # mean repair, replace, component_downtime, system_downtime
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
replace and own downtime cost (lost production is a system cost and is not
attributed to components).

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

Costs, including cost distributions, are saved with the RBD.
