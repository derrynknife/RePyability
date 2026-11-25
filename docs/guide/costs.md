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
| System | `crew_cost_rate=` | Per unit time per repair crew, busy or not (a salary), with `repair_crews=`. A number. See [paying the repair crews](#paying-the-repair-crews). |
| System | `call_out=` | A call-out crew's `"fee"` per call-out and `"rate"` per unit time engaged, with `repair_crews=`. See [a call-out crew](#a-call-out-crew). |

Which cost is charged when, for a component:

| When | Charged |
|---|---|
| It fails (a replacement or repair on failure) | `repair_cost` and `replace_cost`; for a component repaired imperfectly, `replace_cost` only when it is replaced. |
| A planned replacement (a `"preventive"` task) | The task's `"cost"`, not `replace_cost`. |
| An inspection of a `"condition"` policy | The task's `"inspection_cost"`. |
| A test of an `"inspection"` spec (hidden failures) | That spec's `"cost"`; a failure it finds is then charged `repair_cost` and `replace_cost`. |
| While it is down, for any reason | `downtime_cost` per unit time, and `downtime_cost_rate=` while the system is down too. |

So a $4,000 planned replacement taking 8 hours is
`"preventive": {"policy": "age", "interval": ..., "cost": 4000.0, "duration": 8.0}`
(a number is a fixed time; a model of the time is given the same way), not
`replace_cost`, which a failure is charged.

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

The two downtime costs add up: a component's `downtime_cost` is charged
whenever it is down, and `downtime_cost_rate` whenever the system is, so
an hour a series component is down costs both. With
`"downtime_cost": 500.0` on a series pump and `downtime_cost_rate=1000.0`,
that hour costs 1,500. Price lost production in one place: put it in
`downtime_cost_rate`, and keep `downtime_cost` for what the component's
own outage costs (a hired stand-by unit, a penalty per leg).

Every cost defaults to 0, so any subset can be priced. Costs are
undiscounted (but for the [total cost](#discounting)'s optional discount
rate), and every failure is repaired at the same price (preventive
replacement is [below](system-maintenance.md#preventive-maintenance)). A cost must be finite and
non-negative; an unknown key (such as `repair_costs`) raises `ValueError`
rather than being priced at zero.

### A subsystem's costs

A nested `RepairableRBD` (see [Nested repairable RBDs](repairable.md#nested-repairable-rbds))
brings its costs with it. Whatever it pays (its components' repairs and
tasks, their downtime costs, its own `downtime_cost_rate`, its crews'
salaries and set-ups, an upgrade's capital) is counted in every cost the
outer diagram works out: `expected_cost_rate`, `total_cost`,
`expected_cost`, `cost` and `availability().cost`, `expected_npv`,
`npv_distribution` and `plan_vs_actual`. Each amount keeps its category
and is labelled with where it falls: a component of the subsystem at
node `"pumps"` is `("pumps", "A")` (deeper, the path down), and what the
subsystem pays itself is `"pumps"`'s. Its `downtime_cost_rate` is the
cost of that node being down, so the outer diagram counts it as
`component_downtime`; the outer `system_downtime` stays the outer
system's. Its `acquisition_cost` (its components') is the node's
purchase price.

```python
pumps = RepairableRBD(
    [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
    {"A": unit(0.1, 1.0, repair_cost=200.0), "B": unit(0.1, 1.0, repair_cost=200.0)},
    downtime_cost_rate=50.0,
)
station = RepairableRBD(
    [("s", "pumps"), ("pumps", "C"), ("C", "t")],
    {"pumps": pumps, "C": unit(0.02, 0.5, repair_cost=500.0)},
    downtime_cost_rate=1000.0,
)
pumps.expected_cost_rate()     # -> 36.7769   2 * 200 / 11, and 50 * (1/11)**2
station.expected_cost_rate()   # -> 92.8004   the pumps' 36.7769 and the rest's 56.0235
window = station.expected_cost(100.0)
window.by_component   # -> {'C': 962.28, ('pumps', 'A'): 1819.83, ('pumps', 'B'): 1819.83, 'pumps': 40.76}
```

A subsystem runs on its own: nothing outside it changes what happens in
it (it has crews of its own), so its costs are its own methods'. The
exact values add its expected costs; a simulation adds, simulation by
simulation, those of a run of the subsystem alone, whose random numbers
are named by its place in the outer diagram, so that it draws exactly
what it draws inside the outer run (on either engine). Held working
(`working_nodes`), a subsystem costs nothing; held broken, its own
downtime cost throughout. Two things refuse, naming the node: a
subsystem with a downtime cost in a shutdown's scope (the shut holds it
off line as the outer diagram sees it, which its own downtime does not
take), and, for a simulation, one whose draws do not all come from a
stream (a model other than a surpyval parametric one). A plan against a
log compares what the subsystem's log gives (its downtime and its crews)
and carries the rest as planned (`not_compared`): the log is the
subsystem's, not its components'.

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

With nothing priced, not even an [acquisition cost](#the-total-cost-of-ownership),
`has_costs` is `False` and there is no cost model to evaluate:
`expected_cost_rate`, `total_cost`, `expected_cost` and `cost` refuse (a
`routes.Refused`, a `ValueError`), as every analysis refuses a diagram with
nothing it works on.

### Paying the repair crews

With [limited repair crews](repairable.md#repair-crews), a crew more
shortens the queue for repairs, but it is paid whether it works or not.
`crew_cost_rate=` prices each of the `repair_crews` crews per unit time:
the crews cost `repair_crews · crew_cost_rate` per unit time from the
start, whatever happens, and every cost result shows it under the
category `"crews"`. Here a second crew saves less lost production than
its salary:

```python
def crewed(crews):
    return RepairableRBD(
        edges,
        {
            "A": unit(0.1, 1.0, repair_cost=200.0),
            "B": unit(0.1, 1.0, repair_cost=200.0),
            "C": unit(0.02, 0.5, repair_cost=500.0, replace_cost=1500.0),
        },
        downtime_cost_rate=1000.0,
        repair_crews=crews,
        crew_cost_rate=20.0,
    )

crewed(1).expected_cost_rate()   # -> 152.77   20.0 of it the crew's
crewed(2).expected_cost_rate()   # -> 161.33
```

Over a window `T` the crews cost `n c T`, and discounted at a rate `r`,
`n c (1 − exp(−r T)) / r`; a simulation adds the same amount to every
run:

```python
crewed(1).expected_cost(100.0).by_category["crews"]   # -> 2000.0
crewed(1).expected_cost(100.0, discount_rate=0.05).by_category["crews"]   # -> 397.30
```

### A call-out crew

A contract crew that comes only when repairs queue is paid only then.
`call_out={"when_waiting": k, "fee": f, "rate": c}` is one crew more: when
a job falls due, finds every crew busy and is the `k`-th waiting, the
call-out crew comes at once and starts the waiting job the next free crew
would have taken. While it is engaged there is a crew more. It leaves at
the first end of a job that leaves a crew with no waiting job to take
(the crews are alike: the job it has then, if any, is finished by the crew
that came free), and jobs that wait while it is engaged call no other.
Each call-out costs `f` and its time engaged `c` per unit time, under
`"crews"` with any salaries.

With exponential lives and repairs the crews' Markov chain has the
call-out crew exactly, in the long run, over time and discounted;
otherwise the simulations follow the same rule (in Python). What a
contract is worth is the difference from the same system without it:

```python
def on_call(call_out=None):
    return RepairableRBD(
        edges,
        {
            "A": unit(0.1, 1.0, repair_cost=200.0),
            "B": unit(0.1, 1.0, repair_cost=200.0),
            "C": unit(0.02, 0.5, repair_cost=500.0, replace_cost=1500.0),
        },
        downtime_cost_rate=1000.0,
        repair_crews=1,
        call_out=call_out,
    )

contract = {"when_waiting": 1, "fee": 300.0, "rate": 50.0}
on_call().expected_cost_rate()           # -> 132.77
on_call(contract).expected_cost_rate()   # -> 129.69   the contract pays

states = on_call(contract).crew_states()
float(states.probabilities @ states.call_out_rate)      # -> 0.0254   call-outs per unit time
float(states.probabilities @ states.call_out_engaged)   # -> 0.0148   of the time engaged

with_it = on_call(contract).expected_npv(100.0, discount_rate=0.05)
without = on_call().expected_npv(100.0, discount_rate=0.05)
with_it.npv - without.npv                # -> 37.74   the contract's value
with_it.cost_by_category["crews"]        # -> 157.27  what it costs, discounted
```

`npv_distribution` prices simulated histories, which do not record the
call-outs: it refuses a call-out crew with a fee or a rate, and `cost()`
gives the spread of the undiscounted cost.

A crew cost needs `repair_crews`: as many crews as are needed (the
default) cannot be priced, and the constructor refuses it.

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
With nothing at all priced, `cost()` refuses, and `result.cost` is `None`.

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
and refuses the rest with the reason, as it refuses a diagram with nothing
priced. For the spread of a window's cost, simulate it with `cost()`.

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

### Net present value

A plan is judged by what it leaves you with, not only by what it costs.
`expected_npv(horizon, discount_rate=, value_rate=)` gives the **net present
value**: what the system earns while it works (`value_rate` per unit time),
less every cost, each counted at what it is worth now. It is the average over
every way the failures could fall, worked out exactly with no simulation, from
the same expected events and expected time up as `expected_cost` and
`mission_availability`: each cost is discounted at its own time (a block
replacement at its exact time, not in the middle of a year), and the value is
discounted as it is earned. The acquisition cost is paid at the start. It does
not say how much a bad run could lose: that needs the simulated distribution.

```python
npv = line.expected_npv(87600.0, discount_rate=seven, value_rate=120.0, periods=10)
npv.cost                # -> 114528.4   the same present value as expected_cost(...).total
npv.value               # -> 7563068.0  120 per hour while the pump works
npv.npv                 # -> 7448539.5
npv.undiscounted_npv    # -> 10257843.3  the same ten years, not discounted
```

The **equivalent annual cost**, `eac`, is the steady cost per unit time over
the horizon that is worth the same as all the costs (the annuity form:
`cost / ((1 − e^(−r·H)) / r)`, in the models' unit of time). It is the figure
to compare plans of different lives. With no discounting it is the total cost
over the horizon divided by its length, and over a long horizon it approaches
`expected_cost_rate()`.

```python
npv.eac * 8760          # -> 15760.9   a year, for the ten
line.expected_npv(1e7).eac   # -> 1.487   undiscounted and long: the cost rate, 1.4851, and the purchase
```

`periods` (a number of equal periods, or the ends of the periods in the
models' time) cuts the answer into a **cash-flow table**, `npv.cash_flows`:
one row for each period, category (repair, replace, preventive, inspection,
downtime, set-up, acquisition and `"value"`) and component, with its present
value and its undiscounted amount, costs negative and value positive, so the
present values add to the `npv`. It is for applying your own tax,
depreciation and financial-year conventions: which period is which year is
yours to say. Start from the components' current states with `state=`, as for
`expected_cost`.

```python
npv.cash_flows[3]["category"]   # 'value'
sum(row["present_value"] for row in npv.cash_flows)   # -> 7448539.5
```

The average hides the risk: a plan with a slightly lower mean and a much
better worst case is often the one to choose. `npv_distribution(horizon,
discount_rate=, value_rate=, mc_samples=, seed=, hurdle=, tail=0.1)`
simulates the runs themselves (the histories of `simulate_timelines`, by
seed), prices each with the diagram's own costs at the times they fell, and
gives each run's net present value (`samples`) and their spread: `p10`,
`p50` and `p90`, the share of runs below a `hurdle`, and `cvar`, the mean of
the worst `tail` of the runs. Its `mean` estimates `expected_npv`'s, within
`mean_interval()`; `percentile_interval(q)` gives a percentile's confidence
interval by the binomial method. It works where the exact methods refuse
(crews with other lives, maintenance groups, opportunistic renewals at
shutdowns), as the simulation does. A shutdown's cost is charged at each of
its starts; a component it holds off line is not charged for the hold, and a
failure or renewal in the shut is charged as anywhere else. A
cost charged at an event the histories do not record (a standby group's
units' failures, tests, planned replacements that take no time, a set-up
cost, a calendar policy's prices) is not left out: the method refuses the
diagram, naming the cost. A cost given as a distribution is charged at its
mean (`priced_at_mean`).

```python
runs = line.npv_distribution(87600.0, discount_rate=seven, value_rate=120.0,
                             mc_samples=10_000, seed=1, hurdle=7.4e6)
runs.mean                      # -> 7448443.9   expected_npv's 7448539.5, within mean_interval()
runs.p10, runs.p90             # -> (7417097.8, 7478039.2)
runs.probability_below_hurdle  # -> 0.0268
runs.cvar                      # -> 7404250.8   the worst tenth of the runs, on average
```

**Contracts that pay on availability.** An availability contract charges a
penalty for each point of availability below a target in each period, and
may pay a bonus for each point above it: a threshold, which an expected value
cannot price. `AvailabilityContract(periods, target, penalty_per_point,
bonus_per_point=0, cap=None)` states the terms (a point is one percentage
point; `cap` is the most a period can charge or pay), and
`npv_distribution(..., contract=)` pays them run by run, from the share of
each period the system was up in that run, at the period's end. The payments
are kept apart (`contract_samples`, `contract_payoff`), with each run's
availability by period (`period_availability`).

**Comparing two plans.** A choice between plans is about the difference.
`compare_npv(other, horizon, ...)` runs both plans from one seed, a component
in the same place in both drawing the same random numbers (as `compare`
does), so each run's difference comes from how the plans differ rather than
from luck. Identical plans differ by exactly 0 in every run. Here a pump
repaired in 5 hours on average against one repaired in 10, under monthly
terms of 99%, over a year at 8%:

```python
import math
import surpyval as surv
from repyability import AvailabilityContract, RepairableRBD

def pump_line(mttr):
    return RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {"reliability": surv.Exponential.from_params([1e-3]),
               "repairability": surv.Exponential.from_params([1 / mttr]),
               "repair_cost": 500.0}},
    )

terms = AvailabilityContract(730.0, target=0.99, penalty_per_point=1000.0,
                             bonus_per_point=100.0, cap=5000.0)
r = math.log(1.08) / 8760
runs = pump_line(10.0).npv_distribution(8760.0, discount_rate=r, value_rate=10.0,
                                        mc_samples=2000, seed=1, contract=terms)
runs.mean, runs.contract_payoff   # -> (73784, -5546)   penalties cost 5,546 on average
gain = pump_line(5.0).compare_npv(pump_line(10.0), 8760.0, discount_rate=r,
                                  value_rate=10.0, contract=terms,
                                  mc_samples=2000, seed=1)
gain.mean                         # -> 4767   the faster repair is worth this much more
gain.interval.lower, gain.interval.upper    # -> (4656, 4879)
gain.interval.standard_error, gain.independent_standard_error   # -> (56.9, 139.8)
gain.probability_other_better     # -> 0.0005
```

Two independent runs of the same size would have pinned the difference down
less than half as well (`independent_standard_error`).

### Downtime you can be sure of

"I want to be 95% sure I don't lose more than 100 hours to downtime this
year: what is the least I need to pay?" `least_spend_for_downtime` answers
that. A design meets the limit when an upper confidence bound on its
downtime's 95th percentile is under it: an order statistic, so that the
simulation's own noise cannot make a design look safe. The levers each
have their price (copies of a node at a price each, a node's maintenance
intervals at the cost of their tasks), and spend has no bound unless you
give `budget=`:

```python
one_pump = RepairableRBD(
    [("s", "p"), ("p", "t")],
    {"p": {"reliability": surv.Exponential.from_params([1e-3]),
           "repairability": surv.Exponential.from_params([0.02])}},
)
sure = one_pump.least_spend_for_downtime(8760.0, 100.0, redundancy={"p": 5000.0},
                                     mc_samples=1000, seed=1)
sure.met, sure.design.copies, sure.design.spend   # -> (True, {'p': 1}, 5000.0)
round(sure.design.estimate, 1), round(sure.design.bound, 1)   # -> (80.6, 90.2)
round(sure.frontier[0].estimate)   # -> 752   one pump's bad year, in hours
```

`frontier` lists the designs from the cheapest, each with its spend and
its bad-year downtime, so you see what each extra dollar buys. The
target is all system downtime by default; `downtime="unplanned"` counts
failures only, and `measure="cost"` prices each hour at the diagram's
`downtime_cost_rate` (and any `value_rate=` lost). Every design reports
both its `planned` and its `unplanned` downtime, the mean and the bad
year's, in hours and cost, so the trade-off shows. Cutting a wearing
unit's unplanned downtime with shorter intervals costs planned shutdown:

```python
worn_unit = RepairableRBD(
    [("s", "p"), ("p", "t")],
    {"p": {"reliability": surv.Weibull.from_params([4000.0, 3.0]),
           "repairability": surv.Exponential.from_params([1 / 72.0]),
           "repair_cost": 2000.0,
           "preventive": {"interval": 8000.0, "cost": 300.0,
                          "duration": surv.ExactEventTime.from_params([8.0])}}},
)
pm_intervals = {"p": [4000.0, 2000.0, 1000.0, 500.0]}
trade = worn_unit.least_spend_for_downtime(8760.0, 10.0, downtime="unplanned",
                                      intervals=pm_intervals, mc_samples=1000,
                                      seed=3)
trade.design.intervals, round(trade.design.planned["bad_hours"])   # -> ({'p': 500.0}, 136)
[round(d.unplanned["bad_hours"]) for d in trade.frontier]   # -> [347, 307, 183, 81, 0]
everything = worn_unit.least_spend_for_downtime(8760.0, 10.0, intervals=pm_intervals,
                                           mc_samples=1000, seed=3)
everything.met, everything.floor   # -> (False, 'planned')
```

No unplanned downtime in a bad year needs 136 hours of planned shutdown.
Counting all downtime, 10 hours cannot be reached by intervals, and the
result says which floor stopped the levers (`floor`, and `stopped` in
words): here the planned task downtime; otherwise the budget or the
levers' caps (`max_units`, the last interval). `objective="cvar"` targets
the mean of the worst 5% instead (its bound is normal, approximate). A
`UnitPlan` takes the same method, through its one-node diagram.

When downtime is jobs waiting for a crew, crews and call-out are levers
too. `crews={"price": p, "max": m}` adds a repair crew a step to the
diagram's `repair_crews`, each at `p` and its salary at the
`crew_cost_rate`; `call_out=[contract, ...]` tries call-out contracts in
order, each priced by the fees and time engaged its own runs give. Two
pumps in parallel with one crew are both down 40% of the time; a second
crew, or a call-out at the first job waiting, makes it a quarter:

```python
pumps = RepairableRBD(
    [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    {n: {"reliability": surv.Exponential.from_params([0.02]),
         "repairability": surv.Exponential.from_params([0.02])} for n in "ab"},
    repair_crews=1, crew_cost_rate=2.0,
)
staffed = pumps.least_spend_for_downtime(
    8760.0, 1.0, crews={"price": 100.0},
    call_out={"when_waiting": 1, "fee": 50.0, "rate": 3.0},
    mc_samples=1000, seed=5,
)
[(d.crews, d.call_out is not None, round(d.spend)) for d in staffed.frontier]   # -> [(1, False, 0), (1, True, 10947)]
[round(d.unplanned["mean_hours"]) for d in staffed.frontier]   # -> [3495, 2186]
```

The call-out contract is cheaper than a second crew's 17,620 (its price
and a year's salary) for the same downtime: the pumps' 3,504 hours a
year down together become 2,190, as the crews' Markov chain gives.

For hidden failures, `inspections={node: [intervals]}` moves a unit's
test interval through the candidates given, each at its tests' cost.
`upgrades={node: {"reliability": better_life, "cost": price}}` (or a
list of offers, tried in order) buys a better unit from the start, at
its price for each unit, the node's copies included.
A design's spend is the levers' prices and the change in its expected
cost of preventive tasks, inspections and crews over the horizon. The
models are taken as known; `uncertainty=True` runs every design over
`n_draws` draws of the fitted models instead (the same draws for each),
so a life fitted to few failures makes the bad year worse, and the
result's `uncertainty` says what was drawn and that the bound is then
approximate. A
group of units (standby, load-sharing or identical) is one node to the
levers: its copy is a second group in parallel.

For one part replaced at an age or at a failure, over for ever,
`NonRepairable.age_replacement_value(age, discount_rate=)` gives the net
present value (Fox, 1966) and the equivalent annual cost in closed form. At a
discount rate of 0 the equivalent annual cost is `cost_rate(age)`.

```python
from repyability import NonRepairable
bearing = NonRepairable(surv.Weibull.from_params([1000.0, 2.5]),
                        planned_cost=1.0, unplanned_cost=5.0)
bearing.age_replacement_value(493.0).eac                    # -> 0.003462
bearing.age_replacement_value(493.0, discount_rate=seven).eac * 8760   # -> 30.3
```

### Upgrades, retirement and salvage

"Is the upgrade worth it?" and "run it to the end, or retire it now?" are
questions a net present value answers. Three things say them:

- **An upgrade** is on the component: `"upgrade": {"at": time,
  "reliability": new_life, "cost": capital}`. From that time (in the
  diagram's time from 0) its units have the new life: the unit in service
  is replaced by one of it, in no time, and a unit in a repair then comes
  back with it. The capital cost is charged at that time, whatever the
  unit's state, under the category `"capital"`.
- **`retire_at=`** on `expected_npv` retires the system before the
  horizon: nothing is earned or spent after it, and the equivalent annual
  cost is over the time to it.
- **`salvage=`** credits what is left at the end (the retirement, or the
  horizon), discounted from there: a number for the system, or a dict of
  a number for each component; negative for a cost of disposal. The cash
  flows show it under `"salvage"`, and the result's `salvage` is its
  present value (`cost` stays the costs alone).

```python
import math
import numpy as np
import surpyval as surv
from repyability import RepairableRBD

def worn_pump(**more):
    E = surv.Exponential.from_params
    return RepairableRBD(
        [("in", "pump"), ("pump", "out")],
        {"pump": {"reliability": E([1 / 500.0]),
                  "repairability": E([1 / 50.0]),
                  "repair_cost": 2000.0, **more}},
    )

rate, horizon = math.log(1.08) / 8760.0, 8000.0
kept = worn_pump().expected_npv(horizon, discount_rate=rate, value_rate=40.0)
kept.npv                                         # -> 252977.9
better = surv.Exponential.from_params([1 / 2000.0])
upgraded = worn_pump(
    upgrade={"at": 2000.0, "reliability": better, "cost": 30000.0}
).expected_npv(horizon, discount_rate=rate, value_rate=40.0)
upgraded.npv                                     # -> 253936.5
upgraded.cost_by_category["capital"]             # -> 29477.5
retired = worn_pump().expected_npv(
    horizon, discount_rate=rate, value_rate=40.0,
    retire_at=6000.0, salvage=8000.0,
)
retired.npv                                      # -> 199015.8
retired.salvage                                  # -> 7589.2
```

A component's salvage may depend on the age of its unit in service: give
a function of the age (taking an array of ages). Its expected value is
worked out from the age of the unit in service at the end, for a plain
component from new (one with maintenance, tests, an upgrade or a state at
0 takes a number, and says so); a unit down then is credited nothing.

```python
by_age = worn_pump().expected_npv(
    horizon, discount_rate=rate, value_rate=40.0,
    salvage={"pump": lambda age: 8000.0 * np.exp(-age / 1000.0)},
)
by_age.undiscounted_salvage                      # -> 4848.5
```

An upgraded component's values over a horizon are worked out
(`point_availability`, `expected_events`, `expected_cost`,
`expected_npv`): its curve is followed to the upgrade with the old life
and from it with the new, as a [replacement at a given
time](system-maintenance.md#replacement-at-given-times) is. Once
upgraded, the component has the new life for ever, so the long-run
methods (`mean_availability`, `expected_cost_rate`, the importance
measures by default, ...) give the long run once upgraded: the values
of the diagram built with the new life, the time before the upgrade
ignored (`total_cost`, over a horizon from 0, refuses). The simulations follow it too (`availability`, `cost`,
`npv_distribution`, `compare_npv`, in Python): a life that starts at or
after the upgrade's time is drawn from the new model, from the same
random number the old one would have used, and the capital cost is
charged at its time. So `compare_npv` against keeping the old component
gives the upgrade's worth run by run, with its spread. An upgraded
component may also be under block replacement on its own interval
(`"preventive": {"policy": "block", ...}`): it is replaced every
interval with the life in service, and at the upgrade's time as at a
block time (at its cost and in its time, beside the capital cost). Age
replacement, tests and the other tasks are not taken with an upgrade
yet.

### A plan against what happened

Two years into a ten-year plan, the question changes: *how does what actually
happened compare with what we expected, and what is the plan worth now?*
`plan_vs_actual(timelines, now, horizon, ...)` answers it from the outage log.

It does three things. First, it sets the plan's expected cash flows to
`now` (the `expected_npv` table above) beside the cash flows the log
implies, priced with the diagram's own costs: each failure in the log its
repair and replace costs, each planned outage its preventive cost, each
hour a component or the system was down its downtime cost, and each hour
the system was up its value, every amount discounted at the time it fell.
The **variance** is the actual less the expected, by period, category and
component. Amounts are signed as in the cash-flow table (costs negative,
value positive), so a positive variance is in your favour: less spent, or
more earned, than the plan expected. Second, it reads each component's
state now off its log (up for so long since its last repair, or down for so
long, and why) and values the rest of the horizon from there. Third, it adds
what happened so far to that expected rest, and sets the sum beside the
plan's own value: `change` is how much the plan has gained or lost, which is
the variance so far plus how the outlook for the rest has moved.

The log is a [`Timeline`](timelines.md) per component, on the diagram's
clock from 0 (the start of the plan) to `now`, its planned outages flagged.
Here the pump of the net present value example above, two years in:

```python
from repyability import Timeline

log = Timeline.from_outages(
    [(1500, 1512), (2900, 2950), (4100, 4108), (6100, 6190),
     (9800, 9805), (12000, 12040), (15500, 15560), (17400, 17460)],
    end=17520.0,
)
review = line.plan_vs_actual({"pump": log}, 17520.0, 87600.0,
                             discount_rate=seven, value_rate=120.0, periods=10)
log.failures, log.downtime          # (8, 325.0)    the plan expected about 17 failures, down about 173 hours
round(review.variance_so_far)       # -> -26197       against the plan, so far
review.variance_by_category["repair"]            # -> 4367.5    fewer repairs than expected
review.variance_by_category["system_downtime"]   # -> -13893.0  but long outages
review.variance_by_category["value"]             # -> -16671.6  and the production they lost
review.states["pump"]               # NodeState(age=60.0, alive=True): up for 60 hours since its last repair
round(review.plan_npv), round(review.revalued_npv)   # (7448540, 7422361)
round(review.change)                # -> -26178
```

The pump fails at a constant rate, so the outlook for the remaining eight
years does not depend on its state (`rest_npv` is the plan's own
`plan_rest_npv`), and the change is the variance so far. A component that
wears out, or one part way through a planned replacement, moves the outlook
too.

`expected_cash_flows`, `actual_cash_flows` and `variance_cash_flows` are the
three tables, row for row; `rest_cash_flows` is the expected rest on the
plan's clock and periods (a period that `now` falls in is split there).

**What the log cannot tell is not compared.** A cost the log does not
determine is left out of *both* sides and listed in `not_compared`, with the
reason, rather than counted as nil: inspections and tests that find nothing
(they leave no outage), a hidden failure's repair (charged when a test finds
it, which the log does not record), a standby group's units' repairs (its
log is the group's), planned replacements that take no time, a calendar
policy's prices, a group's set-up costs and the acquisition costs. In the
re-valued plan these are carried at the plan's expectation
(`as_planned_so_far`):

```python
review.not_compared[0]["category"]   # 'acquisition': the log does not record purchases
round(review.as_planned_so_far)      # -> -20000
```

**The states now.** Where the diagram's rules let the log give a
component's state, it is read off the log: a unit renewed at each repair is
as old as the time since its last restoration (or, down, has been down since
its last change, for a repair or, flagged planned, for its maintenance); one
replaced at an age, on a block calendar or on condition by maintenance that
takes time is the same, and on a calendar is as far past its last scheduled
time as the clock says. Where the log does not give the state (a nested
diagram, a standby group, an imperfect repair's virtual age, a unit with
hidden failures, a degradation level or stage, replacements that take no
time, a calendar policy, a member of a common-cause group, a unit that may
wait for one of too few repair crews) the method refuses, naming the
component: give its state in `state=` (a [`NodeState`](condition-based.md#describing-the-current-state),
or a nested diagram's dict of them). `states` and `derived_states` say what
was used and what was read off the logs. A component missing from the logs
is refused, never taken to have run without failing; one that never went
down has `Timeline([], end=now)`.

### One call, as JSON

An agent or a service asks an NPV question as one JSON document and gets
one back: `value_plan` takes the diagram as `to_dict()` saves it, the
analysis (`expected_npv`, `expected_npv_uncertainty`, `npv_distribution`,
`compare_npv` with the `"other"` plan, or `plan_vs_actual`) and its
inputs by name, in plain data, and answers with the request it answers,
the units, the route and the result's `to_dict()`, or the refusal and
what to run instead. The request is the record: valuing it again gives
the same answer (a seeded analysis given no seed has one written in).
`npv_schema()` gives the JSON Schema of both documents.

```python
from repyability import value_plan

answer = value_plan({
    "schema": 1,
    "diagram": worn_pump().to_dict(),
    "analysis": "expected_npv",
    "inputs": {"horizon": horizon, "discount_rate": rate,
               "value_rate": 40.0},
})
answer["route"]["route"]                         # -> 'exact'
value_plan(answer["request"]) == answer          # -> True
```

`"task_value"` asks what a service on dates is worth (see
[Is a service worth it?](schedule.md#is-a-service-worth-it)): its inputs
are the `task` (`Task.to_dict()`), the `usage` (a forecast's `start`,
`end` and duty `plan`, as `UsageForecast.to_dict()` has them), the
service's `effect` and `task_value`'s keywords; the diagram is the
unit's plan (`UnitPlan.to_dict()`), needed only with an effect. The
effect is left out for none (the service an accepted cost, valued
exactly), or `{"kind": "restoration", "q": 0.6}` (with its `"model"`,
or `"to_age"`), or `{"kind": "condition", "life": ..., "fresh": [...],
"stale": [...], "lasts": ...}`, the fitted life as a diagram saves a
node's model.

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

Redundancy is a capital decision: the copies are bought now and what they
save comes later, so a `discount_rate` counts against them. And a plant
that is down loses what it would have earned, not only what the outage
costs. `objective="npv"` with a `value_rate` (what the system earns per
unit time while it works) chooses the design with the highest net present
value over the horizon, and gives the `npv`, its split into `value`,
`acquisition_cost` and `running_cost`, and the `eac` (the same as a steady
cost per unit time; `objective="eac"` chooses by it, which over one horizon
is the same design):

```python
import math
hard = math.log(1.30) / 8760      # 30% a year, the models in hours
line.allocate_redundancy(87600.0, discount_rate=hard).units   # {'pump': 1}
best = line.allocate_redundancy(
    87600.0, objective="npv", discount_rate=hard, value_rate=400.0
)
best.units               # {'pump': 2}   an hour down now loses 500
best.npv                 # -> 12314517.8   = value - acquisition_cost - running_cost
best.value               # -> 12385481.5
best.running_cost        # -> 30963.7
best.eac                 # -> -397.67      per hour: it earns more than it costs
```

A value earned while the system works is a cost of its downtime, so the
search is the same one, with an hour down costing `downtime_cost_rate +
value_rate`; with no value it gives the total cost's design. The running
costs, the downtime and the value are the long-run rates over the horizon's
present value, as `total_cost` counts them (the result's `basis` says so);
[`expected_npv`](#net-present-value) values a design from new,
with its availability over time.

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

With limited [repair crews](repairable.md#repair-crews) a copy is one more
job that can wait for a crew: it can keep a crew from another component,
and a copy more can even leave the system down more. Each design is then
worked out whole, from the crews' Markov chain of its components and their
copies (exponential lives and repairs; a call-out crew and the crews'
salaries are counted), the chain counting how many of a component's alike
copies are up rather than telling them apart. The search keeps to what
still holds: no design is down less than with a crew for every job, and
every copy is bought. `"exact"` starts from the greedy design and solves
the chains of the designs that could still beat it by those two, the most
promising first; the result's `basis` says the designs came from the
crews' chain. A design it needs whose chain is too large (15,000 states)
is refused by name: bound the copies with `max_units`, or compare designs
with `compare()`. With crews, whole trains, common-cause groups,
maintenance groups and shutdowns are refused, and a node whose copies
cost nothing to buy needs a `max_units`. (The number of crews is not
chosen: compare the allocations for each. And with at least as many crews
as components, where no job waits, the copies are taken to have crews
too, as without `repair_crews`.)

Otherwise the model and its limits: copies are active and repaired
independently of each other (as many repair crews as failed copies), and
fail independently unless the component is in a common-cause group: then its copies join the
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
