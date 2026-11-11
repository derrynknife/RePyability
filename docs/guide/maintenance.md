# Maintenance policies

!!! tip "Learning this for the first time?"
    This page is the reference. The ideas behind it are taught step by step,
    with worked examples and exercises, in [Lesson
    8](../learn/maintenance.md) (maintaining on purpose).

Two component-level classes price preventive maintenance. They sit at the two
ends of how effective a repair is:

- [`NonRepairable`][repyability.NonRepairable]: the unit is **replaced** on
  failure, as good as new, so every cycle is a statistical renewal. It is
  also the component model a `RepairableRBD` uses internally.
- [`Repairable`][repyability.Repairable]: failures are **repaired** in a way
  that restores the unit only partly (imperfect repair) or not at all
  (minimal repair, "as bad as old"), and the unit is periodically
  **overhauled or replaced** to as good as new. A `Repairable` is a
  standalone tool, not an RBD node; in a `RepairableRBD` a component's
  `"repair"` simulates the same imperfect repair (see [Repairable
  systems](repairable.md#imperfect-repair)).

Fitting the lifetime or recurrence models is surpyval's job; these classes
take fitted models. Theory: [Concepts](../concepts.md#maintenance-models).

## Age replacement (`NonRepairable`)

Replace the unit at age `t` (a planned replacement costing `cp`) or on
failure (unplanned, costing `cu > cp`), whichever comes first. The long-run
cost rate is the expected cost of a cycle over its expected length,

```
cost_rate(t) = (cp · R(t) + cu · F(t)) / ∫₀ᵗ R(u) du
```

and `find_optimal_replacement()` minimises it:

```python
import surpyval as surv
from repyability import NonRepairable

bearing = NonRepairable(surv.Weibull.from_params([1000, 2.5]))
bearing.set_costs_planned_and_unplanned(1, 5)   # cp, cu

bearing.find_optimal_replacement()      # -> 493.05   replace at age 493
policy = bearing.optimal_replacement_policy()
policy.interval                         # -> 493.05
policy.cost_rate                        # -> 0.003462   cost per unit time
bearing.avg_replacement_time(493.05)    # -> 470.15  ∫₀ᵗ R(u) du, the mean cycle length
```

For an exponential lifetime, or a Weibull with shape 1 or less, the unit
does not wear out and preventive replacement never pays: the interval is
`inf` and the cost rate is the run-to-failure rate `cu / MTTF`. The same goes
for any lifetime whose best replacement age is no cheaper than running to
failure (a Gamma with shape below 1, say, or a Weibull of shape 1 or less with
zero-inflation or an offset). When some units never fail (a limited failure
population), running to failure costs nothing in the long run, since those
units are kept, so the interval is `inf` at a cost rate of 0.

An offset (a failure-free period) can make replacement pay even without
wear-out: no unit fails before the offset, so replacing just as it ends
avoids the failures that follow.

```python
dormant = NonRepairable(surv.Weibull.from_params([10, 0.5], gamma=1000))
dormant.set_costs_planned_and_unplanned(1, 5)
dormant.find_optimal_replacement()                 # -> 1000.0   at the offset
dormant.optimal_replacement_policy().cost_rate     # -> 0.001    cp / 1000
```

```python
steady = NonRepairable(surv.Exponential.from_params([0.001]))
steady.set_costs_planned_and_unplanned(1, 5)
steady.optimal_replacement_policy().cost_rate   # -> 0.005   = 5 / 1000
```

`set_costs_planned_and_unplanned` raises `ValueError` unless both costs are
finite, `cp` is not negative and `cp < cu`. Set the costs before asking for a
cost rate or a policy: without them `cost_rate`, `find_optimal_replacement()`
and `optimal_replacement_policy()` raise `ValueError` (except that
`find_optimal_replacement()` returns `inf` without them in the cases above).

A non-parametric lifetime (a Kaplan–Meier fit, say) works too, for a unit on
its own (a diagram refuses one: see [Building an RBD](building.md)). Its survival
function is taken as linear between the estimate's time points, from 1 at age
0, and held at its last value beyond them; the cost rate is then exact, and
the search covers ages up to the last time point, since the estimate says
nothing about later ones.

### As a repairable component

The second argument of `NonRepairable` is the time to replace (repair) the
unit; it defaults to zero (an instant swap). With it, the unit has a
long-run availability and failure frequency, which is how a `RepairableRBD`
uses it:

```python
pump = NonRepairable(
    surv.Weibull.from_params([1000, 2.5]),
    surv.Exponential.from_params([1 / 24]),   # mean 24 h to replace
)
pump.mean_availability()     # -> 0.9737   MTTF / (MTTF + MTTR)
pump.mean_unavailability()   # -> 0.02634
pump.failure_frequency()     # -> 0.001097   1 / (MTTF + MTTR)
```

`reset()` and `next_event()` are the event interface a `RepairableRBD`
simulation calls: `next_event()` alternates between drawing a time to failure
from `reliability` (returned with `False`) and a time to repair from
`time_to_replace` (returned with `True`), and `reset()` starts again with a
failure.

To price an age- or block-replacement policy for a component in its system
(where a planned stop may or may not halt production), give the component a
`"preventive"` schedule in a `RepairableRBD`: see
[Costs](system-maintenance.md#preventive-maintenance).

## Replacement over a calendar of shuts (`NonRepairable`)

Age replacement has one planned price at every moment. In a plant, the price
depends on the date: replacing a bearing in the annual shut costs little, in a
quarterly shut more, at a weekly stop more again, on any other day more still
(or it is not allowed), and at a failure most of all.
[`optimal_calendar_policy`][repyability.NonRepairable.optimal_calendar_policy]
finds the long-run optimal policy over such a calendar
(opportunity-based age replacement, after Dekker et al.). The state is the
date and the part's age, and the policy is a **threshold for each date**:
replace the part if it is at least `a*(t)` old.

Times are numbers in the life model's unit (days here), from 0: turning dates
into them is Reliafy's.

```python
import surpyval as surv
from repyability import Calendar, NonRepairable, Opportunity

bearing = NonRepairable(surv.Weibull.from_params([1000.0, 2.5]))
plant = Calendar(
    [
        Opportunity("weekly", 3.0, period=7.0),
        Opportunity("quarterly", 2.0, period=91.0),
        Opportunity("annual", 1.0, period=364.0),
    ],
    failure=20.0,      # by default the part's unplanned_cost
    anytime=4.0,       # any other day; None: not allowed; by default its planned_cost
)
policy = bearing.optimal_calendar_policy(plant)

policy.cost_rate             # -> 0.006943   a day (2.53 a year)
policy.replacement_rates     # a day; a year:  annual 0.962, failure 0.078,
                             #   weekly, quarterly and any day 0
policy.cost_rates            # a day; a year:  annual 0.962, failure 1.566
policy.schedule(kinds=("quarterly", "annual"))
# -> [(0.0,   'annual',    88.375, 0.962),   (time, kind, threshold, chance)
#     (91.0,  'quarterly', 221.375, 0.0),
#     (182.0, 'quarterly', 323.75, 0.0),
#     (273.0, 'quarterly', 574.0, 0.0)]
policy.next_replacement(age=60.0, now=364.0)
# -> PlannedReplacement(time=728.0, kind='annual', price=1.0)
```

An [`Opportunity`][repyability.Opportunity] repeats every `period` from its
`offset`, or falls on given `dates`.

- **What the policy does, and costs.** `replacement_rates` and `cost_rates`
  give, by kind, the replacements and the cost a unit of time in the long
  run; the costs sum to `cost_rate`. Here the annual shut does all the
  planned work: 0.96 replacements a year at 1, with 0.08 failures a year at
  20, and no other kind is ever used.
- **Each shut's threshold, and whether it is used.** `schedule()` lists each
  opportunity of a cycle as `(time, kind, threshold, chance)`: replace the
  unit there if it is at least `threshold` old, which happens with `chance`
  in the long run. The threshold at a shut depends on how far the next
  cheaper one is (the quarterly shut a quarter before the annual waits for a
  unit of 574 days, the one three quarters before it for one of 221), but at
  these prices the annual shut replaces units before they reach any of them,
  so each quarterly shut's chance is 0.
- **The unit in service.** `next_replacement(age, now)` says when it will be
  replaced, if it does not fail first: a unit 60 days old at the annual shut
  is younger than its threshold, about 88 days, so it is kept to the next
  one.

![The threshold age at each weekly, quarterly and annual shut over a year](img/calendar-thresholds.svg)

The chart is drawn by `scripts/calendar_policy_chart.py`. The quarterly and
annual thresholds, with what the simple policies cost over ten years of the
plant ([`calendar_policy_cost`][repyability.NonRepairable.calendar_policy_cost]
prices a rule given as the age from which each kind of replacement replaces):

| Policy | Rule | Cost over 10 years |
|---|---|---|
| **Optimal** | annual from 88 d; quarterly from 221, 324, 574 d; weekly as charted | **25.27** |
| Every annual shut | `{"annual": 0.0}` | 25.48 |
| Age replacement on any day | `{"anytime": 493.0}` (the textbook interval at 4 and 20) | 50.41 |
| Run to failure | `{}` | 82.05 |

### How it is worked out

`method="auto"` (the default) chooses among these:

- **`"dp"`**: dynamic programming on a grid (its step about the mean life
  over 1,000, put on the opportunities' own times). Over a periodic
  calendar whose cycle (the least common multiple of its periods) has at
  most 20,000 steps, relative value iteration gives the cost rate exactly for
  the grid; a failure renews half a step later on average, so the grid's
  error is second order in the step. With no opportunities and a planned
  price on any day, it is age replacement: threshold 493 days, 0.0034620 a
  day, against the textbook 493.05 and 0.0034620.
- **`"integration"`**: between opportunities in continuous time, on
  quadrature nodes (32 a gap); only with no replacement on any day. It agrees
  with `"dp"` as the step shrinks.
- **`"sampling"`**: for a cycle too long to sweep (co-prime periods of 7, 93,
  400 and 1,301 days repeat every 928,000 years). The cost rate is the mean
  over `mc_samples` sampled times of the cycle of a window's rate, each window
  an exact solve, with its standard error (`cost_rate_se`); the threshold at a
  time is a solve over a window from it, deterministic. Each window takes a
  second or more, so `mc_samples` (64 by default) sets the cost, and
  `"auto"` warns when it falls to sampling. A sampled policy's rates by kind
  (`replacement_rates`, `cost_rates`) and the schedule's chances are
  estimates from a new unit run in, with no standard error: only the cost
  rate has one, and the rates by kind move with the seed (by some 10% at
  8 samples).

`policy.method` and `policy.basis` (`"exact"`, `"differenced"` or
`"sampled"`) say which was used.

**How exact it is.** The grid's step is not an argument: it is a numerical
setting, sized from the life and put on the shuts' periods (here 0.875 days),
not a choice of when decisions are made, which the calendar says by its
opportunities. A periodic calendar keeps its period on the grid whatever
its offsets: an offset or date off the grid (a shut 0.1234 days, three
hours, into the day, as dates turned into model time give) is laid at the
grid time nearest it, within half a step, with a warning, and the policy
is the aligned calendar's. On the grid the cost rate is exact for decisions at its
times, and within about the square of the step of deciding at any moment:
for this plant 2.5272582 a year, against 2.5272583 on a grid four times as
fine. With no replacement on any day, `method="integration"` works between
the shuts in continuous time, exact but for its quadrature (2.5272576 a
year here). A plan of given dates is extended past its
last date by `tail`: `"repeat"` (repeat it) or `"anytime"` (only the
replacement on any day). Both are solved, and `policy.determined_until` is the
first time at which the two differ: the policy before it does not depend on
what comes after the plan. If at some time the ages that replace are not all
those from the youngest (a life whose hazard falls), the policy says so
(`threshold_only` is False) and warns.

The policy prices a part alone. Given to a `RepairableRBD` component (see
[Replacement on a calendar policy](system-maintenance.md#replacement-on-a-calendar-policy)),
it is followed as worked out, in a maintenance group too, whose stops are
then chances to renew the part that the policy did not plan for: it is not
re-optimised for them.

## Overhaul under minimal repair (`Repairable`)

Each failure is minimally repaired at cost `cr`; every `t` the unit is
overhauled to as good as new at cost `co > cr`. With `Λ(t)` the expected
number of failures by `t` (the cumulative intensity), the long-run cost rate
is the Barlow–Hunter policy,

```
cost_rate(t) = (cr · Λ(t) + co) / t
```

For a power-law (Crow–AMSAA) process `Λ(t)` is analytic:

```python
from surpyval.recurrent import CrowAMSAA
from repyability import Repairable

gearbox = Repairable(CrowAMSAA.from_params([1500, 1.5]))
gearbox.set_repair_and_overhaul_costs(100, 10_000)   # cr, co

gearbox.is_simulated                    # False: E[N(t)] is analytic
gearbox.find_optimal_overhaul_interval()   # -> 51299.3
gearbox.optimal_overhaul_policy().cost_rate   # -> 0.5848
gearbox.cost(5000)                      # -> 10608.6   the cost of one 5000-hour cycle
gearbox.cost_rate(5000)                 # -> 2.122     overhauling every 5000 hours instead
```

A finite optimum needs wear-out (Crow–AMSAA `beta > 1`). With `beta <= 1`
repairs never become more frequent, overhauls never pay, and the interval is
`inf` with the cost rate of the repairs alone:

```python
steady_gearbox = Repairable(CrowAMSAA.from_params([1500, 0.9]))
steady_gearbox.set_repair_and_overhaul_costs(100, 10_000)
steady_gearbox.optimal_overhaul_policy().interval   # inf
```

## Imperfect repair (`Repairable`)

Most repairs sit between the two extremes: each one turns the unit's
virtual age back part of the way. Give `Repairable` a surpyval
`GeneralizedRenewal` model (Kijima I or II, restoration factor `q` between 0
for perfect repair and 1 for minimal repair) and the same policies apply.
`E[N(t)]` then has no closed form and is estimated by simulation, so pass a
`seed` for a reproducible answer and `mc_samples` to trade speed for
precision (default 1000). The simulated numbers below (marked `~>`) vary a
little with the seed and with surpyval's simulator; the cost rate is flat
near its minimum, so the interval found varies more than the cost rate:

```python
from surpyval.recurrent import GeneralizedRenewal

model = GeneralizedRenewal.fit_from_parameters(
    [1500, 2.0], q=0.4, kijima="i", dist=surv.Weibull
)
compressor = Repairable(model)
compressor.set_repair_and_overhaul_costs(100, 1000)

compressor.is_simulated   # True
overhaul = compressor.optimal_overhaul_policy(seed=0, mc_samples=200)
overhaul.interval         # ~> 7600   between 7,400 and 7,900
overhaul.cost_rate        # ~> 0.294
```

`max_interval` bounds the search (by default 15 times the mean of the
baseline lifetime). If surpyval's simulator reports that it stopped
resolving failures before the horizon (as it can close to minimal repair,
`q` near 1, where the unit's virtual age reaches ages at which the survival
underflows double precision), the search shortens its horizon to what the
simulation can resolve, so near-minimal repair needs no special handling.
The simulated search never returns `inf`:
if the cost rate is still falling at the horizon (overhauls never pay, or
the optimum lies beyond it), it returns the horizon with a warning that says
whether a larger `max_interval` could help. The optimum grows as repairs
become more effective.

### Replace at the N-th failure

Instead of renewing at a fixed age, repair each failure and replace the unit
at its N-th failure. The cost rate is `(cr · (N − 1) + co) / E[T_N]`, with
`E[T_N]` the expected time to the N-th failure:

```python
limit = compressor.optimal_failure_limit_policy(seed=0, mc_samples=500)
limit.failure_count   # ~> 14   replace at the 13th or 14th failure
limit.cost_rate       # ~> 0.285
compressor.find_optimal_replacement_failure_count(seed=0, mc_samples=500)   # ~> 14
compressor.expected_time_to_nth_failure(5, seed=0, mc_samples=500)   # ~> 4360
```

The result is a [`FailureLimitPolicy`][repyability.FailureLimitPolicy].
`max_failures` bounds the search (by default 30); if the optimum equals the
bound, raise it. The cost rate is usually flat near its minimum, so the
optimal count can move by several failures with the seed or `mc_samples`
while the cost rate barely changes: read it as a region, and compare cost
rates. These methods need a simulation-backed (imperfect-repair) model and
raise `ValueError` otherwise.

For the minimal-repair limit (`q = 1`, a power-law process), `E[T_n]` is exact
and needs no simulation:

```python
from repyability import minimal_repair_time_to_nth_failure

minimal_repair_time_to_nth_failure(alpha=1500, beta=2.0, n=5)   # -> 3271.4
```

It is `alpha · Γ(n + 1/beta) / Γ(n)`.

### Repair, overhaul and replace in cycles

Real plants often work in two levels. Failures are repaired on the spot, which leaves the unit as bad as it was. Every so often the unit is **overhauled**, which makes it younger but not new. After a few overhauls it is **replaced**, which clears the age the overhauls could not. The questions are how often to overhaul (`interval`) and how many periods to run before replacing (`cycles`).

The overhaul takes off a share of the unit's age: `overhaul_factor` is the share it *keeps*, from 0 (as good as new) up to, not including, 1 (no effect). The unit's failures follow a power-law or other intensity model (an analytic model, such as `CrowAMSAA`), and repairs, overhauls and replacements take no time:

```python
from surpyval.recurrent import CrowAMSAA

pump = Repairable(CrowAMSAA.from_params([100.0, 2.5]))
pump.set_repair_and_overhaul_costs(cr=1.0, co=5.0)     # a repair, an overhaul

# Replace every 280 time units, no overhauls (cost 20 for a replacement):
pump.overhaul_replace_cost_rate(280.0, 1, 0.6, 20.0)   # -> 0.1183
# Overhaul every 178, replace every second overhaul period:
pump.overhaul_replace_cost_rate(177.7, 2, 0.6, 20.0)   # -> 0.1172

best = pump.optimal_overhaul_replace_policy(0.6, 20.0, max_cycles=5)
best.interval, best.cycles   # -> (177.70161320528848, 2)
```

Behind the numbers: period `j` starts at the age `v` the overhauls have left (`v_0 = 0`, `v_j = f (v_(j-1) + T)` for the share `f` kept), and expects `Λ(v + T) − Λ(v)` failures, `Λ` being the model's cumulative intensity. A replacement cycle of `K` periods then costs `cr` times the repairs, plus `(K − 1)` overhauls and one replacement, and lasts `K·T`; the long-run cost rate is that cost over that length. It is exact, with no simulation. With one period (`cycles=1`) it is the periodic replacement of `optimal_overhaul_policy`, at the replacement cost.

Take the result as a comparison of a few policies, not a sharp optimum. The cost rate is flat, here the best of two periods saves about 1% over replacing every 280, and it need not fall steadily with the number of periods: it can have a local minimum at a few periods and a slightly lower value at many (overhaul for ever, never replace). A result equal to `max_cycles` may mean more periods pay, so try a few limits. A unit that does not wear out is never overhauled (`interval` is `inf`, `cycles` is 1). The result is an [`OverhaulReplacePolicy`][repyability.OverhaulReplacePolicy]. It needs an analytic (`cif`) model: a simulation-backed imperfect-repair model counts failures from new only and is refused.

## From the part's state: options, delays and renewals (`NonRepairable`)

These answer questions about one part, asked from its age now, in the
part's time unit (turning dates into times is yours). All are exact except
where said.

**Which option is cheapest?** The plan replaces the part on fixed times,
the next `due_in` from now and the following one `every` later.
`compare_replacement_options` costs each way of handling the next
replacement to the following one: do it as planned, now, deferred, moved to
a shut with its own price, or skipped.

```python
import surpyval as surv
from repyability import NonRepairable

bearing = NonRepairable(
    surv.Weibull.from_params([365.0, 2.0]), planned_cost=150.0, unplanned_cost=400.0
)
choice = bearing.compare_replacement_options(
    200.0, 30.0, 180.0, defer_by=[30.0], shuts={"shut": (90.0, 100.0)}
)
[(o.name, round(o.cost)) for o in choice.options]
# -> [('replace', 277), ('now', 269), ('defer by 30', 289), ('shut', 257), ('skip', 274)]
choice.best.name   # -> 'shut'
```

For a new part this agrees with `skip_next_replacement`.

**What does a delay cost?** `delay_effect(age, planned_in, done_in)` is the
extra chance the part fails first, and the extra expected cost, if the
replacement planned for `planned_in` is done at `done_in`.
`cost_of_waiting_for_visit` rounds the due time up to the next visit of an
`Opportunity` (or a list of times); `cost_of_missed_visit` is the delay to
the next visit, weighted by the chance the visit is missed; `slack_effect`
is the long-run cost and risk of letting a task due at age `T` be done up
to `slack` later.

```python
late = bearing.delay_effect(200.0, 10.0, 40.0)
round(late.added_failure_probability, 4)   # -> 0.0934
round(late.added_cost, 2)                  # -> 23.36
```

**A part renewed at every service.** `renewal_forecast` gives the chance
the part fails before each service, and the expected failures, renewals and
cost over a window, from its age and how far it is into the interval, and
says whether renewing at every service beats running to failure and the
best age replacement.

```python
seal = NonRepairable(
    surv.Weibull.from_params([6000.0, 2.2]), planned_cost=200.0, unplanned_cost=900.0
)
plan = seal.renewal_forecast(944.0, 500.0, resets=False, age=340.0)
[round(r, 4) for r in plan.risk_before_service]   # -> [0.0024, 0.0042]
round(plan.expected_cost, 2), plan.worth_it       # -> (407.06, False)
```

**A window's cost and its spread.** A budget starts from the part as it
is and should carry a range, not one figure. `cost_distribution` gives,
over a window from the part's state, the chance of every number of
failures and planned replacements, and so of every cost: the mean, the
standard deviation, percentiles (10th, 50th and 90th by default) and the
chance of exceeding a budget. The policy is run to failure (no `every`),
age replacement (`resets=True`) or block replacement (`resets=False`), as
for `renewal_forecast`; replacements take no time, and an event at the
window's very end falls in the next one.

```python
bearing = NonRepairable(
    surv.Weibull.from_params([365.0, 2.0]), planned_cost=100.0, unplanned_cost=400.0
)
shuts = bearing.cost_distribution(
    1086.0, 180.0, resets=False, age=224.0, since_renewal=159.0, budget=1500.0
)
round(shuts.mean, 1)                          # -> 1155.0
[float(c) for c in shuts.percentiles]         # -> [600.0, 1000.0, 1800.0]
round(shuts.probability_above_budget, 3)      # -> 0.151
[round(p, 3) for p in shuts.failure_probabilities[:4]]   # -> [0.224, 0.362, 0.262, 0.112]
run = bearing.cost_distribution(1086.0, age=224.0)
round(run.mean, 1), round(run.probability_above(1500.0), 3)   # -> (1372.9, 0.442)
```

Costs come in steps (here $100 a shut, $400 a failure), so a percentile
jumps from one outcome to the next: read the outcomes too
(`shuts.outcomes()`, each count of failures and planned replacements with
its cost and chance, most likely first). How it is worked out:

- **The counts decide the cost**, `unplanned × failures + planned ×
  planned replacements`, so everything follows from their joint
  distribution, `probabilities[i, j]`.
- **Age replacement and run to failure:** the history is a run of
  cycles, each ending in a failure before age `T` or a planned
  replacement at exactly `T` (the part in now's first cycle from its age).
  Planned cycles all last `T`, so `i` failures and `j` planned
  replacements take the failure cycles' total plus `j T` in any order; the
  totals' distributions are convolutions of the failure cycle's, on a grid
  sized from the life's spread.
- **Block replacement:** the planned replacements are at fixed times, so
  their number is known, and the failures are a sum of independent counts,
  one per stretch between them.
- **Precision:** the grid's error falls as its step squared. The answer is
  worked out again on a grid twice as coarse, and `precision` is the
  largest change of any probability (about three times the error), plus
  any mass left out; typically 1e-9 to 1e-6.
- **`method="simulate"`** (with `mc_samples` and `seed`) draws the part's
  cycles instead, as a check; it reports `standard_error` for the mean.

**Each of a run of windows.** A budget passed on by period needs each
window's own spend and what could happen in it. `cost_by_window` takes the
windows' `ends` (or a `width` and a `count`) and the same policy and state,
and gives each window's distribution, worked out through every earlier
window: a failure in one renews the part for the next, so a later window is
not a part started afresh. Each window's outcomes are also given in four
classes, by what a planner acts on: nothing, planned work only, one
failure, several failures.

```python
years = bearing.cost_by_window(
    [263.0, 629.0, 994.0, 1359.0], 180.0, resets=False, age=224.0,
    since_renewal=159.0, budget=600.0,
)
[round(float(m), 1) for m in years.mean]     # -> [330.0, 382.3, 382.1, 382.2]
years.outcome_classes          # -> ('nothing', 'planned only', 'one failure', 'several failures')
[round(float(p), 3) for p in years.outcome_probabilities[0]]   # -> [0.0, 0.708, 0.262, 0.031]
[round(float(p), 3) for p in years.probability_above_budget]   # -> [0.031, 0.063, 0.063, 0.064]
round(years.mean.sum(), 1), round(years.total.mean, 1)         # -> (1476.5, 1476.5)
[float(c) for c in years.total.percentiles]                    # -> [800.0, 1600.0, 2000.0]
```

The windows' means add up to the whole span's, but their ranges do not:
the four windows' 90th percentiles sum to $2,400, and the whole span's is
$2,000 (`total`, the span's own distribution). A later window's first event
comes from the part in service at its start, put in at the last renewal
before it (the part in now, a planned replacement in the chain from now,
or a renewal after a failure, by the renewal measure), and from there the
counts are worked out as for one window from now. Under block replacement
each window's planned replacements are known.

**A layered package of visits.** A cheap task at every visit and a deeper
one at every `k`-th: `optimal_layered_visits(light, deep, visit_cost=...)`
finds the visit interval and the ratio `k` that cost least in the long run.
The visits keep to fixed times, a part that fails is replaced at once, and
with one layer and no visit cost it is the best block-replacement interval.

```python
from repyability import optimal_layered_visits

filter_ = NonRepairable(
    surv.Weibull.from_params([90.0, 3.0]), planned_cost=20.0, unplanned_cost=1000.0
)
pump = NonRepairable(
    surv.Weibull.from_params([700.0, 2.5]), planned_cost=400.0, unplanned_cost=2500.0
)
plan = optimal_layered_visits(filter_, pump, visit_cost=50.0)
plan.ratio, round(plan.interval, 1), round(plan.cost_rate, 3)   # -> (10, 29.9, 5.836)
```

## The four ways of managing a failure mode (`NonRepairable`)

For one failure mode, four numbers sets compare the usual tasks: how often
it fails, what it costs a unit of time, and how much risk is left. All are
exact, in the part's time unit.

```python
from repyability import (
    NonRepairable, failure_finding, on_condition,
    optimal_failure_finding_interval, run_to_failure, scheduled_replacement,
)

run_to_failure(bearing).cost_rate                    # ~> 1.2366  per day
plan = scheduled_replacement(bearing)                # at the cheapest age
round(plan.interval), round(plan.cost_rate, 4)       # -> (298, 1.117)
plan.worth_it                                        # -> True
```

**Failure finding** is for a protective device whose failure nobody sees
until it is tested. A test finds a failed device and repairs it as new. The
mean unavailability is the average chance it is failed when it is needed;
with a demand rate it gives the rate of multiple failures (a demand met by
a failed device).

```python
valve = NonRepairable(surv.Exponential.from_params([0.001]))   # per year
test = failure_finding(valve, 2.0, demand_rate=0.02)
round(test.mean_unavailability, 6)   # -> 0.000999   near lambda * tau / 2
optimal_failure_finding_interval(valve, target_unavailability=0.001).interval  # ~> 2.0
```

Without a target, `optimal_failure_finding_interval` takes the test, repair
and consequence costs and the demand rate, and finds the cheapest interval.

**On condition** is for a failure that gives warning: the P-F interval is
the time from the warning to the failure. Inspect every so often, finding the
warning with some probability each time:

```python
on_condition(30.0, 20.0, onset_rate=1 / 400).detection_probability   # -> 1.0
on_condition(30.0, 60.0, onset_rate=1 / 400).detection_probability   # -> 0.5
```

`pf_interval` may be a fitted surpyval model instead of a number, and
`optimal_on_condition_interval` finds the cheapest interval from the prices.

## Running harder against wearing faster

When how the unit is run changes its life (a fitted regression model of
its life against a temperature or a load), `operating_schedule_value` and
`best_switch_time` value an operating schedule for one unit replaced at
each failure: the value it earns, its failures' cost and the NPV, and the
best time to change from one way of running to another. See [Is it worth
running hot?](condition-based.md#is-it-worth-running-hot).

## Results

Both policy types are small typed results:
[`MaintenancePolicy`][repyability.MaintenancePolicy] (`interval`,
`cost_rate`) and [`FailureLimitPolicy`][repyability.FailureLimitPolicy]
(`failure_count`, `cost_rate`). A calendar's policy is a
[`CalendarPolicy`][repyability.CalendarPolicy] (`cost_rate`,
`replacement_rates`, `cost_rates`, `schedule()`, `next_replacement()`). Costs are per event and undiscounted.
