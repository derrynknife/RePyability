# Maintenance policies

Two component-level classes price preventive maintenance. They sit at the two
ends of how effective a repair is:

- [`NonRepairable`][repyability.NonRepairable]: the unit is **replaced** on
  failure, as good as new, so every cycle is a statistical renewal. It is
  also the component model a `RepairableRBD` uses internally.
- [`Repairable`][repyability.Repairable]: failures are **repaired** in a way
  that restores the unit only partly (imperfect repair) or not at all
  (minimal repair, "as bad as old"), and the unit is periodically
  **overhauled or replaced** to as good as new. A repaired unit does not
  renew, so a `Repairable` is a standalone tool, not an RBD node.

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
`inf` and the cost rate is the run-to-failure rate `cu / MTTF`. Only these two
cases are recognised. Another lifetime without wear-out (a Gamma with shape
below 1, say) gets a large, finite age at nearly that rate.

```python
steady = NonRepairable(surv.Exponential.from_params([0.001]))
steady.set_costs_planned_and_unplanned(1, 5)
steady.optimal_replacement_policy().cost_rate   # -> 0.005   = 5 / 1000
```

`set_costs_planned_and_unplanned` raises `ValueError` unless `cp < cu`. Set
the costs before asking for a policy: without them
`optimal_replacement_policy()` raises `ValueError`, and
`find_optimal_replacement()` raises `AttributeError` (unless it can return
`inf` without them, as above).

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
`seed` for a reproducible answer and `n_simulations` to trade speed for
precision (default 1000):

```python
from surpyval.recurrent import GeneralizedRenewal

model = GeneralizedRenewal.fit_from_parameters(
    [1500, 2.0], q=0.4, kijima="i", dist=surv.Weibull
)
compressor = Repairable(model)
compressor.set_repair_and_overhaul_costs(100, 1000)

compressor.is_simulated   # True
overhaul = compressor.optimal_overhaul_policy(seed=0, n_simulations=200)
overhaul.interval         # -> 7909.0
overhaul.cost_rate        # -> 0.2965
```

`max_interval` bounds the search (by default 15 times the mean of the
baseline lifetime). The simulated search never returns `inf`: if overhauls
never pay, or the optimum lies beyond the horizon, it returns `max_interval`
itself, so a result equal to it calls for a longer horizon. There is a limit
the other way too. Close to minimal repair (`q` near 1), once the survival at
the unit's virtual age falls below double precision, surpyval's simulator
stops resolving failures (it warns that sequences stalled), which can drag the
optimum to the horizon; if that warning appears, lower `max_interval`. The
optimum grows as repairs become more effective.

### Replace at the N-th failure

Instead of renewing at a fixed age, repair each failure and replace the unit
at its N-th failure. The cost rate is `(cr · (N − 1) + co) / E[T_N]`, with
`E[T_N]` the expected time to the N-th failure:

```python
limit = compressor.optimal_failure_limit_policy(seed=0, n_simulations=500)
limit.failure_count   # -> 14   replace at the 14th failure
limit.cost_rate       # -> 0.286
compressor.find_optimal_replacement_failure_count(seed=0, n_simulations=500)   # -> 14
compressor.expected_time_to_nth_failure(5, seed=0, n_simulations=500)   # -> 4378.1
```

The result is a [`FailureLimitPolicy`][repyability.FailureLimitPolicy].
`max_failures` bounds the search (by default 30); if the optimum equals the
bound, raise it. The cost rate is usually flat near its minimum, so the
optimal count can move by several failures with the seed or `n_simulations`
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

## Results

Both policy types are small typed results:
[`MaintenancePolicy`][repyability.MaintenancePolicy] (`interval`,
`cost_rate`) and [`FailureLimitPolicy`][repyability.FailureLimitPolicy]
(`failure_count`, `cost_rate`). Costs are per event and undiscounted.
