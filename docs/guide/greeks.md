# Sensitivities: the Greeks

An option's *Greeks* say how its price moves when something it depends on
moves: delta for the price of what it is written on, gamma for how delta
itself moves, theta for the passing of time, vega for the volatility, rho
for the interest rate. The measures on this page ask the same questions of
a system's availability or reliability: how far it moves when a component
or a lever does, whether two improvements reinforce each other, what is
moving it now, and whose uncertainty makes it uncertain. They are named
after the Greeks so that the set reads as one family:

| Greek | What it asks | Method | Adds up |
|---|---|---|---|
| Delta | How far does the system move per unit of a component's availability? | `birnbaum_importance` | |
| Parameter deltas | Which lever moves it most: a life, a repair, the maintenance, a crew? And per unit spent? | `parameter_sensitivity` | |
| DIM | What share of a change lies in each component, or each group of levers? | `differential_importance` | to 1 |
| Gamma | Are two improvements complements or substitutes? | `joint_importance` | |
| Theta | Which component is moving the availability now? | `availability_rate` (`reliability_rate`) | to the system's rate |
| | Which component caused the system's failures? | `barlow_proschan_importance` | to 1 |
| Vega | Whose parameter uncertainty makes the answer uncertain? | `uncertainty_importance` | to 1 (by the delta method) |
| Rho | How sensitive is a plan's value to the discount rate? | to come, with the plan's net present value | |

[Importance measures](importance.md) is the reference for each (and
[Reliability of a system](reliability.md#whose-uncertainty-widens-the-interval)
for vega); this page runs one system through all of them.

## What they share

- **The same arguments.** On a `RepairableRBD` each is a long-run value by
  default. `x` evaluates it at times from new, `window` over
  `[0, window)`, and `state=` starts from the components' current states,
  as `point_availability` and `mission_availability` take them (theta,
  a rate, needs its times; vega names its quantity, `of=`, and takes its
  times as `x`). On a `NonRepairableRBD` they take the time `x`. All but
  vega take `working_nodes` and `broken_nodes`, which hold nodes working
  or failed as for every importance measure.
- **Exact where the structure is.** With independent components the
  system's availability is multilinear in theirs, so delta, gamma, the
  shares among components and theta's split are exact given the
  components' values (which, over time, are worked out numerically on a
  grid); a lever's derivative is a central difference of the system's own
  value; vega is the delta method's linear approximation, or Sobol indices
  estimated from draws. With limited repair crews or common-cause groups
  the components are not independent, and the measures follow the crews'
  and the groups' Markov chains instead, where they can (see the last
  section). `analysis_routes()` says, for a given diagram and without
  running anything, which of these is exact, numerical or refused, and
  why.
- **Shares that add up.** The differential importance, theta's parts,
  Barlow–Proschan's shares and vega's (by the delta method) add up to the
  whole, so they can be shown as a breakdown. Delta, the parameter deltas
  and gamma rank; they do not add up.

## One station, every measure

A pumping station: two pumps in parallel feed a valve. Times are in days.
The pumps wear out over about 40 days and take a day to repair; the valve
lasts longer, is repaired in half a day, and is replaced at age 80 days, a
quarter of a day's work:

```python
import surpyval as surv
from repyability import RepairableRBD

E, W = surv.Exponential.from_params, surv.Weibull.from_params

def unit(life, repair, **more):
    return {"reliability": life, "repairability": repair, **more}

station = RepairableRBD(
    [("s", "pump1"), ("s", "pump2"),
     ("pump1", "valve"), ("pump2", "valve"),
     ("valve", "t")],
    {
        "pump1": unit(W([40, 2.0]), E([1.0])),
        "pump2": unit(W([40, 2.0]), E([1.0])),
        "valve": unit(
            W([160, 1.5]),
            E([2.0]),
            preventive={"policy": "age", "interval": 80.0, "duration": E([4.0])},
        ),
    },
)
station.mean_availability()   # -> 0.99463
```

### Delta: how far each component moves the station

The Birnbaum importance is `∂A/∂A_i`, the station's change per unit of a
component's availability:

```python
station.birnbaum_importance()   # {'pump1': 0.0273, 'pump2': 0.0273, 'valve': 0.9992}
station.birnbaum_importance(x=[5.0, 40.0])["pump1"]   # array([0.005 , 0.0279])
```

The valve is in series: the station works whenever it does, unless both
pumps are down. A pump matters only while the other is down, 2.7% of the
time in the long run, and a fifth as often at five days, while both are
new.

### Parameter deltas: which lever

`parameter_sensitivity()` differentiates the long-run availability in each
lever: here the life and repair models' parameters and the maintenance's
interval and duration (and, where there are any, one more standby unit or
repair crew):

```python
levers = station.parameter_sensitivity()
levers["valve"]["reliability.beta"]                   # -> 0.001286
levers["valve"]["repairability.failure_rate"]         # -> 0.001055
levers["pump1"]["repairability.failure_rate"]         # -> 0.000729
levers["valve"]["preventive.duration.failure_rate"]   # -> 0.000622
levers["valve"]["preventive.interval"]                # -> 0.00003
```

Each is per unit of its lever, and the units differ: the valve's Weibull
shape has none, a repair rate is per day, the interval is in days. Given
what a unit of each costs, `unit_costs=` ranks them by availability per
unit spent; the shares below put them on one footing instead.

### DIM: shares of a change

The differential importance is each component's (or lever's) share of the
change when they all change together, so the shares add up, and a group's
is the sum of its members'. Among the components, moved by as much each:

```python
station.differential_importance()   # {'pump1': 0.0259, 'pump2': 0.0259, 'valve': 0.9482}
```

Over the levers, each moved by the same fraction of itself, the way that
improves the station (`improving=True`), and grouped by kind:

```python
kinds = {
    "lives": [(n, f"reliability.{p}")
              for n in ("pump1", "pump2", "valve") for p in ("alpha", "beta")],
    "repairs": [(n, "repairability.failure_rate")
                for n in ("pump1", "pump2", "valve")],
    "maintenance": [("valve", "preventive.interval"),
                    ("valve", "preventive.duration.failure_rate")],
}
gain = station.differential_importance(
    over="parameters", change="proportional", improving=True, groups=kinds
)
gain["lives"]         # -> 0.4003
gain["repairs"]       # -> 0.2536
gain["maintenance"]   # -> 0.346
```

Among the components the valve holds 95% of the change, but the levers
spread the gain: 40% of it lies in the lives, a quarter in the repairs, and
a third in the valve's maintenance, its two levers alone.

### Gamma: complements or substitutes

The joint importance `∂²A/∂A_i ∂A_j` says whether improving two together
is worth more than the sum of improving each:

```python
joint = station.joint_importance()
joint[("pump1", "pump2")]   # -> -0.9954
joint[("pump1", "valve")]   # -> 0.0274
```

The pumps are substitutes (negative): either does the other's job, so
improving both buys less than the sum of improving each. A pump and the
valve are complements (positive), but weakly: a better valve makes a
better pump worth a little more.

### Theta: what is moving the station now

The station's availability changes as its components age, fail and come
back. `availability_rate(x)` gives its rate of change, from new, split
among the components: `dA/dt = Σ I_B^i dA_i/dt`, each term what that
component is doing to the station then:

```python
rate = station.availability_rate([5.0, 81.0])
rate.rate[0]                 # -> -9.9e-05
rate.node_rate["valve"][0]   # -> -8.7e-05
rate.node_rate["pump1"][0]   # -> -6e-06
```

At five days the availability is falling by about 1e-4 a day, nine-tenths
of it the valve's, wearing in series. At 80 days every valve still in its
first life is replaced, together, and the availability jumps; the jumps
are reported apart, split among the components that make them:

```python
rate.jump_times[0]           # -> 80.0
rate.node_jumps["valve"][0]  # -> -0.7017
rate.node_rate["valve"][1]   # -> 0.0518
```

Just after 80 days the station is down for the replacement with
probability 0.70, the chance that its valve reached 80 days without
failing: of a fleet of stations commissioned together, seven in ten would
be down for it at once, a reason to stagger their first replacements. A
day later, the replacements still under way are ending at 5% a day.

### Theta, integrated: who caused the failures

Integrated, the parts give the Barlow–Proschan importance: each
component's share of the station's failures, the ones in which its failure
found the station up and left it down:

```python
station.barlow_proschan_importance()   # {'pump1': 0.1305, 'pump2': 0.1305, 'valve': 0.7389}
station.barlow_proschan_importance(window=100.0)["valve"]   # -> 0.7567
```

The valve causes three of the station's failures in four, in the long run
and over its first 100 days alike; a pump causes one only when the other
is down already. A planned outage is no failure: the replacements at 80
days count in neither.

### Vega: whose uncertainty widens the answer

The models above are known exactly. Fitted to failure data, they are
estimates, and the answer is uncertain; vega says whose uncertainty makes
it so, and so where more data would narrow it most. Give the station's
pumps a life fitted to 12 failures, and its valve one fitted to 30:

```python
import numpy as np

pump_fit = surv.Weibull.fit(W([40, 2.0]).qf(np.linspace(0.04, 0.96, 12)))
valve_fit = surv.Weibull.fit(W([160, 1.5]).qf(np.linspace(0.02, 0.98, 30)))
fitted = RepairableRBD(
    [("s", "pump1"), ("s", "pump2"),
     ("pump1", "valve"), ("pump2", "valve"),
     ("valve", "t")],
    {
        "pump1": unit(pump_fit, E([1.0])),
        "pump2": unit(pump_fit, E([1.0])),
        "valve": unit(
            valve_fit,
            E([2.0]),
            preventive={"policy": "age", "interval": 80.0, "duration": E([4.0])},
        ),
    },
)
spread = fitted.mean_availability_uncertainty(n_draws=400, seed=0)
spread.interval(0.9)   # (0.9937, 0.9954)
vega = fitted.uncertainty_importance()
vega.first_order["valve"]              # -> 0.81
vega.first_order[("pump1", "pump2")]   # -> 0.19
```

Each draw gives the fits plausible parameters, from their covariance, and
works the station's availability out as the station's own is worked out.
The two pumps share one fit, so they are one input, keyed by the pair.
Four-fifths of the variance of the long-run availability is the valve's
fit: the valve is in series, and its life sets how often the station
stops, where a pump's failure stops it only while the other is down. More
valve failures, not pump failures, would narrow the interval.

The question decides whose data to collect. Unattended, never repaired, the
station is a `NonRepairableRBD` on the same diagram, and its reliability
from new is the pumps' to lose:

```python
from repyability import NonRepairableRBD

unattended = NonRepairableRBD(
    [("s", "pump1"), ("s", "pump2"),
     ("pump1", "valve"), ("pump2", "valve"),
     ("valve", "t")],
    {"pump1": pump_fit, "pump2": pump_fit, "valve": valve_fit},
)
unattended.uncertainty_importance(20.0).first_order[("pump1", "pump2")]   # -> 0.79
unattended.uncertainty_importance(of="mean").first_order[("pump1", "pump2")]   # -> 0.95
```

Unrepaired, a pump's failure is for good, and the pair's wear sets the
station's life: most of the variance of the reliability at 20 days, and
nearly all of the MTTF's, is the pumps' fit. By default each input's share
is the delta method's, which adds up to 1; `method="sobol"` estimates Sobol
indices from draws instead, which take the quantity's nonlinearity in, and
`sampling="sobol"` takes the draws from a scrambled Sobol sequence, which
settles with fewer of them than random draws do.

### Rho: to come

Rho, the sensitivity of a plan's value to the discount rate, waits for the
plan's net present value. `total_cost` already discounts the costs of
ownership (`discount_rate=`, see [Costs](costs.md#discounting)).

## Reading them together

- **Delta and DIM** say the valve is where the station's availability is
  won or lost: 95% of an equal improvement in every component's
  availability.
- **The levers** say how: of the gain from improving every lever by the
  same fraction, a third lies in the valve's maintenance alone, more than
  in all the repairs together.
- **Gamma** says the pumps are substitutes, so improving both pays less
  than the sum, and a pump and the valve need no common campaign.
- **Theta** says what moves the availability from new: the valve's wear,
  and its first replacement, due at 80 days for every valve that has not
  failed by then.
- **Barlow–Proschan** says the valve causes three failures in four.
- **Vega** says the valve's failure data, not the pumps', is what to
  collect more of, for the station as it is run; left unattended, the
  pumps' would be.

## Which class has which

| Measure | `NonRepairableRBD` | `RepairableRBD` | `FaultTree` |
|---|---|---|---|
| `birnbaum_importance` | at `x` | long run, `x`, `window`, `state` | at `t` |
| `parameter_sensitivity` | at `x` | long run, `x`, `window`, `state`; `of="cost_rate"` | |
| `differential_importance` | at `x` | long run, `x`, `window`, `state` | at `t` |
| `joint_importance` | at `x` | long run, `x`, `window`, `state` | at `t` |
| Theta | `reliability_rate(x)` | `availability_rate(x)`, from new or `state` | |
| `barlow_proschan_importance` | whole life, or by `x` | long run, `window`, `state` | |
| `uncertainty_importance` | at `x`; `of=` the MTTF, a B-life, a time to a reliability | long run; `of=` the availability at `x` or over missions `x`, from new or `state`, or the cost rate | |

With limited repair crews or common-cause groups, the measures follow the
crews' and the groups' Markov chains: theta and the Barlow–Proschan shares
split the chains' transitions by the component, or common cause, that makes
each (#199). With common-cause groups `joint_importance` is refused, as
their members cannot be held.
