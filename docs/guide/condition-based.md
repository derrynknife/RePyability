# Condition-based evaluation

The methods on [Reliability of a system](reliability.md) treat every
component as new. In service, each component has already run for a while,
and telemetry says how long. Condition-based evaluation conditions every
component on **its own** current age and propagates that through the diagram
exactly: a "digital twin" of one particular system rather than the fleet
average. The theory is in
[Concepts](../concepts.md#condition-based-evaluation).

## Describing the current state

Each component's state is a [`NodeState`][repyability.NodeState]:

- `age` (default 0): the operating time it has accumulated, in the models'
  time unit;
- `alive` (default `True`): whether it is still working. A failed component
  contributes zero reliability, whatever its age.

Pass a dict of node → `NodeState`; a node left out is treated as new.

```python
import surpyval as surv
from repyability import NodeState, NonRepairableRBD

rbd = NonRepairableRBD(
    [("s", "pump1"), ("s", "pump2"),
     ("pump1", "valve"), ("pump2", "valve"),
     ("valve", "t")],
    {
        "pump1": surv.Weibull.from_params([100, 2]),
        "pump2": surv.Weibull.from_params([100, 2]),
        "valve": surv.Weibull.from_params([200, 1.5]),
    },
)
state = {
    "pump1": NodeState(age=80),    # well worn
    "pump2": NodeState(age=10),    # nearly new
    "valve": NodeState(age=40),
}
```

## Reliability from now

`sf_given_state(x, state)` is the probability the system survives a further
`x` from now. Each component conditions on its age,
`R_i(x | age_i) = R_i(age_i + x) / R_i(age_i)`, and the conditioned
reliabilities go through the same exact system computation as `sf`.

```python
rbd.sf_given_state(20, state)   # -> 0.9063   the next 20 time units
rbd.sf(20)                      # -> 0.9674   what a new system would give
rbd.sf_given_state(20, {})      # -> 0.9674   an empty state is exactly sf
rbd.sf_given_state([0, 20, 40], state)   # array([1.    , 0.9063, 0.7494])
```

At `x = 0` it is 1: whatever is alive now has not failed yet. A component
known to have failed takes zero:

```python
rbd.sf_given_state(20, {"pump1": NodeState(alive=False), "valve": NodeState(age=40)})
# -> 0.8915   pump2 now carries the duty alone
```

## Remaining useful life

`remaining_life(target, state)` is the further time until the system
reliability falls to `target`: the condition-based counterpart of
`time_to_reliability`. `remaining_life(1 − x/100, state)` is the conditional
B*x* life.

```python
rbd.remaining_life(0.9, state)   # -> 20.99
rbd.remaining_life(0.9, {})      # -> 38.84   a new system (= time_to_reliability(0.9))
```

The wear on pump1 and the valve has cut the time to 90% reliability almost in
half. `upper_bound=` limits the search, as for `time_to_reliability`.

## Importance from now

`importances_given_state(x, state)` evaluates the Birnbaum and criticality
importances at the conditioned reliabilities, so the ranking reflects today's
wear:

```python
live = rbd.importances_given_state(20, state)
live["birnbaum"]      # {'pump1': 0.0713, 'pump2': 0.2805, 'valve': 0.9768}
live["criticality"]   # {'pump1': 0.0549, 'pump2': 0.2857, 'valve': 1.0}
live["birnbaum"]["pump2"]   # -> 0.2805
rbd.birnbaum_importance(20)["pump2"]   # -> 0.038   as new, the pumps are equal
```

With pump1 worn, the system now leans on pump2.

## What can take a state

- Parametric and non-parametric distributions and `RegressionNode`s age.
  For them, `NodeState(age=0)` is the same as leaving the node out.
- A fixed-probability component given any state in which it is alive
  contributes 1 from now on: its chance of failing has been resolved by
  observing it working. Left out of the state, it keeps its usual
  reliability `1 − q`. So for such a component a `NodeState()` is *not* the
  same as leaving it out.
- Standby, repeated-standby, load-sharing and repeated nodes and nested RBDs
  raise `ValueError` if given a state.
- The methods raise `NotImplementedError` on an RBD with common-cause groups,
  and `TypeError`/`ValueError` for a malformed state (not a dict, a value
  that is not a `NodeState`, an unknown node, the input or output node, or a
  negative age).

The structure is the durable asset; the state is transient input supplied at
each evaluation, and it is not saved with the RBD.

## Covariate-dependent components

A component's life often depends on how it is run: temperature, load, speed,
duty. With a **regression** model fitted in surpyval (accelerated failure
time, proportional hazards, proportional odds, …), a
[`RegressionNode`][repyability.RegressionNode] makes it an ordinary RBD node.
Fitting stays in surpyval; here we fit synthetic data to have a model to use:

```python
import numpy as np
from repyability import RegressionNode

# Field data: hours to failure and the temperature (°C) each unit ran at.
rng = np.random.default_rng(1)
temperature = rng.uniform(20, 80, size=300)
hours = rng.weibull(2.0, size=300) * 5000 * np.exp(-0.02 * (temperature - 50))
model = surv.WeibullAFT.fit(hours, Z=temperature.reshape(-1, 1))
```

### Fixed operating conditions

Pin the covariates this component runs at with `covariates=`. Its
reliability is the model's survival there, `R(t) = model.sf(t, Z)`:

```python
cool = RegressionNode(model, covariates=[30])
hot = RegressionNode(model, covariates=[70])
cool.sf(3000)   # array([0.8474])
hot.sf(3000)    # array([0.4363])
hot.sf(3000)[0]   # -> 0.4363
```

A node model returns an array even for a scalar time; the RBD methods convert
to floats. The node takes part in everything an ordinary distribution does:
system reliability, importance, MTTF and the condition-based methods above.

```python
motor = NonRepairableRBD([("s", "m"), ("m", "t")], {"m": hot})
motor.sf(3000)                                         # -> 0.4363
motor.sf_given_state(1000, {"m": NodeState(age=2000)}) # -> 0.628
motor.remaining_life(0.9, {"m": NodeState(age=2000)})  # -> 266.7
```

Because the covariates live on the node, `age` keeps its one meaning
(operating time survived) for every node type.

### A load schedule

When the conditions change over the component's life, pass a surpyval
`StepSchedule` (a piecewise-constant covariate path) as `schedule=` instead.
The reliability is then the exact survival along that path,
`model.sf_tvc(t, schedule)`:

```python
from surpyval import StepSchedule

# 30 °C for the first 2000 hours, then 70 °C for the rest of its life
ramp = RegressionNode(
    model, schedule=StepSchedule.from_changepoints([0, 2000], [[30.0], [70.0]])
)
ramp.sf(1500)[0]   # -> 0.9603   still on the cool segment: same as `cool`
ramp.sf(4000)[0]   # -> 0.4599   between cool (0.743) and hot (0.226)
```

Conditioning on age works along the path: a unit 2500 hours in has spent
2000 of them cool and 500 hot, and its next 1000 hours are all hot.

```python
ramped = NonRepairableRBD([("s", "m"), ("m", "t")], {"m": ramp})
ramped.sf_given_state(1000, {"m": NodeState(age=2500)})   # -> 0.7036
```

- Give exactly one of `covariates` and `schedule`; the node checks the model
  at construction and raises `ValueError` if it cannot evaluate it (for
  example, a covariate vector of the wrong width).
- Schedules work for accelerated-failure-time and proportional- or
  additive-hazards models, not proportional-odds, and need a surpyval that
  provides `sf_tvc`.
- `mean()` and `random()` (used for MTTF) need a proper parametric lifetime:
  they work for AFT, proportional-odds and parametric models; a
  semi-parametric Cox model has no defined mean and says so.
- The node saves with the RBD; the fitted model round-trips through
  `surpyval.from_dict`.
