# Tutorial: from a system to a decision

This walkthrough takes a small but realistic system and answers the questions
a reliability engineer actually asks of it: *how reliable is it, when should
we service it, what is the weak link, what would more redundancy buy, what
does its remaining life look like once it is in the field, where is its
redundancy weaker than it looks, and what will it cost to run.* It visits the
main capabilities in the order you would reach for them; the
[user guide](guide/index.md) covers each one in full. Every block below runs
as written, in order.

We will model a **pumping skid**:

```text
                ┌── pump1 ──┐            ┌── filterA ──┐
   inlet ───────┤           ├── ctrl ───┤             ├─────── outlet
                └── pump2 ──┘            └── filterB ──┘
```

Two redundant pumps feed a single controller, which feeds two redundant
filters. The pumps and filters are each a parallel pair (either one carries
the duty), but the controller stands alone: a **single point of failure**.
Keep an eye on it; the analysis will keep pointing back to it.

## 1. Get the component models

RePyability consumes *already-fitted* lifetime models. Fitting failure data
to a distribution is [surpyval](https://github.com/derrynknife/SurPyval)'s
job. With field data for the pumps (hours to failure, and hours so far for
pumps still running), you would fit a Weibull like this:

```python
import numpy as np
import surpyval as surv

# Illustrative field data for 40 pumps, 19 of them still running.
rng = np.random.default_rng(7)
lives = surv.Weibull.from_params([12000, 1.8]).qf(rng.uniform(size=40))
observed_to = rng.uniform(4000, 16000, size=40)
pump_hours = np.minimum(lives, observed_to)
still_running = (lives > observed_to).astype(int)   # 1 = censored

pump_model = surv.Weibull.fit(x=pump_hours, c=still_running)
pump_model.params   # array([1.2125e+04, 1.44e+00]): alpha, beta
```

Here we will assume the fits have been done and write the parameters
directly. Times are in operating hours.

```python
reliabilities = {
    "pump1":   surv.Weibull.from_params([12000, 1.8]),
    "pump2":   surv.Weibull.from_params([12000, 1.8]),
    "ctrl":    surv.Weibull.from_params([40000, 1.2]),
    "filterA": surv.Weibull.from_params([9000, 2.5]),
    "filterB": surv.Weibull.from_params([9000, 2.5]),
}
```

The controller has a long characteristic life (`α = 40000`), but the filters
wear in fast once they are near their life (`β = 2.5`).

## 2. Build the RBD

An RBD is its **edges** (a directed graph from one input to one output) plus a
model per intermediate node. The input/output nodes are inferred as the
unique source and sink.

```python
from repyability import NonRepairableRBD

edges = [
    ("inlet", "pump1"), ("inlet", "pump2"),
    ("pump1", "ctrl"),  ("pump2", "ctrl"),
    ("ctrl", "filterA"), ("ctrl", "filterB"),
    ("filterA", "outlet"), ("filterB", "outlet"),
]
rbd = NonRepairableRBD(edges, reliabilities)
```

## 3. How reliable is it?

```python
rbd.sf(4000)      # -> 0.9091   reliability at 4000 h
rbd.ff(4000)      # -> 0.0909   the complement, unreliability
rbd.sf([2000, 4000, 8000])      # an array in, an array out
```

A scalar time returns a float; an array returns a numpy array. Reliability is
computed **exactly** (not by simulation) from the diagram, so these calls are
cheap and repeatable.

Mean time to failure is a simulated quantity; seed it for reproducibility:

```python
rbd.mean_time_to_failure(seed=0)   # -> 8250.3   h (Monte-Carlo)
```

## 4. When should we service it?

Invert the reliability to turn a target into a time:

```python
rbd.time_to_reliability(0.95)   # -> 2884.2   h until reliability falls to 0.95
rbd.bx_life(10)                 # -> 4187.6   h, the B10 life (10% failed)
```

If you want the 95%-reliability point as a maintenance trigger, service the
skid by about 2900 h.

## 5. What is the weak link?

Two questions, two tools.

**At design time, before you trust any data**, use structural importance. It
sets every component reliability to ½ and measures how often each node is
pivotal, so it depends only on the diagram:

```python
rbd.structural_importance()
# {'pump1': 0.188, 'pump2': 0.188, 'ctrl': 0.562, 'filterA': 0.188, 'filterB': 0.188}
rbd.structural_importance()["ctrl"]   # -> 0.5625
```

The controller scores **3× its neighbours** purely because it is unredundant.

**With the models in hand**, use Birnbaum importance: how much a small change
in each node's reliability moves the system:

```python
rbd.birnbaum_importance(4000)
# {'pump1': 0.12, 'pump2': 0.12, 'ctrl': 0.968, 'filterA': 0.114, 'filterB': 0.114}
rbd.birnbaum_importance(4000)["ctrl"]   # -> 0.968
```

Same verdict, sharper: at 4000 h the system's reliability is almost entirely
hostage to the controller.

## 6. What would more redundancy buy?

Say a second controller costs 15 000, a pump 8 000 and a filter 1 200. What
is the most reliable skid for a budget of 50 000 (the original units
included)?

```python
costs = {"pump1": 8000, "pump2": 8000, "ctrl": 15000, "filterA": 1200, "filterB": 1200}
best = rbd.allocate_redundancy(costs, budget=50_000, t=4000)
best.units          # {'pump1': 1, 'pump2': 1, 'ctrl': 2, 'filterA': 1, 'filterB': 2}
best.reliability    # -> 0.9778   against 0.9091 as built
best.cost           # -> 49600.0
```

The money goes on a second controller first, then a third filter. The
allocation is exact: it is the best design within the budget, not a
heuristic. See [Design and allocation](guide/design.md) for targets, caps
and costs other than money.

## 7. Which measurement should we invest in?

Suppose you can afford to pin down *one* fitted parameter with more data.
Which one most changes the answer? Parameter sensitivity chains the Birnbaum
importance with each parameter's effect on its node:

```python
rbd.parameter_sensitivity(4000)["ctrl"]
# {'alpha': 1.72e-06, 'beta': 0.132}
rbd.parameter_sensitivity(4000)["ctrl"]["beta"]   # -> 0.132
```

The controller's **shape** `β` dwarfs its scale `α` (per unit of each):
system reliability barely moves with the controller's characteristic life but
is sensitive to *how* its hazard grows. Spend the test budget resolving the
shape.

## 8. Going live: a condition-based "digital twin"

Everything above assumed brand-new components. In service, each unit has run
for a while and telemetry tells you how long. Feed that in and every answer
re-computes for *this* skid, not the fleet average.

Say pump1 is well worn, pump2 is nearly new, and the controller has a lot of
hours on it:

```python
from repyability import NodeState

state = {
    "pump1": NodeState(age=9000),
    "pump2": NodeState(age=1000),
    "ctrl":  NodeState(age=30000),
}

rbd.sf_given_state(2000, state)   # -> 0.9293   reliability over the next 2000 h
rbd.sf(2000)                      # -> 0.9709   what a new skid would give
```

The remaining useful life to a 0.9 reliability target has dropped by more
than a third against a fresh skid:

```python
rbd.remaining_life(0.9, state)    # -> 2603.0   h
rbd.remaining_life(0.9, {})       # -> 4187.6   h (fresh)
```

And the *live* importance ranking has shifted. With pump1 worn, the system
now leans on the healthy pump2:

```python
rbd.importances_given_state(2000, state)["birnbaum"]
# {'pump1': 0.065, 'pump2': 0.216, 'ctrl': 0.984, 'filterA': 0.021, 'filterB': 0.021}
rbd.importances_given_state(2000, state)["birnbaum"]["pump2"]   # -> 0.216
```

Stream a new `state` each time fresh readings arrive; every update is a
cheap, exact re-evaluation. (Only lifetime distributions age; standby and
composite nodes cannot take a state.)

## 9. When redundancy is optimistic: dependent failures

Steps 1–8 assumed the five components fail **independently** and each runs at
a **fixed** load. Both assumptions flatter a redundant system. RePyability
lets you relax each one where it matters, and because the system quantity
stays exact you see precisely what the optimism was worth. The theory behind
all three is in [Concepts](concepts.md).

### The pumps share the load

The parallel-pump model assumed the surviving pump is unaffected when its
sibling drops. In reality it then carries the *whole* duty, runs harder, and
ages faster. Refit the pump with **load as a covariate** and model the pair as
a load-sharing group: each pump carries half of a total load of `2.0` while
both run, and the group needs at least one:

```python
from repyability import LoadSharingModel, RegressionNode

# Illustrative field data: pump lives at various (normalised) loads.
rng = np.random.default_rng(3)
pump_loads = rng.uniform(0.5, 2.0, size=500)
life = rng.exponential(scale=12000 * np.exp(-0.8 * (pump_loads - 1.0)), size=500)
pump = surv.ExponentialAFT.fit(life, Z=pump_loads.reshape(-1, 1))   # fitted in surpyval

pumps = LoadSharingModel([pump, pump], load=2.0, k=1)   # 2 units, need >= 1
pumps.sf(4000)       # -> 0.8343   the pair, sharing the load
pumps.is_simulated   # False: an exact (hypoexponential) group lifetime
```

Modelled *independently*, the same two pumps each pinned at the half-load
would read higher:

```python
p = RegressionNode(pump, covariates=[1.0]).sf(4000)[0]
1 - (1 - p) ** 2   # -> 0.9119
```

A single pump at the half-load reads `0.703`, so the load transfer has eaten
more than a third of the pair's redundancy margin (`0.834 − 0.703` against
`0.912 − 0.703`), and more as it wears: `pumps.sf(8000)` is `0.573` against
`0.744`.

### A common cause on the filters

The two filters are the same part from the same shelf, so one bad batch, or
one contamination event upstream, can blind both at once. That is a
**common-cause failure**, and no amount of *structural* redundancy defends
against it. Attach a beta-factor group (here 8% of a filter's failures are
shared) to the otherwise unchanged skid:

```python
from repyability import BetaFactor, CCFGroup

rbd_ccf = NonRepairableRBD(
    edges, reliabilities,
    ccf_groups=[CCFGroup(["filterA", "filterB"], BetaFactor(0.08))],
)
rbd_ccf.sf(4000)   # -> 0.9023   against 0.9091 with independent filters
```

Modest at the skid level *here*, because the filters are not the weak link,
but the coupling is exact, and on a system that leans on its redundant pair
it is the difference between a design that meets its target and one that
only appears to. `BetaFactor(0)` recovers the independent number;
`MGL(β, γ, …)` handles a cause that fails *some* but not all of a larger
group.

### A duty that ramps up

Finally, drop the fixed-load assumption. Say the skid is commissioned gently
and then, at 3000 h, pushed to a harsher continuous duty. Fit a regression
model with the duty as a covariate and give the node a **schedule** instead of
a single covariate vector:

```python
from surpyval import StepSchedule

# Illustrative field data: lives of units run at various duties (0 to 1).
rng = np.random.default_rng(5)
duty_history = rng.uniform(0, 1, size=400)
run_hours = rng.weibull(2.0, size=400) * 9000 * np.exp(-0.6 * duty_history)
duty_unit = surv.WeibullAFT.fit(run_hours, Z=duty_history.reshape(-1, 1))

# benign until 3000 h, then a harsher duty for the rest of life
duty = StepSchedule.from_changepoints([0, 3000], [[0.0], [1.0]])
node = RegressionNode(duty_unit, schedule=duty)
node.sf(4000)[0]   # -> 0.7024   just after the step up
node.sf(6000)[0]   # -> 0.3799   the harsher duty has now done real damage
```

Held at the benign duty the same unit would read `0.762` and `0.554`.
Reliability is the exact survival *along* the duty path, and conditioning on
the component's age (`NodeState(age=...)`) gives its go-forward reliability
from wherever it sits on that path: the digital twin of step 8, now with a
load history.

Each of these is an ordinary node: it saves with the RBD and takes part in
the same exact system computation as everything above.

## 10. If the skid is repairable: availability and cost

If a failed component is repaired rather than replaced, give each one a
*repairability* (time-to-repair) distribution and ask about availability
instead of reliability. Price the repairs and the lost production to get the
cost of running it:

```python
from repyability import RepairableRBD

repair = surv.Exponential.from_params([1 / 48])   # mean 48 h to repair
components = {
    name: {"reliability": model, "repairability": repair}
    for name, model in reliabilities.items()
}
components["ctrl"]["repair_cost"] = 5000.0
components["filterA"]["replace_cost"] = 600.0
components["filterB"]["replace_cost"] = 600.0

rep = RepairableRBD(edges, components, downtime_cost_rate=2000.0)   # per hour down

rep.mean_availability()     # -> 0.99867   long-run, exact
rep.mean_up_time()          # -> 34601.7   h between outages, on average
rep.mean_down_time()        # -> 46.07     h per outage
rep.expected_cost_rate()    # -> 2.942     cost per operating hour, long run
```

Long-run quantities are exact and need no simulation. The time-resolved
availability, and which component causes the outages, come from a seeded
discrete-event simulation:

```python
result = rep.availability(t_simulation=20000, N=2000, seed=0)
result.availability[-1]   # -> 0.9975   availability at 20 000 h
result.criticalities.failure_criticality_index.per_system_failure["ctrl"]
# -> 0.918   the controller caused 92% of the outages
```

The controller again: a second one (step 6) is the obvious investment. See
[Repairable systems](guide/repairable.md) and [Costs](guide/costs.md) for the
rest of the repairable toolkit.

## 11. Save the model

The structure and its fitted models round-trip to plain JSON, so you can build
the RBD once and reload it wherever the analysis runs (a dashboard, a
scheduled job, a notebook):

```python
saved = rbd.to_json()                   # a JSON string
reloaded = NonRepairableRBD.from_json(saved)
reloaded.sf(4000)                       # -> 0.9091
```

Node state is *not* saved: the structure is the durable asset; state is
transient input you supply at evaluation time.

## Where to next

- **[User guide](guide/index.md)**: every method, argument and return
  contract, with runnable examples.
- **[Concepts](concepts.md)**: the theory behind these numbers: path and cut
  sets, the importance-measure family and when to use each, conditioning,
  availability and cost, and the dependent-failure models from step 9.
- **[API reference](api.md)**: the generated signatures and docstrings.
