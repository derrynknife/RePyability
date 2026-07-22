# Tutorial: from a system to a decision

This walkthrough takes a small but realistic system and answers the questions a
reliability engineer actually asks of it — *how reliable is it, when should we
service it, what is the weak link, what does its remaining life look like once it
is in the field, and where is its redundancy weaker than it looks.* Every
capability in RePyability shows up here in the order you would reach for it.

We will model a **pumping skid**:

```text
                ┌── pump1 ──┐            ┌── filterA ──┐
   inlet ───────┤           ├── ctrl ───┤             ├─────── outlet
                └── pump2 ──┘            └── filterB ──┘
```

Two redundant pumps feed a single controller, which feeds two redundant
filters. The pumps and filters are each a parallel pair (either one carries the
duty), but the controller stands alone — a **single point of failure**. Keep an
eye on it; the analysis will keep pointing back to it.

## 1. Get the component models

RePyability consumes *already-fitted* lifetime models. Fitting failure data to a
distribution is [surpyval](https://github.com/derrynknife/SurPyval)'s job — you
would normally do:

```python
import surpyval as surv

# From observed failure/censoring data (illustrative):
pump_model = surv.Weibull.fit(failures=pump_hours, censored=still_running)
```

Here we will just assume the fits have been done and write the parameters
directly. Times are in operating hours.

```python
import surpyval as surv

reliabilities = {
    "pump1":   surv.Weibull.from_params([12000, 1.8]),
    "pump2":   surv.Weibull.from_params([12000, 1.8]),
    "ctrl":    surv.Weibull.from_params([40000, 1.2]),
    "filterA": surv.Weibull.from_params([9000, 2.5]),
    "filterB": surv.Weibull.from_params([9000, 2.5]),
}
```

The controller has a long characteristic life (`α = 40000`) but the filters wear
in fast once they are near their life (`β = 2.5`).

## 2. Build the RBD

An RBD is its **edges** (a directed graph from one input to one output) plus a
model per intermediate node. The input/output nodes are inferred as the unique
source and sink.

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
computed **exactly** (not by simulation) via a memoised decomposition over the
diagram, so these calls are cheap and repeatable.

Mean time to failure is a simulated quantity — seed it for reproducibility:

```python
rbd.mean_time_to_failure(seed=0)   # ~ 8250 h (Monte-Carlo)
```

## 4. When should we service it?

Invert the reliability to turn a target into a time:

```python
rbd.time_to_reliability(0.95)   # -> 2884 h  until reliability falls to 0.95
rbd.bx_life(10)                 # -> 4188 h  the B10 life (10% failed)
```

If you want the 95%-reliability interval as a maintenance trigger, service the
skid by ~2900 h.

## 5. What is the weak link?

Two questions, two tools.

**At design time, before you trust any data**, use structural importance — it
sets every component reliability to ½ and measures how often each node is
pivotal, so it depends only on the diagram:

```python
rbd.structural_importance()
# -> {'ctrl': 0.562, 'pump1': 0.188, 'pump2': 0.188,
#     'filterA': 0.188, 'filterB': 0.188}
```

The controller scores **3× its neighbours** purely because it is unredundant.

**With the models in hand**, use Birnbaum importance — how much a small change in
each node's reliability moves the system:

```python
rbd.birnbaum_importance(4000)
# -> {'ctrl': 0.968, 'pump1': 0.120, 'pump2': 0.120,
#     'filterA': 0.114, 'filterB': 0.114}
```

Same verdict, sharper: at 4000 h the system's reliability is almost entirely
hostage to the controller. If you are going to add redundancy anywhere, add it
there.

## 6. Which measurement should we invest in?

Suppose you can afford to pin down *one* fitted parameter with more data. Which
one most changes the answer? Parameter sensitivity chains the Birnbaum
importance with each parameter's effect on its node:

```python
rbd.parameter_sensitivity(4000)["ctrl"]
# -> {'alpha': 2e-06, 'beta': 0.132}
```

The controller's **shape** `β` dwarfs its scale `α`: system reliability barely
moves with the controller's characteristic life but is sensitive to *how* its
hazard grows. Spend the test budget resolving the shape.

## 7. Going live: a condition-based "digital twin"

Everything above assumed brand-new components. In service, each unit has run for
a while and telemetry tells you how long. Feed that in and every answer
re-computes for *this* skid, not the fleet average.

Say pump1 is well-worn, pump2 is nearly new, and the controller has a lot of
hours on it:

```python
from repyability import NodeState

state = {
    "pump1": NodeState(age=9000),
    "pump2": NodeState(age=1000),
    "ctrl":  NodeState(age=30000),
}

rbd.sf_given_state(2000, state)   # -> 0.9293  reliability of the next 2000 h
rbd.sf(2000)                      # -> 0.9709  what a new skid would give
```

The remaining useful life to a 0.9 reliability target has dropped by more than a
third against a fresh skid:

```python
rbd.remaining_life(0.9, state)    # -> 2603 h
rbd.remaining_life(0.9, {})       # -> 4188 h  (fresh)
```

And the *live* importance ranking has shifted — with pump1 worn, the system now
leans on the healthy pump2:

```python
rbd.importances_given_state(2000, state)["birnbaum"]
# -> {'ctrl': 0.984, 'pump2': 0.216, 'pump1': 0.065, ...}
```

Stream a new `state` each time fresh readings arrive; every update is a cheap,
exact re-evaluation. (Only lifetime distributions age; standby/composite nodes
are out of scope for the condition-based methods in this release.)

## 8. When redundancy is optimistic: dependent failures

Steps 1–7 assumed the five components fail **independently** and each runs at a
**fixed** load. Both assumptions flatter a redundant system. RePyability lets you
relax each one where it matters — and because the system quantity stays exact,
you see precisely what the optimism was worth. The theory behind all three is in
[Concepts](concepts.md).

### The pumps share the load

The parallel-pump model assumed the surviving pump is unaffected when its sibling
drops. In reality it then carries the *whole* duty, runs harder, and ages faster.
Refit the pump with **load as a covariate** and model the pair as a load-sharing
group — each pump carries half of a total load of `2.0` while both run, and the
group needs at least one:

```python
import surpyval as surv
from repyability import LoadSharingModel

pump = surv.ExponentialAFT.fit(pump_hours, Z=pump_loads)   # fitted in surpyval
pumps = LoadSharingModel([pump, pump], load=2.0, k=1)       # 2 units, need >= 1

pumps.sf(4000)     # -> 0.868   the pair, sharing the load
```

Modelled *independently* — the same two pumps each pinned at the half-load — the
redundant pair would read `0.920`. The load transfer has quietly eaten a third of
the pair's redundancy margin, and more as it wears: `0.641` vs `0.764` at 8000 h.
Identical Exponential-baseline units like these get the **exact** hypoexponential
group lifetime, so `pumps.is_simulated` is `False` — no simulation involved.

### A common cause on the filters

The two filters are the same part from the same shelf, so one bad batch — or one
contamination event upstream — can blind both at once. That is a **common-cause
failure**, and no amount of *structural* redundancy defends against it. Attach a
beta-factor group (here 8% of a filter's failures are shared) to the otherwise
unchanged skid:

```python
from repyability import CCFGroup, BetaFactor

rbd_ccf = NonRepairableRBD(
    edges, reliabilities,
    ccf_groups=[CCFGroup(["filterA", "filterB"], BetaFactor(0.08))],
)

rbd_ccf.sf(4000)   # -> 0.9023   vs 0.9091 with independent filters
```

Modest at the skid level *here*, because the filters are not the weak link — but
the coupling is exact, and on a system that leans on its redundant pair it is the
difference between a design that meets its target and one that only appears to.
`BetaFactor(0)` recovers the independent number; `MGL(β, γ, …)` handles a cause
that fails *some* but not all of a larger group.

### A duty that ramps up

Finally, drop the fixed-load assumption. Say the skid is commissioned gently and
then, at 3000 h, pushed to a harsher continuous duty. Fit a regression model with
the duty as a covariate and give the node a **schedule** instead of a single
covariate vector:

```python
from surpyval import StepSchedule
from repyability import RegressionNode

duty_unit = surv.WeibullAFT.fit(run_hours, Z=duty_history)   # fitted in surpyval
# benign until 3000 h, then a harsher duty for the rest of life
duty = StepSchedule.from_changepoints([0, 3000], [[0.0], [1.0]])
node = RegressionNode(duty_unit, schedule=duty)

node.sf(4000)      # -> 0.822   just after the step up
node.sf(6000)      # -> 0.603   the harsher duty has now done real damage
```

Held at the benign duty the same unit would read `0.861` and `0.736`. Reliability
is the exact survival *along* the duty path, and conditioning on the component's
age (`NodeState(age=...)`) gives its go-forward reliability from wherever it sits
on that path — the digital twin of step 7, now with a load history.

Each of these is an ordinary node: it serialises with the RBD and takes part in
the same exact system computation as everything above.

## 9. If the skid is repairable: availability

If a failed component is repaired rather than replaced, give each one a
*repairability* (time-to-repair) distribution and ask about availability instead
of reliability.

```python
import surpyval as surv
from repyability import RepairableRBD

components = {
    "pump1": {
        "reliability":   surv.Weibull.from_params([12000, 1.8]),
        "repairability": surv.Exponential.from_params([1 / 48]),  # ~48 h MTTR
    },
    # ... same for the other components ...
}
rep = RepairableRBD(edges, components)

rep.mean_availability()                       # closed-form long-run availability
result = rep.availability(t_simulation=20000, N=10_000, seed=0)
result.availability                           # availability at each event time
```

Long-run availability is exact and needs no simulation; the time-resolved
availability curve and its criticality measures come from a seeded
discrete-event simulation.

## 10. Persist the model

The structure and its fitted models round-trip to plain JSON, so you can build
the RBD once and reload it wherever the analysis runs (a dashboard, a scheduled
job, a notebook):

```python
rbd.to_json()                       # -> a JSON string
NonRepairableRBD.from_json(saved)   # reconstruct it
```

Node state is *not* serialised — the structure is the durable asset; state is
transient input you supply at evaluation time.

## Where to next

- **[Concepts](concepts.md)** — the theory behind these numbers: path/cut sets,
  the full importance-measure family and when to use each, how conditioning
  works, and the dependent-failure models from step 8 (load sharing, common
  cause, and covariate/time-varying reliability).
- **[User guide](guide.md)** — the reference for every method, argument, and
  return contract.
- **[API reference](api.md)** — the generated signatures and docstrings.
