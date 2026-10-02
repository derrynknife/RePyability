# Phased missions

A flight takes off, cruises and lands. Each phase needs different
equipment: both engines to take off, either one to cruise, both again to
land. The components are the same aircraft's throughout, so an engine lost
in the cruise, which the cruise tolerates, is still lost at landing. The
phases are not independent, and the mission's reliability is not the
product of theirs. `PhasedMission` works it out.

## A mission

Each phase is a name, a duration and a `NonRepairableRBD` over the mission's
components. A node name is the same component in every phase that uses it,
with the same model, and its life runs through the whole mission, whether
or not the current phase uses it:

```python
import surpyval as surv
from repyability import NonRepairableRBD, PhasedMission

engine = surv.Weibull.from_params([1500.0, 1.4])
gear = surv.Weibull.from_params([4000.0, 2.0])
nav = surv.Exponential.from_params([2e-4])
parts = {"e1": engine, "e2": engine, "gear": gear, "nav": nav}

def diagram(edges):
    used = {n for edge in edges for n in edge} - {"s", "t"}
    return NonRepairableRBD(edges, {n: parts[n] for n in used})

both_engines = diagram([("s", "e1"), ("e1", "e2"), ("e2", "gear"), ("gear", "t")])
either_engine = diagram([("s", "e1"), ("s", "e2"), ("e1", "nav"), ("e2", "nav"), ("nav", "t")])
flight = PhasedMission([
    ("take-off", 0.2, both_engines),
    ("cruise", 10.0, either_engine),
    ("landing", 0.3, both_engines),
])
flight.unreliability()    # -> 0.003963
failing = flight.phase_failure_probabilities()
failing["cruise"]         # -> 0.002039
failing["landing"]        # -> 0.001917   mostly an engine lost in the cruise
```

The mission succeeds if every phase's diagram works to the phase's end
(its components only fail, so working at the end is working throughout).
`phase_failure_probabilities` gives the chance of failing in each phase:
getting through the earlier ones and not this one. They add up to the
mission's `unreliability()`, `1 - reliability()` computed to its own
precision when it is small.

Treating the phases as independent, each with its own diagram's chance of
lasting from its start to its end, misses the engine lost in the cruise,
and halves the risk:

```python
alone = (
    float(both_engines.sf(0.2))
    * float(either_engine.sf(10.2)) / float(either_engine.sf(0.2))
    * float(both_engines.sf(10.5)) / float(both_engines.sf(10.2))
)
1 - alone                 # -> 0.002083
```

- **Components.** Any node model with `sf` and `ff`: surpyval distributions,
  fixed probabilities (which fail at the start, if at all, as a demand
  does), nested RBDs and the composite nodes. A repeated node is the
  component it repeats, as in any diagram. A phase can leave a component
  out; it still ages.
- **Phases.** In order, with distinct names; a phase of no duration needs
  its diagram working at that instant (a check before take-off, say).
- **Not modelled.** Repair between or during phases, a component ageing
  faster in one phase than another (it has one lifetime model), and
  common-cause groups, which a phase may not have.

## Exact and simulated

`reliability()`, `unreliability()` and `phase_failure_probabilities()` are
exact by default. Each component's life through the mission is a chain of
independent segments, one per phase, the `k`-th survived with probability
`R(T_k) / R(T_{k-1})` (Esary and Ziehms, 1975): a component works at the end
of phase `j` if it survives its first `j` segments. Each phase's structure
over the segments is a binary decision diagram, built from its diagram's
modules and core as the exact engine reduces it (after Zang, Sun and
Trivedi, 1999), with each component's segments decided together, in phase
order; the mission is the phases' diagrams combined by conjunction. No path
set is listed, so a meshed phase, such as a chain of bridges with tens of
thousands of path sets, is exact in a fraction of a second. A diagram of more
than a million nodes refuses the exact values, and says to simulate.
Setting `repyability.rbd.phased_mission.METHOD = "paths"` decomposes the
phases' minimal path sets over the segments instead, as before: slower
beyond the smallest missions.

`method="simulate"` draws each component's life once per mission instead:
`mc_samples` missions (by default 10,000), seeded with `seed`.
`reliability_interval` gives the simulated reliability with a confidence
interval, and takes the Monte-Carlo options of the other simulations:
`tolerance` (simulate in batches of `mc_samples` until the interval is that
narrow, up to `max_samples`), `confidence` and `antithetic` pairs.

```python
interval = flight.reliability_interval(1_000_000, seed=1)
interval.estimate    # -> 0.996079   simulated
flight.reliability() # -> 0.996037
```
