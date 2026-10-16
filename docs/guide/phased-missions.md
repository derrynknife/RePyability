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

## Importance and cut sets

Which component matters most over the whole mission, and which
combinations of failures end it? A component is the same through every
phase, so its importance holds it across all of them: *working throughout*
(every phase's segment of its life survived) or *failed from the start*
(failed in the first phase, so in every phase). `birnbaum_importance()` is
the mission's reliability with the component working throughout less its
reliability with it failed from the start; `improvement_potential()`,
`risk_achievement_worth()`, `risk_reduction_worth()` and
`criticality_importance()` are built on the same held values, as the
diagrams' are. Each is the mission's decision diagram replayed with the
component's segments so held, exact, at the mission's end; there is no
simulated version. Here any component failed from the start fails the
take-off or the cruise, so the risk achievement worth cannot tell them
apart, but the risk reduction worth can: the navigation unit, alone in
series through the cruise, accounts for half the mission's failures, each
engine for a quarter, the gear for almost none.

```python
raw = flight.risk_achievement_worth()
raw["nav"] == raw["gear"]       # -> True
rrw = flight.risk_reduction_worth()
rrw["nav"]                      # -> 2.0545
rrw["e1"]                       # -> 1.3191
rrw["gear"]                     # -> 1.0017
criticality = flight.criticality_importance()
criticality["nav"]              # -> 0.5133   the share of failures nav accounts for
```

`minimal_cut_sets()` gives the combinations: each a set of events
`(component, phase)`, "the component has failed by the end of that phase",
that together fail the mission. A failure by one phase is one by every
later phase, so a cut set keeps the weakest events that still fail the
mission: an engine lost by the end of the take-off is a cut set of the
take-off, but the landing needs both engines too, so the engine lost by the
end of the landing, whenever it was lost, is the cut set listed. The
phases' own cut sets, dated, less those another implies, are the mission's
(the usual phased-mission minimisation).

```python
cuts = flight.minimal_cut_sets()
sorted(sorted(cut) for cut in cuts)
# -> [[('e1', 'landing')], [('e2', 'landing')], [('gear', 'landing')], [('nav', 'cruise')]]
```

`fussell_vesely()` is the share of the mission's failure probability that
involves each component: the probability that some cut set naming it, in
any phase, has occurred, over the mission's unreliability. The union is
exact, from a decision diagram of the cut sets' events over the segments;
`method="rare_event"` sums the cut sets' probabilities instead, the
approximation many PRA tools report, which over-estimates the union when
cut sets overlap (here none share a component, so the two agree).

```python
fv = flight.fussell_vesely()
fv["nav"]                       # -> 0.5143
fv["e1"]                        # -> 0.2426
flight.fussell_vesely(method="rare_event")["e1"]   # -> 0.2426
```

`analysis_routes()` reports each analysis as exact, or refused when the
mission's diagram (for `fussell_vesely`, the cut sets' too) would pass
`MAX_NODES`, and `reliability_interval` as simulated.
