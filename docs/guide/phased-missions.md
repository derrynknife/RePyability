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
- **Not modelled here.** Repair between or during phases, and
  common-cause groups, which a phase may not have. A block on standby, a
  block that cannot fail in a phase, and a block's life model or
  covariates by phase are set on a mission drawn as
  [one diagram](#one-diagram-each-phase-setting-its-blocks).

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

## One diagram, each phase setting its blocks

Often there is one drawing of the system, and what changes from phase to
phase is how its blocks behave: the auxiliary power unit waits on standby,
the navigation unit is checked and cannot fail during the take-off, the
landing gear is not needed in the cruise, the engine vote needs both
engines to take off and land but one to cruise, an engine runs hotter in
the climb. `PhasedMission.from_diagram` takes the one diagram and a
`MissionPhase` per phase, each naming the blocks it sets:

```python
from repyability import BlockSetting, MissionPhase, PerfectReliability

aircraft = NonRepairableRBD(
    [("s", "e1"), ("s", "e2"), ("e1", "engines"), ("e2", "engines"),
     ("engines", "nav"), ("nav", "gear"), ("gear", "t")],
    {"e1": engine, "e2": engine, "engines": PerfectReliability,
     "nav": nav, "gear": gear},
    k={"engines": 1},
)
flight = PhasedMission.from_diagram(aircraft, [
    MissionPhase("take-off", 0.2, k={"engines": 2}, blocks={"nav": "guaranteed"}),
    MissionPhase("cruise", 10.0, blocks={"gear": "not_needed"}),
    MissionPhase("landing", 0.3, k={"engines": 2}),
])
flight.unreliability()           # -> 0.003983
```

Each block has one clock through the mission, its age, and one life: it
fails once it has used up its life's hazard, and how a phase sets it says
whether its clock runs and whether it uses hazard:

- **`"working"`** (a block not named): it runs, ages and can fail.
- **`"guaranteed"`**: it runs and ages, but cannot fail in this phase (its
  hazard is taken as zero there). Afterwards it carries on from the age it
  reached. A block that failed in an earlier phase stays failed.
- **`"not_needed"`**: the phase takes it as working, whatever its
  condition, but it runs, ages and can fail all the same; failed, it is
  failed for the later phases that need it (the same as leaving it out of a
  phase's own diagram).
- **`"standby"`**: it sits idle until the phase needs it, then runs, ages
  and can fail for the rest of the phase. In a later phase it follows that
  phase's setting: on standby again, it goes idle unless needed at once.
  By default it is cold standby (idle, it neither ages nor can fail) and
  its switch is perfect and instant; `BlockSetting` sets both:
  - `dormancy_factor=f`, from 0 to 1: warm standby. Idle for a time `d` it ages
    `f * d` on its life, so it can fail while idle (an exponential life of
    rate `l` at rate `f * l`); a failure while idle is found when the block
    is called on, and the next spare is called instead. `f = 1` is hot
    standby, ageing as if running.
  - `start_probability=p`: called on, the block starts with
    probability `p`; otherwise it has failed (a failure on demand) for the
    rest of the mission, and the next spare is called. One outcome per
    block and phase.

  Both mean what a `StandbyModel`'s settings of those names mean. A
  `StandbyModel`'s `switching_probability` (a failed switch that ends
  the arrangement) has no setting here: draw a switch that can fail as a
  block of its own, in series with the spares it serves.

  A switch that takes time is not modelled: the mission's blocks are not
  repaired, so any time with the structure down fails the mission, and a
  delay is a failure to start (`start_probability`).
- **covariates**: a block whose model is a `RegressionNode` with fixed
  covariates can be given its covariates in a phase,
  `BlockSetting(covariates=[...])` (with a state too, if wanted:
  `BlockSetting(state="standby", covariates=[...])`). Its life follows
  each phase's covariates along its own clock, its age carried over, as
  a `RegressionNode` along a step schedule does (an accelerated-failure-time
  model by its accumulated exposure, a proportional-hazards one by its
  hazard at each phase's values). A phase that does not name it uses the
  node's own covariates.
- **a model**: `BlockSetting(model=...)` gives the block a life model of
  its own in the phase, in place of its node's (see
  [below](#a-different-life-in-one-phase)).
- **`k`**: a phase can set the number of inputs a k-out-of-n node needs.

**When a block on standby starts.** Whenever the phase's structure does not
work (at the phase's start, or when a running block fails), taking the
blocks that are running and not failed as working (and those not needed,
and guaranteed ones not failed), the idle standby blocks that have not
failed are called on in the order the phase's `blocks` lists them until it
works: each starts if it has not failed idle and it starts on demand, and
has failed otherwise. Then each block just started, latest first, goes back to idle if
the structure works without it, so only the blocks needed start. If no
choice makes the structure work, the mission fails then.

**Exact and simulated.** A mission with no block on standby is exact, as
any phased mission is: each block's segment of a phase is survived with its
life's probability for the stretch (1 in a phase it is guaranteed), so the
decision diagram, the importance measures and the cut sets all work as
above. A block on standby starts when the others fail, so the blocks are
no longer independent, and the mission is not a product over phases:

- **Exact, by a Markov chain**, where every block's life has a constant
  failure rate in every phase (exponential lives, and exponential
  regression blocks at each phase's covariates). A block's age then does
  not matter, only whether it is running, idle or failed, so the mission
  is a Markov chain over the blocks' joint states: a running block fails
  at its rate in the phase (none when guaranteed), an idle one at its
  dormancy times that rate, and at each failure the spares are called on
  by the rule above, each start a branch with its switching probability.
  The chain is carried through each phase exactly (a matrix exponential),
  its states reset at each phase's start as the settings say. It gives the
  reliability, each phase's failure probability, the reliability over
  time, the take-overs and the drivers, up to 20,000 states a phase.
- **Simulated otherwise**: each block's life drawn once per mission (and
  one start outcome per spare and phase), and the missions played
  through the phases together, the take-overs worked out mission by
  mission, on the same rule as the chain (the tests hold them to each
  other). `reliability()` then refuses (pointing to `method="simulate"`
  and `reliability_interval`).

The importance measures and cut sets refuse a mission with a block on
standby either way (pointing to `failure_drivers`), and
`analysis_routes()` says which analysis is which, and why.

### A different life in one phase

Sometimes a block is, for one phase, a different animal: a pump fails
far more often while it starts up than once it runs steadily, an engine
at take-off power wears faster than in the cruise. Drawing a second
block for the phase would give it a second life, as if a new pump took
over; what is wanted is the same pump, on another failure behaviour for
a while. `BlockSetting(model=...)` gives the block that model in the
phase, in place of its node's.

The block keeps one clock and one budget of hazard through the mission.
Entering a phase at age `a`, it uses `H(a + d) - H(a)` of the phase's
model's cumulative hazard over the phase's duration `d`, and carries its
age, and the hazard it has left, into the next phase. So the mission's
reliability for one block is `exp(-sum of each phase's H(a + d) - H(a))`,
each phase's `H` its model's, `a` the block's age at the phase's start.
Here a pump runs a two-hour start-up on a model of early failures, then
100 hours of duty on its own:

```python
import math

pump = surv.Weibull.from_params([2000.0, 1.5])
start_up = surv.Weibull.from_params([300.0, 0.6])
plant = NonRepairableRBD([("s", "pump"), ("pump", "t")], {"pump": pump})
run = PhasedMission.from_diagram(plant, [
    MissionPhase("start-up", 2.0, blocks={"pump": BlockSetting(model=start_up)}),
    MissionPhase("duty", 100.0),
])
run.reliability()                # -> 0.940864
by_hand = math.exp(
    -(start_up.Hf(2.0) - start_up.Hf(0.0)) - (pump.Hf(102.0) - pump.Hf(2.0))
)
round(float(by_hand), 6)         # -> 0.940864
run.phase_failure_probabilities()["start-up"]   # -> 0.04827
```

The phase's model must be one the node could have been given itself
(not a probability of failing, which has no age to carry, and not for a
standby group, a load-sharing or repeated node or a nested diagram,
whose units take none). It goes with any state: on standby the block is
idle until needed, then runs on the phase's model; guaranteed, it ages
but cannot fail, so the model does not matter there; not needed, it runs
and can fail on the phase's model. The mission stays exact unless a block
is on standby, as above.

Covariates by phase are the special case of this: `covariates=z` is the
node's own regression model at `z` as the phase's model. With both, the
model must be a `RegressionNode` with fixed covariates, run at the given
ones. One fitted regression model at two covariate vectors carries the
block's age as covariates do (an accelerated-failure-time model by its
accumulated exposure); between two different models the age itself is
carried. The summary's settings hold the model in its saved form.

### The whole mission at once

`summary()` gives everything about the mission from one run: exact where
the mission is, else from one simulation. It holds the reliability (with a
confidence interval when simulated), the reliability over a grid of times
across the mission (20 steps a phase, each boundary included), each phase's
failure probability, share of the failures, failure probability given it
was reached, the reliability at its end, its blocks' settings, how often
each standby block was started in it, and its top driver; the drivers; and
the riskiest phase.

```python
result = flight.summary()
result.method                    # -> 'exact'
result.riskiest_phase            # -> 'cruise'
landing = result.phases[2]
landing.conditional_failure      # -> 0.00198
landing.top_driver               # -> 'e1'
```

`reliability_over_time(t)` is the probability that the mission has not
failed by each mission time, on one clock across the phases. It may step
down at a phase's start, where the phase needs more than the earlier ones
did: here the landing needs both engines again, and an engine lost in the
cruise fails the mission as the landing starts.

```python
flight.reliability_over_time([0.2, 5.0, 10.2, 10.5])
# -> [0.999992, 0.999033, 0.996154, 0.996017]
```

`failure_drivers()` says which blocks drove the failures: for each block
and phase, the fall in the phase's failure probability if the block never
failed (the mission worked out again with it held working). The total is
the fall in the mission's unreliability. An engine matters most at the
landing, where it is needed again:

```python
drivers = flight.failure_drivers()
drivers.totals["nav"]            # -> 0.002054
drivers.falls["e1"]["landing"]   # -> 0.000954
```

With an auxiliary power unit on standby to stand in for an engine, the
mission is simulated. In the cruise one engine is enough, so the unit stays
idle; at the landing it starts whenever an engine was lost:

```python
apu = surv.Exponential.from_params([1e-3])
with_apu = NonRepairableRBD(
    [("s", "e1"), ("s", "e2"), ("s", "apu"), ("e1", "engines"),
     ("e2", "engines"), ("apu", "engines"), ("engines", "nav"),
     ("nav", "gear"), ("gear", "t")],
    {"e1": engine, "e2": engine, "apu": apu, "engines": PerfectReliability,
     "nav": nav, "gear": gear},
    k={"engines": 1},
)
standing_by = {"apu": "standby"}
backed_up = PhasedMission.from_diagram(with_apu, [
    MissionPhase("take-off", 0.2, k={"engines": 2},
                 blocks={"nav": "guaranteed", **standing_by}),
    MissionPhase("cruise", 10.0, blocks={"gear": "not_needed", **standing_by}),
    MissionPhase("landing", 0.3, k={"engines": 2}, blocks=standing_by),
])
backed_up.analysis_routes()["summary"].route   # -> 'simulated'
result = backed_up.summary(mc_samples=200_000, seed=1)
result.reliability               # -> 0.9980
result.phases[2].takeovers["apu"]   # -> 0.0018
```

### A pumping mission: start-up, duty, standby

A pumping station runs three phases on one diagram: two pumps in
parallel feeding a valve. In the two-hour start-up the duty pump has just
been overhauled and is guaranteed, and the second pump waits on warm
standby (ageing at a fifth of the rate while idle, starting nine times in
ten); the 100 hours of duty need the valve and one pump, the second still
on standby; in the 50 hours on standby the valve is shut (not needed) and
the duty pump becomes the spare. Every life is exponential, so the whole
mission is exact:

```python
plant = NonRepairableRBD(
    [("s", "p1"), ("s", "p2"), ("p1", "valve"), ("p2", "valve"), ("valve", "t")],
    {"p1": surv.Exponential.from_params([0.004]),
     "p2": surv.Exponential.from_params([0.006]),
     "valve": surv.Exponential.from_params([0.001])},
)
spare = BlockSetting(state="standby", dormancy_factor=0.2, start_probability=0.9)
pumping = PhasedMission.from_diagram(plant, [
    MissionPhase("start-up", 2.0, blocks={"p1": "guaranteed", "p2": spare}),
    MissionPhase("duty", 100.0, blocks={"p2": spare}),
    MissionPhase("standby", 50.0, blocks={"valve": "not_needed", "p1": spare}),
])
result = pumping.summary()
result.method                          # -> 'exact'
round(result.reliability, 4)           # -> 0.6959
[round(ph.failure_probability, 4) for ph in result.phases]
# -> [0.002, 0.2074, 0.0947]
round(result.phases[1].takeovers["p2"], 4)   # -> 0.2671
```

`result.times` and `result.reliability_over_time` are the whole mission's
curve on one clock, ready to plot with the phases' `start` and `end`.
Simulated (here asked for), the summary also carries a band about the
curve, `lower` and `upper`, at its `confidence` (two-sided at each time):

```python
played = pumping.summary(method="simulate", mc_samples=50_000, seed=1,
                         confidence=0.9)
played.interval.lower <= result.reliability <= played.interval.upper   # -> True
```

**Not modelled yet.** A switch that takes time (see above); repair;
common-cause groups. Exact values with a standby block and a life whose
rate is not constant are not worked out: such a mission is simulated.
