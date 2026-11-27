# RePyability

[![actions](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml/badge.svg)](https://github.com/derrynknife/RePyability/actions/workflows/actions.yml)
[![docs](https://github.com/derrynknife/RePyability/actions/workflows/docs.yml/badge.svg)](https://derrynknife.github.io/RePyability/)

Reliability engineering tools, open source, for students through to
practising professionals.

RePyability builds and analyses systems as reliability block diagrams (RBDs),
taking already-fitted lifetime models (from
[surpyval](https://github.com/derrynknife/SurPyval) or anything exposing
`sf`/`ff`) as its components:

- **Fault trees**: static fault trees (OR, AND and VOTE gates, repeated
  events) evaluated exactly, with common-cause groups, cut sets, importance
  measures, the mean time to the top event, B-lives and hazard, parameter
  sensitivities, uncertainty from fitted event models, and conversion to
  and from block diagrams; NOT gates (exact, with prime implicants),
  dynamic gates (priority AND, cold, warm and hot spares, sequences and
  functional dependencies: the top event probability over time, integrated
  where each is an independent module, simulated otherwise), and export to
  Open-PSA.
- **Reliability**: exact system reliability, hazard and conditional survival;
  the exact MTTF (or simulated, with confidence intervals); B*X* life;
  uncertainty intervals on the reliability, MTTF, B*X* life and time to a
  reliability from uncertain (fitted) component models.
- **Networks**: networks whose links fail, undirected or directed: the
  exact reliability of the connection between two terminals, several, or
  every node (K-terminal and all-terminal reliability), by a decision
  diagram that keeps meshed networks fast; with repaired links, the
  connection's availability over time and in the long run, and how often
  it is lost.
- **Phased missions**: missions through phases (take-off, cruise,
  landing), each with its own diagram over the same components: the exact
  mission reliability and the chance of failing in each phase, with
  common-cause groups whose causes fail their members from the phase they
  strike in, and blocks repaired while the mission runs.
- **Importance**: Birnbaum, improvement potential, RAW, RRW, criticality,
  Fussell–Vesely, structural importance and parameter sensitivity, on
  repairable systems in the long run or over time.
- **Sensitivity, the Greeks**: how far the system moves with each
  component (delta) and with each lever, a life, a repair, the maintenance
  or a crew; the shares of a change, which add up (differential
  importance); whether two improvements are complements or substitutes
  (gamma); which component is moving the availability now, and which
  caused the failures (theta, Barlow–Proschan); and whose parameter
  uncertainty widens the answer (vega).
- **Live state**: reliability, remaining life and importance given each
  component's current age, and covariate-dependent components.
- **Redundancy and dependence**: cold, warm and hot standby; repeated nodes;
  load sharing; junctions (a vote point that never fails); beta-factor and
  MGL common-cause groups, splitting a probability or a failure rate, in
  block diagrams, repairable systems and fault trees.
- **Event trees**: an initiating event's frequency through the success
  or failure of each barrier to each outcome's frequency (`EventTree`),
  barriers given as fault trees that share events or as protection layers
  (LOPA), exact, with each event's importance to an outcome
  ([the guide](docs/guide/event-trees.md)).
- **Safety functions**: voted channels with proof tests, diagnostics,
  repair and common cause (`SafetyFunction`): the PFDavg, the PFH, the
  spurious-trip rate and the SIL band from one call, exact, with IEC
  61508-6 Annex B's simplified formulas beside them
  ([the guide](docs/guide/safety-functions.md)).
- **Repairable systems**: exact long-run availability, failure frequency and
  MUT/MDT/MTBF; exact availability over time and over a mission, and the
  unavailability to its own precision (a PFD of 1e-17); simulated
  histories with criticality measures; shared repair crews, exact for
  exponential components in the long run and numerical over time, with
  their importance; repairable standby groups (a duty unit and its spares,
  repaired one at a time); imperfect repair (Kijima's virtual age), with
  replacement at the N-th failure; repair again or replace for a fleet
  whose units differ (a joint frailty model of repairs and write-offs:
  the best rule for a new unit, and each unit's decision from its
  frailty); and uncertainty intervals on the
  availability and the cost rate from uncertain (fitted) models of the
  lives, repairs and maintenance.
- **Capacity**: how much a system delivers, from its components'
  capacities (with several levels, or degrading through stages): the exact
  distribution of its capacity at a time or in the long run, the
  probability of meeting a demand, and the production availability.
- **Spares**: how many spares each component uses over a horizon, for a
  system or a fleet, and the stock that meets a fill rate or a stock-out
  target for a replenishment lead time, with interchangeable components'
  spares pooled on one shelf.
- **Timelines**: up/down histories, from outage logs or simulated, with
  their measures (time up, failures and who caused them, first failure);
  merged as a diagram's structure, so a system's history follows from its
  components', with the component behind each of its outages.
- **Schedules on the calendar** (`repyability.schedule`): which dates a
  rule gives (every so many days, weeks, months or years; named weekdays;
  the first Saturday or the last Friday of the month; the first Tuesday of
  November every year; given dates; with seasons and excluded dates), and
  when tasks fall due on them, with two limits, crew visits, and the work
  by month. Plain data in and out, no clock read.
- **Simulation**: seeded Monte-Carlo run to a tolerance, antithetic pairs,
  control variates from the system's exact twin, parallel runs, runs split
  across machines and merged, or sharded through any executor (Ray, Dask,
  a batch system), comparisons of
  designs with common random numbers, and small failure probabilities by
  rare-event simulation (subset simulation, cross-entropy importance
  sampling); repairable systems simulated compiled, with numba installed.
- **Cost, design and maintenance**: exact and simulated running costs,
  including scheduled (age or block) preventive replacement at system level,
  replacement on condition at periodic inspections (by age or by a measured
  level), opportunistic maintenance of grouped components at each other's
  stops, shutdowns, preventive maintenance that restores a unit only
  partly, and tasks counted in running hours, with the downtime split into
  planned and unplanned,
  with its intervals chosen for a cost or availability target, hidden
  failures found by periodic inspection, with the test intervals chosen for a
  PFDavg target, the total cost of ownership, discounted to a present
  value from new or in the long run, optimal redundancy allocation (for the
  lowest total cost of a repairable system, too, with whole trains given
  copies together), reliability
  allocation by the classic named methods (equal and ARINC-style
  apportionment, minimum effort, cost-based), availability allocation to
  MTTF and MTTR targets, and age-replacement and overhaul policies.
- **Saving**: diagrams to and from JSON, seeded results that repeat, and
  every result as plain data (`to_dict()`, ready for `json.dumps`).

```python
import surpyval as surv
from repyability import NonRepairableRBD

rbd = NonRepairableRBD(
    [("s", "pump1"), ("s", "pump2"), ("pump1", "valve"), ("pump2", "valve"), ("valve", "t")],
    {
        "pump1": surv.Weibull.from_params([100, 2]),
        "pump2": surv.Weibull.from_params([100, 2]),
        "valve": surv.Weibull.from_params([200, 1.5]),
    },
)
rbd.sf(50)                     # 0.839: system reliability at t = 50
rbd.birnbaum_importance(50)    # which component matters most
```

New to reliability engineering? The documentation includes
[Learn](https://derrynknife.github.io/RePyability/learn/), a short course
that teaches system reliability from a single part's lifetime to designing
and maintaining whole systems, working every idea out by hand and then with
RePyability, with exercises and worked answers.

## Install

RePyability needs Python 3.11 or later. Install it from
[PyPI](https://pypi.org/project/repyability/):

```bash
pip install repyability
```

Repairable systems simulate about ten times as fast with the optional
compiled engine, which needs [numba](https://numba.pydata.org) (available for
64-bit Linux, macOS on Apple silicon and Windows):

```bash
pip install "repyability[fast]"
```

## What each model can do

The analyses each model type has, and how each family of them is worked
out for each kind of node or component: the gaps are where a cell says
— or ✗. The tables are made from the code (`python -m
repyability.tests.capabilities` writes them, and a test checks they are
current).

<!-- capabilities:start -->

Which analyses each model type has (the method; — where it has none):

| | `NonRepairableRBD` | `RepairableRBD` | `FaultTree` | `Network` | `PhasedMission` | `EventTree` |
|---|---|---|---|---|---|---|
| **Reliability and lifetime** | | | | | | |
| Reliability R(t) | `sf` | — | `sf` | `sf` | `sf` | — |
| Unreliability F(t) | `ff` | — | `ff` | `ff` | `ff` | — |
| Density, hazard, cumulative hazard | `df` | — | `df` | `df` | — | — |
| Mean time to (first) failure | `mean` | — | `mean` | `mean` | — | — |
| Each node's own R(t), F(t) and MTTF | `node_sf` | — | — | — | — | — |
| B-life, time to a reliability | `bx_life` | — | `bx_life` | `bx_life` | — | — |
| Given the system's age | `cs` | — | — | — | — | — |
| Given the components' ages or states | `sf_given_state` | — | — | — | — | — |
| Mean residual and remaining life | `mean_residual_life` | — | — | — | — | — |
| Random lifetimes | `random` | — | `random` | `random` | — | — |
| Top event probability | — | — | `top_event_probability` | — | — | — |
| Each phase's failure probability | — | — | — | — | `phase_failure_probabilities` | — |
| Reliability over a mission's time | — | — | — | — | `reliability_over_time` | — |
| Expected time a mission runs (to failure or its end) | — | — | — | — | `expected_mission_time` | — |
| A mission's results from one run | — | — | — | — | `summary` | — |
| **Availability and outages** | | | | | | |
| Reliability and mean time to the first failure | — | `first_failure_sf` | — | — | — | — |
| Long-run availability | — | `mean_availability` | — | `mean_availability` | — | — |
| Each node's availability | — | `node_availability` | — | — | — | — |
| Availability over time | — | `point_availability` | `point_unavailability` | `point_availability` | — | — |
| Over a mission | — | `mission_availability` | — | `mission_availability` | — | — |
| Unavailability (precise when small) | — | `mean_unavailability` | `mean_unavailability` | `mean_unavailability` | — | — |
| Failure frequency, MTBF | — | `system_failure_frequency` | `system_failure_frequency` | `system_failure_frequency` | — | — |
| Mean up and down times | — | `mean_up_time` | — | — | — | — |
| Expected failures and events in a window | — | `expected_events` | `expected_failures` | `expected_failures` | — | — |
| Simulated availability (distribution) | — | `availability` | — | — | — | — |
| Simulated histories | — | `simulate_timelines` | — | — | — | — |
| Each simulation's values, common random numbers | — | `simulation_samples` | — | — | — | — |
| Repair crews' jobs and chain | — | `crew_jobs` | — | — | — | — |
| Compare two designs | `compare` | `compare` | `compare` | `compare` | `compare` | — |
| **Costs and spares** | | | | | | |
| Long-run cost rate | — | `expected_cost_rate` | — | — | — | — |
| Expected cost in a window | — | `expected_cost` | — | — | — | — |
| Expected net present value | — | `expected_npv` | — | — | — | — |
| A plan against what happened | — | `plan_vs_actual` | — | — | — | — |
| Simulated net present value (distribution) | — | `npv_distribution` | — | — | — | — |
| Best time to switch blocks' running (simulated NPV) | — | `best_switch_time` | — | — | — | — |
| Two plans' net present values compared | — | `compare_npv` | — | — | — | — |
| Least spend for a bad run's downtime (simulated) | — | `least_spend_for_downtime` | — | — | — | — |
| Cost in each of a run of windows | — | `cost_by_window` | — | — | — | — |
| Simulated cost (distribution) | — | `cost` | — | — | — | — |
| Total cost of ownership | — | `total_cost` | — | — | — | — |
| Each component's actions per unit time | — | `node_action_rates` | — | — | — | — |
| A replaced component's renewal cycle | — | `replacement_cycle` | — | — | — | — |
| Planned and unplanned downtime, failures the plan avoids | — | `maintenance_effect` | — | — | — | — |
| Spares demand and stock | — | `spares_demand` | — | — | — | — |
| Repairable spares over bases and a depot | — | `spares_front` | — | — | — | — |
| **Structure** | | | | | | |
| Minimal path and cut sets | `minimal_cut_sets` | `minimal_cut_sets` | `minimal_cut_sets` | `minimal_cut_sets` | `minimal_cut_sets` | — |
| Prime implicants (NOT gates) | — | — | `prime_implicants` | — | — | — |
| Each path set's probability | `path_set_probabilities` | `path_set_probabilities` | — | — | — | — |
| Cut sets ranked by probability | — | — | `ranked_cut_sets` | — | — | — |
| Irrelevant components | `find_irrelevant_components` | `find_irrelevant_components` | — | — | — | — |
| Structure sizes (counts, depth, core) | `structure_summary` | `structure_summary` | — | — | — | — |
| The core that does not reduce | `core_structure` | `core_structure` | — | — | — | — |
| Junctions (drawing nodes) | `junctions` | `junctions` | — | — | — | — |
| Convert to and from an RBD | — | — | `to_rbd` | `to_rbd` | `from_diagram` | — |
| Capacity distribution | `capacity_distribution` | `capacity_distribution` | — | — | — | — |
| Capacity over time | — | `point_capacity` | — | — | — | — |
| How each analysis is computed | `analysis_routes` | `analysis_routes` | `analysis_routes` | `analysis_routes` | `analysis_routes` | `analysis_routes` |
| Save and load (dict, JSON) | `to_dict` | `to_dict` | `to_dict` | `to_dict` | `to_dict` | `to_dict` |
| Export to Open-PSA (MEF) | — | — | `to_open_psa` | — | — | — |
| Import from Open-PSA (MEF) | — | — | `from_open_psa` | — | — | — |
| **Event trees (an initiating event through its barriers)** | | | | | | |
| Each sequence's probability and frequency | — | — | — | — | — | `sequence_probabilities` |
| Each outcome's frequency | — | — | — | — | — | `outcome_frequencies` |
| Each sequence as one fault tree (its prime implicants) | — | — | — | — | — | `sequence_tree` |
| **Safety functions (voted channels, proof tests, diagnostics)** | | | | | | |
| PFDavg, and where it comes from | — | `SafetyFunction.pfd_avg` | — | — | — | — |
| PFH (frequency of a dangerous failure) | — | `SafetyFunction.pfh` | — | — | — | — |
| Spurious-trip rate | — | `SafetyFunction.spurious_trip_rate` | — | — | — | — |
| SIL band, safe failure fraction | — | `SafetyFunction.sil` | — | — | — | — |
| Every result in one call, votes in series | — | `SafetyFunction.results` | — | — | — | — |
| IEC 61508-6's simplified formulas, to compare | — | `SafetyFunction.simplified_formulas` | — | — | — | — |
| How each result is computed | — | `SafetyFunction.analysis_routes` | — | — | — | — |
| **Importance measures** | | | | | | |
| Birnbaum | `birnbaum_importance` | `birnbaum_importance` | `birnbaum_importance` | `birnbaum_importance` | `birnbaum_importance` | `birnbaum_importance` |
| Structural | `structural_importance` | `structural_importance` | `structural_importance` | `structural_importance` | — | — |
| Improvement potential | `improvement_potential` | `improvement_potential` | `improvement_potential` | `improvement_potential` | `improvement_potential` | — |
| Risk achievement worth | `risk_achievement_worth` | `risk_achievement_worth` | `risk_achievement_worth` | `risk_achievement_worth` | `risk_achievement_worth` | — |
| Risk reduction worth | `risk_reduction_worth` | `risk_reduction_worth` | `risk_reduction_worth` | `risk_reduction_worth` | `risk_reduction_worth` | — |
| Criticality | `criticality_importance` | `criticality_importance` | `criticality_importance` | `criticality_importance` | `criticality_importance` | — |
| Fussell-Vesely | `fussell_vesely` | `fussell_vesely` | `fussell_vesely` | `fussell_vesely` | `fussell_vesely` | `fussell_vesely` |
| Differential (DIM) | `differential_importance` | `differential_importance` | `differential_importance` | `differential_importance` | `differential_importance` | — |
| Joint (second order) | `joint_importance` | `joint_importance` | `joint_importance` | `joint_importance` | `joint_importance` | — |
| Barlow-Proschan | `barlow_proschan_importance` | `barlow_proschan_importance` | `barlow_proschan_importance` | `barlow_proschan_importance` | `barlow_proschan_importance` | — |
| Shares of a safety function's PFDavg, common cause apart | — | `pfd_breakdown` | — | — | — | — |
| Each block's part in each phase's failures | — | — | — | — | `failure_drivers` | — |
| Rate of change over time | `reliability_rate` | `availability_rate` | — | — | — | — |
| Uncertainty importance | `uncertainty_importance` | `uncertainty_importance` | — | — | — | — |
| Every measure in one call | `importances` | `importances` | `importances` | `importances` | `importances` | — |
| All measures given a state | `importances_given_state` | — | — | — | — | — |
| Top event given each event | — | — | `conditional_probabilities` | — | — | — |
| Parameter sensitivity, levers | `parameter_sensitivity` | `parameter_sensitivity` | `parameter_sensitivity` | `parameter_sensitivity` | `parameter_sensitivity` | — |
| Tornado data (each lever at its low and high) | `tornado` | `tornado` | `tornado` | `tornado` | — | — |
| **Uncertainty** | | | | | | |
| Of the reliability or availability | `sf_uncertainty` | `mean_availability_uncertainty` | `sf_uncertainty` | `sf_uncertainty` | `reliability_interval` | — |
| Of the MTTF | `mean_uncertainty` | — | `mean_uncertainty` | `mean_uncertainty` | — | — |
| Of the B-life | `bx_life_uncertainty` | — | `bx_life_uncertainty` | `bx_life_uncertainty` | — | — |
| Of the cost rate | — | `expected_cost_rate_uncertainty` | — | — | — | — |
| Of the net present value | — | `expected_npv_uncertainty` | — | — | — | — |
| **Design and maintenance** | | | | | | |
| Reliability allocation | `simple_allocation` | `simple_allocation` | — | — | — | — |
| Availability, MTTF and MTTR allocation | — | `availability_allocation` | — | — | — | — |
| Redundancy allocation | `allocate_redundancy` | `allocate_redundancy` | — | — | — | — |
| Reliability-redundancy allocation | `allocate_reliability_redundancy` | — | — | — | — | — |
| Cost-reliability front | `redundancy_front` | — | — | — | — | — |
| Optimal replacement intervals | — | `optimal_replacement_intervals` | — | — | — | — |
| Optimal proof-test intervals | — | `optimal_inspection_intervals` | — | — | — | — |

How each family of analyses is worked out on a diagram with each kind of node or component: **E** exact, **N** numerical (to a stated accuracy), **S** simulated, **✗** refused as not worked out (naming the methods, or counting them, where only some are), – where the diagram has nothing the analysis works on. Made from `analysis_routes()` on the test catalogue's diagrams.

<details><summary><code>NonRepairableRBD</code>, by node</summary>

| | Reliability | MTTF | B-life | Given a state | Importance | Sensitivity | Uncertainty | Allocation | Capacity | Simulation |
|---|---|---|---|---|---|---|---|---|---|---|
| A surpyval model | E | N | N | E N | E N | E N | S | E N | – | S |
| A fitted model (with its covariance) | E | N | N | E N | E N | E N | S | E N | – | S |
| A fit with a cure share and zero-inflation | E | N | N | E N | E N | E N | S | E N | – | S |
| A probability per demand | E | – | – | E | E | E N | S | E N | – | S |
| A mixture (`MixtureModel`) | E | N | N | E N | E N | E N | S | E N | – | S |
| A regression node (covariates) | E | N | N | E N | E N | E N | S | E N | – | S |
| A joint model's unit, given its readings | N | N | N | N | N | N | S | N | – | S |
| A degrading node | N | N | N | N | N | N | S | N | N | S |
| A fitted degradation process | E | N | N | E N | E N | E N | S | E N | – | S |
| Cold standby (convolved) | N | N | N | N | N | N | S | N | – | S |
| Standby with no reliability of its own | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | N; ✗ `allocate_redundancy` | – | S |
| A Markov component with a lifetime | E | N | N | E N | E N | E N | S | E N | – | S |
| Load sharing | E | N | N | E N | E N | E N | S | E N | – | S |
| A repeated node | E | N | N | E N | E N | E N | S | E N | – | S |
| A repeated standby node | N | N | N | N | N | N | S | N | – | S |
| A nested RBD | E | N | N | E N | E N | E N | S | E N | – | S |
| A junction | E | N | N | E N | E N | E N | S | E N | – | S |
| A beta-factor common-cause group | E | ✗ | N | ✗ | E N | E N | S; ✗ `mean_uncertainty` | E N | – | S |
| An MGL common-cause group | E | ✗ | N | ✗ | E N | E N | S; ✗ `mean_uncertainty` | E N | – | S |
| An alpha-factor common-cause group | E | ✗ | N | ✗ | E N | E N | S; ✗ `mean_uncertainty` | E N | – | S |
| Capacities | E | N | N | E N | E N | E N | S | E N | E | S |
| A structure too meshed to work out | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | S |

</details>

<details><summary><code>RepairableRBD</code>, by component</summary>

| | Long run | Frequency | Over time | Costs | Importance | Sensitivity | Uncertainty | Allocation | Intervals | Spares | Capacity | Simulation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Any life and repair | E | E | N | E N | E N | E N | S | E | – | N | – | S |
| A fitted life | E | E | N | N | E N | E N | S | E | – | N | – | S |
| A fitted Crow-AMSAA process | ✗ | ✗ | N | N; ✗ `expected_cost_rate` | N; ✗ 10 of 11 | ✗ | S; ✗ `mean_availability_uncertainty`, `expected_cost_rate_uncertainty` | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| A fitted G1, ARA or ARI process | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| A probability per demand | ✗ | ✗ | ✗ | ✗ `expected_npv` | ✗ | ✗ | ✗ `mean_availability_uncertainty`, `expected_npv_uncertainty` | ✗ | – | N | ✗ `capacity_distribution` | S |
| A probability per demand, with a demand rate | E | E | N | E N | E N | E N | S | E | – | N | – | S |
| A probability per demand, at set times | ✗ | ✗ | N | N; ✗ `expected_cost_rate` | N; ✗ 10 of 11 | ✗ | S; ✗ `mean_availability_uncertainty`, `expected_cost_rate_uncertainty` | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| A probability per demand, on a demand plan | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| A life with a demand mode beside it | E | E | N | E N | E N | E N | S | E | – | N | – | S |
| A mixture life | E | E | N | E N | E N | E N | S | E | – | N | – | S |
| A fitted degradation process life | E | E | N | N | E N | E N | S | E | – | N | – | S |
| Operated part of the time (duty) | N | N | N | N | N | N | S | N | – | N | – | S |
| Tasks counted in running hours (duty) | N | N | N | N | N | N | S | N | N | N | – | S |
| A standby or load-sharing life, simulated | ✗ | ✗ | ✗ | ✗ `expected_npv` | ✗ | ✗ | ✗ `mean_availability_uncertainty`, `expected_npv_uncertainty` | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| k-out-of-n | E | E | N | N | E N | E N | S | E | – | N | – | S |
| A nested RBD | E | E | N | E N | E N | E N | S | E | – | N | – | S |
| A junction | E | E | N | E N | E N | E N | S | E | – | N | – | S |
| Capacities | E | E | N | N | E N | E N | S | E | – | N | E N | S |
| Age replacement | N | N | N | N | N | N | S | N | N | N | – | S |
| Block replacement | N | N | N | N | N | N | S | N | – | N | – | S |
| Opportunistic maintenance | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ `optimal_replacement_intervals` | ✗ | ✗ `capacity_distribution` | S |
| Replacement on condition | N | N | N | N | N | N | S | N | – | ✗ | – | S |
| Replacement on condition by level | N | N | N | N | N | N | S | N | – | ✗ | – | S |
| Hidden failures, tested | N | N | N | N | N | N | S | N | – | N | – | S |
| Tests that take time | N | N | N | N | N | N | S | N | – | N | – | S |
| Tests that can miss a failure | N | N | N | N | N | N | S | N | – | N | – | S |
| An upgrade (another life from a time) | E | E | N | E N | E N | E N | S | E; ✗ `allocate_redundancy` | – | ✗ | – | S |
| An upgrade, age-replaced | N | N | N | N | N | N | S | N; ✗ `allocate_redundancy` | ✗ `optimal_replacement_intervals` | ✗ | – | S |
| An upgrade, tested | N | N | N | N | N | N | S | N; ✗ `allocate_redundancy` | ✗ `optimal_inspection_intervals` | N | – | S |
| An upgrade, the unit in service kept | N | N | N | N | N | N | S | N; ✗ `allocate_redundancy` | ✗ `optimal_replacement_intervals` | ✗ | – | S |
| A shelf of spares | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| Imperfect repair (Kijima) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| Minimal repair | ✗ | ✗ | N | N; ✗ `expected_cost_rate` | N; ✗ 10 of 11 | ✗ | S; ✗ `mean_availability_uncertainty`, `expected_cost_rate_uncertainty` | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| Minimal repair, repairs that take time | ✗ | ✗ | N | N; ✗ `expected_cost_rate` | N; ✗ 10 of 11 | ✗ | S; ✗ `mean_availability_uncertainty`, `expected_cost_rate_uncertainty` | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| Limited repair crews | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | – | ✗ | ✗ `capacity_distribution` | S |
| A standby group | E | E | N | E N | E N | E N | S | E | – | ✗ | – | S |
| A Markov component (states and rates) | E | E | N | E N | E N | E N | S | E | – | ✗ | – | S |
| A Markov component with a capacity in each state | E | E | N | N | E N | E N | S | E | – | ✗ | E N | S; ✗ `availability` |
| A beta-factor group, revealed failures | E | E | N | N | E N | E N | S | E | – | N | – | S |
| A beta-factor group, tested | E | E | N | E N | E N | E N | S | E | – | N | – | S |
| An alpha-factor group, revealed failures | E | E | N | N | E N | E N | S | E | – | N | – | S |
| A group whose tests take time | N | N | N | N | N | N | S | N | – | N | – | S |
| A structure too meshed to work out | ✗ | ✗ | ✗ | E; ✗ `expected_cost`, `expected_npv` | ✗ | ✗ | S; ✗ `mean_availability_uncertainty`, `expected_npv_uncertainty` | ✗ | – | N | – | S |

</details>

<!-- capabilities:end -->

## When is a simulation needed?

RePyability computes exactly by default: by closed forms, or by
deterministic numerical methods (quadrature, convolution, renewal equations)
that give the same answer every run, to a small stated error. It simulates
only when

- the answer is itself random: sampled lifetimes or histories, or the
  spread of an outcome over a window (a cost's percentiles, how many
  failures);
- components depend on each other in a way no exact method solves in
  general; or
- an exact method could exist but has not been written yet.

**Exact or numerical, in any architecture** (series, parallel,
*k*-out-of-*n*, bridges and other meshed diagrams, components in several
places, nested diagrams):

- **components:** surpyval lifetime distributions, degradation processes
  (Wiener and gamma, as the time to their threshold) and fixed
  probabilities (a diagram refuses a non-parametric fit, such as
  Kaplan–Meier: fit a parametric distribution in surpyval); a unit of a
  surpyval joint model of readings and failures, given its readings
  (`JointNode`, from now); repeated nodes; a component given as a Markov chain
  (`MarkovComponent`: its states, the rates between them and the states
  it works in, with a lifetime or restored by its own transitions);
  standby of Exponential units
  (any number operating, any dormancy); cold standby with one or two
  units operating (any units) or more (identical units); warm standby
  with one unit operating, and hot standby (any units); spares that may
  fail to switch in or to start, in each of these; load sharing of
  identical units; common-cause groups, for the system's reliability,
  importance, sensitivity and redundancy allocation (and, splitting the
  failure rate, its MTTF and the analyses given its members' one age:
  [wear-out lives and ages](docs/guide/common-cause-ages.md));
- **non-repairable questions:** reliability, hazard, MTTF, B*X* life and
  importance measures at any time, also given each component's current age;
  the distribution of the system's capacity;
- **repairable questions,** for independent components repaired when they
  fail, replaced on age or block schedules or on condition at inspections,
  or tested for hidden failures (any life, the tests and repairs instant or
  taking time, the tests finding every failure or missing some): the
  long-run availability,
  failure frequency, MUT/MDT/MTBF and cost rate; the availability over time
  and over a mission, and the expected failures, outages, downtime and cost
  over a window, from new or from the components' current states (their
  ages, the repairs going on, where they are in their calendars, or the
  long run); the capacity distribution in the long run and over time, with
  the production availability of a window; and the spares used over a
  horizon, and the stock to hold for a lead time (under block replacement,
  with repairs and block replacements in no time). Common-cause groups of
  members with exponential lives, tested or repaired, take nothing away:
  the long-run values, importance and allocations stay exact, and the
  values over time from new numerical;
- **fault trees of repaired or tested events** (repeated events, VOTE and
  NOT gates included): the top event's unavailability over time, its
  average over a window and in the long run, and, for a coherent tree, its
  failure frequency and expected occurrences, from the tree's own decision
  diagram over the events' curves.

**Simulated**, or refused with the simulation to run instead:

| Situation | Today | Must it be simulated? |
|---|---|---|
| **What you ask** | | |
| Sampled lifetimes or histories, and distributions or percentiles of an outcome over a window | Simulated (from new, or from the components' current states); each simulation's histories, the system's and its components', kept whole as timelines by `simulate_timelines` (from new) | Yes: the answer is a sample. Its mean over a window (failures, outages, downtime, cost, the capacity delivered) is exact, from new or from a state: `expected_events`, `expected_cost`, `mission_capacity`. |
| The reliability and mean time to the first failure of a repairable system whose components are not all exponential in a redundant structure, or that has maintenance, tests, standby or common-cause groups | Simulated on request (`first_failure_sf(method="simulate")`); bracketed without simulating, wherever the expected events are worked out, by `first_failure_bounds` (between one less the expected system failures and one less the unavailability; the upper bound is 1 where the system takes planned outages), and the mean bounded below by `mean_time_to_first_failure_bound` | Not in general: lives and repairs written as mixtures of Erlang phases, and the common-cause and standby groups' chains made absorbing, could give it numerically; tests and maintenance at fixed intervals have no such chain. The bracket is tight while the expected failures are well under 1. |
| Comparing two designs (`compare`) | Exact where both designs' expected values are (their MTTFs; a repairable system's mission availability and expected cost), with no simulation (#236); otherwise, or on request (`method="simulate"`, `control_variate=False`), simulated with common random numbers | Only where a design's own values are simulated. |
| The uncertainty from fitted component parameters (`sf_uncertainty`, `mean_uncertainty`, `bx_life_uncertainty`, `time_to_reliability_uncertainty`; for a repairable system `mean_availability_uncertainty`, `point_availability_uncertainty`, `mission_availability_uncertainty`, `expected_cost_rate_uncertainty`) | Sampled over the parameters, randomly or quasi-randomly (`sampling="sobol"`), each draw exact or numerical as the diagram's own value is | Sampling is the method. |
| Small failure probabilities, with a node only simulations take | Rare-event simulation (`unreliability_interval`); `ff` refused | Only while the node has no reliability of its own: an exact diagram gives `ff` directly, to full precision however small (a numerical node, such as a cold-standby group of non-exponential units, to its own accuracy, about 1e-6). |
| **Components** | | |
| A probability of failing per demand (fail to start) in a repairable diagram | With a `"demand_rate"` (demands arriving as a Poisson process, each failing the unit with the probability): an exponential life of rate the demand rate times the probability, so exact or numerical on every route, as that life is. With `"demand_times"` (demands at set times, a recurrence rule's through `rule_times`): its values over a window exact (a sum over the demands), the long run, mean lives and importances refused, the rest simulated. With a `"demand_plan"` (a rate that changes): an exponential life in demands on the plan's clock, as a duty plan that changes, so its values over a window numerical where its repairs take no time, else simulated. Without any: simulated (a unit fails at once with the probability, or never), the exact and numerical analyses refused, naming the key. Beside a life that wears out (`"demand": {"probability": p, "rate": d}`): competing risks renewed together, a life of the life's hazard plus `d p`, so exact or numerical on every route, as a life taken by its functions is | Partly: with demands at a rate, its failures are a Poisson process; demands at set times end, so they have no long run; a plan whose unit takes time to repair is no renewal process on the clock, as a duty plan that changes is not. A life with demands at set times or on a plan beside it in one component (draw them as two in series) is not modelled. |
| Warm standby with two or more units operating, of non-exponential units | Simulated in the system's simulations; the analyses that need the group's reliability refused | Yes, in general: which spare is switched in where, and how much each has aged, branch with the order of the failures. With one unit operating it is numerical, hot standby (k-out-of-*n*) exact, and Exponential units exact (a Markov chain), spares that may fail to switch in or start included. |
| Cold standby with three or more different units operating, and load sharing of different units | Simulated in the system's simulations; the analyses that need the group's reliability refused | Yes, in general: which spare goes where, and how old the others are, branch with the order of the failures. Identical units are numerical in both, and so are two different units operating; Exponential units in standby are exact. |
| Anything such a node is part of | Simulated in the system's simulations; the analyses that need the node's reliability refused | Only while the node has no reliability of its own. |
| Common-cause groups: analyses given ages for members of different ages or a group splitting a failure probability, and the MTTF of such a group | Refused (a simulated MTTF leaves a probability split out) | No. A group splitting the failure rate (`basis="rate"`) is exact given its members' one age, as a share of their hazard is one hazard only then, and a shock at its own rate that fails members of any lives and ages is an exponential node in series with them, and `shock_nodes` writes out shocks that fail each member with a probability ([the guide](docs/guide/common-cause-ages.md)). A group splitting the failure rate has an exact MTTF, and the simulations draw its shared shocks. Importance, parameter sensitivity and redundancy allocation are exact with groups (a beta-factor member's copies join its group), and parameter uncertainty is sampled with them. |
| **Architecture and maintenance** | | |
| Phased missions and networks too large for their decision diagrams | Refused, pointing to `method="simulate"` | Only in practice: the diagrams grow with the phases' and the network's width rather than their paths, so meshed missions and networks are exact (a network grid of 100 nodes in a second and a half); one of 121 nodes passes the limit, `repyability.network.MAX_STATES`, which can be raised. |
| Phased missions with blocks repaired during their phases, or on standby | Exact (`reliability`, `phase_failure_probabilities`, the measures a block is held in) where every life has a constant failure rate in every phase and every repair is exponential; simulated otherwise (`reliability_interval`, `summary`, `method="simulate"`), the exact routes refusing | Not for wear-out lives: a block's age then matters to when it is back or called on. Exponential lives and repairs are a Markov chain over the blocks' states, carried through the phases, a phase's down states absorbing. |
| Block diagrams too meshed for their decision diagrams | Simulated (lifetimes, availability, cost and timelines, in Python); the exact and numerical analyses refused | Only in practice: the diagram grows with how wide the mesh is rather than with its paths, so most meshes are exact (a 10 × 10 grid in 0.04 seconds, a random mesh of 60 nodes and 345 links in 2); one of 70 nodes and 485 links passes the limit, `repyability.rbd.bdd.STEP_LIMIT`, which can be raised. |
| Common-cause groups in a repairable diagram | For exponential lives, tested or repaired: the long run, importance and the allocations exact (a beta-factor member's copies join its group, and the availability allocations keep the members' availability), and numerical where the members' tests and repairs take time (#220): a fixed or an exponential time exactly, any other length on pieces of the time since the test (#106); the values over time numerical (the groups' Markov chains), from new, from the members' states or from the long run (#106), and the simulations draw the shared causes, from new or from the members' states, in Python. Shocks at a rate of their own (`Shocks`: each failing each member with a probability, the members of any lives, each then repaired or found at its test on its own): for exponential members, tested or repaired, of rates, repairs and tests of their own, exact as the groups that split the rate are (the long run, importance, the values over time, a member's own availability and spares, and with repair crews), and simulated, in Python, for members of any lives, maintained on a schedule, repaired imperfectly, in a maintenance group or started from a state (a tested member perhaps failed, unseen, by a shock since its last test). Members of other lives in the groups that split the rate refused, as are copies of members whose tests or repairs take time, tests and repairs that can run into the next test, and members held | Other lives: a shared cause has no one rate for members of different ages, so they need a model first. Shocks at their own rate are that model for members of any lives, exact for exponential members and otherwise simulated; their chain would need each member's age, so yes, in general. Without repairs, [the guide](docs/guide/common-cause-ages.md) has a second: a share of the hazard of members of one age. |
| Shared repair crews | For exponential lives and repairs, the long run and importance exact, and the values over time numerical (the same Markov chain, followed by uniformization), and the redundancy allocation exact (each design's chain, its copies queueing for the crews), but the allocations of availability targets; common-cause groups in the same chain, a strike's members queueing in the diagram's order; other lives simulated. The maintenance and test intervals are chosen as if every repair started at once on request (`assume_unlimited_crews=True`), for a plan to simulate with the crews | Other lives: yes, in general. |
| Standby groups (a duty unit and its spares, repaired) | For exponential units, the long run and importance exact, and the values over time numerical (the units' Markov chain, followed by uniformization); other units simulated, but a cold pair (one needed), which is numerical: with one repair at a time its cycle, in the long run and, with a sure switch, over time (a renewal equation on a grid); with both repaired at once a Markov renewal process on the other unit's elapsed repair, in the long run | Other units: yes, in general, but for the pair; its values over time with both repaired at once or a switch that can fail, warm spares and more units could be numerical. |
| Markov components (states and the rates between them) | The long run, importance and long-run capacity exact (the chain's long-run distribution), and the values over time, the expected events and costs over a window and the capacity over time numerical (the chain followed from its start by uniformization); the simulations draw its times in each state and its next states, in Python. Refused: repair crews in its diagram, membership of a common-cause group, maintenance, tests and spares on it (write them as states and rates), the exact first failure (bracketed by `first_failure_bounds`), the spares analyses, and a simulation that follows its states' capacities | No: the chain is exact. The first failure of a diagram with one is the joint chain with its failed states made absorbing; the simulations could follow the capacity of the state it is in. |
| Opportunistic maintenance (renewals at a group's stops) | Simulated | Yes: each member's renewals depend on the others' ages. |
| Imperfect repair (Kijima), with or without replacement at the *N*-th failure | Simulated; but minimal repair (`q = 1`, or a fitted Crow-AMSAA, Duane or Cox-Lewis process) is numerical over a window from new, and its long run, replaced at the *N*-th failure or at an age, is exact (a renewal cycle; Cox-Lewis's with a failure limit by quadrature, #402); without a replacement its long run is refused | Yes, in general: a repair does not renew the unit. Minimal repair's values over time with replacement could be numerical, as a renewal of its cycle. |
| Preventive maintenance that restores a unit only partly (Kijima), with or without replacement at every *N*-th task | Simulated (availability, cost, timelines and the spares used, in Python), from new or from the unit's virtual age; the exact and numerical analyses refused | Yes, in general: a task does not renew the unit, so its failures are not a renewal process's. |
| Shutdowns: outages that hold the system or some blocks off line for a fixed time, the blocks' work done in them | On a period, over blocks that fail independently: exact or numerical as the blocks' own values are (the long run, the values over time, the expected events, costs and NPV), "older than" numerical; the simulations follow the windows, in Python. At given times (outages on a timeline), the same over a horizon where the blocks worked on are renewed at each, or kept unless due (numerical, stretch by stretch), and no long run (refused: the outages end); so too a block renewed at several shutdowns, each renewal its own shut's length. With a restoration at a shut, a block tested at given times, or with blocks tied together (crews, common causes, maintenance groups): simulated. The importance measures, capacities, first failure, spares, allocations, parameter uncertainty and interval searches refused for now | A block tested at given times: no, its recursion could take the times as the renewed and the kept block's do. A restoration: yes (the unit's virtual age is not a renewal's). The rest are not taken in yet. |
| Planned against unplanned downtime, and the failures a plan avoids (`maintenance_effect`) | Exact or numerical where the diagram's and its run-to-failure twin's values over a window are, over blocks that fail independently: each block's probability of having failed over time, beside its downtime, from its curve (maintenance and timed tests at age or on a calendar, shuts on a period or at given times, nested RBDs), in the long run too (where there is one); the simulations follow it path by path, on both engines. With planned outages and limited repair crews or common-cause groups, or with partial restoration: simulated | Crews and common-cause groups, no: their chains could keep a block in planned work apart from one in a repair, as a single block's curve does. |
| Replacement on condition by a measured degradation level | Numerical, from new, from a state (a level measured now, new on its calendar, or in a repair or replacement) and in the long run: the values over time and the long run follow the level's distribution on a grid of levels from one inspection to the next, as replacement on condition by age follows ages; the simulations (in Python) give the spread | No: between inspections the level is a Markov process. |
| An upgraded part (another life from a time) | Numerical over a horizon (the availability, expected events, costs and NPV), from new or from a state, on its own or with block or age replacement, or with tests and repairs in no time that find every failure: the unit in service replaced at the upgrade; kept in service until its next failure or due replacement (`"replace_in_service": False`) on its own or with age replacement, each unit with the life in service when it is put in; the simulations follow it, in Python; the long run is that once upgraded. Kept in service with block replacement or tests, or tested with tests or repairs that take time or tests that miss failures: simulated, the values over a horizon refused | No: a unit kept under block replacement is the stretch with the upgrade split as age replacement's is, and the tests that take time could be followed on each side of it, as tests in no time are. |
| A shelf of spares (a component's `"spares"`: a stock reordered one for one, each order arriving a lead time later) | Simulated (availability, cost, timelines, the spares used and the stock to hold, in Python; the compiled engine refuses it); the exact and numerical analyses refuse the node, but those of the node probabilities given or of the structure alone | Yes, in general: a repair that finds the shelf empty waits for the next order to arrive, so each unit's downtime depends on the demands before it. A unit replaced at a constant rate with no stock is a renewal with its repair lengthened by the lead time, and a stock that never runs out is no shelf. |
| Spares with repair crews, standby groups, opportunistic maintenance, imperfect repair or partial restoration | Refused, pointing to `spares_demand(method="simulate")`. Otherwise numerical, demand and stock: block-replaced components' whether their repairs and block replacements take time or not (from a typical replacement in the long run), but for one dead on arrival while they may take none, and tested components', whatever their tests and repairs take and whether their tests miss failures | Yes, in general: a crew's queue, a group's switching or stops, an imperfect repair and a partial restoration make the replacements depend on more than each unit's own lives. A unit dead on arrival whose renewals may take no time: no, the replacements at one instant could be counted. |

Where a system's expected values over a window are exact,
`availability()` and `cost()` give them by default, with no error, and the
simulations give their spread. Where a system needs simulating for a few of
its nodes (a standby group of other lives, a nested RBD sharing a crew, a
unit repaired imperfectly or restored partly by its maintenance), their mean intervals by default take each
simulation's expected values given those nodes' histories, every other node
exact given their states: a fraction of the error of the simulations' own.
`conditional=True` simulates only those nodes, for the same means in less
time.

For your own diagram, `analysis_routes()` says how each analysis will be
computed (exact, numerical, simulated or refused) and why, without running
anything; a refusal's message names the simulation to run instead. The
[guide](https://derrynknife.github.io/RePyability/guide/saving/#what-is-exact-and-what-is-simulated)
lists every method's route.

## Documentation
The full documentation — tutorial, user guide, concepts and API reference — is
hosted at
**[derrynknife.github.io/RePyability](https://derrynknife.github.io/RePyability/)**.

It is built with [MkDocs](https://www.mkdocs.org/) from the sources in `docs/`
and `mkdocs.yml`, and published to GitHub Pages on every push to `main`. To
build and serve it locally:

```bash
pip install -e .[docs]
mkdocs serve            # then open http://127.0.0.1:8000
```

The source lives in `docs/` and `mkdocs.yml`; start with `docs/index.md`.

## User stories

[`stories/`](stories/README.md) holds the questions users and agents should
be able to answer with RePyability, with how we will know they are answered.
Questions and bug reports are welcome in the
[issue tracker](https://github.com/derrynknife/RePyability/issues).

## Related

[Reliafy](https://reliafy.com) is a hosted, browser-based app built on RePyability and SurPyval, for building reliability block diagrams and analysing their reliability and availability without writing code.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md): setting up, the checks a change
needs (`scripts/check.sh`), and the parts of the package that must agree
with each other.
