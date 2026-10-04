# RePyability roadmap

The bigger ideas for RePyability: features that need design, research or
several weeks of work. They live here, not in the issue tracker, so the
issues stay a list of things someone could start this week. The outcomes
they serve, and how we will know they are met, are the user stories in
[`stories/`](stories/README.md).

## How this file works

- **An idea starts here**, as a section with the problem, why it matters,
  its rough size, what it depends on, and a status:
  - *idea*: worth doing, not designed;
  - *designed*: the approach is settled (the design is summarised here);
  - *scheduled*: broken into issues that are being worked on.
- **When an idea is scheduled**, its design is split into a few actionable
  issues, linked from its section here.
- **When it is done**, its section moves to "Done" with a line and the
  release; the changelog is the full record.
- **Ideas we have decided against** go under "Not planned", with the
  reason, so they are not raised again.

---

## The NPV simulator: what is a maintenance plan worth?

*Status: designed. Size: large (phased, several weeks). Stories: MX-01 to
MX-17 in [`stories/maintenance-economics.md`](stories/maintenance-economics.md).*

**Problem.** Maintenance plans, spares, crews and redundancy are decided on
money, at a cost of capital, over an asset's life. RePyability prices a
plan per hour in the long run; 0.12 added a `discount_rate` to
`total_cost` and `allocate_redundancy` that discounts that long-run rate.
It does not value a plan from where the plant is, over a calendar, with
what the plant earns while it runs. The question users and agents ask is
"which plan creates the most value, at the least cost, and how sure are
we?", and today they answer it by hand.

**Why it can be exact.** By linearity of expectation the expected present
value of a cost charged at events is the cost times the discounted expected
number of events, and of a cost or value accruing per hour the discounted
integral of the (un)availability:

```text
E[PV(events)] = c * integral_0^H exp(-delta t) dM(t)
PV(value)     = integral_0^H exp(-delta t) * v(t) * A(t) dt
```

`M(t)` and `A(t)` are what `expected_events(t)` and `point_availability(t)`
already give exactly, from new or from a state, with maintenance,
inspections and (for exponential units) crews. So the expected NPV is exact
wherever those are, and simulation is needed only for what is not linear
in the events: percentiles, the chance of missing a hurdle, availability
contracts with penalties. The equivalent annual cost `EAC = delta * PV`
reduces to `expected_cost_rate` as `delta -> 0`, a regression check for
every plan. A prototype on `expected_cost(t)` (two Weibull pumps feeding a
compressor, 10 years at 8%) moved the best pump PM interval from 10,000 h
undiscounted to 14,000 h: deferring spend is worth something.

**Design.** Three layers, all plain data so they save and load:

- **Plan**: the actions, mixed freely. Periodic maintenance (as today);
  outages on dated calendars with a duration and a scope (renew some
  components, restore others to a virtual age, take the system down);
  recurring campaigns with opportunistic work; upgrades that swap a
  component's life model at a capital cost; crews and call-out contracts;
  retirement with a salvage value from each component's age. Dated
  renewals generalise block replacement from multiples of an interval to
  any list of times, so the engine work starts from `_block_replacement`.
- **Cash flows**: value (a rate, capacity times a price curve, any `v(t)`),
  event costs (repair, replace, preventive, inspection, set-up), rate costs
  (downtime, crews, holding spares), capital (acquisition, upgrades,
  salvage), each with its own escalation; real or nominal rates.
- **Valuation**: the rate, the horizon, a calendar that maps model time to
  dates, and the method: exact where the curves are (fixed-time events
  discounted at their exact times, not grid midpoints), simulated where
  needed (each event discounted at its time, plans compared with common
  random numbers). The result: NPV, EAC, the undiscounted total, a
  cash-flow table by year, category and component, how it was computed,
  and the units.

**Phases** (each becomes issues when scheduled; the stories' building
blocks B1 to B7):

1. Expected NPV and EAC on the exact curves, with a constant value rate and
   the cash-flow table (B1). Validated against Fox's (1966) closed form
   for discounted age replacement and the `delta -> 0` limit.
2. Value and prices in time, escalation and the calendar (B2).
3. Plans on a calendar: dated outages, campaigns, upgrades, salvage (B3).
4. The simulated NPV distribution and non-linear payoffs (B4).
5. Crews as cash flows and the call-out option, valued in the crews'
   Markov chain with "surge crew active" states (B5).
6. `objective="npv"` / `"eac"` in the optimisers, with their constraints,
   and rho, the discount-rate sensitivity (B6); parameter uncertainty in
   the NPV (B7, on 0.12's #200).

**For agents** (B10). Every input and result is JSON with units; every NPV
analysis is in `analysis_routes()`; refusals say what to do; a valuation is
reproducible from its saved inputs. The 0.12 persona check's integrator
findings (#223, #231, #235) are the starting list.

**Depends on.** Nothing new in SurPyval for phases 1 to 6.

---

## Live NPV: keep the plan's value current as the plant changes

*Status: designed (architecture below; details settle in phase 1 of the
NPV simulator). Size: large. Depends on: the NPV simulator. Stories: MX-18
to MX-25.*

**Problem.** A plan's value changes every time the plant does: a vibration
reading rises, a pump is replaced, a seal fails, new failure data narrow a
fit, the forward curve moves. The aim is that RePyability holds the plant
as it is now and re-values (and, when asked, re-optimises) the plan as each
event arrives, so that Reliafy can keep every maintenance plan optimal by
NPV through all the churn of real maintenance work, and an agent can say at
any moment what the plan is worth, what changed it, and what to do next.

**What exists.** Every exact analysis of a `RepairableRBD` starts from the
components' current states (`state=`, `NodeState`: age, up or down, how
long down, whether in maintenance, the phase on a calendar, or
stationary), and `simulate_timelines` does too (#163). A `NonRepairableRBD`
gives the reliability, remaining life and importance given those states.
What is missing is the state that sensors and repair histories give, the
events that move it, and a valuation that follows it.

**Design.**

- **A live plant object** (working name `Plant`): the `RepairableRBD`, the
  `Plan`, the `Valuation`, each component's current state, and the log of
  events, as of a time. It is event-sourced: the state is the fold of the
  events, so events can arrive late or out of order (a work order closed a
  week after the job) and the result is the same as in order, and any past
  valuation can be rebuilt.
- **Events**:
  - `reading(node, level, at)`: a condition measurement. The component's
    life becomes its remaining-life distribution from that level, from a
    SurPyval degradation model (gamma or Wiener process, or a path model);
  - `replaced(node, at)`: age 0; `repaired(node, at, virtual_age=)`: an
    imperfect repair (Kijima, with SurPyval's fitted restoration factor);
  - `failed(node, at)`: down, queued for a crew if crews are limited,
    spares drawn from stock; `inspected(node, at, found=)`: a test of a
    hidden failure;
  - `refit(component_type, model)`: a new life model (with its covariance)
    for every component of that type, each keeping its own age or
    condition;
  - `prices(curve)`, `plan(changes)`, `covariates(node, values, at)` (a
    load or temperature change for a covariate-dependent component).
- **State.** `NodeState` gains what the events need: a condition (the
  conditioned life model it implies), a virtual age, and a per-node model
  override; or a richer per-node state object that carries them, so that
  `age` keeps its single meaning (operating time).
- **Incremental re-valuation.** Each component's curves (availability,
  expected events) depend only on its own model, state and plan, so they
  are cached by that key and an event recomputes only the components it
  touched; independent components recombine through the structure
  function, and dependent modules (crews, standby groups, common-cause
  groups) are recomputed module by module, as 0.12's conditional runs
  already split them (#189). A price change re-values without touching the
  reliability at all. The live result must equal a valuation from scratch
  with the same state, to the exact methods' precision. Target: about a
  second per event for a 50-component plant with exact curves.
- **Attribution.** Each event's effect on the NPV is reported (the
  difference between successive valuations), and the effects since any
  time add up to the total change, so "why did the plan's value drop?" has
  an answer by event and component.
- **Re-optimisation** (receding horizon): from the state now, the plan
  that maximises NPV within crews, outage windows, budgets and
  availability or PFDavg limits, warm-started from the current plan;
  changes proposed only when they beat the current plan by a margin, so
  the plan does not churn on sensor noise, each with its value ("bring pump
  2's replacement forward to the March outage: +$45k").
- **Change report.** For each update: the NPV move, whether it crossed a
  threshold, and whether the recommended next actions changed and why.
  Reliafy turns that into alerts.
- **Reproducibility.** The plant, its events and each valuation save to
  JSON; a saved valuation reproduces exactly (with its seed where it
  simulates).

**Who does what.**

- **SurPyval**: a degradation model's remaining life from a current level
  as a life model RePyability can take (`sf`, `ff`, `qf`, `mean`,
  `random`); per-unit virtual age from a repair history (there since
  0.23, #615); cheap refits or sequential updates as failures arrive, with
  the covariance. (See SurPyval's roadmap.)
- **RePyability**: the plant object, the state, the valuation, the
  attribution, the re-optimisation, the change report.
- **Reliafy**: ingesting readings and work orders, keeping each plant,
  running updates, alerts, and the MCP tools that let agents ask "what is
  the plan worth now, what changed, what should we do?".

**Open questions.**

- Time base: calendar time against operating hours (components that run
  part-time age by their run hours); the plant needs a utilisation per
  component to map one to the other.
- What a live update does when an exact route is refused (a crew-limited
  plant of non-exponential units): simulate with common random numbers to
  a tolerance, and say so.
- Covariate-driven condition (temperature, load) against degradation
  signals: both are "condition", through different life models
  (`RegressionNode` against a degradation model's remaining life).
- How far to trust a re-optimisation made on uncertain models: report the
  decision's robustness (does the best action change across the parameter
  draws of B7?).

---

## Decision analytics on the NPV

*Status: idea. Size: medium each. Depends on: the NPV simulator.*

Questions that become one or two calls once plans have a value:

- **Price-aware outage timing**: place planned outages against a forward
  price curve, not only choose how often (MX-10, MX-23).
- **Value of information**: what better data are worth (more teardowns, a
  sensor), from the NPV lost to parameter uncertainty (MX-15; on #196 and
  #200).
- **Real options**: run, refurbish, replace or life-extend as decisions
  taken over time from the state now (MX-08, MX-12).
- **Fleet capital rationing**: across many plants, the portfolio of
  interventions with the highest NPV for a budget.
- **Procurement break-even**: the most a more reliable part is worth, or
  the life it needs at a quoted price (MX-07).
- **NPV sensitivities**: rho, and the Greeks of the NPV rather than of the
  availability (0.12's #192 machinery with the NPV as the quantity),
  ranked per dollar.
- **Spares and maintenance together**: stock levels and intervals chosen
  jointly, since each changes the other's value (MX-14).
- **Shadow prices**: carbon or safety costs as cash flows, so "value" is
  more than margin.

---

## Simpler implementations

*Status: scheduled. Issues: #204 (umbrella), #205, #206, #207, #208,
#209.*

Remove the code that is there only for speed where one clean design does
as well: one copy of a component's events in the compiled loop (#206);
timelines from the event loop's recording only (#205); one design of the
decision-diagram search and replay (#207); one implementation of sorting a
run's changes (#208); random streams whose draws depend only on the seed,
the stream, the simulation and the draw (#209). Fewer copies that must
agree means fewer places for the bugs that review rounds keep finding.

---

## Ongoing practice: hardening

Hardening is continuous, not a one-off review before 1.0. It is how every
change is made:

- **Review rounds.** A periodic persona check of the installed release,
  done as users (and agents) use it: a plant engineer, a safety analyst, a
  statistician, an integrator and a new user following the docs, each
  answer checked independently. Each round files concrete issues, fixed in
  the next. Rounds so far: 0.11 (#184), 0.12 (#213).
- **Scenario cards in the regular tests.** Each persona's study becomes a
  card in `repyability/tests/scenarios`: its questions answered through
  the public API, checked against an independent oracle (a closed form,
  a Markov chain, an enumeration), with each question not yet answerable
  as a strict xfail citing its story or issue, so a fix shows up as a
  failing run until the card is updated. They run on every push, so they
  are small and exact (SurPyval's practitioner cards are the model). The
  stories in `stories/` are the cards' source.
- **Every analysis on every kind of diagram.** The routes test covers each
  node kind and spec key, with a guard that fails when one is missing
  (#239).
- **The docs say what the code does.** The docs test checks every shown
  output, not only `# ->` comments; tests run from the installed wheel
  (#167).
- **Concurrency.** Simulations on a shared diagram from several threads
  (#216).

---

## Not planned

- **Fitting lifetime models in RePyability.** Fitting stays in SurPyval;
  RePyability takes fitted models (CLAUDE.md).
- **Plotting and dashboards.** These stay in Reliafy; RePyability stays
  free of plotting dependencies.

---

## Done

- **The Greeks** (0.12, #197): the sensitivity measures as one family:
  delta (#191), parameter sensitivity (#192), differential importance
  (#193), gamma (#194), theta and Barlow–Proschan (#195, #199), vega
  (#196, #200).
- **Discounted total costs and train-wise redundancy** (0.12, #184): the
  first step of the NPV simulator.
- **Exact by default** (0.12, #186, #187, #189): a run's expected values
  exact, or taken given its dependent modules.
