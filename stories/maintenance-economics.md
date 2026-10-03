# Maintenance economics: the net present value of a plan

Stories MX-01 to MX-17. Status: all **open** or **partial** as of 0.11
(October 2026).

## Why

Every cost RePyability computes is undiscounted. `expected_cost_rate`,
`total_cost` and `expected_cost` all say so, and `allocate_redundancy`
states "Discounting is not modelled". But the people who decide on
maintenance plans, spares, crews and redundancy decide on money, at a cost
of capital, over a horizon. Over 20 years discounting changes which option
wins. The end-to-end pump-station study in #184 (item 3) found exactly that
and had to discount by hand.

The question they ask is not "what is the cheapest plan per hour in the
long run" but "**which plan creates the most value, at the least cost,
over the life of the asset, and how sure are we?**"

## Background

### The expected NPV is a linear functional of what is already exact

By linearity of expectation, the expected present value of a cost charged
at events is the cost times the discounted expected number of events:

```text
E[PV] = c * integral_0^H exp(-delta t) dM(t),      delta = ln(1 + r)
```

with `M(t)` the expected number of those events by `t`, which
`expected_events(t)` already gives exactly on a grid of times. Costs and
value that accrue per hour discount against the point availability:

```text
PV(downtime cost) = integral exp(-delta t) * c_d * (1 - A(t)) dt
PV(value)         = integral exp(-delta t) * v(t) * A(t)       dt
```

So expected NPV needs only `A(t)` and `M(t)`, which the engine already has
for independent components, maintenance, inspections and (for exponential
components) repair crews. Two consequences:

- **The long-run rate is not enough.** Discounting weighs the early years
  most, and from new a wear-out system fails less than its long-run rate.
  A prototype on top of `expected_cost` (two Weibull pumps in parallel
  feeding a compressor, 10 years at 8%, 900 per hour of margin) put the
  undiscounted cheapest pump PM interval at 10,000 h and the highest-NPV
  one at 14,000 h: deferring spend is worth something.
- **The like-for-like objective is the equivalent annual cost**,
  `EAC = delta * PV` (or the annuity form over a finite horizon): the
  discounted counterpart of `expected_cost_rate`, comparable across plans
  of different lengths. As `delta -> 0` it must reduce to the existing cost
  rate, which gives every story a free regression check.

Distributional questions (percentiles, the chance of missing a hurdle rate,
contract penalties on annual availability) are not linear in the events
and need the simulated distribution, each event discounted at its own time.

### Building blocks

The stories need these. They are capabilities, not a plan of work; one
change may deliver several.

| ID | Building block |
|---|---|
| **B1** | Expected NPV and EAC on the exact curves: a constant value rate, event and rate costs, acquisition cost; a cash-flow table by year, category and component. Fixed-time events (block or dated maintenance) discounted at their exact times, not at grid midpoints. |
| **B2** | Value that varies in time: a price curve, capacity times price, any `v(t)`; escalation per cost stream; real or nominal rates; a calendar mapping model time to dates. |
| **B3** | Planned maintenance on a calendar: outages at given dates, with a duration and a scope (renew some components, restore others to a virtual age, take the system down); recurring campaigns with opportunistic work; mid-life upgrades that swap a component's life model at a capital cost; retirement with a salvage value from each component's age. |
| **B4** | The simulated NPV distribution: each simulated event discounted at its time; percentiles, the chance of missing a hurdle, conditional value at risk; non-linear payoffs such as availability-contract penalties; plans compared with common random numbers. |
| **B5** | Crews as cash flows and as options: crew salaries as rate costs; a contract crew called in when the repair queue reaches a threshold, at a call-out fee and a premium rate. |
| **B6** | `objective="npv"` / `"eac"` in the optimisers (`optimal_replacement_intervals`, `optimal_inspection_intervals`, `allocate_redundancy`), with constraints kept (availability, PFDavg); the discount-rate sensitivity (rho). |
| **B7** | Parameter uncertainty carried into NPV: the fitted models' uncertainty propagated to the NPV and to the decision (see the uncertainty-importance issue, #196). |

Valuing from the components' current states uses the existing `state=`
(`NodeState`), which the NPV must accept wherever the curves do.

### Shape of an answer

Not an API decision, but what every story's answer should carry:

- the **NPV** (value less costs, or costs alone when no value is given),
  the **EAC**, and the undiscounted total for comparison;
- a **cash-flow table** by year, category (repair, replace, preventive,
  inspection, downtime, setup, crews, capital, salvage, value) and
  component, so users can apply their own tax, depreciation or WACC
  conventions outside the package;
- how it was computed (exact, numerical or simulated, as
  `analysis_routes()` reports for every other analysis), and a clear
  refusal, with the reason, where it cannot be.

---

## Maintenance engineer

### MX-01: Does my PM interval still hold at our cost of capital?

- **Persona:** a maintenance engineer setting a component's age-replacement
  interval, asked by finance to justify it at the company's WACC.
- **Story:** As a maintenance engineer, I want the replacement interval
  that minimises the discounted cost, so that the interval I set is the
  one the business case assumes.
- **In their words:** "What's the best replacement age for this bearing at
  an 8% discount rate, and how much does it differ from the undiscounted
  one?"
- **Setting:** the bearing of Lesson 8 and the maintenance guide:
  `NonRepairable(surv.Weibull.from_params([1000, 2.5]))` with
  `cp = 1`, `cu = 5`, whose undiscounted optimum is 493.0 h at a cost rate
  of 0.003462 per hour.
- **Acceptance criteria:**
  - The optimal interval and its EAC at 0%, 8% and 15% a year (time unit
    stated), and the EAC curve over a range of intervals.
  - At 0% the answer equals `optimal_replacement_policy()` (493.0 h,
    0.003462) to the optimiser's tolerance.
  - It matches Fox's (1966) closed form for discounted age replacement
    over an infinite horizon, integrated independently:
    `PV(T) = [c_u * int_0^T e^(-delta t) f(t) dt + c_p * e^(-delta T) R(T)] / (1 - L(delta))`,
    `L(delta) = int_0^T e^(-delta t) f(t) dt + e^(-delta T) R(T)`.
  - The same question at system level (`RepairableRBD` with
    `"preventive"`) agrees with the component answer for a single node.
- **Needs:** B1, B6.
- **References:** Fox (1966), *Age replacement with discounting*;
  Jardine & Tsang, *Maintenance, Replacement and Reliability*, ch. 2–3.
- **Status:** open.

### MX-02: Overhaul the gearbox now, or defer it to next financial year?

- **Persona:** a maintenance planner with a repairable gearbox under
  minimal repair and a budget that resets on 1 July.
- **Story:** As a maintenance planner, I want the value of each overhaul
  date, so that I can defend deferring (or not deferring) an overhaul
  across a budget boundary.
- **In their words:** "If I push the gearbox overhaul from May to August,
  what does it cost us in expected failures, and what do we gain from
  spending later?"
- **Setting:** the gearbox of the maintenance guide (`Repairable`, minimal
  repair, `optimal_overhaul_policy().cost_rate` 0.5848), on a calendar
  with the overhaul as a dated action.
- **Acceptance criteria:**
  - The NPV of the plan for each candidate overhaul date, the difference
    between dates, and the expected number of failures before each.
  - With no discounting and the overhaul on its optimal cycle, the cost
    rate equals `optimal_overhaul_policy().cost_rate`.
  - An independent check: the expected failures before the overhaul from
    the power-law cumulative intensity, discounted by hand.
- **Needs:** B1, B2 (calendar), B3 (dated overhaul).
- **Status:** open.

### MX-03: Bundle the compressor overhaul into the pump shutdown?

- **Persona:** a maintenance planner at a plant where every stop has a
  fixed set-up cost (isolation, permits, scaffolding, lost production).
- **Story:** As a maintenance planner, I want to compare separate stops
  with one campaign that does both jobs, so that I can see whether sharing
  the set-up cost is worth doing some work early or late.
- **In their words:** "Should we do the compressor overhaul in the same
  shutdown as the pump replacements every four years, instead of on its
  own schedule?"
- **Setting:** a pump pair and a compressor in series (as in the
  prototype), with a maintenance group's set-up cost charged once per stop
  (the existing `"group"` and `"opportunity"` keys), compared with a
  four-yearly campaign.
- **Acceptance criteria:**
  - The NPV and EAC of each plan, with the set-up cost and downtime shown
    separately in the cash-flow table.
  - The campaign interval (and which jobs join it) that maximises NPV.
  - Undiscounted, the separate-stops plan's cost rate equals today's
    `expected_cost_rate` for the same group.
- **Needs:** B1, B3 (campaigns), B6.
- **Status:** partial: groups and opportunistic maintenance exist
  undiscounted; campaigns on a calendar do not.

### MX-04: Proof-test intervals under a SIL target, at least cost

- **Persona:** a functional safety engineer who must meet a PFDavg target
  and would like to spend as little as possible doing it.
- **Story:** As a functional safety engineer, I want the test plan with
  the lowest present cost that still meets the PFDavg target, so that
  compliance is met without over-testing.
- **In their words:** "What's the cheapest proof-test plan for the two
  pressure switches that keeps PFDavg under 1e-4, and should the tests be
  staggered?"
- **Setting:** the 1oo2 trip of #184 (item 4) with a beta factor of 10%,
  where staggering two annual tests by six months takes PFDavg from 9.6e-4
  to 4.9e-4.
- **Acceptance criteria:**
  - The plan (intervals and offsets) minimising PV of tests plus downtime
    subject to PFDavg <= target, with its PFDavg and PV.
  - The offset (at least the half-interval stagger) is searched, not fixed
    at 0.
  - PFDavg checked against the closed form for a 1oo2 group with beta
    (IEC 61508-6, annex B), computed independently.
- **Needs:** B1, B6 (constrained, with offsets).
- **References:** IEC 61508-6; #184 item 4.
- **Status:** partial: `optimal_inspection_intervals` meets the target
  undiscounted, without offsets.

---

## Asset manager and finance

### MX-05: Is the standby pump worth buying?

- **Persona:** an asset manager writing the business case for a capital
  purchase, doing life-cycle costing.
- **Story:** As an asset manager, I want the NPV, payback and IRR of adding
  a standby unit, so that the capital request states its return the way
  finance expects.
- **In their words:** "We're thinking of putting in a standby pump.
  Over 15 years at 8%, does it pay for itself, and when?"
- **Setting:** `alone` and `with_standby` from the costs guide (Weibull
  1000/2.5 pumps, 5000 per replacement, 1000 per preventive replacement,
  500 per hour of lost production; best cost rates 13.14 alone and 6.985
  with the standby), with an acquisition cost for the second pump.
- **Acceptance criteria:**
  - The NPV of each design, the incremental NPV of the standby, its
    discounted payback year and its IRR (the rate at which the incremental
    NPV is zero).
  - Each design at its own best PM interval (the interval changes with the
    design, as the guide shows).
  - The yearly cash-flow table reconciles to the NPV.
  - An independent check: discounting `expected_cost(t)` on a fine grid by
    hand gives the same PV within the grid's error.
- **Needs:** B1, B6.
- **References:** IEC 60300-3-3 (life cycle costing); ISO 15663.
- **Status:** partial: the undiscounted comparison exists; the prototype
  discounting of `expected_cost(t)` works by hand.

### MX-06: Should we add a fourth pump train?

- **Persona:** an asset manager deciding on redundancy for a cooling-water
  pump station over a 20-year life.
- **Story:** As an asset manager, I want redundancy allocation to optimise
  NPV, so that the design it recommends is the one with the best return,
  not the lowest undiscounted cost.
- **In their words:** "Two-of-three or two-of-four pump trains? Over 20
  years at 8%, which is worth more?"
- **Setting:** the 2-of-3 station of #184, with lives fitted in SurPyval
  from censored field data, one repair crew, PM and proof tests; the
  alternative is a fourth train (seal, bearing and motor in series).
- **Acceptance criteria:**
  - The NPV of each design and the winner at 0% and at 8%; the story
    exists because the winner can flip.
  - `allocate_redundancy(..., objective="npv")` (or equivalent) returns the
    same design as comparing the NPVs by hand.
  - A whole train (a set of nodes) can be the unit copied (#184 item 2).
- **Needs:** B1, B6; per-train allocation (#184 item 2).
- **Status:** open.

### MX-07: What's the most we should pay for Vendor B's valve?

- **Persona:** a procurement or reliability engineer comparing quotes for
  a more reliable part.
- **Story:** As an engineer comparing vendors, I want the break-even price
  of a more reliable component, so that I know the most a reliability
  improvement is worth before negotiating.
- **In their words:** "Vendor B's valve has a characteristic life 40%
  longer. It costs more. What's the most we should pay for it?"
- **Setting:** the plant of Lesson 7 (two pumps and a valve;
  `expected_cost_rate()` 121.23, and 44.63 with the valve perfect), with
  Vendor B's valve as a different Weibull.
- **Acceptance criteria:**
  - The PV of running cost with each valve, and the break-even price
    difference (the PV saved).
  - An upper bound: the PV saved by a perfect valve
    (`working_nodes=["valve"]`), which no real valve can exceed.
  - The inverse question: the life Vendor B's valve must have to break
    even at a quoted price.
- **Needs:** B1.
- **Status:** partial: the undiscounted saving per hour is available today.

### MX-08: Refurbish the compressor in year 8, or replace it with the new model?

- **Persona:** an asset manager planning mid-life works.
- **Story:** As an asset manager, I want to value a mid-life upgrade that
  changes a component's life model, with salvage at the end, so that
  refurbish, replace and run-on can be compared on one basis.
- **In their words:** "In year 8 we can refurbish the compressor or put in
  the new model. Which is worth more over the remaining 12 years?"
- **Setting:** the compressor of the prototype (Weibull 30,000 h, 2.2),
  with a refurbishment that restores it to a virtual age (Kijima) and a
  replacement with a different Weibull at a capital cost.
- **Acceptance criteria:**
  - The NPV of run-on, refurbish and replace, with salvage value at the
    horizon from each component's age.
  - Retiring early is not free: the salvage value enters the cash-flow
    table.
- **Needs:** B1, B3 (upgrades, salvage).
- **Status:** open.

### MX-09: What's our P90, and how likely are we to breach the availability contract?

- **Persona:** a commercial manager whose contract pays a penalty when a
  year's availability falls below 95%.
- **Story:** As a commercial manager, I want the distribution of NPV and
  the expected penalty, so that I can price the contract's risk and report
  a P90 to the board.
- **In their words:** "What's the P10/P50/P90 NPV of this plant over the
  contract, and what's the chance we pay the availability penalty in any
  given year?"
- **Setting:** any of the plants above, with a penalty per percentage point
  of annual availability below 95%.
- **Acceptance criteria:**
  - The NPV percentiles, the chance of missing a hurdle NPV, and the
    present value of expected penalties.
  - The penalty is computed from each simulated year's availability, not
    from the mean availability (the story exists because those differ).
  - The simulated mean NPV agrees with the exact expected NPV (B1) within
    its confidence interval.
  - A hand-written Monte Carlo of the yearly availabilities agrees within
    its confidence interval.
- **Needs:** B1, B4.
- **Status:** open.

---

## Operations and trading

### MX-10: Which month should the outage go in?

- **Persona:** a generator's asset manager working with the trading desk.
- **Story:** As an asset manager, I want to value a planned outage at
  each candidate date against the forward price curve, so that the outage
  goes where it costs the least margin, net of the extra failure risk of
  waiting.
- **In their words:** "We need a 21-day outage next year. Which month
  costs us least, given the forward curve and the chance the unit fails
  before we get to it?"
- **Setting:** a generating unit with a wear-out (Weibull) life, a 21-day
  planned outage that renews it, and a seasonal forward price curve.
- **Acceptance criteria:**
  - The NPV for each candidate start month, the best month, and its value
    over the worst.
  - The probability of a forced outage before the planned one, for each
    month.
  - An independent check: enumerating the twelve months by hand, with the
    lost margin of each outage window from the curve.
- **Needs:** B1, B2 (price curve, calendar), B3 (dated outage).
- **Status:** open.

### MX-11: Is it worth running hot?

- **Persona:** an operations manager who can raise output at the cost of
  faster wear.
- **Story:** As an operations manager, I want to value running at a higher
  load against the life it consumes, so that I can decide when the extra
  margin is worth it.
- **In their words:** "If we run at 70 °C instead of 30 °C we make more,
  but the bearings wear out faster. Does it pay?"
- **Setting:** the load schedule of the condition-based guide (30 °C for
  2000 h, then 70 °C), with value proportional to load.
- **Acceptance criteria:**
  - The NPV of each operating schedule, with value and maintenance cost
    separated.
  - The switch-over time that maximises NPV.
- **Needs:** B1, B2; covariate-dependent lives in a `RepairableRBD`, which
  today only the non-repairable side supports.
- **Status:** open.

### MX-12: Pump 1 is at 18,000 hours. Replace it at the March outage, or run it?

- **Persona:** a reliability engineer making a live decision about one
  unit, from its history.
- **Story:** As a reliability engineer, I want to value the options for a
  unit from its current condition, so that the decision uses what we know
  about this unit, not a new one.
- **In their words:** "Pump 1 has run 18,000 hours since its last repair.
  Do we replace it at the March outage, refurbish it, or run it to
  failure?"
- **Setting:** a repairable pump whose repairs are imperfect, with the
  truth of SurPyval's repairable-fleet scenario card (Kijima type I,
  `q = 0.4`, lives Weibull 30 months, 2.5), and a dated outage in March.
- **Acceptance criteria:**
  - The NPV of each option from the unit's current virtual age
    (`state=`), with salvage at the horizon.
  - An independent check: simulating forward from the virtual age by hand.
  - The fit side (the virtual age from the unit's own history) comes from
    SurPyval (its #581); RePyability takes the state.
- **Needs:** B1, B3; SurPyval #581.
- **Status:** open.

---

## Crews, spares and data

### MX-13: One crew, two crews, or a call-out contract?

- **Persona:** a maintenance manager at a plant run with one crew.
- **Story:** As a maintenance manager, I want the value of a second crew
  and of a call-out contract, so that I can choose between hiring, keeping
  a contractor on retainer, or neither.
- **In their words:** "Is a second crew worth its salary? What if we only
  call a contractor when two pumps are down at once?"
- **Setting:** a 2-of-3 pump station of exponential units with one repair
  crew (as in #184 item 5); the contract crew is called when the repair
  queue reaches 1 or 2, at a call-out fee and a premium hourly rate.
- **Acceptance criteria:**
  - The NPV with 1 and 2 crews, net of crew cost; the NPV with the call-out
    at each threshold, and the best threshold.
  - The option's value: NPV with the call-out minus without, compared with
    the cost of a permanent second crew.
  - An exact independent check: the continuous-time Markov chain built in
    the test with scipy, its discounted reward
    `PV = e_0^T (delta I - Q)^(-1) r` from the all-up state, sharing no
    code with `_crew_chain.py`.
- **Needs:** B1, B5.
- **Status:** partial: `expected_cost(t)` already runs over time with
  `repair_crews` for exponential components, so the static crew comparison
  can be discounted by hand; the call-out option does not exist.

### MX-14: How many seals should we hold, with a 12-week lead time?

- **Persona:** a stores or reliability engineer setting stock levels.
- **Story:** As a stores engineer, I want the stock level that maximises
  NPV, so that the capital tied up in spares is weighed against the
  downtime a stock-out causes.
- **In their words:** "How many seals should we keep for the 20 stations,
  given they take 12 weeks to arrive?"
- **Setting:** the spares guide's fleet of 20 with
  `spares_stock(2016.0, fill_rate=0.95, fleet=20)`, with a holding cost
  per unit per year and the downtime cost of waiting for a part.
- **Acceptance criteria:**
  - The NPV for each stock level and the best one, alongside the stock a
    fill-rate target gives.
  - Pooled stock across interchangeable parts (#183) valued the same way.
- **Needs:** B1, B2 (holding cost as a rate cost); #183.
- **Status:** partial: stock for a fill rate or stock-out target exists,
  without costs.

### MX-15: What is better data worth?

- **Persona:** a reliability engineer deciding whether to gather more
  failure data before committing to a plan.
- **Story:** As a reliability engineer, I want the expected value of
  better information about a component's life, so that I can justify (or
  not) tearing down more units or fitting a sensor.
- **In their words:** "Is it worth tearing down five more pumps to firm up
  the Weibull before we set the PM interval?"
- **Setting:** the tutorial's 40 pumps, 19 still running, fitted in
  SurPyval; the decision is the PM interval of MX-01's kind.
- **Acceptance criteria:**
  - The expected NPV lost to parameter uncertainty (the expected value of
    perfect information) for the PM-interval decision.
  - The expected value of the sample information from n more teardowns.
  - An independent check: sampling the fitted parameters from their
    covariance by hand, re-optimising the interval for each draw, and
    averaging.
- **Needs:** B1, B6, B7.
- **References:** #196 (uncertainty importance).
- **Status:** open.

---

## Agents

### MX-16: One call that answers "what is this plan worth?"

- **Persona:** an agent (Reliafy's, or any working through the MCP tools
  or the Python API) answering an engineer's question in a chat.
- **Story:** As an agent, I want one call that returns the NPV, the EAC,
  the yearly cash-flow table and how it was computed, and a clear refusal
  when it cannot, so that I do not hand-roll discounting and get it wrong.
- **In their words:** (the engineer) "What's this maintenance plan worth
  over 15 years at 7%?"
- **Acceptance criteria:**
  - The answer's shape above: NPV, EAC, undiscounted total, cash-flow
    table, method.
  - `analysis_routes()` reports the NPV analyses like every other: exact,
    numerical, simulated or refused, with the method's own reason.
  - Units stated in the result (time unit of the models, the rate's
    period), so an agent cannot mix hours and years silently.
- **Needs:** B1; `analysis_routes` coverage.
- **Status:** open.

### MX-17: The plan said one thing; the outage log says another

- **Persona:** an agent or engineer reviewing a plan against what
  happened.
- **Story:** As an engineer, I want to compare a plan's expected cash flows
  with those implied by the real outage history, so that I can see where
  the assumptions were wrong and re-value the plan from today.
- **In their words:** "We've had two years on this plan. How does what
  actually happened compare with what we expected, and what's the plan
  worth now?"
- **Setting:** a diagram with an outage log (`timelines`, and Reliafy's
  `system_history`), the plan valued at the start, and the components'
  current states.
- **Acceptance criteria:**
  - Expected and actual cash flows per year so far, by category and
    component.
  - The plan re-valued from the current states (`state=`) for the rest of
    the horizon.
- **Needs:** B1, B3; the timelines' measures.
- **Status:** open.
