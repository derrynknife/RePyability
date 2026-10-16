# Calendar replacement policy: decisions to take or verify

This file accompanies the brief *"Long-run optimal replacement policy over an
outage calendar"* (use case UC-13 in `README.md`), built in full:

- `Opportunity`, `Calendar`: `_calendar.py`.
- Phase 1, `CalendarPolicy` and `NonRepairable.optimal_calendar_policy` /
  `calendar_policy_cost`: `_replacement.py`.
- Phase 2: `_integration.py`.
- Phase 3 and the routing: `_replacement.py`.
- The guide section with its chart: `docs/guide/maintenance.md`, drawn by
  `scripts/calendar_policy_chart.py`.
- The `RepairableRBD` hook.

Each item below is a place where I chose something the brief left open,
departed from it, or could not check it. Each says what was done and what
you need to decide. Items are grouped; **★** marks those that most need your
eye.

## A. Things I could not verify

1. **★ The prototype was not available.** `calendar_replacement.py` was in
   neither the repo nor the session. As a result, phase 1 tests 4, 5 and 7
   check numbers I worked out myself on a plant of my own. That plant is a
   Weibull life of scale 1000 days and shape 2.5, a weekly stop at 3, a
   quarterly shut at 2, an annual shut at 1, any day at 4 and a failure at
   20. The prototype's own figures are not checked:
   - 492 days against 493;
   - DP 17.1, replace at every annual shut 17.6, age replacement 29.6, run to
     failure 37.5;
   - "$4 against $3 a week later".

   *Decide:* commit the prototype, or its calendar and prices, and I will add
   its figures as tests.

   What was checked on my plant, over ten years:

   | Policy | Cost |
   |---|---|
   | Optimal | 25.27 |
   | Replace at every annual shut | 25.48 |
   | Age replacement on any day | 50.41 |
   | Run to failure | 82.05 |

   The skip test `replace(annual, 60) is False` and
   `replace(annual, 300) is True` holds. The annual threshold is 88 days.
2. **Test 6 (horizon independence).** On my plant:
   - year 3's thresholds from 10-, 20- and 40-year solves are identical, as
     asserted;
   - from a 5-year solve they are identical too, where the brief expected up
     to about 10 days of end effect;
   - from a 4-year solve they differ by up to 14 days.

   The test asserts at most 10 days for 5 years, which holds.
3. **The literature reference** in the docstrings is "Dekker et al., early
   1990s", as the brief gives it, with no full citation. *Decide:* give the
   reference you want quoted.

## B. Where I departed from the brief

4. **★ A failure renews half a step later on average.** The brief's
   recursion puts the unit fitted after a failure in step `n` at age 0 at
   `n+1`: `(1−p)(c_f + V(0))`. The cost rate is then low by about half a step
   for each failure:
   - 1.7e-4 relative for the collapse test at a step of 1 day, which fails
     the brief's own 1e-5 target;
   - the threshold comes out at 493 days rather than the prototype's 492.

   I use `(1−p)(c_f + ½(V(0) + V(1)))` instead: the new unit is 0 or 1 step
   old, half each, which is a mean of half a step, as a failure uniform in
   the step gives. The error becomes second order:
   - 1.3e-7 at a step of 1 day, 3e-6 at 2 days, 5e-6 at 4 days;
   - threshold 493 days, against the textbook 493.05.

   *Decide:* keep this, or revert to the brief's recursion and relax test 1
   to about 2e-4.
5. **The step is not an argument (decided).** It is a numerical
   setting, not a choice of when decisions are made; the calendar says
   that, by its opportunities. It is sized from the models, as CLAUDE.md
   requires, so that no result depends on the unit of time:
   - about the mean life over 1,000 (`STEPS_PER_LIFE`);
   - reduced to a whole fraction of the largest step on which every
     period, offset and date lies, so the dates fall on the grid;
   - for the plant in days this gives 0.875 days.

   The tests set it through the internal solver
   (`_replacement.optimal_calendar_policy(..., step=)`).
6. **The cycle limit is 20,000 steps** (`MAX_CYCLE`), not "about 20 years of
   days" (7,300). Relative value iteration on 20,000 steps is a few seconds a
   sweep at worst. *Decide:* the limit.
7. **The settings are module constants, not arguments.** These are:
   - `MAX_CYCLE`, `TOLERANCE`, `MAX_LIVES`, `STEPS_PER_LIFE`, `FAR`, `SPAN`,
     `SETTLE_GAPS` and `SETTLE_SPANS` in `_replacement.py`;
   - `NODES`, `AGES`, `MAX_AGES` and `MAX_OPPORTUNITIES` in
     `_integration.py`.

   This follows the package's other modules. The method takes only `step`,
   `max_age`, `method`, `tail`, `mc_samples` and `seed`. *Decide:* whether any
   should be arguments.
8. **The method has more arguments than the brief's signature:** `tail`,
   `mc_samples` and `seed`. `mc_samples` and `seed` are the package's names
   for a sampled run (CLAUDE.md), default 64.
9. **What `"part"` means on a `Calendar`.** The brief makes `anytime=None`
   mean "not allowed" and also says `cp` is the default. Both are kept with a
   sentinel:
   - `Calendar(opportunities, failure="part", anytime="part")`;
   - `"part"` takes the part's `unplanned_cost` / `planned_cost`;
   - `anytime="part"` on a part with no planned cost allows no replacement
     on any day;
   - `None` means not allowed.

   *Decide:* keep the sentinel, or change the defaults.
10. **Rounding dates onto the grid.** Each date is rounded to its nearest
    grid time on its own, so periods do not drift, and a date moves by half a
    step at most. The brief's "warn if a date moved by more than step/2"
    therefore cannot fire. I warn instead when an opportunity's times move by
    more than 1% of the step. For example, an annual shut every 365.25 days
    on a 1-day grid warns: "moved by up to 0.5". *Decide:* the threshold for
    the warning, or no warning.
11. **`Calendar.planned_cost(times, step=None)` takes a step.** Without one,
    only exact times match an opportunity. With one, an opportunity counts at
    the grid time nearest it, as the policy sees it.
12. **`Calendar.period` is in time.** It is the least common multiple of the
    periods, each read as a fraction with a denominator of at most 10⁶. For
    example, 365.25 alone gives 365.25, and 7 with 365.25 gives 10,227. The
    grid's own cycle, in steps, is kept separately. An `Opportunity`'s times
    run both ways from its offset (`offset + k·period` for every whole `k`).

## C. Choices the brief left open

13. **★ How a dated plan goes on after its last date.**
    - **`tail="repeat"` (the default)** repeats the plan from its first date
      to its last, closed by the mean gap between its dates:
      `(last − first) · m/(m−1)` for `m` dates, or the date itself when there
      is only one. Repeats run forward only; nothing is put before the plan.
      Dates at 364, 728 and 1,092 thus give exactly the annual calendar's
      rate.
    - **`tail="anytime"`** leaves only replacement on any day after the
      plan, alongside the calendar's periodic opportunities.
    - **Both tails are solved.** `determined_until` is the first time at
      which their thresholds, or the kinds of replacement open, differ.
    - **The cost rate of a dated calendar always comes from the repeated
      plan**, even with `tail="anytime"`, because only the repeated plan
      defines a long run.

    *Decide:* the repeat period, and what `cost_rate` should mean for a plan.
14. **★ How `method="auto"` routes.**
    - A periodic calendar with a short cycle goes to relative value
      iteration (exact on the grid).
    - A periodic calendar with a long cycle goes to sampling.
    - **A dated plan goes to dynamic programming, not sampling.** The brief
      routes "non-periodic" calendars to sampling, but sampling uniform
      phases fits only periodic schedules; a plan's dates are fixed.
    - `"auto"` never picks integration, which must be asked for.
    - `method="dp"` on a long cycle differences a long finite horizon
      (`basis="differenced"`) rather than refusing.
    - `method="sampling"` on a dated plan is refused.
15. **What sampling samples.** It draws one uniform time over the whole
    cycle and shifts every schedule by it, rather than drawing an
    independent uniform phase for each schedule. This is right for any
    periods, and the same thing for co-prime ones. Each window is differenced
    over the longer of the longest period and the mean life. Each window
    then runs on for a settling length of 2 × the longest gap plus 3 × the
    age a unit outlives with chance 1e-3. On the co-prime calendar
    (7/93/400/1,301 days), a sample takes about 0.2 s, and doubling the
    settling length changes the rate by less than its standard error.
16. **How a sampled or long-horizon policy gives its thresholds.**
    The threshold at a time is worked out in cached blocks of steps, each solved over
    a window long enough for its end not to reach the block. No random
    numbers are involved, so it is deterministic.
17. **★ Pricing a given rule (`calendar_policy_cost`).** A rule is a dict of
    the age from which each kind replaces (opportunity names, plus
    `"anytime"`); kinds left out never replace.
    - **Where kinds coincide,** the kind with the youngest age in the rule
      replaces, and the cheapest wins a tie.
    - **The cost rate** comes from relative value iteration with the
      decisions fixed. The brief suggests the stationary distribution of a
      one-cycle matrix instead; both give the same `g`, and this reuses the
      same code.
    - **The return value is a float.**

    *Decide:* the name, the dict form, and whether it should return a result
    that carries its basis.
18. **What the policy gives (revised after review).**
    - **No per-time `threshold(t)` or `replace(t, age)`:** dropped as of
      no use. Most of their values, at the weekly stops and on ordinary
      days, are never reached.
    - **`replacement_rates` and `cost_rates`:** by kind, the long-run
      replacements and cost a unit of time, failures as `"failure"`. The
      costs sum to `cost_rate`.
      - On a periodic calendar, both are exact for the grid. They come
        from the long-run share of units at each age, carried forward
        from a new unit until the ages at the start of a cycle settle.
      - For a sampled, long or dated calendar, they are estimated from a
        new unit run in over a settling length.
    - **`schedule(kinds=(), until=None)`:** lists `(time, kind, threshold,
      chance)`, where `chance` is the long-run chance of a replacement
      there, or from a new unit at 0 where the calendar does not repeat.
      A shut the policy never uses has a chance of 0.
      - By default it lists the opportunities, not the days on which only
        the any-day replacement is open.
      - `until` defaults to the period, or to the plan's end; a sampled
        policy needs it given.
    - **`next_replacement(age, now=0)`:** when the unit in service will
      be replaced if it does not fail first, at which opportunity and
      price (`PlannedReplacement`). A unit of age 0 has just been put in,
      so it is not replaced again at that time.

    *Decide:* the names, and whether rates by kind should be given per
    unit of time (as now) or per cycle.
19. **A replace set that is not a threshold.** When the ages that replace
    are not all those from the youngest (for example a log-normal life),
    the schedule's threshold is the youngest age that replaces,
    `threshold_only` is False, and a warning is given. The rates by kind
    then take every age from the youngest, so they no longer sum exactly
    to the cost rate.
20. **The NPV hook.** `_sweep` takes `beta`, a discount per step that
    defaults to 1. There is no public API yet, as the brief asks.

## D. Phase 2 (integration)

21. **The quadrature.**
    - **Product integration:** each piece's increase in the distribution,
      times the mean of the integrand at its ends. It is exact in the
      density, so a density that is infinite at 0 is no problem.
    - **Not used:** Simpson or Gauss rules, `numerical_convolution` and
      `_quadrature`.
    - **32 pieces per gap** (`NODES`).
    - **Convergence:** relative value iteration, not policy iteration with a
      linear solve.

    It agrees with the grid as the step shrinks (2e-6 at a step of 1 day).
22. **The age grid is aligned with the shortest gap.** Its spacing is a whole
    fraction of the shortest gap. Without that, daily opportunities came out
    0.3% *below* age replacement, which is impossible. With it they match to
    3e-9. It needs as many ages as the gap demands: integration refuses age
    grids beyond 20,000 points, and cycles beyond 20,000 opportunities. The
    daily test takes about 8 s.
23. **Dated plans in phase 2** are supported with `tail="repeat"` only. The
    policy is given from the plan's first date onwards.

## E. The `RepairableRBD` hook (deliverable 6)

24. **★ How a component uses the policy.** A component takes
    `"preventive": {"policy": "calendar", "calendar": <CalendarPolicy>}`.
    - At each renewal, the event loop asks the policy for the first time at
      which the unit's age reaches the threshold, counting from when the
      unit was put in, and schedules the replacement then.
    - **Planned replacements are charged the calendar's price** for that
      opportunity, so the spec takes no `"cost"` (and no `"interval"`).
    - **Failures are charged the diagram's own `replace_cost`** and
      `repair_cost`. The calendar's failure price only shapes the policy.
    - **The policy is applied as given, not re-optimised.** It was optimised
      for a part replaced at once. If the diagram's repairs or maintenance
      take time, its outages are not what the policy assumed.

    *Decide:* whether the failure price should come from the calendar.
25. **★ It is simulation only.**
    - Every exact or numerical analysis refuses it, through the same check
      as an opportunistic group member (`_require_no_opportunities`). That
      covers availability over time, long-run values, the exact twin,
      expected cost, spares and so on.
    - **This holds even for a component alone,** whose `g` the policy itself
      gives exactly.
    - **numba refuses it** (`_compiled.unsupported`), so it runs in the
      Python loop.
    - **Saving it is refused** (`to_json` raises).
    - **A state's phase is refused,** because a calendar's times are
      absolute; an age is taken.
    - **Checked:** the catalogue has a diagram for it; the routes, levers and
      capability tests pass; and the simulated cost of a lone bearing
      matches the policy's `g` within 3 standard errors over 40 years.

    *Decide:* whether the analyses of a lone calendar component should use
    the policy's `g` (exact) rather than refuse.
26. **Not done: maintenance groups.** The brief says the hook "ties into the
    existing opportunistic-maintenance groups". A calendar component can
    still be given a `"group"`, but nothing makes the other members'
    stops opportunities for it, or the reverse, and this is untested.
    *Decide:* what a group should do with a calendar component (its
    opportunities as the group's stops?).

## F. Around the build

27. **UC-7's `replacement_policies` and UC-12's
    `optimal_next_replacement`.** Decided: `replacement_policies` is
    removed (a special case of `optimal_calendar_policy`), with the
    `older_than` rule it added to `condition_cycle`;
    `optimal_replacement_policy` is kept. Done (#348):
    `optimal_next_replacement(policy=)` takes the `cost_rate` of a
    `CalendarPolicy` (or a `MaintenancePolicy`) as its baseline; the
    age replacement's stays the default.
28. **The chart.** It is an SVG committed under `docs/guide/img/`, drawn by
    `scripts/calendar_policy_chart.py` with matplotlib, outside the package
    (which keeps no plotting dependency). Run the script again after a change
    to the policy.
29. **CLAUDE.md.** The scope note gains a line: a calendar of opportunities
    in model time is RePyability's, and laying it on real dates is Reliafy's.
30. **Failures that were already on dev, fixed here:**
    - `test_every_export_is_a_node_or_said_not_to_be` had been failing
      since `ReplacementPlan` was added; it is now listed as not a node,
      with `Opportunity` and `Calendar`;
    - `test_api_docstrings` had been failing for the cost parameters of
      `NonRepairable`, `ReplacementPlan` and `replacement_policies`; this was
      fixed in #340.
