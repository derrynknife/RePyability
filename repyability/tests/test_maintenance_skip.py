"""Whether to skip the next planned replacement right after a failure
(``NonRepairable.skip_next_replacement``; #319, UC-6 of
``repyability/maintenance/README.md``)."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad

from repyability import NonRepairable, SkipValue

LIFE = surv.Weibull.from_params([365.0, 2.0])
BEARING = NonRepairable(LIFE, planned_cost=150.0, unplanned_cost=400.0)
YEAR = 365.25


def per_day(rate):
    """The continuous rate a day of a rate ``rate`` a year."""
    return math.log1p(rate) / YEAR


def weibull_lives(rng, size):
    return 365.0 * np.sqrt(-np.log(rng.random(size)))


def simulated_fixed(days, every, planned, unplanned, rate, n, seed):
    """The NPV of skipping on a plan on fixed times, simulated plainly:
    done, a new unit at ``days`` and its failures to ``days + every``;
    skipped, the failures of the unit in now (and its replacements) from
    ``days`` to ``days + every``. The rest is the same both ways."""
    rng, r = np.random.default_rng(seed), per_day(rate)

    def failures(start, keep_from):
        time, value = np.full(n, start), np.zeros(n)
        while True:
            time = time + weibull_lives(rng, n)
            hit = time < days + every
            if not hit.any():
                return value
            counted = hit & (time >= keep_from)
            value[counted] += unplanned * np.exp(-r * time[counted])
            time = np.where(hit, time, np.inf)

    done = planned * math.exp(-r * days) + failures(days, days)
    skipped = failures(0.0, days)
    difference = done - skipped
    return difference.mean(), difference.std() / math.sqrt(n)


def simulated_resetting(every, planned, unplanned, rate, n, seed, years):
    """The present values of each choice on a plan that resets, simulated
    plainly over ``years``, the first cycle cut off at ``every`` (done) or
    ``2 every`` (skipped): their means, and the error of their
    difference."""
    rng, r = np.random.default_rng(seed), per_day(rate)

    def value(first):
        time, total = np.zeros(n), np.zeros(n)
        cut = np.full(n, first)
        live = np.ones(n, dtype=bool)
        while live.any():
            life = weibull_lives(rng, live.sum())
            failed = life < cut[live]
            time[live] += np.minimum(life, cut[live])
            costs = np.where(failed, unplanned, planned)
            total[live] += costs * np.exp(-r * time[live])
            cut[live] = every
            live = time < years * YEAR
        return total

    done, skipped = value(every), value(2.0 * every)
    error = math.sqrt((done.var() + skipped.var()) / n)
    return done.mean(), skipped.mean(), error


def test_the_bearing_on_a_plan_that_resets():
    # UC-6: every 180 days, 150 planned, 400 at a failure, 8% a year.
    value = BEARING.skip_next_replacement(
        180, resets=True, discount_rate=per_day(0.08)
    )
    assert isinstance(value, SkipValue)
    assert value.time_to_next == 180.0
    assert value.discount_rate == per_day(0.08)
    assert value.npv == pytest.approx(26.558, abs=1e-3)
    assert value.npv == value.replace - value.skip
    assert value.best == "skip"
    done, skipped, error = simulated_resetting(
        180.0, 150.0, 400.0, 0.08, 40_000, 3, 300
    )
    assert abs(value.npv - (done - skipped)) < 4 * error
    # Each is the present value of the whole future (300 years of it
    # simulated: what is left is discounted away).
    assert value.replace == pytest.approx(done, rel=2e-3)
    assert value.skip == pytest.approx(skipped, rel=2e-3)


@pytest.mark.parametrize("days", [1.0, 30.0, 120.0, 180.0])
def test_the_bearing_on_a_plan_on_fixed_times(days):
    value = BEARING.skip_next_replacement(
        180, resets=False, time_to_next=days, discount_rate=per_day(0.08)
    )
    assert value.time_to_next == days
    assert value.npv == value.replace - value.skip
    mean, error = simulated_fixed(
        days, 180.0, 150.0, 400.0, 0.08, 300_000, int(days)
    )
    assert abs(value.npv - mean) < 4 * error


def test_the_fixed_answer_by_the_time_to_the_next_replacement():
    # On a 180-day plan, too frequent for the bearing (the best age is
    # near 298 days), skipping always pays, the more the sooner the next
    # replacement; on a 365-day plan it stops paying some 200 to 300 days
    # out.
    r = per_day(0.08)

    def npv(every, days):
        return BEARING.skip_next_replacement(
            every, resets=False, time_to_next=days, discount_rate=r
        )

    values = [npv(180, d).npv for d in (1, 30, 90, 180)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(149.06, abs=0.01)
    assert values[-1] == pytest.approx(35.26, abs=0.01)
    assert npv(365, 10).npv > 0 > npv(365, 300).npv
    assert npv(365, 10).best == "skip"
    assert npv(365, 300).best == "replace"


def test_the_fixed_costs_are_those_to_the_following_replacement():
    # Done: the replacement at d, the new unit's failures to d + T, and
    # the failures before d; skipped: the failures to d + T. Each from
    # the part's failures over a window (UC-5, discounted by hand), the
    # failures after d from a new unit of the same life.
    r, every, days = per_day(0.08), 180.0, 30.0
    value = BEARING.skip_next_replacement(
        every, resets=False, time_to_next=days, discount_rate=r
    )
    from repyability import RepairableRBD

    unit = RepairableRBD(
        [("s", "u"), ("u", "t")],
        {
            "u": {
                "reliability": LIFE,
                "repairability": "instant",
                "replace_cost": 400.0,
            }
        },
    )
    failures = unit.expected_cost(
        np.array([every, days, days + every]), discount_rate=r
    ).mean
    later = math.exp(-r * days)
    assert value.replace == pytest.approx(
        150.0 * later + later * failures[0] + failures[1], rel=1e-12
    )
    assert value.skip == pytest.approx(failures[2], rel=1e-12)


def exponential_closed_forms(rate, every, planned, unplanned, r, days):
    """The two plans' NPVs for an exponential life, written out: its
    failures come at ``rate`` whatever its age."""
    total = r + rate

    def length(cut):
        return -math.expm1(-total * cut) / total

    def cost(cut):
        failed = rate * length(cut)
        return unplanned * failed + planned * math.exp(-total * cut)

    once, twice = length(every), length(2 * every)
    resetting = (
        cost(every) - cost(2 * every) + cost(every) * (twice - once) / once
    )

    def failures(t):
        # A new unit and its replacements fail at ``rate`` throughout.
        return unplanned * rate * (-math.expm1(-r * t) / r if r else t)

    later = math.exp(-r * days)
    fixed = (
        planned * later
        + later * failures(every)
        - (failures(days + every) - failures(days))
    )
    return resetting, fixed


@pytest.mark.parametrize("yearly", [0.0, 0.08, 0.5])
def test_an_exponential_life_against_its_closed_forms(yearly):
    rate, every, days = 1 / 200.0, 90.0, 40.0
    part = NonRepairable(
        surv.Exponential.from_params([rate]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )
    resetting, fixed = exponential_closed_forms(
        rate, every, 150.0, 400.0, per_day(yearly), days
    )
    r = per_day(yearly)
    value = part.skip_next_replacement(every, resets=True, discount_rate=r)
    assert value.npv == pytest.approx(resetting, rel=1e-8)
    value = part.skip_next_replacement(
        every, resets=False, time_to_next=days, discount_rate=r
    )
    assert value.npv == pytest.approx(fixed, rel=1e-6)
    # With no ageing, a planned replacement only costs: skipping saves.
    assert resetting > 0 and fixed > 0


def test_a_resetting_plan_with_discounting_values_the_whole_future():
    # Done, the plan from new, A(T) / (r M(T)); skipped, the longer first
    # cycle and then the plan from new: written out for an exponential
    # life, whose cycle values have closed forms.
    rate, every, r = 1 / 200.0, 90.0, per_day(0.08)
    part = NonRepairable(
        surv.Exponential.from_params([rate]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )
    total = r + rate

    def length(cut):
        return -math.expm1(-total * cut) / total

    def cost(cut):
        return 400.0 * rate * length(cut) + 150.0 * math.exp(-total * cut)

    whole = cost(every) / (r * length(every))
    skipped = cost(2 * every) + (1 - r * length(2 * every)) * whole
    value = part.skip_next_replacement(every, resets=True, discount_rate=r)
    assert value.replace == pytest.approx(whole, rel=1e-8)
    assert value.skip == pytest.approx(skipped, rel=1e-8)


def test_with_no_discounting_a_resetting_plan_is_the_classic_comparison():
    # Undiscounted, skipping lengthens the first cycle: it saves its cost
    # difference, less the long-run cost rate over the time it adds. Each
    # choice is its first cycle's cost less its length at that rate.
    every, planned, unplanned = 180.0, 150.0, 400.0

    def mean_length(cut):
        return quad(lambda x: float(LIFE.sf(x)), 0, cut, epsrel=1e-13)[0]

    def cycle_cost(cut):
        kept = float(LIFE.sf(cut))
        return unplanned * (1 - kept) + planned * kept

    rate_of_cost = cycle_cost(every) / mean_length(every)
    classic = (
        cycle_cost(every)
        - cycle_cost(2 * every)
        + rate_of_cost * (mean_length(2 * every) - mean_length(every))
    )
    value = BEARING.skip_next_replacement(every, resets=True)
    assert value.discount_rate == 0.0
    assert value.npv == pytest.approx(classic, rel=1e-9)
    # Each choice's whole future is endless: only the difference is
    # finite (with a rate the same fields hold the present values).
    assert value.replace == value.skip == math.inf
    assert value.best == "skip"
    assert json.loads(json.dumps(value.to_dict()))["npv"] == value.npv
    # And a tiny rate is next to it.
    tiny = BEARING.skip_next_replacement(
        every, resets=True, discount_rate=1e-9
    )
    assert tiny.npv == pytest.approx(classic, rel=1e-6)


@pytest.mark.parametrize("resets", [True, False])
def test_a_rate_that_discounts_the_costs_away_is_warned_of(resets):
    # A rate a year given with a life in days (#231): both plans warn as
    # a RepairableRBD's costs do, naming the life's mean.
    with pytest.warns(UserWarning, match="shortest mean life") as record:
        value = BEARING.skip_next_replacement(
            180.0,
            resets=resets,
            time_to_next=None if resets else 30.0,
            discount_rate=0.08,
        )
    assert "exp(-25.9)" in str(record[0].message)
    assert math.isfinite(value.npv)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        BEARING.skip_next_replacement(
            180.0,
            resets=resets,
            time_to_next=None if resets else 30.0,
            discount_rate=per_day(0.08),
        )


def test_the_rate_is_a_unit_of_the_life_and_the_answer_the_same_in_hours():
    # The same bearing with its life in hours, and the rate an hour.
    in_hours = NonRepairable(
        surv.Weibull.from_params([365.0 * 24.0, 2.0]),
        planned_cost=150.0,
        unplanned_cost=400.0,
    )
    r = per_day(0.08)
    for resets, time_to_next in ((True, None), (False, 30.0)):
        days = BEARING.skip_next_replacement(
            180.0, resets=resets, time_to_next=time_to_next, discount_rate=r
        )
        hours = in_hours.skip_next_replacement(
            180.0 * 24.0,
            resets=resets,
            time_to_next=None if time_to_next is None else 24.0 * time_to_next,
            discount_rate=r / 24.0,
        )
        assert hours.npv == pytest.approx(days.npv, rel=1e-6)
        assert hours.replace == pytest.approx(days.replace, rel=1e-6)
        assert hours.skip == pytest.approx(days.skip, rel=1e-6)


def test_the_value_goes_through_json():
    value = BEARING.skip_next_replacement(
        180, resets=False, time_to_next=30, discount_rate=per_day(0.08)
    )
    plain = json.loads(json.dumps(value.to_dict()))
    assert plain["resets"] is False
    assert plain["time_to_next"] == 30.0
    assert plain["npv"] == pytest.approx(123.08, abs=0.01)
    assert plain["best"] == "skip"
    assert set(plain) == {
        "resets",
        "every",
        "time_to_next",
        "discount_rate",
        "replace",
        "skip",
        "npv",
        "best",
    }


def test_the_time_to_the_next_replacement_goes_with_a_fixed_plan_only():
    with pytest.raises(ValueError, match="give no time_to_next"):
        BEARING.skip_next_replacement(180, resets=True, time_to_next=30)
    with pytest.raises(ValueError, match="needs time_to_next"):
        BEARING.skip_next_replacement(180, resets=False)
    with pytest.raises(ValueError, match="at most every"):
        BEARING.skip_next_replacement(180, resets=False, time_to_next=181)
    for time_to_next in (0, -5, math.inf, True, "30"):
        with pytest.raises(ValueError, match="time_to_next"):
            BEARING.skip_next_replacement(
                180, resets=False, time_to_next=time_to_next
            )


@pytest.mark.parametrize("rate", [-0.01, math.inf, math.nan, True, "8%"])
def test_a_discount_rate_is_a_rate_at_least_0(rate):
    with pytest.raises(ValueError, match="discount_rate"):
        BEARING.skip_next_replacement(180, resets=True, discount_rate=rate)


@pytest.mark.parametrize("every", [0, -1, math.inf, True, "180"])
def test_every_is_a_time(every):
    with pytest.raises(ValueError, match="every"):
        BEARING.skip_next_replacement(every, resets=True)


def test_resets_is_true_or_false():
    with pytest.raises(TypeError, match="resets"):
        BEARING.skip_next_replacement(180, resets=1)


def test_the_costs_are_the_part_s_own():
    with pytest.raises(ValueError, match="costs not set"):
        NonRepairable(LIFE).skip_next_replacement(180, resets=True)
    with pytest.raises(ValueError, match="costs not set"):
        NonRepairable(LIFE, planned_cost=150.0).skip_next_replacement(
            180, resets=True
        )
    part = NonRepairable(LIFE)
    part.set_costs_planned_and_unplanned(150.0, 400.0)
    value = part.skip_next_replacement(
        180, resets=True, discount_rate=per_day(0.08)
    )
    assert value.npv == pytest.approx(26.558, abs=1e-3)


def test_a_life_with_no_closed_form_is_refused():
    estimate = surv.KaplanMeier.fit(LIFE.random(50))
    part = NonRepairable(estimate, planned_cost=150.0, unplanned_cost=400.0)
    with pytest.raises(NotImplementedError, match="closed form"):
        part.skip_next_replacement(180, resets=True)
