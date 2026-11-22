"""Opportunities with the prices on the part (UC-45 of
``repyability/maintenance/README.md``): ``Opportunity`` and
``Opportunities`` in model time and on real dates, each part's prices
(``opportunity_cost``, ``opportunity_costs``, ``job_duration``), the
helper that works prices out from a replacement's cost and downtime, and
the deprecated 0.14 and 0.19 forms giving the same numbers."""

import json
import math
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    RBD,
    Calendar,
    NonRepairable,
    Opportunities,
    Opportunity,
    OpportunityPrices,
    RepairableRBD,
    Shutdown,
    Shutdowns,
    __version__,
)
from repyability.schedule import Recurrence, Usage
from repyability.utils.deprecation import NEXT_REMOVAL

LIFE = surv.Weibull.from_params([1000.0, 2.5])
NOW = "2026-11-16"


def old(make):
    """``make()``, with the deprecated forms' warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return make()


def new(make):
    """``make()``, refusing any warning of a deprecated form."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return make()


def same_policy(a, b):
    """Two policies' numbers, bit for bit."""
    assert a.to_dict().keys() == b.to_dict().keys()
    for key, value in a.items():
        other = b[key]
        if isinstance(value, float) and math.isnan(value):
            assert math.isnan(other), key
        else:
            assert json.dumps(value, default=str, sort_keys=True) == (
                json.dumps(other, default=str, sort_keys=True)
            ), key


# ------------------------------------------------- the old forms' numbers


def plant_part(**more):
    return NonRepairable(
        LIFE,
        planned_cost=4.0,
        unplanned_cost=20.0,
        opportunity_costs={"weekly": 3.0, "quarterly": 2.0, "annual": 1.0},
        **more,
    )


PLANT = Opportunities(
    [
        Opportunity("weekly", period=7.0),
        Opportunity("quarterly", period=91.0),
        Opportunity("annual", period=364.0),
    ]
)


def old_plant(anytime=4.0):
    return old(
        lambda: Calendar(
            [
                Opportunity("weekly", 3.0, period=7.0),
                Opportunity("quarterly", 2.0, period=91.0),
                Opportunity("annual", 1.0, period=364.0),
            ],
            failure=20.0,
            anytime=anytime,
        )
    )


def test_model_time_is_bit_identical_to_the_calendar():
    before = old(
        lambda: NonRepairable(LIFE).optimal_calendar_policy(old_plant())
    )
    after = new(lambda: plant_part().optimal_opportunity_policy(PLANT))
    same_policy(before, after)
    assert after.cost_rate == pytest.approx(0.006943, abs=5e-7)
    assert before.schedule() == after.schedule()
    rule = {"annual": 0.0}
    assert old(
        lambda: NonRepairable(LIFE).calendar_policy_cost(old_plant(), rule)
    ) == new(lambda: plant_part().opportunity_policy_cost(PLANT, rule))


def test_model_time_discounted_is_bit_identical():
    rate = math.log(1.07) / 365.25
    stops = Opportunities([Opportunity("stop", period=28.0)], any_day=False)
    before = old(
        lambda: NonRepairable(LIFE).optimal_calendar_policy(
            Calendar(
                [Opportunity("stop", 3.0, period=28.0)],
                failure=20.0,
                anytime=None,
            ),
            discount_rate=rate,
        )
    )
    part = NonRepairable(LIFE, unplanned_cost=20.0, opportunity_cost=3.0)
    after = new(
        lambda: part.optimal_opportunity_policy(stops, discount_rate=rate)
    )
    same_policy(before, after)
    assert after.npv == before.npv and after.eac == before.eac


SITE_PART = NonRepairable(
    LIFE,
    planned_cost=4.0,
    unplanned_cost=20.0,
    opportunity_costs={"stop": 3.0, "turnaround": 0.5},
)
STOP_PART = NonRepairable(
    LIFE, planned_cost=4.0, unplanned_cost=20.0, opportunity_cost=3.0
)
STOP = Opportunity("stop", Recurrence.every(4, "weeks", "2026-11-21"))
TURNAROUND = Opportunity("turnaround", dates=["2027-03-03", "2028-09-09"])


def old_site(any_day=4.0, turnaround=True):
    def make():
        shuts = [
            Shutdown(
                "stop",
                Recurrence.every(4, "weeks", "2026-11-21"),
                replace_cost=3.0,
            )
        ]
        if turnaround:
            shuts.append(
                Shutdown(
                    "turnaround",
                    dates=["2027-03-03", "2028-09-09"],
                    replace_cost=0.5,
                )
            )
        return Shutdowns(shuts, any_day=any_day)

    return old(make)


@pytest.mark.parametrize("discount", [None, 0.07])
@pytest.mark.parametrize("turnaround", [False, True])
def test_a_site_is_bit_identical_to_shutdowns(discount, turnaround):
    site = Opportunities([STOP, TURNAROUND] if turnaround else [STOP])
    part = SITE_PART if turnaround else STOP_PART
    options = dict(unit="days", now=NOW, discount=discount)
    before = old(
        lambda: NonRepairable(
            LIFE, planned_cost=1.0, unplanned_cost=20.0
        ).optimal_calendar_policy(old_site(turnaround=turnaround), **options)
    )
    after = new(lambda: part.optimal_opportunity_policy(site, **options))
    same_policy(before, after)
    assert before.schedule("2029-12-31") == after.schedule("2029-12-31")


def test_a_site_s_rule_cost_is_bit_identical():
    rule = {"stop": 300.0}
    options = dict(unit="days", now=NOW)
    before = old(
        lambda: NonRepairable(
            LIFE, planned_cost=1.0, unplanned_cost=20.0
        ).calendar_policy_cost(old_site(turnaround=False), rule, **options)
    )
    after = new(
        lambda: STOP_PART.opportunity_policy_cost(
            Opportunities([STOP]), rule, **options
        )
    )
    assert before.cost == after.cost
    assert after.prices.price("stop") == 3.0


def test_the_old_forms_on_the_new_method_give_the_same_numbers():
    # A Calendar or Shutdowns given to the new method: its prices are the
    # part's for those opportunities.
    part = NonRepairable(LIFE, planned_cost=1.0, unplanned_cost=20.0)
    via_old = old(lambda: part.optimal_calendar_policy(old_plant()))
    via_new = old(lambda: part.optimal_opportunity_policy(old_plant()))
    same_policy(via_old, via_new)
    assert via_new.prices.row("weekly").source == "opportunity"
    assert via_new.prices.row("any day").source == "collection"
    assert via_new.prices.row("failure").price == 20.0


# ----------------------------------------------------------- the prices


def test_overrides_by_name_and_by_object():
    weekly, annual = Opportunity("weekly", period=7.0), Opportunity(
        "annual", period=364.0
    )
    plant = Opportunities([weekly, annual])
    by_name = NonRepairable(
        LIFE,
        planned_cost=4.0,
        unplanned_cost=20.0,
        opportunity_cost=3.0,
        opportunity_costs={"annual": 1.0},
    )
    by_object = NonRepairable(
        LIFE,
        planned_cost=4.0,
        unplanned_cost=20.0,
        opportunity_cost=3.0,
        opportunity_costs={annual: 1.0},
    )
    a = new(lambda: by_name.optimal_opportunity_policy(plant))
    b = new(lambda: by_object.optimal_opportunity_policy(plant))
    same_policy(a, b)
    rows = {r.name: (r.price, r.source) for r in a.prices.rows}
    assert rows == {
        "weekly": (3.0, "opportunity_cost"),
        "annual": (1.0, "override"),
        "any day": (4.0, "planned_cost"),
        "failure": (20.0, "unplanned_cost"),
    }


def test_a_typo_is_refused_with_the_names():
    part = NonRepairable(
        LIFE,
        planned_cost=4.0,
        unplanned_cost=20.0,
        opportunity_costs={"anual": 1.0},
    )
    with pytest.raises(ValueError, match="'anual'.*'weekly', 'quarterly'"):
        part.optimal_opportunity_policy(PLANT)
    stranger = Opportunity("annual", period=365.0)
    part = NonRepairable(
        LIFE,
        planned_cost=4.0,
        unplanned_cost=20.0,
        opportunity_costs={stranger: 1.0},
    )
    with pytest.raises(ValueError, match="'annual', which is not one"):
        part.optimal_opportunity_policy(PLANT)


def test_with_no_opportunity_cost_an_opportunity_is_any_other_day():
    part = NonRepairable(LIFE, planned_cost=4.0, unplanned_cost=20.0)
    policy = new(lambda: part.optimal_opportunity_policy(PLANT))
    assert {r.name: r.price for r in policy.prices.rows} == {
        "weekly": 4.0,
        "quarterly": 4.0,
        "annual": 4.0,
        "any day": 4.0,
        "failure": 20.0,
    }
    assert policy.prices.row("weekly").source == "planned_cost"
    # Every opportunity at the any-day price is age replacement.
    age = part.optimal_replacement_policy()
    assert policy.cost_rate == pytest.approx(age.cost_rate, rel=1e-5)


def test_an_opportunity_with_no_price_at_all_is_refused():
    part = NonRepairable(LIFE, unplanned_cost=20.0)
    with pytest.raises(ValueError, match="'weekly' has no price"):
        part.optimal_opportunity_policy(PLANT)


def test_a_stop_too_short_for_the_job_is_skipped_and_said():
    site = Opportunities(
        [
            Opportunity(
                "weekly stop", Recurrence.weekly("Sat"), duration="8h"
            ),
            Opportunity("short stop", Recurrence.weekly("Wed"), duration="4h"),
        ],
        any_day=False,
    )
    part = NonRepairable(
        LIFE, unplanned_cost=20.0, opportunity_cost=3.0, job_duration="6h"
    )
    policy = new(
        lambda: part.optimal_opportunity_policy(site, unit="days", now=NOW)
    )
    assert policy.prices.unused == ("short stop",)
    row = policy.prices.row("short stop")
    assert not row.used and math.isnan(row.price)
    assert row.reason == "the 6 h job doesn't fit in the 4 h short stop"
    assert "short stop" not in policy.rule.thresholds
    # The same as the site without it.
    alone = new(
        lambda: part.optimal_opportunity_policy(
            Opportunities([site["weekly stop"]], any_day=False),
            unit="days",
            now=NOW,
        )
    )
    assert policy.cost == alone.cost
    # A job of no stated length uses every stop.
    quick = NonRepairable(LIFE, unplanned_cost=20.0, opportunity_cost=3.0)
    assert (
        new(
            lambda: quick.optimal_opportunity_policy(
                site, unit="days", now=NOW
            )
        ).prices.unused
        == ()
    )


def test_a_job_too_long_in_model_time():
    stops = Opportunities(
        [
            Opportunity("long", period=28.0, duration=1.0),
            Opportunity("short", period=7.0, duration=0.1),
        ]
    )
    part = NonRepairable(
        LIFE,
        planned_cost=4.0,
        unplanned_cost=20.0,
        opportunity_cost=3.0,
        job_duration=0.25,
    )
    policy = new(lambda: part.optimal_opportunity_policy(stops))
    assert policy.prices.unused == ("short",)
    assert "doesn't fit" in policy.prices.row("short").reason
    text = NonRepairable(
        LIFE, planned_cost=4.0, unplanned_cost=20.0, job_duration="6h"
    )
    with pytest.raises(ValueError, match="in model time give it as a number"):
        text.optimal_opportunity_policy(stops)


def test_one_record_serves_two_parts_with_their_own_prices():
    pump = NonRepairable(
        LIFE, planned_cost=4.0, unplanned_cost=20.0, opportunity_cost=1.0
    )
    seal = NonRepairable(
        LIFE, planned_cost=4.0, unplanned_cost=20.0, opportunity_cost=3.5
    )
    stops = Opportunities([Opportunity("stop", period=28.0)])
    a = new(lambda: pump.optimal_opportunity_policy(stops))
    b = new(lambda: seal.optimal_opportunity_policy(stops))
    assert a.prices.price("stop") == 1.0 and b.prices.price("stop") == 3.5
    assert a.cost_rate < b.cost_rate
    # Each is that part's policy over a calendar priced for it.
    for part, policy, price in ((pump, a, 1.0), (seal, b, 3.5)):
        alone = old(
            lambda: NonRepairable(LIFE).optimal_calendar_policy(
                Calendar(
                    [Opportunity("stop", price, period=28.0)],
                    failure=20.0,
                    anytime=4.0,
                )
            )
        )
        same_policy(alone, policy)


def test_where_uses_only_what_takes_that_label_down():
    site = Opportunities(
        [
            Opportunity("site stop", Recurrence.weekly("Sat")),
            Opportunity(
                "line 2 shut",
                Recurrence.every(4, "weeks", "2026-11-18"),
                takes_down=["Line 2"],
            ),
        ],
        any_day=False,
    )
    part = NonRepairable(LIFE, unplanned_cost=20.0, opportunity_cost=3.0)
    line1 = new(
        lambda: part.optimal_opportunity_policy(
            site, where="Line 1", unit="days", now=NOW
        )
    )
    assert line1.prices.unused == ("line 2 shut",)
    assert (
        "doesn't take 'Line 1' down" in line1.prices.row("line 2 shut").reason
    )
    line2 = new(
        lambda: part.optimal_opportunity_policy(
            site, where="Line 2", unit="days", now=NOW
        )
    )
    assert line2.prices.unused == ()


def test_any_day_false_allows_no_other_day():
    part = NonRepairable(LIFE, planned_cost=4.0, unplanned_cost=20.0)
    stops = Opportunities([Opportunity("stop", period=28.0)], any_day=False)
    policy = new(lambda: part.optimal_opportunity_policy(stops))
    assert not policy.prices.row("any day").used
    assert policy.replacement_rates.get("anytime", 0.0) == 0.0


# ------------------------------------------------------------ the helper


def test_the_helper_s_prices_by_hand():
    site = Opportunities(
        [
            Opportunity(
                "weekly stop", Recurrence.weekly("Sat"), duration="8h"
            ),
            Opportunity("short stop", Recurrence.weekly("Wed"), duration="4h"),
            Opportunity("turnaround", dates=["2027-03-03"]),
        ]
    )
    prices = NonRepairable.opportunity_prices(
        site,
        replacement_cost=1_200,
        replacement_duration="6h",
        failure_cost=4_200,
        failure_duration="18h",
        downtime_cost=2_500,
    )
    assert isinstance(prices, OpportunityPrices)
    assert prices.price("weekly stop") == 1_200.0
    assert prices.price("turnaround") == 1_200.0  # no length: used
    assert prices.unused == ("short stop",)
    assert prices.price("any day") == 1_200.0 + 2_500.0 * 6.0 == 16_200.0
    assert prices.price("failure") == 4_200.0 + 2_500.0 * 18.0 == 49_200.0
    json.dumps(prices.to_dict())
    # Applied: the part's own prices, the downtime cost kept off it.
    part = prices.apply(NonRepairable(LIFE))
    assert (part.cp, part.cu, part.opportunity_cost) == (
        16_200.0,
        49_200.0,
        1_200.0,
    )
    assert part.job_duration == "6h" and part.opportunity_costs == {}
    assert not hasattr(part, "downtime_cost")
    policy = new(
        lambda: part.optimal_opportunity_policy(site, unit="days", now=NOW)
    )
    for given, worked_out in zip(policy.prices.rows, prices.rows):
        assert given.name == worked_out.name
        assert given.used == worked_out.used
        if given.used:
            assert given.price == worked_out.price


def test_the_helper_in_model_time():
    stops = Opportunities(
        [
            Opportunity("long", period=28.0, duration=1.0),
            Opportunity("short", period=7.0, duration=0.1),
        ]
    )
    prices = NonRepairable.opportunity_prices(
        stops,
        replacement_cost=1.0,
        replacement_duration=0.25,
        failure_cost=5.0,
        failure_duration=1.0,
        downtime_cost=8.0,
    )
    assert prices.price("long") == 1.0 and prices.unused == ("short",)
    assert prices.price("any day") == 3.0 and prices.price("failure") == 13.0
    with pytest.raises(ValueError, match="in model time"):
        NonRepairable.opportunity_prices(
            stops,
            replacement_cost=1.0,
            replacement_duration="6h",
            failure_cost=5.0,
            failure_duration=1.0,
            downtime_cost=8.0,
        )


def test_a_policy_s_prices_cannot_be_applied():
    policy = plant_part().optimal_opportunity_policy(PLANT)
    with pytest.raises(ValueError, match="only"):
        policy.prices.apply(NonRepairable(LIFE))


# ------------------------------------------------- the opportunities


def test_a_mix_of_clocks_is_refused():
    with pytest.raises(ValueError, match="'stop' is in model time"):
        Opportunities(
            [
                Opportunity("stop", period=7.0),
                Opportunity("shut", Recurrence.weekly("Sat")),
            ]
        )
    with pytest.raises(ValueError, match="period= is for model time"):
        Opportunity("x", Recurrence.weekly("Sat"), period=7.0)
    with pytest.raises(ValueError, match="time_zone is for"):
        Opportunities(
            [Opportunity("stop", period=7.0)], time_zone="Australia/Brisbane"
        )
    with pytest.raises(ValueError, match="model time a duration is a number"):
        Opportunity("stop", period=7.0, duration="8h")
    with pytest.raises(ValueError, match="takes_down"):
        Opportunity("stop", period=7.0, takes_down=[])


@pytest.mark.parametrize(
    "opportunity",
    [
        Opportunity("stop", period=28.0, offset=3.0, duration=0.5),
        Opportunity("turnaround", dates=[100.0, 400.0]),
        Opportunity(
            "weekly stop",
            Recurrence.weekly("Sat"),
            duration="8h",
            at="21:00",
            takes_down=["Line 2"],
        ),
        Opportunity(
            "turnaround",
            dates=["2027-03-03", "2028-09-09"],
            then_every="4 years",
            duration="21d",
        ),
        Opportunity(
            "service", every=Usage(15_000, "km"), also_every="6 months"
        ),
    ],
)
def test_an_opportunity_is_saved_and_loaded(opportunity):
    data = json.loads(json.dumps(opportunity.to_dict()))
    assert Opportunity.from_dict(data) == opportunity
    hash(opportunity)


def test_opportunities_are_saved_and_loaded():
    site = Opportunities(
        [STOP, TURNAROUND], any_day=False, time_zone="Australia/Brisbane"
    )
    data = json.loads(json.dumps(site.to_dict()))
    assert Opportunities.from_dict(data) == site
    # A saved Shutdowns loads, its prices kept as the deprecated ones.
    saved = old_site().to_dict()
    loaded = new(lambda: Opportunities.from_dict(saved))
    part = NonRepairable(LIFE, planned_cost=1.0, unplanned_cost=20.0)
    options = dict(unit="days", now=NOW)
    same_policy(
        old(lambda: part.optimal_calendar_policy(old_site(), **options)),
        new(lambda: part.optimal_opportunity_policy(loaded, **options)),
    )


def test_the_part_s_prices_are_saved_with_it():
    part = NonRepairable(
        LIFE,
        opportunity_cost=1.0,
        job_duration="6h",
        opportunity_costs={Opportunity("annual", period=364.0): 0.5},
    )
    rbd = RepairableRBD([("s", "p"), ("p", "t")], {"p": part})
    back = RBD.from_dict(json.loads(json.dumps(rbd.to_dict()))).components["p"]
    assert back.opportunity_cost == 1.0 and back.job_duration == "6h"
    assert back.opportunity_costs == {"annual": 0.5}


def test_part_prices_are_checked():
    with pytest.raises(ValueError, match="opportunity_cost"):
        NonRepairable(LIFE, opportunity_cost=-1.0)
    with pytest.raises(ValueError, match="job_duration"):
        NonRepairable(LIFE, job_duration=0.0)
    with pytest.raises(ValueError, match="job_duration"):
        NonRepairable(LIFE, job_duration="six hours")
    with pytest.raises(TypeError, match="keys"):
        NonRepairable(LIFE, opportunity_costs={3: 1.0})
    with pytest.raises(ValueError, match="opportunity_costs"):
        NonRepairable(LIFE, opportunity_costs={"a": -1.0})


def test_the_site_s_dates_and_outages_need_no_prices():
    site = Opportunities(
        [Opportunity("weekly stop", Recurrence.weekly("Sat"), duration="8h")]
    )
    assert site.dates(NOW, "2026-11-30")["weekly stop"][0].isoformat() == (
        "2026-11-21"
    )
    outages = site.on_diagram(NOW, "2026-11-30", unit="hours")
    assert outages["weekly stop"]["duration"] == 8.0
    part = NonRepairable(LIFE, planned_cost=1.0, unplanned_cost=5.0)
    route = site.route(part, unit="days", now=NOW)
    assert route.route == "exact"


# --------------------------------------------------------- deprecations


@pytest.mark.parametrize(
    "make, match",
    [
        (
            lambda: Opportunity("w", 3.0, period=7.0),
            r"Opportunity\(name, cost, ...\) puts a price.*opportunity_cost",
        ),
        (
            lambda: Opportunity("w", period=7.0, cost=3.0),
            r"Opportunity\(..., cost=\) puts a price",
        ),
        (
            lambda: Calendar(),
            "Calendar is deprecated: use Opportunities",
        ),
        (
            lambda: Shutdown("s", Recurrence.weekly("Sat"), replace_cost=1.0),
            "Shutdown is deprecated: use Opportunity",
        ),
        (
            lambda: Shutdowns([]),
            "Shutdowns is deprecated: use Opportunities",
        ),
        (
            lambda: Opportunities(any_day=4.0),
            r"Opportunities\(any_day=<price>\) puts a price",
        ),
        (
            lambda: NonRepairable(
                LIFE, planned_cost=1.0, unplanned_cost=5.0
            ).optimal_calendar_policy(Opportunities()),
            "optimal_calendar_policy is deprecated: use "
            "NonRepairable.optimal_opportunity_policy",
        ),
        (
            lambda: NonRepairable(
                LIFE, planned_cost=1.0, unplanned_cost=5.0
            ).calendar_policy_cost(Opportunities(), {}),
            "calendar_policy_cost is deprecated: use "
            "NonRepairable.opportunity_policy_cost",
        ),
    ],
)
def test_each_old_form_warns_with_its_replacement(make, match):
    with pytest.warns(FutureWarning, match=match) as caught:
        make()
    assert any(NEXT_REMOVAL in str(w.message) for w in caught)


def test_the_0_14_positional_form_still_reads():
    legacy = old(lambda: Opportunity("annual", 1.0, 364.0, 182.0))
    assert (legacy.cost, legacy.period, legacy.offset) == (1.0, 364.0, 182.0)
    overhaul = old(
        lambda: Opportunity.from_recurrence(
            "overhaul",
            1.0,
            Recurrence.yearly(month=11, weekday="Tuesday", nth=1),
            "2026-11-14",
            "2029-12-31",
            unit="days",
        )
    )
    assert overhaul.cost == 1.0 and overhaul.dates == (353.0, 724.0, 1088.0)


def test_what_0_20_deprecates_goes_in_0_21():
    # Calendar, Shutdown, Shutdowns, the prices on opportunities and the
    # two calendar method names: removed in 0.21. This fails as a reminder
    # once the version reaches it.
    version = tuple(map(int, __version__.split(".")))[:2]
    assert version < tuple(map(int, NEXT_REMOVAL.split(".")))


def test_prices_go_through_json():
    policy = plant_part().optimal_opportunity_policy(PLANT)
    data = json.loads(json.dumps(policy.to_dict()))
    assert [r["name"] for r in data["prices"]["rows"]] == [
        "weekly",
        "quarterly",
        "annual",
        "any day",
        "failure",
    ]
    assert np.isfinite(data["cost_rate"])
