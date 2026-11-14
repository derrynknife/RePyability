"""Duty plans (``repyability.schedule.DutyPlan``): the parts, layering
from specific to general, time zones, and saving.

Every number here is worked by hand in the comments, from the clock
times and the calendar.
"""

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from repyability.schedule import (
    Continuous,
    DutyPlan,
    Rate,
    Roster,
    Runs,
    Shifts,
)
from repyability.schedule._duty import part_from_dict


def hours(plan, start, end, **kw):
    """The hours a plan gives from ``start`` to ``end``."""
    f = plan.forecast(start, end, **kw)
    return f.at(end)["hours"] - f.at(start)["hours"]


# --- the parts -------------------------------------------------------------


def test_runs_shifts_roster_hours():
    # 12:31 to 03:27 the next day: 11 h 29 min + 3 h 27 min = 14 h 56 min.
    run = Runs([("2026-11-14 12:31", "2026-11-15 03:27")])
    assert run.hours == pytest.approx(14 + 56 / 60)
    assert Shifts("Mon-Fri", "06:00", "22:00").hours_a_week == 80.0
    assert Shifts("Fri", "22:00", "06:00").hours_a_week == 8.0
    # 4 on, 4 off from Monday 2 Nov 2026, 12 h a day: 2-5 Nov and 10-13
    # Nov on, so 8 days in 2-15 Nov, 96 h.
    roster = DutyPlan(Roster((4, 4), "2026-11-02", "07:00", "19:00"))
    assert hours(roster, "2026-11-02", "2026-11-16") == pytest.approx(96.0)
    assert Roster((4, 4), "2026-11-02", "07:00", "19:00").on(
        datetime(2026, 11, 10).date()
    )


def test_continuous_and_rates():
    # all the time: 24 h a day; 3 h a day; 15 h in each calendar month
    assert hours(DutyPlan(Continuous()), "2027-01-01", "2027-01-08") == 168.0
    assert hours(DutyPlan(Rate(3, per="day")), "2027-01-01", "2027-01-11") == (
        pytest.approx(30.0)
    )
    month = DutyPlan(Rate(15, per="month"))
    f = month.forecast("2027-01-01", "2027-04-01")
    assert f.by_period("month").usage["hours"] == pytest.approx((15, 15, 15))
    # half of February (14 of 28 days) is half its 15 h
    assert hours(month, "2027-02-01", "2027-02-15") == pytest.approx(7.5)
    week = DutyPlan(Rate(1200, per="week", meter="km"))
    f = week.forecast("2026-11-16", "2026-11-30")
    assert f.meters == ("km",)
    assert f.at("2026-11-30")["km"] == pytest.approx(2400.0)


def test_describe():
    assert (
        Shifts("Mon-Fri", "06:00", "22:00").describe() == "Mon-Fri 06:00-22:00"
    )
    assert Rate(15, per="month").describe() == "about 15 h a month"
    assert Rate((2, 4), per="day").describe() == "about 2 to 4 h a day"
    assert Rate(1200, per="week", meter="km").describe() == (
        "about 1,200 km a week"
    )
    assert Roster((4, 4), "2026-11-02", "07:00", "19:00").describe() == (
        "4 on, 4 off, 07:00-19:00"
    )
    assert Runs([("2026-11-14 12:31", "2026-11-15 03:27")]).describe() == (
        "exact runs"
    )


@pytest.mark.parametrize(
    "make, words",
    [
        (lambda: Shifts("Mon", "06:00", "06:00"), "same time"),
        (
            lambda: Shifts(
                "Mon-Fri",
                "06:00",
                "14:00",
                from_="2027-01-01",
                until="2026-01-01",
            ),
            "after from_",
        ),
        (
            lambda: Shifts(
                [("Mon", "06:00", "14:00"), ("Mon", "13:00", "20:00")]
            ),
            "overlap",
        ),
        (lambda: Shifts("Funday", "06:00", "14:00"), "days of the week"),
        (lambda: Runs([("2026-11-15", "2026-11-14")]), "stop after"),
        (
            lambda: Runs(
                [
                    ("2026-11-14 01:00", "2026-11-14 05:00"),
                    ("2026-11-14 04:00", "2026-11-14 06:00"),
                ]
            ),
            "overlap",
        ),
        (lambda: Rate(30, per="day"), "more hours"),
        (lambda: Rate((4, 2), per="day"), "at most"),
        (lambda: Rate(3, per="fortnight"), "per must be"),
        (lambda: Roster((4,), "2026-11-02", "07:00", "19:00"), "pattern"),
        (lambda: DutyPlan(), "at least one part"),
        (
            lambda: DutyPlan(Continuous(), time_zone="Mars/Olympus"),
            "not a time zone",
        ),
        (lambda: DutyPlan(Continuous(), utilisation=1.5), "utilisation"),
        (lambda: DutyPlan(Continuous(), season="Smarch"), "season"),
        (lambda: DutyPlan(Continuous(), fill="all"), "fill"),
        (
            lambda: DutyPlan(
                Continuous(),
                Rate(3, per="day", meter="tonnes"),
                per_hour={"tonnes": 4},
            ),
            "both",
        ),
    ],
)
def test_refusals_say_what_to_do(make, words):
    with pytest.raises((ValueError, TypeError), match=words):
        make()


def test_overlapping_parts_are_refused_naming_both():
    with pytest.raises(ValueError) as err:
        DutyPlan(
            Shifts("Mon-Fri", "06:00", "22:00", until="2027-03-15"),
            Rate(3, per="day", from_="2027-03-01"),
        )
    assert "Mon-Fri 06:00-22:00" in str(err.value)
    assert "about 3 h a day" in str(err.value)
    with pytest.raises(ValueError, match="specific parts overlap"):
        DutyPlan(
            Continuous(),
            specific=[
                Runs([("2026-11-14 12:00", "2026-11-14 18:00")]),
                Rate(
                    3, per="day", from_="2026-11-14 17:00", until="2026-11-15"
                ),
            ],
        )


def test_a_later_background_part_ends_the_one_before():
    # Mon-Fri one shift (16 h) until 1 Mar, two shifts with a break from
    # then (06:00-14:00 and 14:30-22:30, also 16 h a day, but 22:30).
    plan = DutyPlan(
        Shifts("Mon-Fri", "06:00", "22:00"),
        Shifts(
            [("Mon-Fri", "06:00", "14:00"), ("Mon-Fri", "14:30", "22:30")],
            from_="2027-03-01",
        ),
    )
    f = plan.forecast("2027-02-22", "2027-03-08")
    # week of 22 Feb: 80 h; week of 1 Mar: 80 h
    assert f.at("2027-03-01")["hours"] == pytest.approx(80.0)
    assert f.at("2027-03-08")["hours"] == pytest.approx(160.0)
    # On Mon 1 Mar at 22:15 the second pattern is running (14:30-22:30).
    assert f.at("2027-03-01 22:15")["hours"] - f.at("2027-03-01 14:30")[
        "hours"
    ] == pytest.approx(7.75)
    assert f.describe() == (
        "Mon-Fri 06:00-22:00 to 1 Mar, then Mon-Fri 06:00-14:00 and "
        "Mon-Fri 14:30-22:30"
    )


# --- options ---------------------------------------------------------------


def test_season_outages_utilisation_and_other_meters():
    shifts = Shifts("Mon-Fri", "06:00", "22:00")
    # April to October only: in a week of March nothing; a week of April 80
    seasonal = DutyPlan(shifts, season="Apr-Oct")
    assert hours(seasonal, "2027-03-08", "2027-03-15") == 0.0
    assert hours(seasonal, "2027-04-05", "2027-04-12") == 80.0
    # an outage from Tue 10:00 to Wed 10:00: Tue 06-10 and Wed 10-22, 16 h
    # lost... Tue loses 12 h (10-22), Wed loses 4 h (06-10): 64 h in the
    # week, and Wednesday's run, cut at its start, is a run of its own.
    down = DutyPlan(shifts, outages=[("2027-04-06 10:00", "2027-04-07 10:00")])
    f = down.forecast("2027-04-05", "2027-04-12")
    assert f.at("2027-04-12")["hours"] == pytest.approx(64.0)
    assert f.at("2027-04-12")["starts"] == 5.0
    # an outage in the middle of a run splits it into two runs
    split = DutyPlan(
        shifts, outages=[("2027-04-05 10:00", "2027-04-05 11:00")]
    )
    f = split.forecast("2027-04-05", "2027-04-06")
    assert f.at("2027-04-06")["starts"] == 2.0
    assert f.at("2027-04-06")["hours"] == 15.0
    # 90% utilisation: 72 h and 4.5 starts expected in the week
    f = DutyPlan(shifts, utilisation=0.9).forecast("2027-04-05", "2027-04-12")
    assert f.at("2027-04-12")["hours"] == pytest.approx(72.0)
    assert f.at("2027-04-12")["starts"] == pytest.approx(4.5)
    # 40 t an hour and one cycle a run
    f = DutyPlan(
        shifts, per_hour={"tonnes": 40}, per_run={"cycles": 1}
    ).forecast("2027-04-05", "2027-04-12")
    assert f.meters == ("hours", "starts", "tonnes", "cycles")
    assert f.at("2027-04-12")["tonnes"] == pytest.approx(3200.0)
    assert f.at("2027-04-12")["cycles"] == 5.0


# --- time zones ------------------------------------------------------------


@pytest.mark.parametrize(
    "zone, forward, back",
    [
        ("Australia/Sydney", 7.0, 9.0),
        ("Australia/Brisbane", 8.0, 8.0),
        (None, 8.0, 8.0),
    ],
)
def test_a_night_shift_across_daylight_saving(zone, forward, back):
    # Sydney's clocks go forward at 02:00 on Sun 4 Oct 2026 and back at
    # 03:00 on Sun 4 Apr 2027; Brisbane has no daylight saving.
    plan = DutyPlan(Shifts("Sat", "22:00", "06:00"), time_zone=zone)
    assert hours(plan, "2026-10-03", "2026-10-05") == pytest.approx(forward)
    assert hours(plan, "2027-04-03", "2027-04-05") == pytest.approx(back)


def test_times_that_do_not_exist_or_happen_twice():
    plan = DutyPlan(
        Runs([("2026-10-04 02:30", "2026-10-04 05:00")]),
        time_zone="Australia/Sydney",
    )
    f = plan.forecast("2026-10-04", "2026-10-05")
    # 02:30 does not exist: the run starts at 03:00, 2 h to 05:00
    assert f.at("2026-10-05")["hours"] == pytest.approx(2.0)
    assert any("does not exist" in n and "03:00" in n for n in f.notes)
    # 02:30 on 4 Apr 2027 happens twice: the first (AEDT); to 03:30 AEST
    # is 2 real hours
    plan = DutyPlan(
        Runs([("2027-04-04 02:30", "2027-04-04 03:30")]),
        time_zone="Australia/Sydney",
    )
    f = plan.forecast("2027-04-04", "2027-04-05")
    assert f.at("2027-04-05")["hours"] == pytest.approx(2.0)
    assert any("happens twice" in n for n in f.notes)
    # answers carry the zone
    assert f.start.tzinfo is not None
    assert str(f.start) == "2027-04-04 00:00:00+11:00"


def test_a_moment_given_with_a_zone_is_converted():
    plan = DutyPlan(Continuous(), time_zone="Australia/Brisbane")
    f = plan.forecast("2027-01-01 00:00+00:00", "2027-01-02")
    assert str(f.start) == "2027-01-01 10:00:00+10:00"
    assert f.at("2027-01-02")["hours"] == pytest.approx(14.0)


# --- from specific to general ----------------------------------------------


def generator_plan(**kw):
    return DutyPlan(
        background=Rate(3, per="day"),
        specific=[
            Runs(
                [
                    ("2026-11-14 12:31", "2026-11-15 03:27"),
                    ("2026-11-15 12:30", "2026-11-16 03:30"),
                ],
                from_="2026-11-14",
            )
        ],
        **kw,
    )


def test_layers_say_where_the_hours_come_from():
    f = generator_plan().forecast("2026-11-14", "2027-11-14")
    assert f.describe() == "exact runs to 16 Nov 03:30, then about 3 h a day"
    layers = [(s.layer, s.description, s.exact) for s in f.stretches]
    assert layers == [
        ("specific", "exact runs", True),
        ("background", "about 3 h a day", False),
    ]
    # Without from_, the runs cover from their first start: the morning
    # of 14 Nov is the background's.
    plan = DutyPlan(
        background=Rate(3, per="day"),
        specific=[Runs([("2026-11-14 12:31", "2026-11-15 03:27")])],
    )
    assert plan.forecast("2026-11-14", "2026-11-20").describe() == (
        "about 3 h a day to 14 Nov 12:31, then exact runs to 15 Nov 03:27, "
        "then about 3 h a day"
    )


def test_pro_rata_handover_adds_no_duty():
    # Tue 16 Nov: the runs give 03:30 h to 03:30; the background gives the
    # 20.5 h left of the day 3 x 20.5/24 = 2.5625 h; never 3.5 + 3.
    f = generator_plan().forecast("2026-11-14", "2026-11-20")
    days = f.by_period("day").usage["hours"]
    assert days[0] == pytest.approx(11 + 29 / 60)  # 12:31 to midnight
    assert days[1] == pytest.approx(3 + 27 / 60 + 11.5)  # to 03:27, 12:30 on
    assert days[2] == pytest.approx(3.5 + 3 * 20.5 / 24)
    assert days[3:] == pytest.approx((3.0, 3.0, 3.0))


@pytest.mark.parametrize(
    "plan, start, end, biggest",
    [
        # mid-day: a specific rate of 5 h a day to noon, then 3 h a day
        (
            DutyPlan(
                Rate(3, per="day"),
                specific=[
                    Rate(
                        5,
                        per="day",
                        from_="2027-01-01",
                        until="2027-01-05 12:00",
                    )
                ],
            ),
            "2027-01-01",
            "2027-01-10",
            5.0,
        ),
        # at midnight
        (
            DutyPlan(
                Rate(3, per="day"),
                specific=[
                    Rate(5, per="day", from_="2027-01-01", until="2027-01-05")
                ],
            ),
            "2027-01-01",
            "2027-01-10",
            5.0,
        ),
        # mid-shift: the same shifts, specific, to 12:00 on Wed 6 Jan
        (
            DutyPlan(
                Shifts("Mon-Fri", "06:00", "22:00"),
                specific=[
                    Shifts(
                        "Mon-Fri",
                        "06:00",
                        "22:00",
                        from_="2027-01-04",
                        until="2027-01-06 12:00",
                    )
                ],
            ),
            "2027-01-04",
            "2027-01-11",
            16.0,
        ),
    ],
)
def test_aligned_handovers_stay_within_the_parts(plan, start, end, biggest):
    # Each day at most the bigger of the two parts' day totals.
    days = plan.forecast(start, end).by_period("day").usage["hours"]
    assert max(days) <= biggest + 1e-9
    # the handover day of the rates: 5 x 12/24 + 3 x 12/24 = 4 h
    if biggest == 5.0 and "12:00" in str(plan.specific[0].until):
        assert days[4] == pytest.approx(4.0)


def test_a_shift_partly_covered_counts_only_its_uncovered_part():
    # Background Mon-Fri 06:00-22:00; specific runs cover Mon 4 Jan 2027
    # 00:00 to 12:00 with one run 08:00-10:00. Monday: 2 h + 12:00-22:00
    # of the shift, 10 h; and the shift cut at 12:00 is no new start.
    plan = DutyPlan(
        Shifts("Mon-Fri", "06:00", "22:00"),
        specific=[
            Runs(
                [("2027-01-04 08:00", "2027-01-04 10:00")],
                from_="2027-01-04",
                until="2027-01-04 12:00",
            )
        ],
    )
    f = plan.forecast("2027-01-04", "2027-01-05")
    assert f.at("2027-01-05")["hours"] == pytest.approx(12.0)
    assert f.at("2027-01-05")["starts"] == 1.0


def test_handover_across_daylight_saving():
    # Sydney, Sun 4 Oct 2026 is 23 real hours. Runs to 01:00 (1 h), then
    # 3 h a day spread over the day's real length: 3 x 22/23 more.
    plan = DutyPlan(
        Rate(3, per="day"),
        specific=[Runs([("2026-10-04 00:00", "2026-10-04 01:00")])],
        time_zone="Australia/Sydney",
    )
    f = plan.forecast("2026-10-03", "2026-10-06")
    days = f.by_period("day").usage["hours"]
    assert days == pytest.approx((3.0, 1.0 + 3 * 22 / 23, 3.0))


def test_top_up_fills_the_budget_and_never_adds():
    def plan(run_hours, fill):
        return DutyPlan(
            Rate(15, per="month"),
            specific=[
                Runs(
                    [
                        (
                            "2026-11-02 08:00",
                            datetime(2026, 11, 2, 8)
                            + timedelta(hours=run_hours),
                        )
                    ],
                    from_="2026-11-01",
                    until="2026-11-11",
                )
            ],
            fill=fill,
        )

    # 10 h of runs to 11 Nov; top up: November gets 15 h in all, the 5 h
    # left spread over 11-30 Nov; December its full 15 h.
    f = plan(10, "top_up").forecast("2026-11-01", "2027-01-01")
    months = f.by_period("month").usage["hours"]
    assert months == pytest.approx((15.0, 15.0))
    assert "topped up" in f.describe()
    assert f.fill == "top_up"
    # 20 h of runs, more than the budget: nothing added (never below 0)
    f = plan(20, "top_up").forecast("2026-11-01", "2027-01-01")
    assert f.by_period("month").usage["hours"] == pytest.approx((20.0, 15.0))
    # pro rata: 10 h + 15 x 20/30 of the month left
    f = plan(10, "pro_rata").forecast("2026-11-01", "2027-01-01")
    assert f.by_period("month").usage["hours"] == pytest.approx((20.0, 15.0))
    assert "topped up" not in f.describe()


def test_top_up_across_a_month_end():
    # Runs 30 Nov 20:00 to 1 Dec 04:00 (4 h in each month), covering to
    # 2 Dec: November's budget is topped up from its own 4 h before the
    # runs (none left uncovered after them), December's from 4 h.
    plan = DutyPlan(
        Rate(15, per="month"),
        specific=[
            Runs(
                [("2026-11-30 20:00", "2026-12-01 04:00")],
                from_="2026-11-30",
                until="2026-12-02",
            )
        ],
        fill="top_up",
    )
    f = plan.forecast("2026-11-01", "2027-01-01")
    assert f.by_period("month").usage["hours"] == pytest.approx((15.0, 15.0))


def handover_plans():
    runs = Runs(
        [
            ("2026-11-14 12:31", "2026-11-15 03:27"),
            ("2026-11-15 12:30", "2026-11-16 03:30"),
        ],
        from_="2026-11-14",
    )
    for zone in (None, "Australia/Sydney"):
        for back in (
            Rate(3, per="day"),
            Rate(20, per="day"),
            Rate((2, 4), per="day"),
            Rate(90, per="month"),
            Shifts("Mon-Sun", "06:00", "22:00"),
            Roster((4, 4), "2026-11-10", "18:00", "06:00"),
            Continuous(),
        ):
            for fill in ("pro_rata", "top_up"):
                yield DutyPlan(
                    back, specific=[runs], time_zone=zone, fill=fill
                )


@pytest.mark.parametrize("plan", list(handover_plans()), ids=str)
def test_days_add_up_and_never_pass_24_hours(plan):
    f = plan.forecast("2026-11-10", "2026-12-20")
    days = f.by_period("day")
    total = sum(days.usage["hours"])
    change = f.at("2026-12-20")["hours"] - f.at("2026-11-10")["hours"]
    assert total == pytest.approx(change, abs=1e-9)
    for got, start, end in zip(days.usage["hours"], days.starts, days.ends):
        assert got <= (end - start) / timedelta(hours=1) + 1e-9
    for row in (days.low, days.high):
        assert max(row["hours"]) <= 24.0 + 1e-9


# --- saving ----------------------------------------------------------------


PARTS = [
    Runs([("2026-11-14 12:31", "2026-11-15 03:27")], until="2026-11-16"),
    Shifts(
        [("Mon-Fri", "06:00", "14:00"), ("Sat", "22:00", "06:00")],
        season=[4, 5, 6],
        utilisation=0.8,
    ),
    Roster(
        (2, 2, 3, 2, 2, 3), "2026-11-02", "07:00", "19:00", from_="2026-11-02"
    ),
    Rate((2, 4), per="day"),
    Rate(1200, per="week", meter="km"),
    Continuous(from_="2026-01-01", until="2026-02-01"),
]


@pytest.mark.parametrize("part", PARTS, ids=lambda p: p.kind)
def test_parts_round_trip(part):
    data = json.loads(json.dumps(part.to_dict()))
    assert data["kind"] == part.kind
    assert part_from_dict(data) == part


def test_plan_round_trips():
    plan = DutyPlan(
        Shifts("Mon-Fri", "06:00", "22:00"),
        Rate(1200, per="week", meter="km"),
        specific=[Runs([("2026-11-14 12:31", "2026-11-15 03:27")])],
        time_zone="Australia/Sydney",
        season="Nov-Feb",
        outages=[("2026-12-24", "2026-12-27")],
        utilisation=0.95,
        per_hour={"tonnes": 40},
        per_run={"cycles": 1},
        fill="top_up",
    )
    data = json.loads(json.dumps(plan.to_dict()))
    assert data["version"] == 1
    again = DutyPlan.from_dict(data)
    assert again == plan
    assert again.season == (1, 2, 11, 12)
    with pytest.raises(ValueError, match="later RePyability"):
        DutyPlan.from_dict({**data, "version": 99})


def test_forecast_to_dict_is_plain_json():
    f = generator_plan().forecast("2026-11-14", "2026-12-14")
    data = json.loads(json.dumps(f.to_dict()))
    assert data["description"] == f.describe()
    assert data["stretches"][0]["layer"] == "specific"
    assert data["plan"]["kind"] == "duty_plan"
    assert np.isclose(data["at_end"]["hours"], f.at(f.end)["hours"])
