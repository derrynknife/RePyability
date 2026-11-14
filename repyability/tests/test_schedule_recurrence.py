"""Recurrence rules and seasons (repyability.schedule, UC-2, UC-4, UC-28,
UC-29 and UC-31 of ``repyability/schedule/README.md``): the hand-worked
dates, and every rule against its definition walked a day at a time."""

import calendar
import json
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from repyability.schedule import Recurrence, RecurrenceNotes, Season
from repyability.schedule._recurrence import _read_rrule

D = date.fromisoformat
WEEKDAYS = list(calendar.day_name)


def iso(days):
    return [d.isoformat() for d in days]


# ---------------------------------------------------------------- hand-worked


def test_a_weekly_task_through_the_rest_of_2026():
    # UC-2: first done Wednesday 15 July 2026; from Sunday 11 October to
    # 31 December: every Wednesday from 14 October, 12 times.
    weekly = Recurrence.every(1, "weeks", "2026-07-15")
    dates = weekly.dates("2026-10-11", "2026-12-31")
    assert iso(dates) == [
        "2026-10-14",
        "2026-10-21",
        "2026-10-28",
        "2026-11-04",
        "2026-11-11",
        "2026-11-18",
        "2026-11-25",
        "2026-12-02",
        "2026-12-09",
        "2026-12-16",
        "2026-12-23",
        "2026-12-30",
    ]
    assert all(d.weekday() == 2 for d in dates)
    assert len(weekly.dates("2026-07-15", "2026-10-10")) == 13
    assert len(weekly.dates("2026-01-01", "2026-12-31")) == 25
    # From new year, 52 weeks and a day: 53 times in 2026.
    new_year = Recurrence.every(7, "days", "2026-01-01")
    assert len(new_year.dates("2026-01-01", "2026-12-31")) == 53


def test_the_first_tuesday_and_the_30th_through_the_rest_of_2026():
    # UC-4.
    tuesday = Recurrence.monthly(weekday="Tuesday", nth=1)
    assert iso(tuesday.dates("2026-10-11", "2026-12-31")) == [
        "2026-11-03",
        "2026-12-01",
    ]
    thirtieth = Recurrence.monthly(day=30)
    assert iso(thirtieth.dates("2026-10-11", "2026-12-31")) == [
        "2026-10-30",
        "2026-11-30",
        "2026-12-30",
    ]


def test_the_first_saturday_of_each_month_from_14_november_2026():
    # UC-29 (and the plan's Example E).
    rule = Recurrence.monthly(weekday="Saturday", nth=1)
    assert iso(rule.dates("2026-11-14", "2027-03-31")) == [
        "2026-12-05",
        "2027-01-02",
        "2027-02-06",
        "2027-03-06",
    ]
    assert rule.describe() == "the first Saturday of every month"


def test_the_first_tuesday_of_november_every_year():
    # UC-28: 2 Nov 2027, 7 Nov 2028, 6 Nov 2029 (2026's was the 3rd,
    # before the period).
    rule = Recurrence.yearly(month=11, weekday="Tuesday", nth=1)
    assert iso(rule.dates("2026-11-14", "2029-12-31")) == [
        "2027-11-02",
        "2028-11-07",
        "2029-11-06",
    ]
    assert rule.describe() == "the first Tuesday of November, every year"
    # 364 or 371 days apart: no fixed period.
    gaps = {
        (b - a).days
        for a, b in zip(
            *(lambda d: (d, d[1:]))(rule.dates("2026-01-01", "2040-12-31"))
        )
    }
    assert gaps == {364, 371}
    assert Recurrence.yearly(month="Nov", weekday="tue", nth=1) == rule


def test_the_fifth_saturdays_of_2027_and_the_months_without_one():
    # UC-29: four, eight months skipped and listed.
    rule = Recurrence.monthly(weekday="Saturday", nth=5)
    assert iso(rule.dates("2027-01-01", "2027-12-31")) == [
        "2027-01-30",
        "2027-05-29",
        "2027-07-31",
        "2027-10-30",
    ]
    notes = rule.notes("2027-01-01", "2027-12-31")
    assert isinstance(notes, RecurrenceNotes)
    assert notes.missing_months == (
        "2027-02",
        "2027-03",
        "2027-04",
        "2027-06",
        "2027-08",
        "2027-09",
        "2027-11",
        "2027-12",
    )
    assert notes.text[0].startswith(
        "8 months have no fifth Saturday and are skipped: Feb 2027"
    )
    assert "skipped" in rule.describe()
    # The fifth from last is the first in a month with five.
    last5 = Recurrence.monthly(weekday="Saturday", nth=-5)
    assert iso(last5.dates("2027-01-01", "2027-12-31")) == [
        "2027-01-02",
        "2027-05-01",
        "2027-07-03",
        "2027-10-02",
    ]


def test_short_months_fall_on_their_last_day_or_are_skipped():
    last, skip = Recurrence.monthly(day=30), Recurrence.monthly(
        day=30, short_months="skip"
    )
    assert last.dates("2027-02-01", "2027-02-28") == (D("2027-02-28"),)
    assert skip.dates("2027-02-01", "2027-02-28") == ()
    assert skip.notes("2027-02-01", "2027-02-28").missing_months == (
        "2027-02",
    )
    assert last.dates("2028-02-01", "2028-02-29") == (D("2028-02-29"),)
    assert (
        len(Recurrence.monthly(day=31).dates("2026-01-01", "2026-12-31")) == 12
    )
    longest = Recurrence.monthly(day=31, short_months="skip")
    assert len(longest.dates("2026-01-01", "2026-12-31")) == 7
    assert len(longest.notes("2026-01-01", "2026-12-31").missing_months) == 5


def test_months_from_an_anchor_keep_its_day():
    # 31 January: a shorter month's last day, counted from the anchor, not
    # from the month before (so 31 March, not 28 March).
    monthly = Recurrence.every(1, "months", "2027-01-31")
    assert iso(monthly.dates("2027-01-01", "2027-05-31")) == [
        "2027-01-31",
        "2027-02-28",
        "2027-03-31",
        "2027-04-30",
        "2027-05-31",
    ]
    quarterly = Recurrence.every(3, "months", "2026-11-30")
    assert iso(quarterly.dates("2026-11-01", "2027-12-31")) == [
        "2026-11-30",
        "2027-02-28",
        "2027-05-30",
        "2027-08-30",
        "2027-11-30",
    ]
    skip = Recurrence.every(1, "months", "2027-01-31", short_months="skip")
    assert len(skip.dates("2027-01-01", "2027-12-31")) == 7


def test_leap_days():
    yearly = Recurrence.every(1, "years", "2028-02-29")
    assert iso(yearly.dates("2028-01-01", "2032-12-31")) == [
        "2028-02-29",
        "2029-02-28",
        "2030-02-28",
        "2031-02-28",
        "2032-02-29",
    ]
    only_leap = Recurrence.yearly(month=2, day=29, short_months="skip")
    assert iso(only_leap.dates("2027-01-01", "2032-12-31")) == [
        "2028-02-29",
        "2032-02-29",
    ]
    assert only_leap.describe() == "29 February, every year (leap years only)"
    assert len(only_leap.notes("2027-01-01", "2032-12-31").missing_months) == 4


def test_every_so_many_weeks_counts_from_the_anchor_week():
    # Weeks start on Monday: the anchor, Wednesday 15 July, is in the week
    # of Monday 13 July, so every other Monday is 27 July, 10 August, ...
    rule = Recurrence.weekly("Mon", every=2, anchor="2026-07-15")
    assert iso(rule.dates("2026-07-01", "2026-08-31")) == [
        "2026-07-27",
        "2026-08-10",
        "2026-08-24",
    ]
    days = Recurrence.weekly("Mon-Fri")
    assert len(days.dates("2026-11-16", "2026-11-29")) == 10
    assert Recurrence.weekly("Fri-Mon").weekdays == (0, 4, 5, 6)
    assert Recurrence.weekly(["sat", "Sunday"]).weekdays == (5, 6)


def test_both_ends_of_a_period_are_included():
    weekly = Recurrence.every(1, "weeks", "2026-10-14")
    assert len(weekly.dates("2026-10-14", "2026-10-28")) == 3
    assert weekly.dates("2026-10-14", "2026-10-14") == (D("2026-10-14"),)
    assert weekly.dates("2026-10-15", "2026-10-20") == ()
    assert weekly.dates("2026-01-01", "2026-10-13") == ()  # before it


def test_given_dates():
    rule = Recurrence.on(["2028-09-09", "2027-03-03", "2027-03-03"])
    assert rule.given == (D("2027-03-03"), D("2028-09-09"))
    assert rule.dates("2027-01-01", "2028-12-31") == rule.given
    assert rule.after("2027-03-03") == D("2028-09-09")
    assert rule.after("2028-09-09") is None
    assert rule.number("2027-12-31") == 1
    assert rule.describe() == "on 3 Mar 2027 and 9 Sep 2028"


# ---------------------------------------------------------------- options


def test_a_season_leaves_out_the_dates_outside_it():
    summer = Recurrence.monthly(
        weekday="Saturday", nth=1, season=Season(months=[10, 11, 12, 1, 2, 3])
    )
    dates = summer.dates("2027-01-01", "2027-12-31")
    assert [d.month for d in dates] == [1, 2, 3, 10, 11, 12]
    assert len(summer.notes("2027-01-01", "2027-12-31").out_of_season) == 6
    assert summer.describe().endswith(", in October to March only")
    across = Recurrence.weekly(
        "Mon", season=Season(start="12-20", end="01-10")
    )
    assert iso(across.dates("2026-12-01", "2027-01-31")) == [
        "2026-12-21",
        "2026-12-28",
        "2027-01-04",
    ]


def test_excluded_dates_are_skipped_or_moved():
    # The last Friday of December 2026 is Christmas Day.
    fridays = dict(weekday="Friday", nth=-1)
    plain = Recurrence.monthly(**fridays)
    assert plain.dates("2026-12-01", "2026-12-31") == (D("2026-12-25"),)
    skip = Recurrence.monthly(**fridays, excluding=["2026-12-25"])
    assert skip.dates("2026-12-01", "2026-12-31") == ()
    assert skip.notes("2026-12-01", "2026-12-31").excluded == (
        D("2026-12-25"),
    )
    before = Recurrence.monthly(
        **fridays, excluding=[("2026-12-24", "2026-12-28")], moved="before"
    )
    assert before.dates("2026-12-01", "2026-12-31") == (D("2026-12-23"),)
    assert before.notes("2026-12-01", "2026-12-31").moved == (
        (D("2026-12-25"), D("2026-12-23")),
    )
    after = Recurrence.monthly(
        **fridays, excluding=[("2026-12-24", "2026-12-28")], moved="after"
    )
    assert after.dates("2026-12-01", "2026-12-31") == (D("2026-12-29"),)
    assert after.describe() == (
        "the last Friday of every month, except 24 Dec 2026 to 28 Dec 2026 "
        "(moved to the day after)"
    )


def test_a_move_across_the_period_edge_is_found():
    # The date moved into the period comes from outside it.
    rule = Recurrence.on(
        ["2027-01-01"],
        excluding=[("2026-12-30", "2027-01-01")],
        moved="before",
    )
    assert rule.dates("2026-12-01", "2026-12-31") == (D("2026-12-29"),)
    assert rule.dates("2027-01-01", "2027-01-31") == ()
    # Two moved onto one day count once.
    daily = Recurrence.every(
        1,
        "days",
        "2027-01-01",
        excluding=[("2027-01-02", "2027-01-03")],
        moved="after",
    )
    assert iso(daily.dates("2027-01-01", "2027-01-05")) == [
        "2027-01-01",
        "2027-01-04",
        "2027-01-05",
    ]


def test_on_or_after_after_includes_and_number():
    rule = Recurrence.monthly(weekday="Saturday", nth=1, anchor="2026-11-01")
    assert rule.on_or_after("2026-11-07") == D("2026-11-07")
    assert rule.after("2026-11-07") == D("2026-12-05")
    assert rule.on_or_after("2026-01-01") == D("2026-11-07")  # the anchor
    assert rule.includes("2027-01-02") and not rule.includes("2027-01-03")
    assert rule.number("2026-10-31") == 0
    assert rule.number("2026-11-07") == 1
    assert rule.number("2027-01-02") == 3
    with pytest.raises(ValueError, match="anchor"):
        Recurrence.monthly(day=1).number("2027-01-01")


# ---------------------------------------------------------------- walked


def month_length(day):
    return calendar.monthrange(day.year, day.month)[1]


def on_pattern(rule, day):
    """Whether ``day`` is one of a rule's raw dates (before options),
    from the definitions, written out plainly."""
    if rule.anchor is not None and day < rule.anchor:
        return False
    if rule.rule == "every":
        a = rule.anchor
        if rule.unit in ("days", "weeks"):
            step = rule.every_n * (7 if rule.unit == "weeks" else 1)
            return (day - a).days % step == 0
        months = (day.year - a.year) * 12 + day.month - a.month
        step = rule.every_n * (12 if rule.unit == "years" else 1)
        if months % step:
            return False
        if rule.short_months == "skip":
            return day.day == a.day
        return day.day == min(a.day, month_length(day))
    if rule.rule == "weekly":
        if day.weekday() not in rule.weekdays:
            return False
        if rule.anchor is None:
            return True
        week = rule.anchor - timedelta(days=rule.anchor.weekday())
        return ((day - week).days // 7) % rule.every_n == 0
    if rule.rule == "yearly" and day.month != rule.month:
        return False
    if rule.anchor is not None:
        if rule.rule == "monthly":
            since = (day.year - rule.anchor.year) * 12
            since += day.month - rule.anchor.month
        else:
            since = day.year - rule.anchor.year
        if since % rule.every_n:
            return False
    if rule.day is not None:
        if rule.short_months == "skip":
            return day.day == rule.day
        return day.day == min(rule.day, month_length(day))
    if day.weekday() != rule.weekdays[0]:
        return False
    if rule.nth > 0:
        return (day.day - 1) // 7 == rule.nth - 1
    return (month_length(day) - day.day) // 7 == -rule.nth - 1


def walked(rule, start, end):
    days, day = [], start
    while day <= end:
        if on_pattern(rule, day):
            days.append(day)
        day += timedelta(days=1)
    return tuple(days)


def random_rule(rng, origin):
    kind = rng.choice(["every", "weekly", "monthly", "yearly"])
    anchor = origin + timedelta(days=int(rng.integers(0, 1500)))
    nth = int(rng.choice([1, 2, 3, 4, 5, -1, -2, -3, -4, -5]))
    weekday = WEEKDAYS[int(rng.integers(0, 7))]
    short = str(rng.choice(["last", "skip"]))
    if kind == "every":
        unit = str(rng.choice(["days", "weeks", "months", "years"]))
        options = (
            {"short_months": short} if unit in ("months", "years") else {}
        )
        return Recurrence.every(
            int(rng.integers(1, 40 if unit == "days" else 5)),
            unit,
            anchor,
            **options,
        )
    every = int(rng.integers(1, 4))
    given = anchor if every > 1 or rng.random() < 0.5 else None
    if kind == "weekly":
        days = sorted(set(rng.choice(WEEKDAYS, size=int(rng.integers(1, 4)))))
        return Recurrence.weekly(days, every=every, anchor=given)
    if rng.random() < 0.5:
        day = int(rng.integers(1, 32))
        pattern = {"day": day, "short_months": short}
    else:
        pattern = {"weekday": weekday, "nth": nth}
    if kind == "monthly":
        return Recurrence.monthly(**pattern, every=every, anchor=given)
    month = int(rng.integers(1, 13))
    if "day" in pattern:
        pattern["day"] = min(
            pattern["day"], calendar.monthrange(2028, month)[1]
        )
    return Recurrence.yearly(month=month, **pattern, every=every, anchor=given)


def test_every_rule_is_its_definition_walked_a_day_at_a_time():
    rng = np.random.default_rng(0)
    origin = date(2026, 1, 1)
    for _ in range(400):
        rule = random_rule(rng, origin)
        start = origin + timedelta(days=int(rng.integers(0, 1800)))
        end = start + timedelta(days=int(rng.integers(0, 900)))
        assert rule.dates(start, end) == walked(rule, start, end), rule
        # ... and dateutil reads its iCalendar text back to the same.
        text = rule.to_rrule(start=start)
        assert _read_rrule(text, start, end) == rule.dates(start, end), text


def test_the_missing_months_are_the_cycle_months_with_no_date():
    rng = np.random.default_rng(1)
    origin = date(2026, 1, 1)
    for _ in range(200):
        nth = int(rng.choice([5, -5]))
        every = int(rng.integers(1, 4))
        # On a month's first day: a day before the anchor is not missing.
        anchor = (origin + timedelta(days=int(rng.integers(0, 400)))).replace(
            day=1
        )
        if rng.random() < 0.5:
            rule = Recurrence.monthly(
                weekday=WEEKDAYS[int(rng.integers(0, 7))],
                nth=nth,
                every=every,
                anchor=anchor,
            )
        else:
            rule = Recurrence.monthly(
                day=int(rng.integers(29, 32)),
                short_months="skip",
                every=every,
                anchor=anchor,
            )
        start = anchor.replace(day=1)
        end = start + timedelta(days=int(rng.integers(30, 900)))
        end = end.replace(day=month_length(end))
        months_with = {d.strftime("%Y-%m") for d in rule.dates(start, end)}
        cycle = set()
        y, m = start.year, start.month
        while (y, m) <= (end.year, end.month):
            if ((y - anchor.year) * 12 + m - anchor.month) % every == 0:
                cycle.add(f"{y:04d}-{m:02d}")
            y, m = (y, m + 1) if m < 12 else (y + 1, 1)
        missing = rule.notes(start, end).missing_months
        assert set(missing) == cycle - months_with


# ---------------------------------------------------------------- words


@pytest.mark.parametrize(
    "rule, words",
    [
        (
            Recurrence.every(1, "days", "2026-01-01"),
            "every day from 1 Jan 2026",
        ),
        (
            Recurrence.every(2, "weeks", "2026-07-15"),
            "every 2 weeks from 15 Jul 2026",
        ),
        (Recurrence.weekly("Saturday"), "every Saturday"),
        (Recurrence.weekly("Mon-Fri"), "every weekday, Monday to Friday"),
        (
            Recurrence.weekly("Mon, Wed, Fri"),
            "every Monday, Wednesday and Friday",
        ),
        (
            Recurrence.monthly(day=30),
            "the 30th of every month (a shorter month's last day)",
        ),
        (Recurrence.monthly(day=1), "the 1st of every month"),
        (
            Recurrence.monthly(
                weekday="Friday", nth=-1, every=3, anchor="2027-01-01"
            ),
            "the last Friday of every 3rd month, from 1 Jan 2027",
        ),
        (
            Recurrence.monthly(weekday="Monday", nth=-2),
            "the second from last Monday of every month",
        ),
        (Recurrence.yearly(month=7, day=1), "1 July, every year"),
        (
            Recurrence.yearly(month=7, day=1, every=2, anchor="2026-01-01"),
            "1 July, every 2nd year, from 1 Jan 2026",
        ),
        (
            Recurrence.weekly(
                "Tue", season=Season(months=[4, 5, 6, 7, 8, 9, 10])
            ),
            "every Tuesday, in April to October only",
        ),
    ],
)
def test_describe_says_the_rule_in_plain_words(rule, words):
    assert rule.describe() == words
    assert words in repr(rule)


def test_a_season_in_words():
    assert Season(months=[1, 3, 5]).describe() == "January, March and May"
    assert Season(months=["Dec", "Jan"]).describe() == "December to January"
    assert Season(start="09-01", end="04-30").describe() == (
        "1 September to 30 April"
    )


# ---------------------------------------------------------------- saving


RULES = [
    Recurrence.every(1, "weeks", "2026-07-15"),
    Recurrence.every(3, "months", "2027-01-31", short_months="skip"),
    Recurrence.weekly("Mon-Fri", excluding=["2026-12-25"], moved="before"),
    Recurrence.weekly("Sat", every=2, anchor="2026-11-02"),
    Recurrence.monthly(weekday="Saturday", nth=5),
    Recurrence.monthly(
        day=30, season=Season(start="09-01", end="04-30"), anchor="2026-01-01"
    ),
    Recurrence.yearly(month="Nov", weekday="Tuesday", nth=1),
    Recurrence.yearly(month=2, day=29, short_months="skip"),
    Recurrence.on(
        ["2027-03-03", "2028-09-09"],
        excluding=[("2028-09-01", "2028-09-30")],
        moved="after",
    ),
]


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.describe())
def test_a_rule_goes_through_json_and_back(rule):
    data = json.loads(json.dumps(rule.to_dict()))
    assert data["kind"] == "recurrence" and data["version"] == 1
    back = Recurrence.from_dict(data)
    assert back == rule
    assert back.dates("2026-01-01", "2029-12-31") == rule.dates(
        "2026-01-01", "2029-12-31"
    )
    assert hash(back) == hash(rule)


def test_the_saved_shape_is_plain():
    rule = Recurrence.yearly(month=11, weekday="Tuesday", nth=1)
    assert rule.to_dict() == {
        "kind": "recurrence",
        "version": 1,
        "rule": "yearly",
        "every": 1,
        "weekday": "Tuesday",
        "nth": 1,
        "month": "November",
    }


def test_a_package_cadence_is_read_as_it_was_stored():
    # Shapes work packages are stored in today: every, unit, anchor and a
    # pattern.
    first_saturday = Recurrence.from_dict(
        {
            "every": 1,
            "unit": "months",
            "anchor": "2026-11-01",
            "pattern": {"weekday": "Saturday", "nth": 1},
        }
    )
    assert iso(first_saturday.dates("2026-11-01", "2027-01-31")) == [
        "2026-11-07",
        "2026-12-05",
        "2027-01-02",
    ]
    assert first_saturday.number("2027-01-02") == 3
    weekly = Recurrence.from_dict(
        {
            "every": 1,
            "unit": "weeks",
            "anchor": "2026-11-02",
            "pattern": {"weekday": "Saturday"},
            "season": {"months": [10, 11, 12]},
        }
    )
    assert iso(weekly.dates("2026-11-01", "2026-11-21")) == [
        "2026-11-07",
        "2026-11-14",
        "2026-11-21",
    ]
    fifteenth = Recurrence.from_dict(
        {
            "every": 2,
            "unit": "months",
            "anchor": "2026-01-20",
            "pattern": {"day": 15},
        }
    )
    assert iso(fifteenth.dates("2026-01-01", "2026-06-30")) == [
        "2026-03-15",
        "2026-05-15",
    ]
    plain = Recurrence.from_dict(
        {"every": 6, "unit": "months", "anchor": "2026-08-31"}
    )
    assert plain == Recurrence.every(6, "months", "2026-08-31")


def test_a_task_interval_is_read_as_it_was_stored():
    assert Recurrence.from_dict(
        {"interval": 3, "unit": "months", "start": "2026-01-15"}
    ) == Recurrence.every(3, "months", "2026-01-15")
    assert Recurrence.from_dict(
        {"interval": 14.0, "unit": "days", "start": "2026-01-15"}
    ) == Recurrence.every(14, "days", "2026-01-15")
    with pytest.raises(ValueError, match="by use"):
        Recurrence.from_dict(
            {"interval": 500, "unit": "hours", "start": "2026-01-15"}
        )
    with pytest.raises(ValueError, match="smaller unit"):
        Recurrence.from_dict(
            {"interval": 1.5, "unit": "months", "start": "2026-01-15"}
        )


def test_a_later_format_or_another_kind_is_refused():
    with pytest.raises(ValueError, match="later version"):
        Recurrence.from_dict({"kind": "recurrence", "version": 99})
    with pytest.raises(ValueError, match="not a recurrence"):
        Recurrence.from_dict({"kind": "task"})
    with pytest.raises(ValueError, match="not a saved rule"):
        Recurrence.from_dict({"name": "x"})


def test_to_rrule_writes_icalendar_text():
    rule = Recurrence.yearly(month=11, weekday="Tuesday", nth=1)
    assert rule.to_rrule(start="2026-01-01") == (
        "DTSTART:20260101T000000\nRRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=+1TU"
    )
    on = Recurrence.on(["2027-03-03", "2028-09-09"], excluding=["2027-03-03"])
    assert _read_rrule(on.to_rrule(), "2027-01-01", "2028-12-31") == (
        D("2028-09-09"),
    )
    with pytest.raises(ValueError, match="start="):
        Recurrence.weekly("Mon").to_rrule()
    with pytest.raises(ValueError, match="move"):
        RULES[2].to_rrule(start="2026-01-01")
    with pytest.raises(ValueError, match="days of the year"):
        RULES[5].to_rrule()


# ---------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    "make, error, match",
    [
        (lambda: Recurrence.every(0, "days", "2026-01-01"), ValueError, "n "),
        (
            lambda: Recurrence.every(1.5, "days", "2026-01-01"),
            ValueError,
            "n ",
        ),
        (
            lambda: Recurrence.every(True, "days", "2026-01-01"),
            ValueError,
            "n ",
        ),
        (
            lambda: Recurrence.every(1, "hours", "2026-01-01"),
            ValueError,
            "unit",
        ),
        (lambda: Recurrence.every(1, "days", None), TypeError, "anchor"),
        (
            lambda: Recurrence.every(1, "days", datetime(2026, 1, 1, 9)),
            TypeError,
            "time of day",
        ),
        (lambda: Recurrence.every(1, "days", "1/1/2026"), ValueError, "YYYY"),
        (lambda: Recurrence.weekly("Tues"), ValueError, "day of the week"),
        (lambda: Recurrence.weekly(3), TypeError, "weekdays"),
        (lambda: Recurrence.weekly("Mon", every=2), ValueError, "anchor"),
        (lambda: Recurrence.monthly(), ValueError, "one of them"),
        (
            lambda: Recurrence.monthly(day=30, weekday="Mon", nth=1),
            ValueError,
            "one of them",
        ),
        (lambda: Recurrence.monthly(day=32), ValueError, "day"),
        (lambda: Recurrence.monthly(day=30, nth=1), ValueError, "nth"),
        (lambda: Recurrence.monthly(weekday="Mon"), ValueError, "nth"),
        (lambda: Recurrence.monthly(weekday="Mon", nth=6), ValueError, "nth"),
        (lambda: Recurrence.monthly(weekday="Mon", nth=0), ValueError, "nth"),
        (
            lambda: Recurrence.monthly(weekday="Mon", nth=1, every=2),
            ValueError,
            "anchor",
        ),
        (
            lambda: Recurrence.monthly(
                weekday="Mon", nth=1, short_months="skip"
            ),
            ValueError,
            "short_months",
        ),
        (
            lambda: Recurrence.monthly(day=30, short_months="first"),
            ValueError,
            "short_months",
        ),
        (lambda: Recurrence.yearly(month=13, day=1), ValueError, "month"),
        (
            lambda: Recurrence.yearly(month="Movember", day=1),
            ValueError,
            "month",
        ),
        (lambda: Recurrence.yearly(month=4, day=31), ValueError, "no day 31"),
        (
            lambda: Recurrence.yearly(
                month=11, day=1, season=Season(months=[4, 5])
            ),
            ValueError,
            "season",
        ),
        (lambda: Recurrence.on([]), ValueError, "at least one"),
        (lambda: Recurrence.on("2027-01-01"), TypeError, "list"),
        (
            lambda: Recurrence.weekly("Mon", moved="after"),
            ValueError,
            "excluded",
        ),
        (
            lambda: Recurrence.weekly("Mon", excluding="2027-01-01"),
            TypeError,
            "excluding",
        ),
        (
            lambda: Recurrence.weekly(
                "Mon", excluding=[("2027-01-05", "2027-01-01")]
            ),
            ValueError,
            "before it starts",
        ),
        (
            lambda: Recurrence.weekly(
                "Mon", excluding=["2027-01-01"], moved="nearest"
            ),
            ValueError,
            "moved",
        ),
        (lambda: Recurrence(), TypeError, "Recurrence.every"),
        (lambda: Season(), ValueError, "one of them"),
        (lambda: Season(months=range(1, 13)), ValueError, "whole year"),
        (lambda: Season(start="13-01", end="01-01"), ValueError, "MM-DD"),
        (lambda: Season(start="01-01", end="01-01"), ValueError, "different"),
    ],
)
def test_inputs_are_checked(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_a_period_cannot_end_before_it_starts():
    with pytest.raises(ValueError, match="before it starts"):
        Recurrence.weekly("Mon").dates("2026-12-31", "2026-10-11")
