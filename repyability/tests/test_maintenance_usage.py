"""Tasks by use on an asset's operating schedule (#319, UC-10 of
``repyability/maintenance/README.md``)."""

import json
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from repyability import (
    Asset,
    Hours,
    Starts,
    Task,
    TaskDates,
    UsageDue,
    Window,
)

SHIFT = Window("Mon-Fri", "06:00", "22:00")
PUMP = Asset(
    "pump 1",
    runs=SHIFT,
    meters={"hours": 12_340, "starts": 4_210},
    read_at="2026-10-12 00:00",
)
SERVICE = Task(
    "pump service", every=Hours(500), asset=PUMP, last_done=12_000, cost=250
)
QUARTER = timedelta(minutes=15)


def walked(windows, reading, read_at, meter, every, last_done, start, end):
    """The times a task by use falls due, by walking the clock a quarter
    of an hour at a time: its definition, written out plainly. Each
    window is (weekdays, start, stop) in quarter hours of the day."""

    def running(moment):
        for days, begins, ends in windows:
            for back in (0, 1):
                day = moment.date() - timedelta(days=back)
                if day.weekday() not in days:
                    continue
                opened = datetime.combine(day, datetime.min.time())
                opened += begins * QUARTER
                closed = datetime.combine(day, datetime.min.time())
                closed += ends * QUARTER
                if ends <= begins:
                    closed += timedelta(days=1)
                if opened <= moment < closed:
                    return opened
        return None

    times, target, moment = [], last_done + every, read_at
    while moment < end:
        opened = running(moment)
        if meter == "starts":
            if opened == moment:
                reading += 1
        elif opened is not None:
            reading += 0.25
        while target <= reading:
            when = moment if meter == "starts" else moment + QUARTER
            if start <= when < end:
                times.append(when)
            target += every
        moment += QUARTER
    return tuple(times)


def test_the_pump_service_through_the_rest_of_2026_and_into_2027():
    # UC-10: 160 hours to 12,500, ten runs: Friday 23 October at the
    # stop; 660 to 13,000, 41 runs and a quarter: Tuesday 8 December,
    # four hours into its run; 1,160 to 13,500: Wednesday 20 January.
    due = SERVICE.due("2026-10-12", "2026-12-31")
    assert isinstance(due, UsageDue)
    assert due.meter == "hours"
    assert due.readings == (12_500.0, 13_000.0)
    assert due.times == (
        datetime(2026, 10, 23, 22, 0),
        datetime(2026, 12, 8, 10, 0),
    )
    assert due.count == 2
    assert due.cost == 500.0
    later = SERVICE.due("2027-01-01", "2027-01-31")
    assert later.times == (datetime(2027, 1, 20, 14, 0),)


def test_the_times_are_the_clock_walked_a_quarter_hour_at_a_time():
    rng = np.random.default_rng(10)
    checked = 0
    while checked < 60:
        windows = []
        for _ in range(int(rng.integers(1, 3))):
            days = tuple(
                sorted(
                    rng.choice(7, size=int(rng.integers(1, 8)), replace=False)
                )
            )
            begins, length = int(rng.integers(0, 96)), int(rng.integers(1, 96))
            windows.append((days, begins, (begins + length) % 96))
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        def clock(quarters):
            return f"{quarters // 4:02d}:{15 * (quarters % 4):02d}"

        runs = [
            Window(",".join(names[d] for d in days), clock(b), clock(e))
            for days, b, e in windows
        ]
        read_at = datetime(2026, 10, 12) + int(rng.integers(0, 96)) * QUARTER
        meter = str(rng.choice(["hours", "starts"]))
        try:
            asset = Asset("a", runs, {meter: 100.0}, read_at)
        except ValueError:
            continue  # windows that overlap
        if meter == "hours":
            every = 0.25 * int(rng.integers(1, 200))
            interval = Hours(every)
        else:
            every = float(rng.integers(1, 6))
            interval = Starts(every)
        # The last time on the walk's grid (a quarter of an hour, or a
        # whole start), past 100 less every, so it is not due already.
        step = 0.25 if meter == "hours" else 1.0
        last_done = 100.0 - step * int(rng.integers(0, round(every / step)))
        task = Task("t", every=interval, asset=asset, last_done=last_done)
        start = read_at.date() + timedelta(days=int(rng.integers(0, 10)))
        end = start + timedelta(days=int(rng.integers(0, 30)))
        due = task.due(start, end)
        expected = walked(
            windows,
            100.0,
            read_at,
            meter,
            every,
            last_done,
            max(read_at, datetime.combine(start, datetime.min.time())),
            datetime.combine(end + timedelta(days=1), datetime.min.time()),
        )
        assert due.times == expected
        checked += 1


def test_a_run_overnight_goes_into_the_next_day():
    night = Asset(
        "pump 2",
        runs=Window("Mon-Fri", "22:00", "06:00"),
        meters={"hours": 0},
        read_at="2026-10-12 00:00",
    )
    task = Task("t", every=Hours(10), asset=night, last_done=0)
    # Monday's run: 22:00 to Tuesday 06:00, 8 hours; the 10th hour is two
    # hours into Tuesday's run, at midnight.
    assert task.due("2026-10-12", "2026-10-14").times[0] == datetime(
        2026, 10, 14, 0, 0
    )


def test_two_shifts_with_a_break():
    shifts = Asset(
        "line",
        runs=[
            Window("Mon-Fri", "06:00", "14:00"),
            Window("Mon-Fri", "14:30", "22:30"),
        ],
        meters={"hours": 0, "starts": 0},
        read_at="2026-10-12 00:00",
    )
    hours = Task("t", every=Hours(12), asset=shifts, last_done=0)
    assert hours.due("2026-10-12", "2026-10-12").times == (
        datetime(2026, 10, 12, 18, 30),
    )
    starts = Task("s", every=Starts(3), asset=shifts, last_done=0)
    assert starts.due("2026-10-12", "2026-10-13").times == (
        datetime(2026, 10, 13, 6, 0),
    )


def test_starts_fall_due_at_a_run_start():
    task = Task("s", every=Starts(5), asset=PUMP, last_done=4_207)
    # 4,210 read on Monday before its start: 4,211 at Monday's, 4,212 at
    # Tuesday's.
    assert task.due("2026-10-12", "2026-10-20").times == (
        datetime(2026, 10, 13, 6, 0),
        datetime(2026, 10, 20, 6, 0),
    )


def test_a_reading_in_a_run_counts_the_rest_of_it():
    mid = Asset(
        "pump 3",
        runs=SHIFT,
        meters={"hours": 100.0, "starts": 10},
        read_at="2026-10-12 10:00",
    )
    hours = Task("t", every=Hours(20), asset=mid, last_done=90)
    # 10 hours left to 110: the 12 hours left of Monday's run cover them.
    assert hours.due("2026-10-12", "2026-10-12").times == (
        datetime(2026, 10, 12, 20, 0),
    )
    starts = Task("s", every=Starts(2), asset=mid, last_done=9)
    # Monday's start is already in the reading: 11 at Tuesday's.
    assert starts.due("2026-10-12", "2026-10-13").times == (
        datetime(2026, 10, 13, 6, 0),
    )


def test_a_later_period_counts_the_hours_before_it():
    due = SERVICE.due("2026-11-01", "2026-12-31")
    assert due.times == (datetime(2026, 12, 8, 10, 0),)


def test_window_hours_and_days():
    assert SHIFT.hours == 16.0
    assert SHIFT.days == (0, 1, 2, 3, 4)
    assert Window("Fri-Mon", "08:00", "09:30").days == (0, 4, 5, 6)
    assert Window("monday, Wed,FRIDAY", "08:00", "09:00").days == (0, 2, 4)
    assert Window("Sat", "18:00", "00:00").hours == 6.0


def test_the_result_goes_through_json():
    plain = json.loads(
        json.dumps(SERVICE.due("2026-10-12", "2026-12-31").to_dict())
    )
    assert plain["times"] == ["2026-10-23T22:00:00", "2026-12-08T10:00:00"]
    assert plain["readings"] == [12500.0, 13000.0]
    assert plain["count"] == 2


def test_a_calendar_task_is_as_it_was():
    weekly = Task("t", every=7, first="2026-07-15")
    assert isinstance(weekly.due("2026-10-11", "2026-12-31"), TaskDates)
    assert weekly.asset is None and weekly.last_done is None


@pytest.mark.parametrize(
    "arguments, error, match",
    [
        (("Mon-Fri", "06:00", "06:00"), ValueError, "same time"),
        (("Mon-Fry", "06:00", "22:00"), ValueError, "days"),
        (("Mon-Wed-Fri", "06:00", "22:00"), ValueError, "days"),
        ((5, "06:00", "22:00"), TypeError, "days"),
        (("Mon", "6am", "22:00"), ValueError, "start"),
        (("Mon", "06:00", 22), TypeError, "stop"),
    ],
)
def test_a_window_is_checked(arguments, error, match):
    with pytest.raises(error, match=match):
        Window(*arguments)


@pytest.mark.parametrize(
    "keywords, error, match",
    [
        (
            {"runs": [SHIFT, Window("Mon", "21:00", "23:00")]},
            ValueError,
            "overlap",
        ),
        ({"runs": []}, TypeError, "runs"),
        ({"meters": {"cycles": 3}}, ValueError, "meters"),
        ({"meters": {"hours": -1}}, ValueError, "hours"),
        ({"meters": {}}, TypeError, "meters"),
        ({"read_at": "12/10/2026"}, ValueError, "read_at"),
    ],
)
def test_an_asset_is_checked(keywords, error, match):
    arguments = {
        "runs": SHIFT,
        "meters": {"hours": 1.0},
        "read_at": "2026-10-12 00:00",
    } | keywords
    with pytest.raises(error, match=match):
        Asset("a", **arguments)


@pytest.mark.parametrize(
    "keywords, error, match",
    [
        ({"first": "2026-01-01"}, ValueError, "first"),
        ({"asset": None}, TypeError, "asset"),
        (
            {
                "every": Starts(10),
                "asset": Asset("a", SHIFT, {"hours": 1}, "2026-10-12"),
            },
            ValueError,
            "starts",
        ),
        ({"last_done": 12_400}, ValueError, "past"),
        ({"last_done": 11_840}, ValueError, "due already"),
        ({"last_done": "12000"}, ValueError, "last_done"),
    ],
)
def test_a_task_by_use_is_checked(keywords, error, match):
    arguments = {
        "every": Hours(500),
        "asset": PUMP,
        "last_done": 12_000,
    } | keywords
    with pytest.raises(error, match=match):
        Task("t", **arguments)


def test_a_calendar_task_takes_no_asset_and_needs_a_first_date():
    with pytest.raises(ValueError, match="by use"):
        Task("t", every=7, first="2026-07-15", asset=PUMP)
    with pytest.raises(TypeError, match="first time"):
        Task("t", every=7)


@pytest.mark.parametrize("amount", [0, -5, float("inf"), True, "500"])
def test_hours_are_more_than_0(amount):
    with pytest.raises(ValueError, match="Hours"):
        Hours(amount)


@pytest.mark.parametrize("amount", [0, 1.5, True, "5"])
def test_starts_are_whole(amount):
    with pytest.raises(ValueError, match="Starts"):
        Starts(amount)


def test_the_period_starts_from_the_reading():
    with pytest.raises(ValueError, match="before the meters"):
        SERVICE.due("2026-10-11", "2026-12-31")
    assert SERVICE.due(date(2026, 10, 12), date(2026, 10, 12)).count == 0
