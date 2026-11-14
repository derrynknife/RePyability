# Schedules

## Duty plans: operating hours on dates

A life fitted in operating hours, or a task every 2,000 running hours,
needs to know how the equipment runs before it can say anything about
dates. A `DutyPlan` says how it runs; its `forecast` is the map between
moments and meter readings, both ways.

### A repeating week

A pump runs Monday to Friday, 06:00 to 22:00, and its hour meter read
12,340 h at the start of Monday 16 November 2026:

```python
from repyability.schedule import DutyPlan, Shifts, Task, Usage

pump = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
usage = pump.forecast("2026-11-16", "2027-12-01", readings={"hours": 12_340})
usage.at("2027-01-01")["hours"]  # -> 12884
```

Its service is every 2,000 operating hours and was last done at 12,000
h, so it is next due at 14,000 h. The meter reaches that at 18:00 on
Thursday 8 April 2027, in the middle of a run; the pump stops at 22:00
that day, which is when the service can be done:

```python
crossing = usage.when(hours=14_000)
str(crossing.moment)  # '2027-04-08 18:00:00'
str(crossing.next_stop_after)  # '2027-04-08 22:00:00'
```

At a flat 80 hours a week the date would have been Saturday 10 April, a
day the pump does not run: the shape of the week matters, not only its
total.

A `Task` with a limit by use takes the same forecast:

```python
service = Task("Service", every=Usage(2000, "hours"), last_done_reading=12_000)
due = service.due("2026-11-16", "2027-11-30", usage=usage)
[str(m) for m in due.moments]  # ['2027-04-08 18:00:00', '2027-09-30 18:00:00']
```

The hours in each calendar month, and the reading at each month's end:

```python
months = usage.by_period("month")
months.usage["hours"][:3]  # (176.0, 368.0, 336.0)
months.at_end["hours"][-1]  # -> 16692
```

### Operating time, the model's clock

A life fitted in operating hours is reached from dates through
`model_time` (operating time since the forecast's start, in the model's
unit) and back through `moment`:

```python
usage.model_time("2026-11-23")  # -> 80
usage.moment(80.0) == usage.moment(usage.model_time("2026-11-20 22:00"))  # True
```

`duty()` is the share of the time running (what a `RepairableRBD`
block's `"duty"` takes) and `rate("starts", per="hours")` one meter per
unit of another:

```python
round(usage.duty(), 4)  # -> 0.4772
usage.rate("starts", per="hours")  # -> 0.0625
```

### The parts of a plan

| Part | Means | Example |
|---|---|---|
| `Runs([(on, off), ...])` | concrete runs | on 14 Nov 12:31, off 15 Nov 03:27 |
| `Shifts(days, start, stop)` | a repeating week; a stop before the start is the next day | `Shifts("Mon-Fri", "06:00", "22:00")` |
| `Roster(pattern, anchor, start, stop)` | days on and off in turn | `Roster((4, 4), "2026-11-02", "07:00", "19:00")` |
| `Rate(amount, per, meter="hours")` | a general rate, each calendar period's amount spread over it | `Rate(3, per="day")`, `Rate(15, per="month")`, `Rate((2, 4), per="day")` |
| `Continuous()` | all the time | a 24/7 process pump |

Options on the plan (and, but for the zone and outages, on each part):
`from_`/`until`, `season="Apr-Oct"`, `outages=[(start, end), ...]`,
`utilisation=0.9` (each planned hour and start counts 0.9, an expected
value), `time_zone="Australia/Sydney"`, and other meters:
`per_hour={"tonnes": 40}`, `per_run={"cycles": 1}`, or a `Rate` of their
own (`Rate(1200, per="week", meter="km")`). Every run adds one to
`"starts"`.

### A concrete run

A standby generator is booked to run from 12:31 on Saturday 14 November
to 03:27 on Sunday, 14 hours 56 minutes. Its meter read 486.2 h; its
500-hour check falls due 13.8 hours into the run, at 02:19:

```python
from repyability.schedule import Runs

generator = DutyPlan(Runs([("2026-11-14 12:31", "2026-11-15 03:27")]))
run = generator.forecast("2026-11-14", "2026-11-16", readings={"hours": 486.2})
str(run.when(hours=500).moment)  # '2026-11-15 02:19:00'
run.at("2026-11-16")["hours"]  # -> 501.13
```

### A general rate, and a range

A conveyor runs about 3 hours a day; its meter read 1,180 h on 1
December 2026 and its idlers are inspected every 250 h (last at 1,000
h). The next falls due at 1,250 h, 23⅓ days on:

```python
from repyability.schedule import Rate

conveyor = DutyPlan(Rate(3, per="day"))
belt = conveyor.forecast("2026-12-01", "2027-12-01", readings={"hours": 1_180})
str(belt.when(hours=1_250).moment)  # '2026-12-24 08:00:00'
```

If the rate is only known as "2 to 4 hours a day", the date is a range:
the middle at 3 h a day, the earliest at 4 and the latest at 2:

```python
ranged = DutyPlan(Rate((2, 4), per="day"))
c = ranged.forecast("2026-12-01", "2027-12-01", readings={"hours": 1_180}).when(hours=1_250)
str(c.earliest), str(c.latest)  # ('2026-12-18 12:00:00', '2027-01-05 00:00:00')
```

A rate is an assumption, and every answer from it says so (`exact` is
False); runs, shifts and rosters are exact.

### From specific to general

A plan is usually exact for the next day or two and rough after. Give
the rough part as the **background** and the exact runs as **specific**
parts on top: the forecast uses the runs where they cover, and the
background the moment they run out, and says which stretch came from
which:

```python
plan = DutyPlan(
    background=Rate(3, per="day"),
    specific=[Runs([("2026-11-14 12:31", "2026-11-15 03:27"),
                    ("2026-11-15 12:30", "2026-11-16 03:30")],
                   from_="2026-11-14")],
)
layered = plan.forecast("2026-11-14", "2027-11-14", readings={"hours": 4_820})
layered.describe()  # 'exact runs to 16 Nov 03:30, then about 3 h a day'
```

A specific part covers from its `from_` to its `until`; runs, by
default, from their first start to their last stop. As the days pass,
add the next runs; nothing else changes, and a new reading re-anchors the
forecast.

**No duty is made at the handover.** By default (`fill="pro_rata"`) the
background applies only to the time the specific parts leave, at its
rate spread by time: the runs above give Monday 16 November 3.5 hours
to 03:30, and the background the 20.5 hours left of the day 3 × 20.5/24
= 2.5625 hours, never a full 3 hours more:

```python
layered.by_period("day").usage["hours"][2]  # -> 6.0625
```

A shift or roster background simply starts again at its next clock time,
counting only the part of a run no specific part covers. For a rate that
is a budget ("15 hours a month"), `fill="top_up"` gives the period in
which the specific parts end what they left of its amount (never less
than nothing), spread over the rest of it; the periods after get the
whole amount. Either way the days add up to the change in the reading.

Two parts of one layer that would count the same time twice are
refused, with a message naming both; a background part with no end ends
where the next one starts (`Shifts(..., from_="2027-03-01")`).

### Time zones

With `time_zone=`, plans are in the site's wall-clock time and count
real hours: in Sydney, a 22:00 to 06:00 shift is 7 hours the night the
clocks go forward and 9 the night they go back. With no zone, clock
times are taken as they are.

```python
night = DutyPlan(Shifts("Sat", "22:00", "06:00"), time_zone="Australia/Sydney")
night.forecast("2026-10-03", "2026-10-05").at("2026-10-05")["hours"]  # -> 7
night.forecast("2027-04-03", "2027-04-05").at("2027-04-05")["hours"]  # -> 9
```

A time that does not exist (02:30 the night the clocks go forward) moves
to the end of the gap (03:00); one that happens twice means the first.
The forecast's `notes` say when either happened. On Windows, install the
`tzdata` package for the zones (RePyability requires it there).

### With no plan

`UsageForecast.calendar(start, end)` is the identity map: calendar time
is operating time, as for a life fitted in calendar days.

### Saving

Every part and plan has `to_dict()` and `from_dict()` (plain JSON, a
`"kind"` on each part, moments as ISO text, the zone by name, and a
`"version"`), and every answer is plain data with `to_dict()`.
