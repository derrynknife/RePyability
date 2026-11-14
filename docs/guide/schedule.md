# Schedules on the calendar

RePyability's models work in model time: a number of hours, days or cycles
from 0, or from a state. `repyability.schedule` connects that to the
calendar. This page covers the first part: **which dates a rule gives**
(`Recurrence`) and **when tasks fall due on them** (`Task`,
`work_by_period`). Duty plans (how many operating hours by when) and
forecasts through them build on these.

The module computes and stores nothing. You pass it dates, rules and what
you know (when a task was last done); it never reads the clock. Dates are
whole dates, as `datetime.date` or ISO text such as `"2026-11-14"`, and
**every period includes both of its ends**: `dates("2026-10-11",
"2026-12-31")` means through 31 December.

## Which dates: `Recurrence`

One type, made with a named constructor for each kind of rule:

| Constructor | Means | Example |
|---|---|---|
| `Recurrence.every(n, unit, anchor)` | every n days, weeks, months or years from a date | every 2 weeks from 15 Jul 2026 |
| `Recurrence.weekly(weekdays, every=1, anchor=None)` | named weekdays, every n weeks | every Saturday; `"Mon-Fri"` |
| `Recurrence.monthly(day= or weekday=, nth=, every=1, anchor=None)` | a day of the month, or its nth weekday | the 30th; the first Saturday; the last Friday of every 3rd month |
| `Recurrence.yearly(month=, day= or weekday=, nth=, every=1)` | a date or weekday pattern in a named month | 1 July; the first Tuesday of November |
| `Recurrence.on(dates)` | given dates | turnarounds on 3 Mar 2027 and 9 Sep 2028 |

```python
from repyability.schedule import Recurrence, Season, Task, work_by_period

first_saturday = Recurrence.monthly(weekday="Saturday", nth=1)
[str(d) for d in first_saturday.dates("2026-11-14", "2027-03-31")]
# ['2026-12-05', '2027-01-02', '2027-02-06', '2027-03-06']
first_saturday.describe()  # 'the first Saturday of every month'

overhaul = Recurrence.yearly(month="November", weekday="Tuesday", nth=1)
[str(d) for d in overhaul.dates("2026-11-14", "2029-12-31")]
# ['2027-11-02', '2028-11-07', '2029-11-06']
```

`describe()` says the rule in plain words, so you can check it is the rule
you meant before you save it.

### Short months, the fifth weekday, and anchors

A day the month doesn't have falls on the month's last day by default, or
is skipped with `short_months="skip"`. Months counted from an anchor keep
the anchor's day: they are not carried to the 28th after February.

```python
monthly = Recurrence.every(1, "months", "2027-01-31")
[str(d) for d in monthly.dates("2027-01-01", "2027-04-30")]
# ['2027-01-31', '2027-02-28', '2027-03-31', '2027-04-30']
```

`nth` runs from 1 to 5, and from -1 (the last) to -5. A fifth Saturday is
in only some months; the others are skipped, and `notes()` says which:

```python
fifth = Recurrence.monthly(weekday="Saturday", nth=5)
len(fifth.dates("2027-01-01", "2027-12-31"))  # -> 4
len(fifth.notes("2027-01-01", "2027-12-31").missing_months)  # -> 8
```

With `every` above 1, weeks, months or years are counted from the
anchor's, so it needs an `anchor`. Weeks start on Monday. No date comes
before an anchor.

### Seasons, excluded dates and moves

Every rule takes the same options:

- `season=`: a `Season` of months (`Season(months=[4, 5, 6, 7, 8, 9,
  10])`) or of days of the year (`Season(start="09-01", end="04-30")`,
  across the new year). Dates outside it don't count.
- `excluding=`: dates, or `(first, last)` ranges, that don't count:
  holidays, a shut. RePyability ships no holiday calendar; you pass the
  dates.
- `moved=`: what an excluded date does: `"skip"` (the default),
  `"before"` (the nearest day before that is not excluded) or `"after"`.

```python
month_end = Recurrence.monthly(
    weekday="Friday", nth=-1, excluding=["2026-12-25"], moved="before"
)
[str(d) for d in month_end.dates("2026-11-01", "2026-12-31")]
# ['2026-11-27', '2026-12-24']
month_end.notes("2026-11-01", "2026-12-31").text
# ('25 Dec 2026 is excluded: moved to 24 Dec 2026.',)
```

### Asking about one date

`on_or_after(day)` and `after(day)` give the next date (or `None`),
`includes(day)` whether the rule falls on it, and `number(day)` how many
dates there are from the anchor through it: a visit's number, for work
done every third visit.

```python
visits = Recurrence.monthly(weekday="Saturday", nth=1, anchor="2026-11-01")
str(visits.after("2026-11-07"))  # '2026-12-05'
visits.number("2027-01-02")  # -> 3
```

### Saving

`to_dict()` gives plain JSON with a `"kind"` and a `"version"`;
`from_dict()` reads it back, and also reads a work package's cadence
(`every`, `unit`, `anchor`, `pattern`) and a task's interval (`interval`,
`unit`, `start`) as earlier tools stored them. `to_rrule()` writes the
rule as iCalendar text, which python-dateutil (and calendar programs) read
back.

```python
Recurrence.from_dict(overhaul.to_dict()) == overhaul  # True
print(overhaul.to_rrule(start="2026-01-01"))
# DTSTART:20260101T000000
# RRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=+1TU
```

## When tasks fall due: `Task`

A `Task` falls due by its rule (`every=`), and optionally by a second limit
(`also=`), whichever comes first.

```python
conveyor = Task(
    "inspect conveyor",
    every=Recurrence.every(1, "weeks", "2026-07-15"),
    cost=250.0,
)
due = conveyor.due("2026-10-11", "2026-12-31")
due.count  # -> 12
due.cost  # -> 3000
```

`due()` returns a `TaskDue`: the due `dates`, which limit set each
(`limits`), when each is assumed done (`done`), how many days late
(`late`) and whether that is within the task's `tolerance` (`on_time`),
the `count`, `cost` and `hours`, and plain-language `notes`.

### When it is done, and what counts again from then

A forecast has to assume when each time is done: on its due date, or with
`visits=` at the first visit on or after it. Then:

- an **interval** (`Recurrence.every`) counts again from when the task was
  done (`last_done=` for the last time before the period);
- a **pattern** (weekly, monthly, yearly, given dates) keeps its own
  dates.

`resets=True` or `False` overrides that for every limit, and
`TaskDue.resets` says which applied.

```python
service = Task(
    "service",
    every=Recurrence.every(3, "months", "2026-01-15"),
    last_done="2026-09-20",
    visits=Recurrence.weekly("Tue"),
    tolerance=3,
)
due = service.due("2026-11-01", "2027-12-31")
[str(d) for d in due.done]
# ['2026-12-22', '2027-03-23', '2027-06-29', '2027-10-05']
due.late  # (2, 1, 6, 6)
due.on_time  # (True, True, False, False)
```

A task already due before the period starts (after `last_done`) is
**overdue**: it comes first, with its missed date, done at the period's
start (`due.overdue` is `True`).

### Two limits

```python
oil = Task(
    "oil change",
    every=Recurrence.every(30, "days", "2027-01-01"),
    also=Recurrence.monthly(day=1),
    last_done="2027-01-10",
)
due = oil.due("2027-01-10", "2027-04-30")
[str(d) for d in due.dates]
# ['2027-02-01', '2027-03-01', '2027-03-31', '2027-04-01']
due.limits  # ('also', 'also', 'every', 'also')
```

Limits by use (every so many operating hours) come with duty plans and
take the same `every=` and `also=`.

## Work by period

`work_by_period(tasks, period, start, end)` counts the tasks done in each
day, week (Monday to Sunday), month, quarter or year, with their hours and
cost. The first and last periods are cut to the dates asked. Tasks with no
hours or no cost are listed in `no_hours` and `no_cost`, so a total isn't
read as all the work.

```python
tasks = [
    Task("inspect", every=Recurrence.weekly("Mon"), duration=1.5, cost=120.0),
    Task("service", every=Recurrence.every(3, "months", "2026-12-20"),
         duration=4.0, cost=480.0),
]
work = work_by_period(tasks, "month", "2026-11-16", "2027-02-28")
work.count  # (3, 5, 4, 4)
work.hours  # (4.5, 10.0, 6.0, 6.0)
work.cost  # (360.0, 960.0, 480.0, 480.0)
```
