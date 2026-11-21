"""The calendar an analysis runs on: when it starts, at which site, to
when, and each unit's duty plan (how many operating hours by when).

A diagram's clock is calendar time from the calendar's start. Each unit
whose life is in operating time reaches it through its own duty plan
``D(t)``, the operating time from the start to ``t``: its life on the
calendar is ``R(D(t))``, a unit put into service at ``t0`` failing when
``D`` has run its operating life on from ``D(t0)``, and a task counted in
operating time falls when ``D`` reaches it. A plan that runs one share of
the time throughout is the block's ``"duty"`` share, worked out exactly; a
plan that changes (weekday shifts, a season) is followed by the
simulations. The duty plans belong to the analysis, not to the units'
plans: the same diagram runs on another site's calendar with other plans.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional

import numpy as np

from repyability.utils.checks import is_number, one_of

from ._meters import UNITS
from ._usage import DutySchedule, UsageForecast


@dataclass(frozen=True)
class CalendarContext:
    """The calendar an analysis runs on (see the module's notes).

    Parameters
    ----------
    usage : UsageForecast
        The calendar's frame: its start ("now", the diagram's 0), end,
        time zone and periods
        (``UsageForecast.calendar(start, end)`` for one with no plan of
        its own).
    duty : mapping, optional
        Node -> its duty plan: a ``UsageForecast`` from the same start to
        at least the same end (``DutyPlan(...).forecast(...)``), or a
        share of the time in (0, 1]. A node left out keeps its spec's
        ``"duty"``, if any.
    unit : str, optional
        The unit of the diagram's models (``"hours"``, the default,
        ``"days"``, ...): of its clock and of each life in operating time.

    Examples
    --------
    >>> from repyability.schedule import CalendarContext, DutyPlan, Shifts
    >>> from repyability.schedule import UsageForecast
    >>> frame = UsageForecast.calendar("2027-01-04", "2027-01-11")
    >>> shifts = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
    ...     "2027-01-04", "2027-01-11"
    ... )
    >>> plan = CalendarContext(frame, {"pump": shifts}).duty_plan("pump")
    >>> plan.times[:3], plan.shares[:3]
    ((0.0, 6.0, 22.0), (0.0, 1.0, 0.0))
    """

    usage: UsageForecast
    duty: Mapping[Any, Any] = field(default_factory=dict)
    unit: str = "hours"

    def __post_init__(self) -> None:
        if not isinstance(self.usage, UsageForecast):
            raise TypeError(
                "usage must be a UsageForecast (DutyPlan(...).forecast(...), "
                "or UsageForecast.calendar(start, end) with no plan); got "
                f"{self.usage!r}."
            )
        one_of("unit", self.unit, tuple(UNITS))
        if not isinstance(self.duty, Mapping):
            raise TypeError(
                "duty must map each node to its duty plan (a "
                "UsageForecast, or a share of the time); got "
                f"{self.duty!r}."
            )
        object.__setattr__(self, "duty", dict(self.duty))
        for node in self.duty:
            self.duty_plan(node)  # checked where it is given

    # -- each unit's plan ---------------------------------------------------

    def duty_plan(self, node) -> Any:
        """Node ``node``'s duty as a block's ``"duty"`` takes it: a share
        of the time, or the steps of its forecast over the diagram's clock
        (a ``DutySchedule`` in the calendar's unit); None for a node the
        calendar gives no plan."""
        if node not in self.duty:
            return None
        given = self.duty[node]
        if is_number(given):
            return float(given)
        if not isinstance(given, UsageForecast):
            raise TypeError(
                f"The duty plan of node {node!r} must be a UsageForecast "
                "(DutyPlan(...).forecast(start, end)) or a share of the "
                f"time; got {given!r}."
            )
        if given.start != self.usage.start:
            raise ValueError(
                f"The duty plan of node {node!r} starts at {given.start}, "
                f"and the calendar at {self.usage.start}: forecast it from "
                "the calendar's start."
            )
        if given.end < self.usage.end:
            raise ValueError(
                f"The duty plan of node {node!r} ends at {given.end}, "
                f"before the calendar does ({self.usage.end}): forecast it "
                "to the calendar's end at least."
            )
        steps = given.schedule(self.unit)
        if len(set(steps.shares)) == 1:
            return float(steps.shares[0])
        return steps

    def model_time(self, moments) -> np.ndarray:
        """The diagram's clock at each moment: real time since the
        calendar's start, in its unit."""
        usage = self.usage
        given = [moments] if isinstance(moments, (str, datetime)) else moments
        seconds = np.array(
            [usage._seconds(m) - usage._start_s for m in given], dtype=float
        )
        return seconds / 3600.0 / UNITS[self.unit]

    def rule_times(self, rule, at=None) -> np.ndarray:
        """A recurrence rule's dates within the calendar, each at the clock
        time ``at`` (midnight by default), on the diagram's clock: those
        after its start (a task due at the start itself is due now, and
        is not taken as done): a ``"preventive"`` spec's ``"replace_at"``
        for a task on the rule.

        Examples
        --------
        >>> from repyability.schedule import CalendarContext, Recurrence
        >>> from repyability.schedule import UsageForecast
        >>> frame = UsageForecast.calendar("2027-01-01", "2028-01-01")
        >>> quarterly = Recurrence.monthly(
        ...     weekday="Monday", nth=1, every=3, anchor="2027-01-01"
        ... )
        >>> CalendarContext(frame, unit="days").rule_times(quarterly).tolist()
        [3.0, 94.0, 185.0, 276.0]
        """
        from ._forecast import _instants, _rule

        usage = self.usage
        dates = _rule(rule).dates(usage.start.date(), usage.end.date())
        times = self.model_time(_instants(dates, usage._clock, at))
        end = self.model_time([usage.end])[0]
        return times[(times > 0.0) & (times <= end)]

    # -- the diagram on it ----------------------------------------------------

    def diagram(self, rbd):
        """``rbd`` (a ``RepairableRBD``) on this calendar: each node the
        calendar gives a plan has it as its ``"duty"``. The diagram
        itself when the calendar gives no plan."""
        from repyability.rbd.repairable_rbd import RepairableRBD

        components = dict(rbd._init_args["components"])
        changed = False
        for node, spec in list(components.items()):
            as_spec = getattr(spec, "spec", None)
            if callable(as_spec) and not isinstance(spec, dict):
                spec = as_spec()
            if not isinstance(spec, dict):
                if node in self.duty:
                    raise ValueError(
                        f"The calendar gives node {node!r} a duty plan, "
                        "and it is not a unit with a life (a spec or a "
                        "UnitPlan): leave it out of the calendar's duty."
                    )
                continue
            plan = self.duty_plan(node)
            if plan is not None:
                spec = {**spec, "duty": plan}
                changed = True
            components[node] = spec
        for node in self.duty:
            if node not in components:
                raise ValueError(
                    f"The calendar gives node {node!r} a duty plan, and "
                    f"the diagram has no node {node!r}."
                )
        if not changed:
            return rbd
        return RepairableRBD(**{**rbd._init_args, "components": components})

    # -- saving --------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The calendar as plain data: its frame, each node's plan (a
        share, or its steps on the diagram's clock) and its unit."""
        plans: Dict[str, Any] = {}
        for node in self.duty:
            plan = self.duty_plan(node)
            plans[str(node)] = (
                plan if isinstance(plan, float) else _steps_dict(plan)
            )
        return {
            "usage": self.usage.to_dict(),
            "duty": plans,
            "unit": self.unit,
        }


def _steps_dict(plan: DutySchedule) -> Dict[str, Any]:
    return {
        "unit": plan.unit,
        "times": list(plan.times),
        "shares": list(plan.shares),
    }


def calendar_of(usage) -> Optional[CalendarContext]:
    """``usage`` if it is a ``CalendarContext``, else None."""
    return usage if isinstance(usage, CalendarContext) else None
