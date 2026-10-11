"""Components replaced when they fail, and what their failures will be over
a period of dates (#319, UC-5 of this package's ``README.md``).

A component's current unit was installed on a date and is working at the
start of the period; each time a unit fails it is replaced, at once, with
a new one. The chance the current unit fails in the period follows from
its life, conditioned on the age it has reached. The expected number of
failures (each a replacement) counts the new units' failures too: it is
the renewal count of a [`RepairableRBD`][repyability.RepairableRBD]'s
component replaced in no time, from that age (``expected_events``, exact,
with no simulation). A life's time unit is days.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

from repyability.maintenance._tasks import _date
from repyability.rbd.results import _ResultMapping
from repyability.utils import checks


@dataclass(frozen=True, init=False)
class Component:
    """A component whose units are replaced with new ones when they fail.

    Parameters
    ----------
    name : str
        What the component is called.
    life : surpyval model
        A unit's life, in days: a fitted surpyval model, or one built with
        ``from_params``.
    installed : datetime.date or str
        The date the current unit was installed, as a date or as text such
        as ``"2026-03-01"``.
    cost : float, optional
        What each replacement costs, at least 0, in your currency. By
        default None: not given, and the forecast has no cost.

    Examples
    --------
    A bearing with a Weibull life (scale 365 days, shape 2), installed on
    1 March 2026 and replaced with a new one each time it fails, from 11
    October to the end of 2026:

    >>> import surpyval as surv
    >>> from repyability.maintenance import Component
    >>> bearing = Component(
    ...     "bearing",
    ...     surv.Weibull.from_params([365.0, 2.0]),
    ...     installed="2026-03-01",
    ...     cost=400.0,
    ... )
    >>> forecast = bearing.failures("2026-10-11", "2026-12-31")
    >>> round(forecast.probability, 4)
    0.2784
    >>> round(forecast.expected_failures, 4)
    0.283
    >>> round(forecast.expected_cost, 2)
    113.2
    """

    name: str
    life: Any
    installed: date
    cost: Optional[float]

    def __init__(self, name: str, life, installed, cost=None) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be text; got {name!r}.")
        checks.no_distribution(life, "life")
        if not callable(getattr(life, "sf", None)):
            raise TypeError(
                "life must be a lifetime model with a survival function "
                f"(sf), such as a fitted surpyval model; got {life!r}."
            )
        if cost is not None and not (
            checks.is_number(cost) and 0.0 <= float(cost) < float("inf")
        ):
            raise ValueError(
                f"cost must be a number, at least 0 (or None); got {cost!r}."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "life", life)
        object.__setattr__(self, "installed", _date(installed, "installed"))
        object.__setattr__(self, "cost", None if cost is None else float(cost))

    def failures(self, start, end) -> "FailureForecast":
        """What the component's failures will be from ``start`` to
        ``end``, both days included, with the current unit working at the
        start.

        Parameters
        ----------
        start, end : datetime.date or str
            The first and last days of the period, as dates or as text
            such as ``"2026-12-31"``.

        Returns
        -------
        FailureForecast
            The chance the current unit fails in the period, the expected
            number of failures (each a replacement), and their expected
            cost.
        """
        from repyability.rbd.node_state import NodeState
        from repyability.rbd.repairable_rbd import RepairableRBD

        start, end = _date(start, "start"), _date(end, "end")
        if end < start:
            raise ValueError(
                f"The period ends ({end}) before it starts ({start})."
            )
        if start < self.installed:
            raise ValueError(
                f"The period starts ({start}) before the current unit was "
                f"installed ({self.installed})."
            )
        age = float((start - self.installed).days)
        days = float((end - start).days + 1)
        surviving = float(self.life.sf(age))
        if not surviving > 0.0:
            raise ValueError(
                f"The life of component {self.name!r} gives no chance of a "
                f"unit working at age {age:g} days, as the current unit is "
                f"at the start ({start})."
            )
        probability = 1.0 - float(self.life.sf(age + days)) / surviving
        unit = RepairableRBD(
            [("s", "unit"), ("unit", "t")],
            {"unit": {"reliability": self.life, "repairability": "instant"}},
        )
        events = unit.expected_events(days, state={"unit": NodeState(age=age)})
        failures = float(events.node_failures["unit"])
        cost = None if self.cost is None else failures * self.cost
        return FailureForecast(
            self.name, start, end, probability, failures, cost
        )


@dataclass
class FailureForecast(_ResultMapping):
    """The failures of a [`Component`][repyability.Component] in a
    period: returned by its ``failures``.

    Attributes
    ----------
    component : str
        The component's name.
    start, end : datetime.date
        The period's first and last days, both included.
    probability : float
        The chance the current unit fails in the period.
    expected_failures : float
        The expected number of failures in the period, each a replacement
        with a new unit: the current unit's, and its replacements'.
    expected_cost : float or None
        Their expected cost: the cost of each replacement times
        ``expected_failures``; None if the component was given no cost.
    """

    component: str
    start: date
    end: date
    probability: float
    expected_failures: float
    expected_cost: Optional[float]
