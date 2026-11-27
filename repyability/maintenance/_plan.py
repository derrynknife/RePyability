"""One unit's maintenance plan (UC-63 of this package's README): its life
and repair models, its tasks, each in its own clock, and its prices, as
plain data.

A ``RepairableRBD`` node takes a ``UnitPlan`` as it takes the spec
dict, which is the plan's shorthand: ``UnitPlan.from_spec`` reads a
spec and ``UnitPlan.spec`` writes the one the diagram's engines
read. What belongs to the system (its structure, system downtime and
value, crews, spares, the site's shutdowns and common-cause groups) stays
on the diagram; a plan names only its maintenance group and its place in
the crews' queue. The diagram checks a plan as it checks a spec, naming
the node. (A plan is a ``UnitPlan``: ``MaintenancePlan`` is the result
of ``RepairableRBD.optimal_replacement_intervals``.)
"""

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, Hashable, Optional, Tuple

#: A task that takes no time.
INSTANT = "instant"


def _put(out: dict, key: str, value, default=None) -> None:
    """Write ``key`` unless ``value`` is its default (not given)."""
    if value is not None and not (
        isinstance(value, str)
        and isinstance(default, str)
        and value == default
    ):
        out[key] = value


def _take(spec: dict, used: set, key: str, default=None):
    """``spec[key]`` (``default`` if absent or None), noting it was read."""
    used.add(key)
    value = spec.get(key)
    return default if value is None else value


def _all_read(what: str, spec: dict, used: set) -> None:
    """Refuse a spec with a key no field of the plan holds: it would be
    dropped quietly."""
    left = sorted(
        str(key) for key in set(spec) - used if spec[key] is not None
    )
    if left:
        raise ValueError(
            f"A {what} spec has key(s) {left} that a UnitPlan does "
            "not hold; check their spelling against the spec's keys "
            "(RepairableRBD.COMPONENT_SPEC_KEYS, PREVENTIVE_KEYS, "
            "INSPECTION_KEYS)."
        )


@dataclass(frozen=True)
class Restoration:
    """How much of its age a preventive task takes away (a task that
    renews the unit, the default, has none).

    Give a Kijima factor (``q`` with ``model`` ``"kijima1"`` or
    ``"kijima2"``: 0 renews, 1 leaves the unit as old as it was, above 1
    leaves it older), a virtual age to restore to (``to_age``), or a
    fitted surpyval ``GeneralizedRenewal`` (``process``). With
    ``replace_every`` ``N``, every ``N``-th task replaces the unit, at
    ``replacement_cost`` (the task's own cost if None). The rules are in
    [Maintaining a system](../guide/system-maintenance.md), "Planned
    maintenance that restores partly".

    Parameters
    ----------
    q : float, optional
        Kijima's restoration factor, 0 or more: 0 renews the unit, 1
        leaves it as old as it was, above 1 leaves it older. Not given with
        ``to_age`` or ``process``.
    model : str, optional
        Kijima's virtual-age model for ``q``: ``"kijima2"`` (the default,
        the factor applied to the whole virtual age) or ``"kijima1"``
        (to the age gained since the last action).
    to_age : float, optional
        A virtual age, 0 or more and in the plan's unit of time, that
        each task leaves the unit at, whatever its age before.
    process : surpyval GeneralizedRenewal, optional
        A fitted renewal process whose Kijima model and factor each task
        takes.
    replace_every : int, optional
        Every ``replace_every``-th task since the unit was last renewed
        replaces it as new instead.
    replacement_cost : float or distribution, optional
        The price of such a replacement, in money; by default the task's
        own ``cost``.
    """

    q: Optional[float] = None
    model: str = "kijima2"
    to_age: Optional[float] = None
    process: Any = None
    replace_every: Optional[int] = None
    replacement_cost: Any = None

    def __post_init__(self):
        given = [
            name
            for name in ("q", "to_age", "process")
            if getattr(self, name) is not None
        ]
        if len(given) > 1:
            raise ValueError(
                "A Restoration takes one of q (a Kijima factor), to_age "
                f"(a virtual age) and process (a fitted model), got {given}."
            )

    def spec(self) -> dict:
        """The keys of a ``"preventive"`` spec this restoration is."""
        out: dict = {}
        if self.to_age is not None:
            out["restore_to"] = self.to_age
        elif self.process is not None:
            out["restoration"] = self.process
        elif self.q is not None:
            out["restoration"] = {"model": self.model, "q": self.q}
        _put(out, "replace_every", self.replace_every)
        _put(out, "replacement_cost", self.replacement_cost)
        return out

    @classmethod
    def _read(cls, spec: dict, used: set) -> Optional["Restoration"]:
        given = _take(spec, used, "restoration")
        kwargs: Dict[str, Any] = {
            "to_age": _take(spec, used, "restore_to"),
            "replace_every": _take(spec, used, "replace_every"),
            "replacement_cost": _take(spec, used, "replacement_cost"),
        }
        if isinstance(given, dict):
            kwargs.update(model=given.get("model"), q=given.get("q"))
        elif given is not None:
            kwargs["process"] = given
        if all(value is None for value in kwargs.values()):
            return None
        return cls(**kwargs)


@dataclass(frozen=True)
class _Task:
    """A preventive task: what each kind shares."""

    #: The spec's ``"policy"`` for this kind, and the one it is by default.
    POLICY: ClassVar[str] = "age"
    DEFAULT_POLICY: ClassVar[str] = "age"
    #: The fields written as the spec's keys of the same name.
    _PLAIN: ClassVar[Tuple[str, ...]] = ()

    def spec(self) -> dict:
        """The ``"preventive"`` spec this task is."""
        out: dict = {}
        if self.POLICY != self.DEFAULT_POLICY:
            out["policy"] = self.POLICY
        for name in self._PLAIN:
            _put(out, name, getattr(self, name), INSTANT)
        restoration = getattr(self, "restoration", None)
        if restoration is not None:
            out.update(restoration.spec())
        return out

    @classmethod
    def _read(cls, spec: dict, used: set):
        kwargs = {
            name: _take(
                spec, used, name, INSTANT if name == "duration" else None
            )
            for name in cls._PLAIN
        }
        if any(f.name == "restoration" for f in fields(cls)):
            kwargs["restoration"] = Restoration._read(spec, used)
        return cls(**kwargs)


_COMMON = ("cost", "duration", "basis")


@dataclass(frozen=True)
class AgePolicy(_Task):
    """Replace (or restore) the unit at age ``interval`` since it was
    last renewed, or at a failure, whichever comes first.

    ``cost`` is each task's price (a number, or a distribution drawn each
    time), ``duration`` its time (a model, or ``"instant"``), and
    ``opportunity`` the age from which a stop of the unit's maintenance
    group renews it early. ``basis="operating"`` counts the times in
    operating hours (put on the clock through the unit's duty). The
    spec's keys are in the [preventive maintenance
    guide](../guide/system-maintenance.md#preventive-maintenance).

    Parameters
    ----------
    interval : float
        The age at which the unit is replaced, in the plan's unit of time,
        more than 0 (``inf``: never).
    cost : float or distribution, optional
        The price of each task, in money: a number, or a distribution of it
        (anything with ``qf`` and ``mean``, such as a fitted surpyval
        model) drawn afresh each time. None prices nothing.
    duration : model, float or "instant", optional
        How long each task takes, in the plan's unit of time: a
        time-to-maintain model or a fixed time, the unit down meanwhile (a
        planned outage); ``"instant"`` (the default) renews it in place.
    opportunity : float, optional
        An age, from 0 to ``interval``, from which a stop of the unit's
        maintenance group (the plan's ``group``) renews it early, as the
        task would; None takes no opportunity.
    restoration : Restoration, optional
        How much of the unit's age each task takes away; None (the
        default) renews it as new.
    basis : str, optional
        What the task's times are counted in: ``"calendar"`` (None, the
        default) or ``"operating"``, the unit's running hours, put on the
        calendar through its ``duty`` (see [Operating part of the
        time](../guide/repairable.md#operating-part-of-the-time)).
    """

    interval: Any = None
    cost: Any = None
    duration: Any = INSTANT
    opportunity: Optional[float] = None
    restoration: Optional[Restoration] = None
    basis: Optional[str] = None

    POLICY: ClassVar[str] = "age"
    _PLAIN: ClassVar[Tuple[str, ...]] = ("interval", "opportunity") + _COMMON


@dataclass(frozen=True)
class BlockPolicy(_Task):
    """Replace (or restore) the unit every ``interval`` whatever its age,
    or at each of the given ``times`` (from 0) and no more. A
    replacement due while the unit is down is skipped.

    Parameters
    ----------
    interval : float, optional
        The time between replacements, in the plan's unit of time (at
        ``interval``, ``2 * interval``, ...); not given with ``times``.
    times : sequence of float, optional
        The times of the replacements, increasing, from 0 on the
        diagram's clock, and none after the last: a plan valued over a
        horizon (see [Replacement at given
        times](../guide/system-maintenance.md#replacement-at-given-times)).
    cost : float or distribution, optional
        The price of each task, in money: a number, or a distribution of it
        (anything with ``qf`` and ``mean``, such as a fitted surpyval
        model) drawn afresh each time. None prices nothing.
    duration : model, float or "instant", optional
        How long each task takes, in the plan's unit of time: a
        time-to-maintain model or a fixed time, the unit down meanwhile (a
        planned outage); ``"instant"`` (the default) renews it in place.
    restoration : Restoration, optional
        How much of the unit's age each task takes away; None (the
        default) renews it as new.
    basis : str, optional
        What the task's times are counted in: ``"calendar"`` (None, the
        default) or ``"operating"``, the unit's running hours, put on the
        calendar through its ``duty`` (see [Operating part of the
        time](../guide/repairable.md#operating-part-of-the-time)).
    """

    interval: Any = None
    times: Optional[Tuple[float, ...]] = None
    cost: Any = None
    duration: Any = INSTANT
    restoration: Optional[Restoration] = None
    basis: Optional[str] = None

    POLICY: ClassVar[str] = "block"
    _PLAIN: ClassVar[Tuple[str, ...]] = ("interval",) + _COMMON

    def spec(self) -> dict:
        """The ``"preventive"`` spec this task is (``"replace_at"`` for
        given ``times``)."""
        out = super().spec()
        if self.times is not None:
            # A plan's times are block replacement by default.
            out.pop("policy", None)
            out["replace_at"] = self.times
        return out

    @classmethod
    def _read(cls, spec: dict, used: set):
        task = super()._read(spec, used)
        times = _take(spec, used, "replace_at")
        if times is None:
            return task
        from dataclasses import replace

        return replace(task, times=times)


@dataclass(frozen=True)
class ConditionPolicy(_Task):
    """Inspect the unit every ``interval`` and replace (or restore) it
    when its chance of failing by the next inspection is above
    ``threshold``, or, for a degradation process, its measured ``level``
    is reached; each inspection costs ``inspection_cost``. See
    [Replacement on
    condition](../guide/system-maintenance.md#replacement-on-condition).

    Parameters
    ----------
    interval : float
        The time between inspections, in the plan's unit of time, more
        than 0; inspections are done while the unit is up, in no time.
    threshold : float, optional
        A probability: the unit is replaced when its chance of
        failing before the next inspection, given its age, is above it.
        Not given with ``level``.
    level : float, optional
        For a life that is a degradation process: the measured level, in
        the process's units, at or past which the unit is replaced
        (simulated only). Not given with ``threshold``.
    inspection_cost : float or distribution, optional
        The price of each inspection, in money (a number, or a
        distribution drawn afresh each time).
    cost : float or distribution, optional
        The price of each replacement, in money: a number, or a
        distribution of it (anything with ``qf`` and ``mean``, such as a
        fitted surpyval model) drawn afresh each time. None prices
        nothing.
    duration : model, float or "instant", optional
        How long each task takes, in the plan's unit of time: a
        time-to-maintain model or a fixed time, the unit down meanwhile (a
        planned outage); ``"instant"`` (the default) renews it in place.
    restoration : Restoration, optional
        How much of the unit's age each task takes away; None (the
        default) renews it as new.
    basis : str, optional
        What the task's times are counted in: ``"calendar"`` (None, the
        default) or ``"operating"``, the unit's running hours, put on the
        calendar through its ``duty`` (see [Operating part of the
        time](../guide/repairable.md#operating-part-of-the-time)).
    """

    interval: Any = None
    threshold: Optional[float] = None
    level: Optional[float] = None
    inspection_cost: Any = None
    cost: Any = None
    duration: Any = INSTANT
    restoration: Optional[Restoration] = None
    basis: Optional[str] = None

    POLICY: ClassVar[str] = "condition"
    _PLAIN: ClassVar[Tuple[str, ...]] = (
        "interval",
        "threshold",
        "level",
        "inspection_cost",
    ) + _COMMON


@dataclass(frozen=True)
class CalendarReplacement(_Task):
    """Replace the unit on a calendar ``policy``'s thresholds: a
    ``CalendarPolicy`` (``NonRepairable.optimal_calendar_policy``) or a
    site's ``ShutdownPolicy``, whose prices are charged; ``opportunity``
    as for ``AgePolicy``. See [Replacement on a calendar
    policy](../guide/system-maintenance.md#replacement-on-a-calendar-policy).

    Parameters
    ----------
    policy : CalendarPolicy or ShutdownPolicy
        The calendar policy whose thresholds say when the unit is
        replaced: at the first time its age reaches one. Its prices are
        charged, so the task takes no ``cost``.
    duration : model, float or "instant", optional
        How long each task takes, in the plan's unit of time: a
        time-to-maintain model or a fixed time, the unit down meanwhile (a
        planned outage); ``"instant"`` (the default) replaces it in place.
    opportunity : float, optional
        An age, from which a stop of the unit's
        maintenance group (the plan's ``group``) renews it early, as the
        task would; None takes no opportunity.
    basis : str, optional
        What the task's times are counted in: ``"calendar"`` (None, the
        default) or ``"operating"``, the unit's running hours, put on the
        calendar through its ``duty`` (see [Operating part of the
        time](../guide/repairable.md#operating-part-of-the-time)).
    """

    policy: Any = None
    duration: Any = INSTANT
    opportunity: Optional[float] = None
    basis: Optional[str] = None

    POLICY: ClassVar[str] = "calendar"
    _PLAIN: ClassVar[Tuple[str, ...]] = ("duration", "opportunity", "basis")

    def spec(self) -> dict:
        """The ``"preventive"`` spec this task is (its policy under
        ``"calendar"``)."""
        out = super().spec()
        _put(out, "calendar", self.policy)
        return out

    @classmethod
    def _read(cls, spec: dict, used: set):
        task = super()._read(spec, used)
        from dataclasses import replace

        return replace(task, policy=_take(spec, used, "calendar"))


@dataclass(frozen=True)
class AtShutdown(_Task):
    """Renew the unit at each of the site's shutdowns named ``shutdown``
    (a name, or a list of names), taking the shut's time; with
    ``older_than``, only if it is that old. ``policy="condition"``
    inspects it there instead, replacing it above ``threshold``. See
    [Shutdowns](../guide/system-maintenance.md#shutdowns).

    Parameters
    ----------
    shutdown : str or list of str
        The name (or names) of the diagram's shutdowns at whose shuts the
        task is done; it takes the shut's time.
    policy : str, optional
        ``"block"`` (the default): renewed at every shut (with
        ``older_than``, only if that old). ``"condition"``: inspected at
        every shut and replaced above ``threshold``.
    older_than : float, optional
        Renewed at a shut only if its age then is at least this, in the
        plan's unit of time.
    threshold : float, optional
        Under ``"condition"``: the probability of failing
        before the next shut, given its age, above which it is replaced.
    inspection_cost : float or distribution, optional
        The price of each inspection (under ``"condition"``), in money
        (a number, or a distribution drawn afresh each time).
    cost : float or distribution, optional
        The price of each renewal, in money: a number, or a distribution of it
        (anything with ``qf`` and ``mean``, such as a fitted surpyval
        model) drawn afresh each time. None prices nothing.
    restoration : Restoration, optional
        How much of the unit's age each task takes away; None (the
        default) renews it as new.
    basis : str, optional
        What the task's times are counted in: ``"calendar"`` (None, the
        default) or ``"operating"``, the unit's running hours, put on the
        calendar through its ``duty`` (see [Operating part of the
        time](../guide/repairable.md#operating-part-of-the-time)).
    """

    shutdown: Any = None
    policy: str = "block"
    older_than: Optional[float] = None
    threshold: Optional[float] = None
    inspection_cost: Any = None
    cost: Any = None
    restoration: Optional[Restoration] = None
    basis: Optional[str] = None

    POLICY: ClassVar[str] = "block"
    DEFAULT_POLICY: ClassVar[str] = "block"
    _PLAIN: ClassVar[Tuple[str, ...]] = (
        "shutdown",
        "older_than",
        "threshold",
        "inspection_cost",
        "cost",
        "basis",
    )

    def spec(self) -> dict:
        """The ``"preventive"`` spec this task is (with its
        ``"shutdown"``)."""
        out = super().spec()
        if self.policy != self.DEFAULT_POLICY:
            out = {"policy": self.policy, **out}
        return out

    @classmethod
    def _read(cls, spec: dict, used: set):
        task = super()._read(spec, used)
        from dataclasses import replace

        return replace(task, policy=_take(spec, used, "policy", "block"))


@dataclass(frozen=True)
class OnRule(_Task):
    """Renew (or restore) the unit on each date of a recurrence ``rule``
    (the first Monday of each quarter, say), at the clock time ``at``: a
    task on the calendar, whose dates need one. A diagram holding it
    refuses to be built without a calendar; ``CalendarContext.diagram``
    puts it on that calendar, the task then replacing the unit at the
    rule's dates within it (``CalendarContext.rule_times``), as
    ``BlockPolicy(times=...)`` does. A replacement due while the unit is
    down is skipped. See [Tasks on a
    rule](../guide/system-maintenance.md#tasks-on-a-rule).

    Parameters
    ----------
    rule : Recurrence
        The recurrence rule (``repyability.schedule.Recurrence``) whose
        dates the task falls on.
    at : str or datetime.time, optional
        The clock time on each date (``"06:00"``); midnight by default.
    cost : float or distribution, optional
        The price of each task, in money: a number, or a distribution of it
        (anything with ``qf`` and ``mean``, such as a fitted surpyval
        model) drawn afresh each time. None prices nothing.
    duration : model, float or "instant", optional
        How long each task takes, in the plan's unit of time: a
        time-to-maintain model or a fixed time, the unit down meanwhile (a
        planned outage); ``"instant"`` (the default) renews it in place.
    restoration : Restoration, optional
        How much of the unit's age each task takes away; None (the
        default) renews it as new.
    """

    rule: Any = None
    at: Any = None
    cost: Any = None
    duration: Any = INSTANT
    restoration: Optional[Restoration] = None

    POLICY: ClassVar[str] = "block"
    DEFAULT_POLICY: ClassVar[str] = "block"
    _PLAIN: ClassVar[Tuple[str, ...]] = ("rule", "at", "cost", "duration")

    def __post_init__(self):
        from repyability.schedule._recurrence import Recurrence

        if not isinstance(self.rule, Recurrence):
            raise TypeError(
                "OnRule's rule must be a Recurrence "
                "(repyability.schedule.Recurrence.monthly(...), ...); got "
                f"{self.rule!r}."
            )
        if self.at is not None:
            from repyability.schedule._meters import clock

            clock(self.at, "at")  # checked where it is given


@dataclass(frozen=True)
class Inspection:
    """Test the unit every ``interval`` (or at the site's ``shutdown``),
    finding a hidden failure; the first at ``offset``, each finding one
    with probability ``coverage`` and every ``full_test`` all of them,
    each taking ``duration`` and costing ``cost``. The spec's keys are in
    [Hidden failures and
    inspection](../guide/system-maintenance.md#hidden-failures-and-inspection).

    Parameters
    ----------
    interval : float, optional
        The time between tests, in the plan's unit of time, more than 0;
        not given with ``shutdown``.
    cost : float or distribution, optional
        The price of each test, in money: a number, or a distribution of it
        (anything with ``qf`` and ``mean``, such as a fitted surpyval
        model) drawn afresh each time. None prices nothing.
    duration : model, float or "instant", optional
        How long each test takes, in the plan's unit of time: a model or a
        fixed time, the unit off line meanwhile; ``"instant"`` (the
        default) takes no time.
    offset : float, optional
        The time of the first test, from 0 to less than ``interval``
        (tests at ``offset``, ``offset + interval``, ...); by default the
        first is at ``interval``.
    coverage : float, optional
        The chance in [0, 1] that a test finds a failure (by default 1); a
        failure a test misses stays hidden until a full test.
    full_test : float, optional
        With a ``coverage`` below 1: the time between full tests, which
        find every failure, a whole multiple of ``interval``.
    shutdown : str or list of str, optional
        The name (or names) of the diagram's shutdowns at whose shuts the
        unit is tested, in place of an ``interval``.
    basis : str, optional
        What the test's times (``interval``, ``offset``, ``full_test``)
        are counted in: ``"calendar"`` (None, the default) or
        ``"operating"``, the unit's running hours, put on the calendar
        through its ``duty`` (see [Operating part of the
        time](../guide/repairable.md#operating-part-of-the-time)).
    """

    interval: Any = None
    cost: Any = None
    duration: Any = INSTANT
    offset: Optional[float] = None
    coverage: Optional[float] = None
    full_test: Optional[float] = None
    shutdown: Any = None
    basis: Optional[str] = None

    def spec(self) -> dict:
        """The ``"inspection"`` spec this task is."""
        out: dict = {}
        for f in fields(self):
            _put(out, f.name, getattr(self, f.name), INSTANT)
        return out

    @classmethod
    def _read(cls, spec: dict, used: set) -> "Inspection":
        return cls(
            **{
                f.name: _take(
                    spec,
                    used,
                    f.name,
                    INSTANT if f.name == "duration" else None,
                )
                for f in fields(cls)
            }
        )


@dataclass(frozen=True)
class Upgrade:
    """From time ``at`` (from 0) the unit has another ``life``: the unit
    in service is replaced by one of it (in no time, or as the plan's
    block or age replacement replaces it), for the capital ``cost``; or,
    with ``replace_in_service=False``, it stays until its next failure
    or due replacement, and the unit put in then has the new life.

    Parameters
    ----------
    at : float
        When the upgrade is made, from 0 on the diagram's clock, in the
        plan's unit of time.
    life : model
        The new unit's time-to-failure model (as the plan's ``life``).
    cost : float, optional
        The capital cost of the upgrade, in money, charged at ``at``.
    replace_in_service : bool, optional
        Whether the unit in service at ``at`` is replaced then (None:
        the default, True), or stays until its next failure or due
        replacement (False).
    """

    at: Any = None
    life: Any = None
    cost: Any = None
    replace_in_service: Any = None

    def spec(self) -> dict:
        """The ``"upgrade"`` spec this task is."""
        out: dict = {"at": self.at, "reliability": self.life}
        _put(out, "cost", self.cost)
        _put(out, "replace_in_service", self.replace_in_service)
        return out

    @classmethod
    def _read(cls, spec: dict, used: set) -> "Upgrade":
        return cls(
            at=_take(spec, used, "at"),
            life=_take(spec, used, "reliability"),
            cost=_take(spec, used, "cost"),
            replace_in_service=_take(spec, used, "replace_in_service"),
        )


@dataclass(frozen=True)
class ImperfectRepair:
    """A repair that restores the unit only partly: Kijima's ``model``
    (``"kijima1"`` or ``"kijima2"``) with factor ``q`` in [0, 1], or a
    fitted renewal ``process`` (G1, ARA, ARI) repairing by its own rule.
    With ``replace_after`` ``N``, the ``N``-th failure since the unit was
    renewed replaces it. See [Imperfect
    repair](../guide/repairable.md#imperfect-repair).

    Parameters
    ----------
    model : str, optional
        Kijima's virtual-age model: ``"kijima1"`` (the default, the
        factor applied to the age gained since the last repair) or
        ``"kijima2"`` (to the whole virtual age).
    q : float, optional
        The restoration factor in [0, 1]: 0 renews the unit at each
        repair, 1 leaves it as old as it was (minimal repair).
    process : surpyval renewal model, optional
        A fitted G1, ARA or ARI renewal process repairing by its own rule,
        in place of ``model`` and ``q``.
    replace_after : int, optional
        The failure, counted since the unit was renewed, at which it is
        replaced rather than repaired.
    """

    model: str = "kijima1"
    q: Optional[float] = None
    process: Any = None
    replace_after: Optional[int] = None

    def spec(self) -> dict:
        """The spec's ``"repair"`` and ``"replace_after"``."""
        out: dict = {}
        if self.process is not None:
            out["repair"] = {"model": "process", "process": self.process}
        elif self.q is not None:
            out["repair"] = {"model": self.model, "q": self.q}
        _put(out, "replace_after", self.replace_after)
        return out


@dataclass(frozen=True)
class Prices:
    """The unit's own prices: each ``repair``, each ``replace`` (a
    failure that uses a spare; each a number or a distribution), its own
    ``downtime`` per unit of time down, and its ``acquisition``, the
    one-off cost of buying it. A task's price is the task's ``cost``.
    The [costs guide](../guide/costs.md) says which price is charged when.

    Parameters
    ----------
    repair : float or distribution, optional
        The price of each repair (each failure), in money: a number, or a
        distribution drawn afresh each time.
    replace : float or distribution, optional
        The price of each replacement on failure (one that uses a spare),
        in money: a number, or a distribution drawn afresh each time.
    downtime : float, optional
        The price per unit of time the unit is down, in money per the
        plan's unit of time, whether or not the system is.
    acquisition : float, optional
        The one-off cost of buying the unit, in money.
    """

    repair: Any = None
    replace: Any = None
    downtime: Any = None
    acquisition: Any = None

    #: The spec's key for each price.
    KEYS: ClassVar[Dict[str, str]] = {
        "repair": "repair_cost",
        "replace": "replace_cost",
        "downtime": "downtime_cost",
        "acquisition": "acquisition_cost",
    }


#: The task classes a plan takes, by the spec key each is under.
PREVENTIVE_TASKS = (
    AgePolicy,
    BlockPolicy,
    ConditionPolicy,
    CalendarReplacement,
    AtShutdown,
    OnRule,
)
_POLICIES = {
    "age": AgePolicy,
    "block": BlockPolicy,
    "condition": ConditionPolicy,
    "calendar": CalendarReplacement,
}


#: The plan's fields written as the spec's keys of the same name.
_OWN_KEYS = (
    "priority",
    "group",
    "standby",
    "duty",
    "covariates",
    "salvage",
    "escalation",
    "retire_at",
    "load_sharing",
    "identical",
    "spares",
    "demand_rate",
)


def _kind(task) -> str:
    if isinstance(task, PREVENTIVE_TASKS):
        return "preventive"
    if isinstance(task, Inspection):
        return "inspection"
    if isinstance(task, Upgrade):
        return "upgrade"
    names = ", ".join(
        cls.__name__ for cls in PREVENTIVE_TASKS + (Inspection, Upgrade)
    )
    raise TypeError(
        f"A UnitPlan's task must be one of {names} "
        f"(repyability.maintenance), got {task!r}."
    )


@dataclass(frozen=True)
class UnitPlan:
    """One unit's maintenance plan: its ``life`` and ``repair`` (time to
    repair) models, its ``tasks`` and its ``prices``.

    A ``RepairableRBD`` node takes a plan as it takes a spec dict (the
    plan's shorthand: ``from_spec`` and ``spec``), so every analysis takes
    it. A plan holds at most one preventive task (bind it to several
    shutdowns with ``AtShutdown(shutdown=[...])``), one ``Inspection`` and
    one ``Upgrade``. ``imperfect_repair`` makes repairs partial;
    ``priority`` (the crews' queue, higher first), ``group`` (its
    maintenance group's name), ``standby`` (a dict, making the node a
    standby group of such units), ``duty`` (the share of the time it
    operates, for a life in operating hours), ``covariates`` (a
    surpyval ``StepSchedule``), ``spares`` (its shelf of spares) and
    ``demand_rate`` (the demands per unit of time on a life that is a
    probability of failing per demand) are as in the spec.

    Three terms are the unit's own money, which a valuation
    (``expected_npv``) takes as it takes its arguments of the same names
    for the node: ``salvage``, what the unit is worth at the end (a
    number, or a function of the age of the unit in service, taking an
    array of ages; negative for a cost of disposal); ``escalation``, the
    growth rate per unit time of every price of the unit (continuous, of
    either sign); and ``retire_at``, when the unit is retired, nothing
    earned or spent after it (a plan valued alone, or the only unit of a
    diagram).

    A plan is valued alone through the diagram of its one node
    (``diagram()``, the node named ``UnitPlan.NODE``): ``expected_npv``,
    ``cash_flows``, ``npv_distribution``, ``plan_vs_actual``,
    ``expected_cost``, ``expected_cost_rate``, ``events_by_period``,
    ``failures_by_period`` and the optimisers are that diagram's, so a plan
    alone and a diagram holding it give the same values by the same code.
    Each takes ``calendar``, a ``schedule.CalendarContext`` whose ``duty``
    names the node ``"unit"``: the diagram on that calendar.

    Parameters
    ----------
    life : model
        The unit's time-to-failure model, in the plan's unit of time: a
        fitted surpyval model, or any model with a life's functions (a
        ``NonRepairableRBD`` included), as a spec's ``"reliability"``.
    repair : model, float or "instant", optional
        The unit's time-to-repair model, as a spec's
        ``"repairability"``; ``"instant"`` repairs it in no time.
    tasks : sequence, optional
        Its tasks: at most one preventive task (``AgePolicy``,
        ``BlockPolicy``, ``ConditionPolicy``, ``CalendarReplacement``,
        ``AtShutdown`` or ``OnRule``), one ``Inspection`` and one
        ``Upgrade``.
    prices : Prices or dict, optional
        The unit's own prices (a dict is read as the fields of a
        ``Prices``); by default none.
    imperfect_repair : ImperfectRepair, optional
        Makes each repair partial; by default a repair renews the unit.
    priority : float, optional
        Its place in the repair crews' queue: higher first.
    group : hashable, optional
        The name of its maintenance group, whose stops it shares.
    standby : dict, optional
        Makes the node a standby group of such units, as a spec's
        ``"standby"``.
    duty : float, optional
        The share of the time it operates, in (0, 1], for a life in
        operating hours.
    covariates : surpyval StepSchedule, optional
        The covariates of a regression life over time.
    salvage : float or callable, optional
        What the unit is worth at the end, in money: a number, or a
        function of the age of the unit in service (taking an array of
        ages); negative for a cost of disposal. In the simulated runs a
        unit down at the end (in its repair, or failed unseen by its
        tests) is worth nothing; the age is its virtual age where it
        keeps one, and its operating time on a duty plan; a group's
        units are each worth the function at its own age (a spare's the
        life it has used up waiting); and inside a nested diagram each
        unit is credited by its own salvage.
    escalation : float, optional
        The continuous growth rate per unit of time of every price of the
        unit, of either sign.
    retire_at : float, optional
        When the unit is retired, in the plan's unit of time from 0;
        nothing is earned or spent after it.
    load_sharing : dict, optional
        Makes the node a load-sharing group of such units, as a spec's
        ``"load_sharing"``.
    identical : dict, optional
        Makes the node a block of identical units, each repaired on its
        own, as a spec's ``"identical"``.
    spares : dict, optional
        Its shelf of spares, as a spec's ``"spares"``: ``"stock"``,
        ``"lead_time"`` and optionally ``"repairs"``.
    demand_rate : float, optional
        The demands per unit of time on a unit whose life is a
        probability of failing per demand, as a spec's ``"demand_rate"``.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability.maintenance import AgePolicy, UnitPlan, Prices
    >>> plan = UnitPlan(
    ...     surv.Weibull.from_params([1000.0, 2.5]),
    ...     surv.Exponential.from_params([0.1]),
    ...     tasks=(AgePolicy(interval=4000.0, cost=20.0),),
    ...     prices=Prices(repair=400.0),
    ... )
    >>> sorted(plan.spec())
    ['preventive', 'reliability', 'repair_cost', 'repairability']
    >>> UnitPlan.from_spec(plan.spec()) == plan
    True
    """

    life: Any
    repair: Any = None
    tasks: Tuple[Any, ...] = ()
    prices: Prices = field(default_factory=Prices)
    imperfect_repair: Optional[ImperfectRepair] = None
    priority: Optional[float] = None
    group: Optional[Hashable] = None
    standby: Optional[dict] = None
    duty: Optional[float] = None
    covariates: Any = None
    salvage: Any = None
    escalation: Optional[float] = None
    retire_at: Optional[float] = None
    load_sharing: Optional[dict] = None
    identical: Optional[dict] = None
    spares: Optional[dict] = None
    demand_rate: Optional[float] = None

    #: The node a plan valued alone is in its one-node diagram
    #: (``diagram()``): what its results' tables name.
    NODE: ClassVar[str] = "unit"

    def __post_init__(self):
        if self.life is None:
            raise ValueError(
                "A UnitPlan needs the unit's life: a fitted model "
                "of its time to failure (such as a surpyval distribution)."
            )
        tasks = self.tasks
        if not isinstance(tasks, (tuple, list)):
            tasks = (tasks,)
        object.__setattr__(self, "tasks", tuple(tasks))
        if isinstance(self.prices, dict):
            object.__setattr__(self, "prices", Prices(**self.prices))
        elif not isinstance(self.prices, Prices):
            raise TypeError(
                "A UnitPlan's prices are a Prices (or a dict of its "
                f"fields: {', '.join(Prices.KEYS)}), got {self.prices!r}."
            )
        if not (
            self.imperfect_repair is None
            or isinstance(self.imperfect_repair, ImperfectRepair)
        ):
            raise TypeError(
                "A UnitPlan's imperfect_repair is an ImperfectRepair, "
                f"got {self.imperfect_repair!r}."
            )
        seen: Dict[str, Any] = {}
        for task in self.tasks:
            kind = _kind(task)
            if kind in seen:
                more = (
                    " A task at several shutdowns is one task: "
                    "AtShutdown(shutdown=[...])."
                    if kind == "preventive"
                    else ""
                )
                raise ValueError(
                    f"A UnitPlan holds one {kind} task, got "
                    f"{seen[kind]!r} and {task!r}.{more}"
                )
            seen[kind] = task

    def task(self, kind: str):
        """The plan's ``"preventive"``, ``"inspection"`` or ``"upgrade"``
        task, or None.

        Parameters
        ----------
        kind : str
            ``"preventive"``, ``"inspection"`` or ``"upgrade"``.
        """
        from repyability.utils.checks import one_of

        one_of("kind", kind, ("preventive", "inspection", "upgrade"))
        return next((t for t in self.tasks if _kind(t) == kind), None)

    def spec(self) -> dict:
        """The spec dict this plan is, as a ``RepairableRBD`` node reads
        it: keys at their defaults left out."""
        out: dict = {"reliability": self.life}
        if self.repair is not None:
            out["repairability"] = self.repair
        for name, key in Prices.KEYS.items():
            _put(out, key, getattr(self.prices, name))
        for task in self.tasks:
            out[_kind(task)] = task.spec()
        if self.imperfect_repair is not None:
            out.update(self.imperfect_repair.spec())
        for key in _OWN_KEYS:
            _put(out, key, getattr(self, key))
        return out

    @classmethod
    def from_spec(cls, spec: dict) -> "UnitPlan":
        """The plan a spec dict states (the spec a ``RepairableRBD`` node
        takes); a key no field holds is refused.

        Parameters
        ----------
        spec : dict
            A spec dict with a ``"reliability"``, as a ``RepairableRBD``
            node takes it.
        """
        if not isinstance(spec, dict) or "reliability" not in spec:
            raise ValueError(
                "A spec is a dict with a 'reliability' (and a "
                f"'repairability'), got {spec!r}."
            )
        used: set = set()
        tasks = []
        preventive = _take(spec, used, "preventive")
        if preventive is not None:
            if not isinstance(preventive, dict):
                raise ValueError(
                    f"A 'preventive' spec is a dict, got {preventive!r}."
                )
            inner: set = set()
            if preventive.get("shutdown") is not None:
                kind: Any = AtShutdown
            elif preventive.get("rule") is not None:
                kind = OnRule
            else:
                policy = preventive.get("policy")
                if policy is None:
                    policy = (
                        "block"
                        if preventive.get("replace_at") is not None
                        else "age"
                    )
                if policy not in _POLICIES:
                    raise ValueError(
                        "A 'preventive' spec's policy is 'age', 'block', "
                        f"'condition' or 'calendar', got {policy!r}."
                    )
                kind = _POLICIES[policy]
                inner.add("policy")
            tasks.append(kind._read(preventive, inner))
            _all_read("'preventive'", preventive, inner)
        inspection = _take(spec, used, "inspection")
        if inspection is not None:
            inner = set()
            tasks.append(Inspection._read(inspection, inner))
            _all_read("'inspection'", inspection, inner)
        upgrade = _take(spec, used, "upgrade")
        if upgrade is not None:
            inner = set()
            tasks.append(Upgrade._read(upgrade, inner))
            _all_read("'upgrade'", upgrade, inner)
        repair = _take(spec, used, "repair")
        replace_after = _take(spec, used, "replace_after")
        imperfect = None
        if repair is not None or replace_after is not None:
            repair = repair or {}
            imperfect = (
                ImperfectRepair(process=repair.get("process"))
                if repair.get("model") == "process"
                else ImperfectRepair(
                    model=repair.get("model", "kijima1"), q=repair.get("q")
                )
            )
            if replace_after is not None:
                from dataclasses import replace

                imperfect = replace(imperfect, replace_after=replace_after)
        plan = cls(
            life=_take(spec, used, "reliability"),
            repair=_take(spec, used, "repairability"),
            tasks=tuple(tasks),
            prices=Prices(
                **{
                    name: _take(spec, used, key)
                    for name, key in Prices.KEYS.items()
                }
            ),
            imperfect_repair=imperfect,
            **{key: _take(spec, used, key) for key in _OWN_KEYS},
        )
        _all_read("component", spec, used)
        return plan

    def to_dict(self) -> dict:
        """The plan as JSON-ready data (as a diagram saves a node's spec);
        ``from_dict`` loads it."""
        from repyability.rbd.serialisation import _serialise_component

        return {
            "kind": "unit_plan",
            "spec": _serialise_component(self.spec()),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnitPlan":
        """The plan ``to_dict`` saved.

        Parameters
        ----------
        data : dict
            What ``to_dict`` returned.
        """
        from repyability.rbd.serialisation import _deserialise_component

        if not isinstance(data, dict) or data.get("kind") != "unit_plan":
            raise ValueError(
                "Not a saved UnitPlan (a dict of kind "
                f"'unit_plan', from UnitPlan.to_dict): {data!r}."
            )
        return cls.from_spec(_deserialise_component(data["spec"]))

    # -- the plan valued alone (UC-65 of this package's README) ------------

    def diagram(self, calendar=None):
        """The ``RepairableRBD`` of this plan's one node (named
        ``UnitPlan.NODE``), on ``calendar`` if given (a
        ``schedule.CalendarContext``; its ``duty`` names the node
        ``"unit"``): what every valuation of the plan alone asks.

        Parameters
        ----------
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        """
        from repyability.rbd.repairable_rbd import RepairableRBD

        node = self.NODE
        edges = [("input", node), (node, "output")]
        if calendar is None:
            return RepairableRBD(edges, {node: self})
        from repyability.schedule._calendar import CalendarContext

        if not isinstance(calendar, CalendarContext):
            raise TypeError(
                "calendar must be a schedule.CalendarContext (its duty "
                f"naming the node {node!r}); got {calendar!r}."
            )
        return calendar.diagram(edges, {node: self})

    def analysis_routes(self, calendar=None) -> dict:
        """How each valuation of the plan alone is worked out (exact,
        numerical, simulated or refused, with the reason), without
        running anything: its diagram's routes, by the plan's method.

        Parameters
        ----------
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        """
        routes = self.diagram(calendar).analysis_routes()
        return {
            mine: routes[theirs]
            for mine, theirs in _PLAN_ROUTES.items()
            if theirs in routes
        }

    def expected_npv(self, horizon, *, calendar=None, **kwargs):
        """The plan's expected net present value over ``horizon``:
        ``RepairableRBD.expected_npv`` of its one-node diagram (the same
        arguments), its salvage, escalation and retirement taken from
        the plan where the arguments leave them out.

        Parameters
        ----------
        horizon : float
            The length of time valued, in the plan's unit of time.
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        **kwargs
            As for ``RepairableRBD.expected_npv``.

        Examples
        --------
        >>> import math
        >>> import surpyval as surv
        >>> from repyability.maintenance import AgePolicy, Prices, UnitPlan
        >>> pump = UnitPlan(
        ...     surv.Weibull.from_params([20000.0, 2.5]),
        ...     surv.Exponential.from_params([1 / 24]),
        ...     tasks=(AgePolicy(interval=12000.0, cost=1500.0),),
        ...     prices=Prices(repair=6000.0, acquisition=40000.0),
        ...     salvage=5000.0,
        ... )
        >>> r = math.log(1.07) / 8760
        >>> result = pump.expected_npv(8 * 8760.0, discount_rate=r)
        >>> round(result.npv), round(result.salvage)
        (-48719, 2910)
        """
        return self.diagram(calendar).expected_npv(horizon, **kwargs)

    def cash_flows(self, horizon, *, calendar=None, **kwargs):
        """The plan's expected cash flows by period and category: the
        table of ``expected_npv`` (the same arguments; ``periods`` sets
        the periods).

        Parameters
        ----------
        horizon : float
            The length of time valued, in the plan's unit of time.
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        **kwargs
            As for ``expected_npv``.
        """
        return self.expected_npv(
            horizon, calendar=calendar, **kwargs
        ).cash_flows

    def npv_distribution(self, horizon, *, calendar=None, **kwargs):
        """The plan's NPV in each of many simulated runs:
        ``RepairableRBD.npv_distribution`` of its one-node diagram.

        Parameters
        ----------
        horizon : float
            The length of time valued, in the plan's unit of time.
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        **kwargs
            As for ``RepairableRBD.npv_distribution``.
        """
        return self.diagram(calendar).npv_distribution(horizon, **kwargs)

    def plan_vs_actual(
        self, timeline, now, horizon, *, calendar=None, **kwargs
    ):
        """The plan against what happened to the unit: its ``Timeline``
        (outage log) to ``now``, as ``RepairableRBD.plan_vs_actual`` of
        its one-node diagram; a ``state`` is the unit's ``NodeState``.

        Parameters
        ----------
        timeline : Timeline
            The unit's outage log up to ``now``.
        now : float
            The time of the last record, in the plan's unit of time.
        horizon : float
            The length of time valued from 0, in the plan's unit of time.
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        **kwargs
            As for ``RepairableRBD.plan_vs_actual``; a ``state`` or
            ``start_state`` is the unit's own ``NodeState``.
        """
        state = kwargs.pop("state", None)
        if state is not None:
            kwargs["state"] = {self.NODE: state}
        start = kwargs.pop("start_state", None)
        if start is not None:
            kwargs["start_state"] = {self.NODE: start}
        return self.diagram(calendar).plan_vs_actual(
            {self.NODE: timeline}, now, horizon, **kwargs
        )

    def expected_cost(self, t, *, calendar=None, **kwargs):
        """The plan's expected cost to ``t``:
        ``RepairableRBD.expected_cost`` of its one-node diagram.

        Parameters
        ----------
        t : float
            The time to which costs are counted, in the plan's unit of
            time.
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        **kwargs
            As for ``RepairableRBD.expected_cost``.
        """
        return self.diagram(calendar).expected_cost(t, **kwargs)

    def expected_cost_rate(self, *, calendar=None) -> float:
        """The plan's long-run expected cost per unit time:
        ``RepairableRBD.expected_cost_rate`` of its one-node diagram.

        Parameters
        ----------
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        """
        return self.diagram(calendar).expected_cost_rate()

    def events_by_period(self, usage, period: str = "month", **kwargs):
        """The plan's expected events (and on request costs) in each
        calendar period: ``schedule.events_by_period`` of its one-node
        diagram; ``usage`` a ``UsageForecast`` or a ``CalendarContext``
        (giving the node ``"unit"`` its duty plan).

        Parameters
        ----------
        usage : UsageForecast or CalendarContext
            The calendar and the unit's duty plan (a ``CalendarContext``'s
            ``duty`` names the node ``"unit"``).
        period : str, optional
            The calendar period counted in: ``"month"`` (the default), as
            ``schedule.events_by_period`` takes.
        **kwargs
            As for ``schedule.events_by_period``.
        """
        from repyability.schedule import events_by_period
        from repyability.schedule._calendar import calendar_of

        rbd = self.diagram(calendar_of(usage))
        return events_by_period(rbd, usage, period, **kwargs)

    def failures_by_period(self, usage, period: str = "month", **kwargs):
        """The plan's expected failures (with the other events) in each
        calendar period, as ``events_by_period``: a unit in a plan is
        worked out by the diagram's engine, its tasks and repairs taking
        their time, on the calendar its duty plan gives.

        Parameters
        ----------
        usage : UsageForecast or CalendarContext
            As for ``events_by_period``.
        period : str, optional
            As for ``events_by_period``.
        **kwargs
            As for ``events_by_period``.
        """
        return self.events_by_period(usage, period, **kwargs)

    def optimal_replacement_interval(self, *, calendar=None, **kwargs):
        """The plan's best age-replacement interval:
        ``RepairableRBD.optimal_replacement_intervals`` of its one-node
        diagram (the same arguments, but ``nodes``).

        Parameters
        ----------
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        **kwargs
            As for ``RepairableRBD.optimal_replacement_intervals``.
        """
        return self.diagram(calendar).optimal_replacement_intervals(**kwargs)

    def least_spend_for_downtime(
        self, horizon, limit, *, calendar=None, **kwargs
    ):
        """The least spend that keeps a bad run's downtime under
        ``limit``: ``RepairableRBD.least_spend_for_downtime`` of its
        one-node diagram (the same arguments; a lever names the node
        ``"unit"``).

        Parameters
        ----------
        horizon : float
            The length of time valued.
        limit : float
            The downtime a bad run must stay under.
        calendar : CalendarContext, optional
            The calendar the diagram runs on, as for ``diagram``.
        **kwargs
            As for ``RepairableRBD.least_spend_for_downtime``.
        """
        return self.diagram(calendar).least_spend_for_downtime(
            horizon, limit, **kwargs
        )

    def optimal_inspection_interval(self, *, calendar=None, **kwargs):
        """The plan's best proof-test interval:
        ``RepairableRBD.optimal_inspection_intervals`` of its one-node
        diagram (the same arguments, but ``nodes``).

        Parameters
        ----------
        calendar : CalendarContext, optional
            The calendar the plan's diagram runs on (its ``duty`` names
            the node ``"unit"``); by default none.
        **kwargs
            As for ``RepairableRBD.optimal_inspection_intervals``.
        """
        return self.diagram(calendar).optimal_inspection_intervals(**kwargs)


#: Each valuation of a plan alone, by the route of its diagram's method.
_PLAN_ROUTES = {
    "expected_npv": "expected_npv",
    "cash_flows": "expected_npv",
    "npv_distribution": "npv_distribution",
    "plan_vs_actual": "plan_vs_actual",
    "expected_cost": "expected_cost",
    "expected_cost_rate": "expected_cost_rate",
    "events_by_period": "expected_events_by_window",
    "failures_by_period": "expected_events_by_window",
    "optimal_replacement_interval": "optimal_replacement_intervals",
    "optimal_inspection_interval": "optimal_inspection_intervals",
    "least_spend_for_downtime": "least_spend_for_downtime",
}
