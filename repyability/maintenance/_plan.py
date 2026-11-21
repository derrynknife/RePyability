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
    ``replacement_cost`` (the task's own cost if None).
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
    operating hours (put on the clock through the unit's duty).
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
    or at each of the given ``times`` (from 0) and no more."""

    interval: Any = None
    times: Optional[Tuple[float, ...]] = None
    cost: Any = None
    duration: Any = INSTANT
    restoration: Optional[Restoration] = None
    basis: Optional[str] = None

    POLICY: ClassVar[str] = "block"
    _PLAIN: ClassVar[Tuple[str, ...]] = ("interval",) + _COMMON

    def spec(self) -> dict:
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
    is reached; each inspection costs ``inspection_cost``."""

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
    as for ``AgePolicy``."""

    policy: Any = None
    duration: Any = INSTANT
    opportunity: Optional[float] = None
    basis: Optional[str] = None

    POLICY: ClassVar[str] = "calendar"
    _PLAIN: ClassVar[Tuple[str, ...]] = ("duration", "opportunity", "basis")

    def spec(self) -> dict:
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
    inspects it there instead, replacing it above ``threshold``."""

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
class Inspection:
    """Test the unit every ``interval`` (or at the site's ``shutdown``),
    finding a hidden failure; the first at ``offset``, each finding one
    with probability ``coverage`` and every ``full_test`` all of them,
    each taking ``duration`` and costing ``cost``."""

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
    in service is replaced by one of it, in no time, for the capital
    ``cost``."""

    at: Any = None
    life: Any = None
    cost: Any = None

    def spec(self) -> dict:
        """The ``"upgrade"`` spec this task is."""
        out: dict = {"at": self.at, "reliability": self.life}
        _put(out, "cost", self.cost)
        return out

    @classmethod
    def _read(cls, spec: dict, used: set) -> "Upgrade":
        return cls(
            at=_take(spec, used, "at"),
            life=_take(spec, used, "reliability"),
            cost=_take(spec, used, "cost"),
        )


@dataclass(frozen=True)
class ImperfectRepair:
    """A repair that restores the unit only partly: Kijima's ``model``
    (``"kijima1"`` or ``"kijima2"``) with factor ``q`` in [0, 1], or a
    fitted renewal ``process`` (G1, ARA, ARI) repairing by its own rule.
    With ``replace_after`` ``N``, the ``N``-th failure since the unit was
    renewed replaces it."""

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
    one-off cost of buying it. A task's price is the task's ``cost``."""

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
)
_POLICIES = {
    "age": AgePolicy,
    "block": BlockPolicy,
    "condition": ConditionPolicy,
    "calendar": CalendarReplacement,
}


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
    operates, for a life in operating hours) and ``covariates`` (a
    surpyval ``StepSchedule``) are as in the spec.

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
        task, or None."""
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
        for key in ("priority", "group", "standby", "duty", "covariates"):
            _put(out, key, getattr(self, key))
        return out

    @classmethod
    def from_spec(cls, spec: dict) -> "UnitPlan":
        """The plan a spec dict states (the spec a ``RepairableRBD`` node
        takes); a key no field holds is refused."""
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
            **{
                key: _take(spec, used, key)
                for key in (
                    "priority",
                    "group",
                    "standby",
                    "duty",
                    "covariates",
                )
            },
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
        """The plan ``to_dict`` saved."""
        from repyability.rbd.serialisation import _deserialise_component

        if not isinstance(data, dict) or data.get("kind") != "unit_plan":
            raise ValueError(
                "Not a saved UnitPlan (a dict of kind "
                f"'unit_plan', from UnitPlan.to_dict): {data!r}."
            )
        return cls.from_spec(_deserialise_component(data["spec"]))
