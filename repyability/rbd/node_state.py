"""The per-node condition input for condition-based ("digital twin")
reliability evaluation.

An RBD's *structure* is static, but in a condition-based setting each component
has a *current state* streamed from sensors -- how much life it has already
accumulated, and whether it is still working. ``NodeState`` carries that
per-node state into ``NonRepairableRBD.sf_given_state``,
``NonRepairableRBD.remaining_life`` and
``NonRepairableRBD.importances_given_state``, and, with how long a down
component has been down and where it is in its calendar, into the analyses
of a ``RepairableRBD`` from now rather than from new (its ``state``
argument). Two optional fields say more about the unit in service now:
the degradation level a sensor measured (``level``), and a life model of
its own (``model``), such as one refitted to it.
"""

import math
from dataclasses import dataclass
from typing import Any, Optional

from repyability.utils.checks import is_number


@dataclass(frozen=True)
class NodeState:
    """The current condition of a single RBD node.

    Passed, as a dict ``{node: NodeState}``, to the condition-based methods
    of [`NonRepairableRBD`][repyability.NonRepairableRBD]
    (``sf_given_state``, ``remaining_life`` and
    ``importances_given_state``). A working component's forward reliability
    is conditioned on the life it has already survived:
    ``R_i(x | age) = R_i(age + x) / R_i(age)``. A failed component
    (``alive=False``) contributes zero reliability regardless of its age. A
    node left out of the dict is treated as new and unconditioned. The state
    is immutable (a frozen dataclass).

    A [`RepairableRBD`][repyability.RepairableRBD] takes the same states,
    with more fields, to start its analyses from now (``state=`` of
    ``point_availability``, ``availability`` and the others): a component
    up is ``age`` since it was last put into service as new; one down has
    been down for ``down_for``, in a repair, or (``maintenance=True``) in
    its preventive maintenance; one on a calendar (block replacement, or
    tests of hidden failures) is ``since_scheduled`` past its last
    scheduled replacement or test; one repaired imperfectly has the
    ``virtual_age`` it had at its last repair; and ``stationary=True``
    puts it in its long-run state instead, for a component long in
    service whose state is not known.

    Two fields say more about the unit in service now, in both kinds of
    diagram. ``level`` is the degradation level measured on a unit whose
    life is a fitted surpyval Wiener or gamma degradation process: its
    life from now is the time its level takes to climb from there to the
    failure threshold. Both processes' increments are independent of
    their past, so once the level is known, how long the unit took to get
    there says nothing more about its life: ``age`` is then not used to
    condition the life, but keeps its one meaning, the operating time,
    for what else needs it (an age replacement falls due when the
    operating time reaches its interval). ``model`` is a life model of
    this unit's own (refitted to it, or a joint model's prediction from
    its readings), used instead of the node's for the unit in service
    now, conditioned on its ``age`` as the node's would be; in a
    ``RepairableRBD`` the units that replace it have the node's own life
    again. A ``level`` applies to the ``model`` when both are given.

    Parameters
    ----------
    age : float, optional
        The component's current age / accumulated operating time (``>= 0``),
        by default ``0.0``. Age 0 is the same as no conditioning whenever the
        component's reliability at time 0 is 1, as for ordinary lifetime
        distributions. In a ``RepairableRBD`` it is on the diagram's
        clock: for a component that operates part of the time (a spec's
        ``"duty"``), the calendar time since it was put into service,
        its operating age being ``duty * age``.
    alive : bool, optional
        Whether the component is currently working, by default ``True``.
    down_for : float, optional
        For a repairable component that is down: how long it has been down
        so far, by default ``0.0`` (it has just gone down). What is left of
        its repair (or maintenance) is conditioned on it.
    maintenance : bool, optional
        For a repairable component that is down: whether it is down for
        its preventive maintenance rather than for a repair, by default
        ``False``.
    since_scheduled : float, optional
        For a repairable component on a calendar: the time since its last
        block replacement, or its last test (of hidden failures, at which
        it was found working; with an ``age`` below it, it was put into
        service since, and has not been tested), by default None: 0, as
        from new.
    stationary : bool, optional
        For a repairable component: start it in its long-run state, by
        default ``False``. Its ``since_scheduled`` still places it on its
        calendar; the other fields must be left at their defaults.
    virtual_age : float, optional
        For a repairable component repaired imperfectly (a spec's
        ``"repair"``, or a fitted ``GeneralizedRenewal`` or Poisson
        process): its virtual age at its last repair, by default None (as
        new); ``age`` is then its operating time since. One down is at
        this virtual age once its repair is over. For a unit of a fitted
        surpyval renewal model, ``unit_states()`` gives its
        ``virtual_age`` now and ``since_failure``: give ``age=
        since_failure`` and ``virtual_age=virtual_age - since_failure``.
        The simulations take it.
    level : float, optional
        For a unit whose life is a surpyval Wiener or gamma degradation
        process (the node's, or ``model``): its degradation level measured
        now, by default None (not measured). Its life from now is the
        first passage from this level to the failure threshold. It must be
        below the threshold (a unit at or past it has failed: give
        ``alive=False``), and for a gamma process, whose level only rises,
        at least the process's starting level ``y0``; a Wiener level may
        be below its start. Replaced on condition by its level, the unit
        goes on from it to its next inspection (``since_scheduled`` says
        when the last was), which needs the level: an age alone is
        refused there.
    model : object, optional
        The life model of this unit, used instead of the node's for the
        unit in service now, by default None (the node's). It may be
        anything the node could itself have been given as its life (a
        fitted surpyval model, a ``RegressionNode``, ...), and a joint
        model's prediction (or ``JointNode``) of this unit from its
        readings, whose age is the prediction's. It is checked against the
        node when the state is used.

    Raises
    ------
    ValueError
        If ``age``, ``down_for``, ``since_scheduled`` or ``virtual_age``
        is negative or not finite, ``down_for`` or ``maintenance`` is given
        for a component that is alive, ``level`` or ``model`` for one that
        is down or with a ``virtual_age``, ``level`` is not a finite
        number, or another field is given with ``stationary``.

    Notes
    -----
    A level or a unit's own model is checked against the node where the
    state is used (``sf_given_state``, ``point_availability``, ...): a
    level needs a degradation process for the unit's life, below its
    threshold, and a model must be one the node could have been given.
    Only lifetime (time-varying) distributions age. A fixed-probability
    component given a state with ``alive=True`` contributes reliability 1
    whatever its age: observing it working resolves its per-demand
    uncertainty. Leave it out of the dict to use its unconditioned
    probability instead. Covariate/load dependence is a property of the
    *component*, not its transient state: a
    [`RegressionNode`][repyability.RegressionNode] stores the covariates, so
    ``age`` keeps a single meaning (operating time) for every node type. A
    unit run differently from its fellows is given its own model (a
    ``RegressionNode`` at its covariates) as ``model``; the covariates
    still belong to that model, not to the state.

    Examples
    --------
    >>> from repyability import NodeState
    >>> NodeState(age=1200)
    NodeState(age=1200, alive=True)
    >>> NodeState(alive=False)
    NodeState(age=0.0, alive=False)

    Two Weibull components in parallel: one 50 hours old and one already
    failed, so the system's reliability over the next 20 hours is that of
    the survivor, ``R(70) / R(50)``:

    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD
    >>> rbd = NonRepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {
    ...         "a": surv.Weibull.from_params([100, 2]),
    ...         "b": surv.Weibull.from_params([100, 2]),
    ...     },
    ... )
    >>> state = {"a": NodeState(age=50), "b": NodeState(alive=False)}
    >>> round(rbd.sf_given_state(20, state), 4)
    0.7866

    A repairable pump down for 2 hours of a repair, one 300 hours old,
    and one tested weekly, last tested 100 hours ago:

    >>> NodeState(alive=False, down_for=2.0)
    NodeState(age=0.0, alive=False, down_for=2.0)
    >>> NodeState(age=300.0)
    NodeState(age=300.0, alive=True)
    >>> NodeState(age=300.0, since_scheduled=100.0)
    NodeState(age=300.0, alive=True, since_scheduled=100.0)

    A bearing whose wear is a gamma process (a mean of 0.5 a year: each
    year's wear a gamma of shape 2 and rate 4), failing at a wear of 10,
    measured at 6 after 3 years, against the same bearing new; its age is
    kept as its operating time:

    >>> from surpyval.degradation import GammaProcessModel
    >>> wear = GammaProcessModel(alpha=2.0, beta=4.0, threshold=10.0)
    >>> bearing = NonRepairableRBD([("s", "b"), ("b", "t")], {"b": wear})
    >>> NodeState(age=3.0, level=6.0)
    NodeState(age=3.0, alive=True, level=6.0)
    >>> measured = {"b": NodeState(age=3.0, level=6.0)}
    >>> round(bearing.sf_given_state(5.0, measured), 4)
    0.9567
    >>> round(bearing.sf_given_state(5.0, {}), 4)
    1.0
    """

    age: float = 0.0
    alive: bool = True
    down_for: float = 0.0
    maintenance: bool = False
    since_scheduled: Optional[float] = None
    stationary: bool = False
    virtual_age: Optional[float] = None
    level: Optional[float] = None
    model: Any = None

    def __post_init__(self) -> None:
        if self.age < 0:
            raise ValueError(
                f"NodeState.age must be non-negative, got {self.age!r}."
            )
        for name in ("age", "down_for", "since_scheduled", "virtual_age"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(
                    f"NodeState.{name} must be finite and non-negative, got "
                    f"{value!r}."
                )
        if self.level is not None and not (
            is_number(self.level) and math.isfinite(float(self.level))
        ):
            raise ValueError(
                "NodeState.level is the degradation level measured now, a "
                f"finite number, got {self.level!r}."
            )
        if self.own_life and not self.alive:
            raise ValueError(
                "NodeState.level and model describe the unit in service, "
                "which is working: a unit that is down has neither (the "
                "unit that replaces it is the node's own)."
            )
        if self.own_life and self.virtual_age is not None:
            raise ValueError(
                "NodeState.virtual_age conditions the node's own life on "
                "its imperfect repairs, and a level or a model of the "
                "unit's own replaces that life: give one or the other."
            )
        if self.alive and (self.down_for or self.maintenance):
            raise ValueError(
                "NodeState.down_for and maintenance describe a component "
                "that is down: give alive=False with them."
            )
        if self.stationary and (
            self.age
            or not self.alive
            or self.down_for
            or self.maintenance
            or self.virtual_age
            or self.own_life
        ):
            raise ValueError(
                "A stationary NodeState is in its long-run state, which "
                "decides its age and whether it is up: give only its "
                "since_scheduled."
            )

    def __repr__(self) -> str:
        fields = [f"age={self.age!r}", f"alive={self.alive!r}"]
        if self.down_for:
            fields.append(f"down_for={self.down_for!r}")
        if self.maintenance:
            fields.append("maintenance=True")
        if self.since_scheduled is not None:
            fields.append(f"since_scheduled={self.since_scheduled!r}")
        if self.stationary:
            fields.append("stationary=True")
        if self.virtual_age is not None:
            fields.append(f"virtual_age={self.virtual_age!r}")
        if self.level is not None:
            fields.append(f"level={self.level!r}")
        if self.model is not None:
            fields.append(f"model={self.model!r}")
        return f"NodeState({', '.join(fields)})"

    @property
    def new(self) -> bool:
        """Whether the state is that of a component new now: up, at age
        0, at the start of its calendar."""
        return (
            self.alive
            and not self.age
            and not self.stationary
            and not self.since_scheduled
            and not self.virtual_age
            and not self.own_life
        )

    @property
    def own_life(self) -> bool:
        """Whether the unit in service now has a life of its own: a
        measured ``level``, or its own ``model``."""
        return self.level is not None or self.model is not None
