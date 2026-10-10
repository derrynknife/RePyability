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
argument).
"""

import math
from dataclasses import dataclass
from typing import Optional


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
    tests of hidden failures) is ``phase`` past its last scheduled
    replacement or test; one repaired imperfectly has the ``virtual_age``
    it had at its last repair; and ``stationary=True`` puts it in its
    long-run state instead, for a component long in service whose state is
    not known.

    Parameters
    ----------
    age : float, optional
        The component's current age / accumulated operating time (``>= 0``),
        by default ``0.0``. Age 0 is the same as no conditioning whenever the
        component's reliability at time 0 is 1, as for ordinary lifetime
        distributions.
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
    phase : float, optional
        For a repairable component on a calendar: the time since its last
        block replacement, or its last test (of hidden failures, at which
        it was found working; with an ``age`` below the phase, it was put
        into service since, and has not been tested), by default None: 0,
        as from new.
    stationary : bool, optional
        For a repairable component: start it in its long-run state, by
        default ``False``. Its ``phase`` still places it on its calendar;
        the other fields must be left at their defaults.
    virtual_age : float, optional
        For a repairable component repaired imperfectly (a spec's
        ``"repair"``, or a fitted ``GeneralizedRenewal`` or Poisson
        process): its virtual age at its last repair, by default None (as
        new); ``age`` is then its operating time since. One down is at
        this virtual age once its repair is over. For a unit of a fitted
        surpyval renewal model, ``unit_states()`` gives its
        ``virtual_age`` now and ``since_failure``: give ``age=
        since_failure`` and ``virtual_age=virtual_age - since_failure``
        (#269). The simulations take it.

    Raises
    ------
    ValueError
        If ``age``, ``down_for``, ``phase`` or ``virtual_age`` is negative
        or not finite, ``down_for`` or ``maintenance`` is given for a
        component that is alive, or another field is given with
        ``stationary``.

    Notes
    -----
    Only lifetime (time-varying) distributions age. A fixed-probability
    component given a state with ``alive=True`` contributes reliability 1
    whatever its age: observing it working resolves its per-demand
    uncertainty. Leave it out of the dict to use its unconditioned
    probability instead. Covariate/load dependence is a property of the
    *component*, not its transient state: a
    [`RegressionNode`][repyability.RegressionNode] stores the covariates, so
    ``age`` keeps a single meaning (operating time) for every node type.

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

    A repairable pump down for 2 hours of a repair, and one 300 hours old:

    >>> NodeState(alive=False, down_for=2.0)
    NodeState(age=0.0, alive=False, down_for=2.0)
    >>> NodeState(age=300.0)
    NodeState(age=300.0, alive=True)
    """

    age: float = 0.0
    alive: bool = True
    down_for: float = 0.0
    maintenance: bool = False
    phase: Optional[float] = None
    stationary: bool = False
    virtual_age: Optional[float] = None

    def __post_init__(self) -> None:
        if self.age < 0:
            raise ValueError(
                f"NodeState.age must be non-negative, got {self.age!r}."
            )
        for name in ("age", "down_for", "phase", "virtual_age"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(
                    f"NodeState.{name} must be finite and non-negative, got "
                    f"{value!r}."
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
        ):
            raise ValueError(
                "A stationary NodeState is in its long-run state, which "
                "decides its age and whether it is up: give only its phase."
            )

    def __repr__(self) -> str:
        fields = [f"age={self.age!r}", f"alive={self.alive!r}"]
        if self.down_for:
            fields.append(f"down_for={self.down_for!r}")
        if self.maintenance:
            fields.append("maintenance=True")
        if self.phase is not None:
            fields.append(f"phase={self.phase!r}")
        if self.stationary:
            fields.append("stationary=True")
        if self.virtual_age is not None:
            fields.append(f"virtual_age={self.virtual_age!r}")
        return f"NodeState({', '.join(fields)})"

    @property
    def new(self) -> bool:
        """Whether the state is that of a component new now: up, at age
        0, at the start of its calendar."""
        return (
            self.alive
            and not self.age
            and not self.stationary
            and not self.phase
            and not self.virtual_age
        )
