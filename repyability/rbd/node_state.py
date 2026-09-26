"""The per-node condition input for condition-based ("digital twin")
reliability evaluation.

An RBD's *structure* is static, but in a condition-based setting each component
has a *current state* streamed from sensors -- how much life it has already
accumulated, and whether it is still working. ``NodeState`` carries that
per-node state into ``NonRepairableRBD.sf_given_state``,
``NonRepairableRBD.remaining_life`` and
``NonRepairableRBD.importances_given_state``.
"""

from dataclasses import dataclass


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

    Parameters
    ----------
    age : float, optional
        The component's current age / accumulated operating time (``>= 0``),
        by default ``0.0``. Age 0 is the same as no conditioning whenever the
        component's reliability at time 0 is 1, as for ordinary lifetime
        distributions.
    alive : bool, optional
        Whether the component is currently working, by default ``True``.

    Raises
    ------
    ValueError
        If ``age`` is negative.

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
    """

    age: float = 0.0
    alive: bool = True

    def __post_init__(self) -> None:
        if self.age < 0:
            raise ValueError(
                f"NodeState.age must be non-negative, got {self.age!r}."
            )
