"""The per-node condition input for condition-based ("digital twin")
reliability evaluation.

An RBD's *structure* is static, but in a condition-based setting each component
has a *current state* streamed from sensors -- how much life it has already
accumulated, and whether it is still working. :class:`NodeState` carries that
per-node state into
:meth:`~repyability.NonRepairableRBD.sf_given_state`,
:meth:`~repyability.NonRepairableRBD.remaining_life` and
:meth:`~repyability.NonRepairableRBD.importances_given_state`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeState:
    """The current condition of a single RBD node.

    A component's forward reliability is conditioned on the life it has already
    survived: ``R_i(x | age) = R_i(age + x) / R_i(age)``. A failed component
    (``alive=False``) contributes zero reliability regardless of its age.

    Parameters
    ----------
    age : float, optional
        The component's current age / accumulated exposure ``X_i`` (``>= 0``),
        by default ``0.0`` (a brand-new component, equivalent to no
        conditioning).
    alive : bool, optional
        Whether the component is currently working, by default ``True``.

    Notes
    -----
    Only lifetime (time-varying) distributions age; a fixed-probability
    component's reliability does not depend on ``age``. Load/stress is not part
    of the state in this release -- it is a planned extension (accelerated-life
    exposure and load sharing), at which point this class gains a further
    field.
    """

    age: float = 0.0
    alive: bool = True

    def __post_init__(self) -> None:
        if self.age < 0:
            raise ValueError(
                f"NodeState.age must be non-negative, got {self.age!r}."
            )
