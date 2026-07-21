"""Common-cause failure (CCF) models for RBD nodes.

Redundant components frequently fail from a *shared* root cause — a common
manufacturing defect, a shared power supply, one maintenance error applied to
every unit — so their failures are correlated rather than independent. The
exact RBD engine assumes independence, and therefore over-estimates redundant
systems; a CCF model injects the shared-cause coupling.

This module provides the **beta-factor** model, the workhorse of
probabilistic-risk assessment: a fraction ``beta`` of a component's failures
come from a cause shared across the whole group (failing every member at once),
and the remaining ``1 - beta`` are independent. A :class:`CCFGroup` binds a set
of member nodes to such a model; pass groups to a ``NonRepairableRBD`` via its
``ccf_groups`` argument.

See issue #44. Partial common-cause models (alpha-factor, Multiple Greek
Letter — where a cause fails *some but not all* of a group) build on the same
conditioning machinery and are a planned extension.
"""

from typing import Any, Collection, Hashable


class BetaFactor:
    """The beta-factor common-cause model.

    A fraction ``beta`` of each component's total failure probability is
    attributed to a cause shared across the whole group (which fails every
    member simultaneously); the remaining ``1 - beta`` is independent.

    Parameters
    ----------
    beta : float
        The common-cause fraction, in ``[0, 1]``. ``beta = 0`` is ordinary
        independence; ``beta = 1`` makes the group fail entirely in unison.
    """

    def __init__(self, beta: float):
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta!r}.")
        self.beta = float(beta)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BetaFactor) and other.beta == self.beta

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.beta))

    def __repr__(self) -> str:
        return f"BetaFactor(beta={self.beta})"


class CCFGroup:
    """A common-cause group: member nodes coupled by a shared failure cause.

    Parameters
    ----------
    members : collection of node names
        The RBD nodes that share the common cause (at least two, distinct).
        Standard CCF theory is for *symmetric* groups, so the members should
        carry identical component models.
    model : BetaFactor
        The common-cause model coupling the members. (Alpha-factor / Multiple
        Greek Letter are a planned extension.)
    """

    def __init__(self, members: Collection[Hashable], model: Any):
        members = tuple(members)
        if len(members) < 2:
            raise ValueError(
                f"A CCF group needs at least 2 members, got {len(members)}."
            )
        if len(set(members)) != len(members):
            raise ValueError(
                f"CCF group members must be distinct, got {list(members)}."
            )
        if not isinstance(model, BetaFactor):
            raise ValueError(
                "CCFGroup model must be a BetaFactor; alpha-factor and "
                "Multiple Greek Letter models are not supported yet."
            )
        self.members = members
        self.model = model

    def __repr__(self) -> str:
        return f"CCFGroup(members={list(self.members)}, model={self.model!r})"
