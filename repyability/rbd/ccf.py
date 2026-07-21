"""Common-cause failure (CCF) models for RBD nodes.

Redundant components frequently fail from a *shared* root cause — a common
manufacturing defect, a shared power supply, one maintenance error applied to
every unit — so their failures are correlated rather than independent. The
exact RBD engine assumes independence, and therefore over-estimates redundant
systems; a CCF model injects the shared-cause coupling.

Two models are provided, both consumed by ``NonRepairableRBD``'s ``ccf_groups``
argument via a :class:`CCFGroup`:

* :class:`BetaFactor` — a fraction ``beta`` of a component's failures come from
  a cause shared across the *whole* group (all members fail together), the rest
  are independent. The workhorse of probabilistic-risk assessment.
* :class:`MGL` — the Multiple Greek Letter model, which additionally captures
  *partial* common causes (a cause failing some but not all of the group) via a
  cascade of conditional probabilities ``beta, gamma, delta, ...``. Beta-factor
  is the two-unit special case.

Both expose :meth:`decompose`, which splits a group's failure probability into
an independent part plus a set of mutually-exclusive *shock* outcomes (each a
subset of members failing together); ``NonRepairableRBD`` evaluates the exact
system reliability by conditioning on those outcomes. See issue #44.

Alpha-factor (a data-estimable reparameterisation of the same multiplicities)
is a planned extension.
"""

from itertools import combinations
from math import comb
from typing import Any, Collection, Hashable, List, Tuple

import numpy as np

# A group's failure decomposition: the per-component independent failure
# probability, and a list of (members-failing-together, probability) shocks.
Decomposition = Tuple[np.ndarray, List[Tuple[frozenset, np.ndarray]]]


class BetaFactor:
    """The beta-factor common-cause model (all-or-nothing).

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

    def required_group_size(self) -> Any:
        return None  # any group of >= 2 members

    def decompose(self, members: Collection[Hashable], Q) -> Decomposition:
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        q_independent = (1.0 - self.beta) * Q
        shocks = [(frozenset(members), self.beta * Q)]
        return q_independent, shocks

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BetaFactor) and other.beta == self.beta

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.beta))

    def __repr__(self) -> str:
        return f"BetaFactor(beta={self.beta})"


class MGL:
    """The Multiple Greek Letter common-cause model.

    The parameters are the conditional probabilities of a common-cause failure
    escalating to the next level: ``beta = P(shared by >= 2 | failed)``,
    ``gamma = P(>= 3 | >= 2)``, ``delta = P(>= 4 | >= 3)``, and so on. The
    number of parameters fixes the group size: ``n`` letters describe a group
    of ``n + 1`` members. ``MGL(beta)`` is exactly :class:`BetaFactor` on a
    two-member group.

    The probability that a common cause fails a *specific* set of ``k`` of the
    ``m`` members is the standard MGL basic-event probability

    ``Q_k = [1 / C(m-1, k-1)] * (rho_1 * ... * rho_k) * (1 - rho_{k+1}) * Q``

    with ``rho_1 = 1``, ``rho_2 = beta``, ``rho_3 = gamma``, ..., and
    ``rho_{m+1} = 0``. These partition each component's total failure
    probability ``Q`` exactly.

    Parameters
    ----------
    *letters : float
        ``beta, gamma, delta, ...``, each in ``[0, 1]``; at least one. A group
        of ``m`` members needs ``m - 1`` letters.
    """

    def __init__(self, *letters: float):
        if len(letters) < 1:
            raise ValueError("MGL needs at least one parameter (beta).")
        for value in letters:
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"MGL parameters must be in [0, 1], got {value!r}."
                )
        self.letters = tuple(float(v) for v in letters)

    @property
    def group_size(self) -> int:
        return len(self.letters) + 1

    def required_group_size(self) -> int:
        return self.group_size

    def _specific_set_prob(self, m: int, k: int, Q: np.ndarray) -> np.ndarray:
        # rho[0..m-1] represents rho_1..rho_m (rho_1 = 1, then the letters).
        rho = [1.0] + list(self.letters)
        prod = 1.0
        for i in range(k):
            prod *= rho[i]
        rho_next = rho[k] if k < m else 0.0
        return (prod * (1.0 - rho_next) / comb(m - 1, k - 1)) * Q

    def decompose(self, members: Collection[Hashable], Q) -> Decomposition:
        members = tuple(members)
        m = len(members)
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        q_independent = self._specific_set_prob(m, 1, Q)
        shocks: List[Tuple[frozenset, np.ndarray]] = []
        for k in range(2, m + 1):
            q_k = self._specific_set_prob(m, k, Q)
            for subset in combinations(members, k):
                shocks.append((frozenset(subset), q_k))
        return q_independent, shocks

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MGL) and other.letters == self.letters

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.letters))

    def __repr__(self) -> str:
        return f"MGL{self.letters}"


class CCFGroup:
    """A common-cause group: member nodes coupled by a shared failure cause.

    Parameters
    ----------
    members : collection of node names
        The RBD nodes that share the common cause (at least two, distinct).
        Standard CCF theory is for *symmetric* groups, so the members should
        carry identical component models.
    model : BetaFactor or MGL
        The common-cause model coupling the members. An :class:`MGL` model
        fixes the group size (``m - 1`` letters for ``m`` members).
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
        if not isinstance(model, (BetaFactor, MGL)):
            raise ValueError(
                "CCFGroup model must be a BetaFactor or MGL; alpha-factor is "
                "not supported yet."
            )
        required = model.required_group_size()
        if required is not None and required != len(members):
            raise ValueError(
                f"{type(model).__name__} describes a group of {required} "
                f"members, but this group has {len(members)}."
            )
        self.members = members
        self.model = model

    def __repr__(self) -> str:
        return f"CCFGroup(members={list(self.members)}, model={self.model!r})"
