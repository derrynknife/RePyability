"""Common-cause failure (CCF) models for RBD nodes.

Redundant components frequently fail from a *shared* root cause — a common
manufacturing defect, a shared power supply, one maintenance error applied to
every unit — so their failures are correlated rather than independent. The
exact RBD engine assumes independence, and therefore over-estimates redundant
systems; a CCF model injects the shared-cause coupling.

Two models are provided, both consumed by ``NonRepairableRBD``'s ``ccf_groups``
argument via a [`CCFGroup`][repyability.CCFGroup]:

* [`BetaFactor`][repyability.BetaFactor] — a fraction ``beta`` of a
  component's failures come from a cause shared across the *whole* group (all
  members fail together), the rest are independent. The workhorse of
  probabilistic-risk assessment.
* [`MGL`][repyability.MGL] — the Multiple Greek Letter model, which
  additionally captures *partial* common causes (a cause failing some but not
  all of the group) via a cascade of conditional probabilities
  ``beta, gamma, delta, ...``. Beta-factor is the two-unit special case.

Both expose ``decompose``, which splits a group's failure probability into
an independent part plus a set of mutually-exclusive *shock* outcomes (each a
subset of members failing together); ``NonRepairableRBD`` evaluates the exact
system reliability by conditioning on those outcomes. See issue #44.

These are the PRA basic-event models, which split each member's failure
*probability* ``Q``; like their textbook form they assume ``Q`` is small (a
mission or proof-test interval, not a whole life). As ``Q`` grows they drift
from a rate-based treatment, and from about ``Q = 0.5`` a redundant group can
come out more reliable than an independent one.

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
    It applies to a group of any size. Use it through a
    [`CCFGroup`][repyability.CCFGroup]; models with equal ``beta``
    compare equal.

    Parameters
    ----------
    beta : float
        The common-cause fraction, in ``[0, 1]``. ``beta = 0`` is ordinary
        independence; ``beta = 1`` makes the group fail entirely in unison.

    Raises
    ------
    ValueError
        If ``beta`` is outside ``[0, 1]``.

    Examples
    --------
    With a failure probability ``Q = 0.2`` per member and ``beta = 0.1``,
    each member fails independently with ``0.9 * 0.2`` and the shared
    cause fails both with ``0.1 * 0.2``:

    >>> from repyability import BetaFactor
    >>> q_independent, shocks = BetaFactor(0.1).decompose(["a", "b"], 0.2)
    >>> round(float(q_independent[0]), 4)
    0.18
    >>> [(sorted(members), round(float(q[0]), 4)) for members, q in shocks]
    [(['a', 'b'], 0.02)]
    """

    def __init__(self, beta: float):
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta!r}.")
        self.beta = float(beta)

    def required_group_size(self) -> Any:
        """The group size this model requires: None, meaning any size.

        [`CCFGroup`][repyability.CCFGroup] checks a model's required size
        against its members; a beta-factor model fits a group of any size
        (two or more members).

        Returns
        -------
        None
            Always None.
        """
        return None  # any group of >= 2 members

    def decompose(self, members: Collection[Hashable], Q) -> Decomposition:
        """Split the failure probability into independent and shared parts.

        Each member's failure probability ``Q`` splits into an independent
        part and one whole-group shock. The RBD calls this at each
        evaluation time with ``Q = 1 - sf(t)`` of the group's first member.

        Parameters
        ----------
        members : collection of hashable
            The group's member node names.
        Q : float or array_like
            Each member's total failure probability (the same for every
            member of a symmetric group), at one or more times.

        Returns
        -------
        q_independent : numpy.ndarray
            Each member's independent failure probability,
            ``(1 - beta) * Q``, at least 1-d.
        shocks : list of (frozenset, numpy.ndarray)
            One shock, ``(frozenset(members), beta * Q)``: the probability
            that the shared cause fails every member.
        """
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
    of ``n + 1`` members. ``MGL(beta)`` is exactly
    [`BetaFactor`][repyability.BetaFactor] on a two-member group.

    The probability that a common cause fails a *specific* set of ``k`` of the
    ``m`` members is the standard MGL basic-event probability

    ``Q_k = [1 / C(m-1, k-1)] * (rho_1 * ... * rho_k) * (1 - rho_{k+1}) * Q``

    with ``rho_1 = 1``, ``rho_2 = beta``, ``rho_3 = gamma``, ..., and
    ``rho_{m+1} = 0``. These partition each component's total failure
    probability ``Q`` exactly.

    Use it through a [`CCFGroup`][repyability.CCFGroup]; models with equal
    letters compare equal.

    Parameters
    ----------
    *letters : float
        ``beta, gamma, delta, ...``, each in ``[0, 1]``; at least one. A group
        of ``m`` members needs ``m - 1`` letters.

    Raises
    ------
    ValueError
        If no letters are given, or any is outside ``[0, 1]``.

    Examples
    --------
    A three-member group with ``beta = 0.1`` and ``gamma = 0.3``, each
    member failing with probability ``Q = 0.1``:

    >>> from repyability import MGL
    >>> mgl = MGL(0.1, 0.3)
    >>> mgl.group_size
    3
    >>> q_independent, shocks = mgl.decompose(["a", "b", "c"], 0.1)
    >>> round(float(q_independent[0]), 4)  # (1 - beta) * Q
    0.09
    >>> for members, q in shocks:
    ...     print(sorted(members), round(float(q[0]), 4))
    ['a', 'b'] 0.0035
    ['a', 'c'] 0.0035
    ['b', 'c'] 0.0035
    ['a', 'b', 'c'] 0.003
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
        """The number of members this model describes, ``len(letters) + 1``."""
        return len(self.letters) + 1

    def required_group_size(self) -> int:
        """The group size this model requires, ``group_size``.

        [`CCFGroup`][repyability.CCFGroup] rejects a group whose number of
        members differs.

        Returns
        -------
        int
            ``len(letters) + 1``.
        """
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
        """Split the failure probability into independent and shared parts.

        With ``m = len(members)``, each member fails alone with
        ``Q_1 = (1 - beta) * Q``, and each specific set of ``k >= 2``
        members fails together with the basic-event probability ``Q_k``
        given in the class description. The RBD calls this at each
        evaluation time with ``Q = 1 - sf(t)`` of the group's first member.

        Parameters
        ----------
        members : collection of hashable
            The group's member node names: ``group_size`` of them, as
            [`CCFGroup`][repyability.CCFGroup] ensures.
        Q : float or array_like
            Each member's total failure probability (the same for every
            member of a symmetric group), at one or more times.

        Returns
        -------
        q_independent : numpy.ndarray
            ``Q_1``, each member's independent failure probability, at
            least 1-d.
        shocks : list of (frozenset, numpy.ndarray)
            ``(subset, Q_k)`` for every subset of ``k = 2, ..., m`` members,
            smaller subsets first.
        """
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

    Pass groups to [`NonRepairableRBD`][repyability.NonRepairableRBD] as
    ``ccf_groups=[...]``. At each time the RBD takes the members' failure
    probability ``Q = 1 - sf(t)``, splits it with the model's
    ``decompose`` into independent failures and mutually exclusive shocks,
    and computes the exact system reliability by conditioning on every
    group's shock outcome. The groups enter the RBD's ``sf`` and ``ff``
    and the methods built on them (``sf`` raises ``NotImplementedError``
    if a member is forced through ``working_nodes``/``broken_nodes``).
    The RBD's Monte-Carlo ``random`` and ``mean`` ignore them, and its
    probability-based importance measures, parameter sensitivity,
    redundancy allocation and condition-based methods raise
    ``NotImplementedError`` when groups are present.

    Parameters
    ----------
    members : collection of node names
        The RBD nodes that share the common cause (at least two, distinct).
        Standard CCF theory is for *symmetric* groups, so the members must
        carry identical component models.
    model : BetaFactor or MGL
        The common-cause model coupling the members. An
        [`MGL`][repyability.MGL] model fixes the group size (``m - 1``
        letters for ``m`` members).

    Raises
    ------
    ValueError
        If there are fewer than two members, a member is listed twice,
        ``model`` is not a ``BetaFactor`` or ``MGL``, or an ``MGL`` model's
        group size differs from the number of members.

    Notes
    -----
    The RBD validates the groups when it is built. It rejects a member that
    is not a component node of the RBD (an unknown node, the input or
    output node, or a repeated node), a node in more than one group, and a
    group whose members carry different models (compared through their
    serialised form; members that cannot be serialised are not compared).

    Examples
    --------
    Two filters in parallel, each failing with probability 0.1, where 10%
    of their failures come from a shared cause:

    >>> from surpyval import FixedEventProbability
    >>> from repyability import BetaFactor, CCFGroup, NonRepairableRBD
    >>> edges = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
    >>> nodes = {n: FixedEventProbability.from_params(0.1) for n in "ab"}
    >>> round(NonRepairableRBD(edges, nodes).sf(), 4)  # independent
    0.99
    >>> group = CCFGroup(["a", "b"], BetaFactor(0.1))
    >>> rbd = NonRepairableRBD(edges, nodes, ccf_groups=[group])
    >>> round(rbd.sf(), 4)  # (1 - 0.01) * (1 - 0.09 ** 2)
    0.982
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
