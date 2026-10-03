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

By default these are the PRA basic-event models, which split each member's
failure *probability* ``Q``; like their textbook form they assume ``Q`` is
small (a mission or proof-test interval, not a whole life). As ``Q`` grows
they drift from a rate-based treatment, and from about ``Q = 0.5`` a
redundant group can come out more reliable than an independent one, so the
RBD warns when a member's ``Q`` passes ``VALIDITY`` (#132). With
``basis="rate"`` they split each member's failure *rate* instead: every
cause is a shock whose cumulative hazard is a fixed fraction of the
members' own, ``H(t) = -log R(t)``. Each member then keeps its own life
distribution, whatever it is, and the model holds over the whole life; it
agrees with the probability split to first order in ``Q``.

Alpha-factor (a data-estimable reparameterisation of the same multiplicities)
is a planned extension.
"""

from itertools import combinations
from math import comb
from typing import Any, Collection, Dict, Hashable, List, Optional, Tuple

import numpy as np

# A group's failure decomposition: the per-component independent failure
# probability, and a list of (members-failing-together, probability) shocks.
Decomposition = Tuple[np.ndarray, List[Tuple[frozenset, np.ndarray]]]

#: How a model splits a member's failure: its probability, or its rate.
BASES = ("probability", "rate")

#: The largest probability of failing at which the probability split is
#: used without a warning: there it is within about 3.5% of the rate-based
#: split (for a parallel pair with ``beta = 0.3``), and it drifts further
#: as ``Q`` grows.
VALIDITY = 0.1


def _check_basis(basis: str) -> str:
    if basis not in BASES:
        raise ValueError(
            f"basis must be 'probability' or 'rate', got {basis!r}."
        )
    return basis


def _hazard(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """The cumulative hazard ``-log R``, from whichever of the probability
    of failing ``Q`` and of surviving ``R`` keeps its precision: a small
    ``Q``'s through ``log1p``, a small ``R``'s directly."""
    with np.errstate(divide="ignore"):
        return np.where(Q <= 0.5, -np.log1p(-Q), -np.log(R))


def _rate_outcomes(
    members: tuple,
    independent: float,
    causes: List[Tuple[frozenset, float]],
    H: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[frozenset, np.ndarray]]]:
    """A rate-based split at cumulative hazards ``H``: each member's
    probability of failing on its own and of not doing so, and the
    probability of each set of members the shared causes fail together
    (see ``_union_outcomes``). Each cause fires by then with probability
    ``1 - exp(-c H)``, independently of the others, ``c`` its fraction of
    the members' hazard."""
    if independent > 0.0:
        r_independent = np.exp(-independent * H)
        q_independent = -np.expm1(-independent * H)
    else:
        r_independent, q_independent = np.ones_like(H), np.zeros_like(H)
    fired = [
        (struck, -np.expm1(-c * H), np.exp(-c * H))
        for struck, c in causes
        if c > 0.0
    ]
    return _union_outcomes(members, q_independent, r_independent, fired)


def _event_outcomes(
    members: tuple,
    independent: float,
    causes: List[Tuple[frozenset, float]],
    Q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[frozenset, np.ndarray]]]:
    """A probability split whose shocks are independent basic events, as
    PRA codes take them (#180): each member fails on its own with
    probability ``independent * Q``, and each shared cause, a fraction
    ``c`` of ``Q``, fires with probability ``c * Q``, independently of the
    others (see ``_union_outcomes``)."""
    fired = [(struck, c * Q, 1.0 - c * Q) for struck, c in causes if c > 0.0]
    return _union_outcomes(
        members, independent * Q, 1.0 - independent * Q, fired
    )


def _union_outcomes(
    members: tuple,
    q_independent: np.ndarray,
    r_independent: np.ndarray,
    fired: List[Tuple[frozenset, np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[frozenset, np.ndarray]]]:
    """Each member's probability of failing on its own and of not doing
    so, and the probability of each set of members the shared causes fail
    together (mutually exclusive outcomes, the union of the causes that
    struck, smaller sets first, then in the members' order; none for the
    empty set), from each cause's probability of firing and of not, as
    ``(members it fails, fires, holds)``, independently of the others.
    Summed outcome by outcome, without differences, so a small
    probability keeps its precision."""
    zero = np.zeros_like(q_independent)
    unions: Dict[frozenset, np.ndarray] = {
        frozenset(): np.ones_like(q_independent)
    }
    for struck, fires, holds in fired:
        after: Dict[frozenset, np.ndarray] = {}
        for union, p in unions.items():
            after[union] = after.get(union, zero) + p * holds
            joined = union | struck
            after[joined] = after.get(joined, zero) + p * fires
        unions = after
    del unions[frozenset()]
    order = {member: i for i, member in enumerate(members)}
    shocks = sorted(
        unions.items(),
        key=lambda item: (len(item[0]), sorted(order[m] for m in item[0])),
    )
    return q_independent, r_independent, shocks


class _Model:
    """What the common-cause models share."""

    basis: str

    def _causes(self, members) -> Tuple[float, list]:
        raise NotImplementedError

    @property
    def shocks(self) -> str:
        """How the shared causes combine: ``"independent"`` events (by
        rate, always), or ``"exclusive"`` ones (the probability split's
        default; see ``MGL``)."""
        return "independent" if self.basis == "rate" else "exclusive"

    def _no_shock(self, members, Q, R) -> np.ndarray:
        """With independent causes: the probability that no shared cause
        has struck, at the members' probabilities of failing ``Q`` and
        surviving ``R``, as a product (never one less the shocks')."""
        _, causes = self._causes(members)
        if self.basis != "rate":
            out = np.ones_like(Q)
            for _, c in causes:
                out = out * (1.0 - c * Q)
            return out
        total = sum(c for _, c in causes)
        if total <= 0.0:
            return np.ones_like(Q)
        return np.exp(-total * _hazard(Q, R))


class BetaFactor(_Model):
    """The beta-factor common-cause model (all-or-nothing).

    A fraction ``beta`` of each component's failures come from a cause
    shared across the whole group (which fails every member
    simultaneously); the rest are independent. It applies to a group of
    any size. Use it through a [`CCFGroup`][repyability.CCFGroup]; models
    with equal ``beta`` and ``basis`` compare equal.

    ``basis`` says what ``beta`` is a fraction of. By default,
    ``"probability"``, of each member's probability of failing ``Q``: the
    shared cause fails the group with ``beta * Q``, and each member fails
    on its own with ``(1 - beta) * Q``. That is the PRA basic-event model,
    for a small ``Q``, as over a mission or a proof-test interval; over a
    whole life it stops describing a lifetime (each member would only ever
    fail with ``1 - beta * (1 - beta)``), and the RBD warns once a member's
    ``Q`` passes 0.1. With ``"rate"``, of each member's failure rate: the
    shared cause is a shock that has not struck by ``t`` with probability
    ``R(t) ** beta``, and each member survives its own causes with
    ``R(t) ** (1 - beta)``, ``R`` the members' reliability. Every member
    keeps its own life distribution, the model holds over the whole life,
    and the RBD's MTTF and simulations include the group. The two agree to
    first order in ``Q``.

    Parameters
    ----------
    beta : float
        The common-cause fraction, in ``[0, 1]``. ``beta = 0`` is ordinary
        independence; ``beta = 1`` makes the group fail entirely in unison.
    basis : {"probability", "rate"}, optional
        What ``beta`` is a fraction of, by default ``"probability"``.

    Raises
    ------
    ValueError
        If ``beta`` is outside ``[0, 1]``, or ``basis`` is neither
        ``"probability"`` nor ``"rate"``.

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

    Split by rate, the shock strikes with ``1 - 0.8 ** 0.1`` and each
    member fails on its own with ``1 - 0.8 ** 0.9``: close to the above at
    this ``Q``, and a lifetime model at any:

    >>> rate = BetaFactor(0.1, basis="rate")
    >>> q_independent, shocks = rate.decompose(["a", "b"], 0.2)
    >>> round(float(q_independent[0]), 4)
    0.1819
    >>> [(sorted(members), round(float(q[0]), 4)) for members, q in shocks]
    [(['a', 'b'], 0.0221)]
    """

    def __init__(self, beta: float, basis: str = "probability"):
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta!r}.")
        self.beta = float(beta)
        self.basis = _check_basis(basis)

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
        evaluation time with ``Q``, the probability that the group's first
        member has failed by then.

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
            Each member's independent failure probability, at least 1-d:
            ``(1 - beta) * Q``, or by rate ``1 - (1 - Q) ** (1 - beta)``.
        shocks : list of (frozenset, numpy.ndarray)
            One shock, ``frozenset(members)`` with the probability that the
            shared cause fails every member: ``beta * Q``, or by rate
            ``1 - (1 - Q) ** beta``.
        """
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        q_independent, _, shocks = self._split(members, Q, 1.0 - Q)
        return q_independent, shocks

    def _split(self, members, Q, R):
        """Each member's probability of failing on its own and of not
        doing so, and the shocks (see ``decompose``), at the members'
        probabilities of failing ``Q`` and of surviving ``R`` (both 1-d
        arrays: by rate, whichever keeps its precision is used)."""
        if self.basis == "rate":
            return _rate_outcomes(
                tuple(members),
                1.0 - self.beta,
                [(frozenset(members), self.beta)],
                _hazard(Q, R),
            )
        q_independent = (1.0 - self.beta) * Q
        shocks = [(frozenset(members), self.beta * Q)]
        return q_independent, 1.0 - q_independent, shocks

    def _causes(self, members) -> Tuple[float, list]:
        """By rate: each member's own causes' fraction of its hazard, and
        each shared cause, as the members it fails and its fraction."""
        return 1.0 - self.beta, [(frozenset(members), self.beta)]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BetaFactor)
            and other.beta == self.beta
            and other.basis == self.basis
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.beta, self.basis))

    def __repr__(self) -> str:
        basis = ", basis='rate'" if self.basis == "rate" else ""
        return f"BetaFactor(beta={self.beta}{basis})"


class MGL(_Model):
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

    That is the default ``basis``, ``"probability"``: the PRA basic-event
    model, for a small ``Q`` (see [`BetaFactor`][repyability.BetaFactor]).
    With ``basis="rate"`` the same fractions split each member's failure
    rate instead: each specific set of ``k`` members has a cause of its
    own, a shock that has not struck by ``t`` with probability
    ``R(t) ** (Q_k / Q)``, independently of the others, and the members it
    strikes fail together. Every member keeps its own life distribution,
    and the model holds over the whole life.

    **How the shocks combine, splitting the probability.** By default
    (``shocks="exclusive"``) the outcomes are mutually exclusive: the
    group is struck by one shared cause at most, set ``S`` with
    probability ``Q_k``, so each member fails with probability ``Q``
    exactly. PRA codes (SAPHIRE, CAFTA, RiskSpectrum) instead take each
    ``Q_k`` as a basic event of its own, independent of the others, so
    that several may strike: ``shocks="independent"``. The two differ at
    second order in ``Q`` (a 2-out-of-3 group of ``MGL(0.2, 0.3)`` at
    ``Q = 0.031`` fails with probability 0.010219 one way and 0.010192
    the other), so match the tool you check against (#180). A
    ``BetaFactor``, or an ``MGL`` model with one shared cause, has one
    shock, and both agree. By rate, the causes strike independently.

    Use it through a [`CCFGroup`][repyability.CCFGroup]; models with equal
    letters, ``basis`` and ``shocks`` compare equal.

    Parameters
    ----------
    *letters : float
        ``beta, gamma, delta, ...``, each in ``[0, 1]``; at least one. A group
        of ``m`` members needs ``m - 1`` letters.
    basis : {"probability", "rate"}, optional
        What the letters split, by default ``"probability"`` (keyword
        only).
    shocks : {"exclusive", "independent"}, optional
        How the shared causes combine when the letters split the
        probability (keyword only): by default ``"exclusive"``, one at
        most; ``"independent"``, as independent basic events. By rate
        they are always independent.

    Raises
    ------
    ValueError
        If no letters are given, any is outside ``[0, 1]``, ``basis`` is
        neither ``"probability"`` nor ``"rate"``, or ``shocks`` is
        neither ``"exclusive"`` nor ``"independent"`` (``"exclusive"`` by
        rate).

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

    def __init__(
        self,
        *letters: float,
        basis: str = "probability",
        shocks: Optional[str] = None,
    ):
        if len(letters) < 1:
            raise ValueError("MGL needs at least one parameter (beta).")
        for value in letters:
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"MGL parameters must be in [0, 1], got {value!r}."
                )
        self.letters = tuple(float(v) for v in letters)
        self.basis = _check_basis(basis)
        if shocks is None:
            shocks = "independent" if self.basis == "rate" else "exclusive"
        if shocks not in ("exclusive", "independent"):
            raise ValueError(
                "shocks must be 'exclusive' or 'independent', got "
                f"{shocks!r}."
            )
        if self.basis == "rate" and shocks != "independent":
            raise ValueError(
                "By rate, every cause strikes independently: shocks="
                "'exclusive' splits the probability (basis='probability')."
            )
        self._shocks = shocks

    @property
    def shocks(self) -> str:
        """How the shared causes combine: ``"exclusive"`` (one at most,
        the probability split's default) or ``"independent"`` (as
        independent basic events; by rate, always)."""
        return self._shocks

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
        evaluation time with ``Q``, the probability that the group's first
        member has failed by then.

        By rate, or with ``shocks="independent"``, the shocks are the sets
        of members the causes that have struck fail between them: several
        causes may strike, and a member fails if any of its causes has, so
        these are the distinct unions, mutually exclusive as the RBD needs,
        each with its probability.

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
            least 1-d (by rate, ``1 - (1 - Q) ** (1 - beta)``).
        shocks : list of (frozenset, numpy.ndarray)
            ``(subset, Q_k)`` for every subset of ``k = 2, ..., m`` members,
            smaller subsets first; by rate, ``(subset, probability)`` for
            every set the struck causes can fail together.
        """
        Q = np.atleast_1d(np.asarray(Q, dtype=float))
        q_independent, _, shocks = self._split(members, Q, 1.0 - Q)
        return q_independent, shocks

    def _split(self, members, Q, R):
        """Each member's probability of failing on its own and of not
        doing so, and the shocks (see ``decompose``), at the members'
        probabilities of failing ``Q`` and of surviving ``R`` (both 1-d
        arrays: by rate, whichever keeps its precision is used)."""
        members = tuple(members)
        if self.basis == "rate":
            independent, causes = self._causes(members)
            return _rate_outcomes(members, independent, causes, _hazard(Q, R))
        if self.shocks == "independent":
            independent, causes = self._causes(members)
            return _event_outcomes(members, independent, causes, Q)
        m = len(members)
        q_independent = self._specific_set_prob(m, 1, Q)
        shocks: List[Tuple[frozenset, np.ndarray]] = []
        for k in range(2, m + 1):
            q_k = self._specific_set_prob(m, k, Q)
            for subset in combinations(members, k):
                shocks.append((frozenset(subset), q_k))
        return q_independent, 1.0 - q_independent, shocks

    def _causes(self, members) -> Tuple[float, list]:
        """By rate: each member's own causes' fraction of its hazard, and
        each shared cause, as the members it fails and its fraction."""
        members = tuple(members)
        m = len(members)
        one = np.ones(1)
        causes = [
            (frozenset(subset), float(self._specific_set_prob(m, k, one)[0]))
            for k in range(2, m + 1)
            for subset in combinations(members, k)
        ]
        return float(self._specific_set_prob(m, 1, one)[0]), causes

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MGL)
            and other.letters == self.letters
            and other.basis == self.basis
            and other.shocks == self.shocks
        )

    def __hash__(self) -> int:
        return hash(
            (type(self).__name__, self.letters, self.basis, self.shocks)
        )

    def __repr__(self) -> str:
        parts = [repr(letter) for letter in self.letters]
        if self.basis == "rate":
            parts.append("basis='rate'")
        elif self.shocks == "independent":
            parts.append("shocks='independent'")
        return f"MGL({', '.join(parts)})"


#: The names of an MGL model's letters, in order: ``beta`` first.
LETTERS = (
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
)


def parameters(model) -> Dict[str, float]:
    """A common-cause model's parameters by name: a ``BetaFactor``'s
    ``beta``, an ``MGL`` model's letters (``beta``, ``gamma``, ``delta``,
    ...)."""
    if isinstance(model, BetaFactor):
        return {"beta": model.beta}
    return {
        LETTERS[j] if j < len(LETTERS) else f"letter{j + 1}": value
        for j, value in enumerate(model.letters)
    }


def with_parameters(model, values: Dict[str, float]) -> "_Model":
    """``model`` with the parameters ``values`` names changed (see
    ``parameters``), its basis kept: a ``ValueError`` for one outside
    ``[0, 1]`` or a name it does not have."""
    current = parameters(model)
    unknown = [name for name in values if name not in current]
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are not parameters of {model!r}, whose "
            f"parameters are {list(current)}."
        )
    current.update({name: float(v) for name, v in values.items()})
    if isinstance(model, BetaFactor):
        return BetaFactor(current["beta"], basis=model.basis)
    return MGL(*current.values(), basis=model.basis, shocks=model.shocks)


def validity_warning(group: "CCFGroup", Q: float) -> None:
    """Warn that ``group``'s probability split is used at a probability of
    failing ``Q`` beyond ``VALIDITY``, and name the rate-based model to
    use over a lifetime."""
    import sys
    import warnings

    model = group.model
    rate: _Model = (
        BetaFactor(model.beta, basis="rate")
        if isinstance(model, BetaFactor)
        else MGL(*model.letters, basis="rate")
    )
    # Point at the first caller outside the package.
    level, frame = 2, sys._getframe(1)
    while frame.f_back is not None:
        name = frame.f_globals.get("__name__", "")
        if not name.startswith("repyability.") or name.startswith(
            "repyability.tests"
        ):
            break
        frame, level = frame.f_back, level + 1
    warnings.warn(
        f"Common-cause group {list(group.members)} ({model!r}): its "
        f"members' probability of failing reaches Q = {Q:.3g} at the times "
        f"evaluated, beyond the {VALIDITY} the probability split is meant "
        "for. It is a rare-event model: about 3.5% off a rate-based split "
        "at Q = 0.1 and further beyond, and from about Q = 0.5 it makes a "
        "redundant group more reliable than an independent one. Over a "
        f"lifetime, split the failure rate: {rate!r}.",
        UserWarning,
        stacklevel=level,
    )


class CCFGroup:
    """A common-cause group: member nodes coupled by a shared failure cause.

    Pass groups to [`NonRepairableRBD`][repyability.NonRepairableRBD] as
    ``ccf_groups=[...]``. At each time the RBD takes the members' failure
    probability ``Q`` (``1 - sf(t)``, or for the RBD's ``ff`` their model's
    own ``ff(t)``, which keeps a small one's precision), splits it with the
    model's ``decompose`` into independent failures and mutually exclusive
    shocks,
    and computes the exact system reliability by conditioning on every
    group's shock outcome. The groups enter the RBD's ``sf`` and ``ff``
    and the methods built on them (``sf`` raises ``NotImplementedError``
    if a member is forced through ``working_nodes``/``broken_nodes``).
    A model that splits the failure *probability* (the default) is meant
    for a small ``Q``: the RBD warns, once per group, when a member's
    ``Q`` passes ``VALIDITY`` (0.1), and its exact ``mean`` refuses the
    group, while its Monte-Carlo ``random`` and ``mean(method="simulate")``
    leave the group out. A model that splits the failure *rate*
    (``basis="rate"``) holds over the whole life: the exact ``mean``
    includes the group, and the simulations draw its shared shocks. The
    RBD's importance measures condition a member on its state through the
    shock outcomes, its parameter sensitivity and uncertainty take a
    group's members together (with the model's own parameters), and its
    redundancy allocation lets a ``BetaFactor`` member's copies join the
    group; its condition-based methods raise ``NotImplementedError``.

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
