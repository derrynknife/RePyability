"""Common-cause groups worked out module by module, or as shock events
(#219).

Given the outcome of a common-cause group's shocks, its members are
independent, of each other and of every other node (see ``ccf.py``). So a
group need only be conditioned on within the smallest module of the
decomposition (see ``modular.py``) that holds its members, its *owner*:
the sum over the group's outcomes of the owner's probabilities given each
is the owner's probability, and everything above it sees the owner as one
independent node. A group whose members only the core joins is owned by
the system. Groups in separate modules then cost a sum each, where
conditioning on every combination of every group's outcomes doubled the
time per group.

Groups owned by one module are conditioned on together, which multiplies
their outcomes: redundant trains, with a group of each kind of component
across them, would double the time per group again. Where a module's
groups have more than ``COMBINATIONS`` outcomes between them, they are
written out instead (see ``Plan``): each shared cause is an event of its
own, a ``Shock``, which every member it strikes needs not to have fired,
and the structure is worked out again with the shocks as repeated nodes,
which its decision diagram takes in its stride. Causes that exclude each
other (the multiple Greek letter model's by default) are taken as
independent ones that fail the same sets of members as often, where
there are such (see ``ccf._as_independent``); where there are not, at
the probabilities evaluated, the groups are conditioned on instead.

``Evaluation`` works out, for given node probabilities, every term's
probabilities once (each owner's summed over its groups' outcomes), and
then the system's, also with nodes held working or failed, or a group's
outcome given: only the terms above them are worked out again, and the
owners among those summed over their outcomes again. ``joints`` gives
what the importance measures need of a node, and ``failed_cut_sets`` each
node's probability of failing with some minimal cut set containing it,
the numerator of its Fussell-Vesely importance (for a structure without
shocks). Every value is a sum of products, so a small probability keeps
its precision.
"""

import math
from dataclasses import dataclass
from itertools import product
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability.rbd.modular import (
    KOON,
    NODE,
    PARALLEL,
    SERIES,
    Decomposition,
    _koon_cut_shares,
    _products_of_others,
)

#: The owner of a group whose members only the core joins: the system.
SYSTEM = -1

#: The most combinations of their outcomes the groups one module holds may
#: have before those whose causes are independent are written out as shock
#: events (see ``Plan``).
COMBINATIONS = 64

#: A group's outcome: its probability, and its members' probabilities of
#: working and of failing given it.
Outcome = Tuple[Any, Dict[Hashable, Any], Dict[Hashable, Any]]


@dataclass(frozen=True)
class Shock:
    """A common-cause group's shared cause as an event of its own: the
    ``index``-th of group ``group``'s causes. It "works" while it has not
    fired."""

    group: int
    index: int

    def __repr__(self) -> str:
        return f"<common cause {self.index} of group {self.group}>"


@dataclass(frozen=True)
class Struck:
    """An appearance of ``shock`` after ``node`` (a member, or a repeat of
    one), in a diagram's structure with the shocks: a repeated node."""

    shock: Shock
    node: Hashable


@dataclass(frozen=True)
class Failed:
    """A member's failure from any of its causes, in a fault tree's
    structure with the shocks: an OR gate of its own failure and its
    shocks."""

    member: Hashable


def _outcomes_of(group, Q, R) -> List[Outcome]:
    """A group's mutually exclusive shock outcomes at its members'
    probabilities of failing ``Q`` and of working ``R``: each subset of
    members a shock fails together, then no shock, every member failing on
    its own."""
    q_independent, r_independent, shocks = group.model._split(
        group.members, Q, R
    )
    outcomes: List[Outcome] = []
    total_shock = np.zeros_like(Q)
    for subset, prob in shocks:
        total_shock = total_shock + prob
        outcomes.append(
            (
                prob,
                {
                    member: (
                        np.zeros_like(Q) if member in subset else r_independent
                    )
                    for member in group.members
                },
                {
                    member: (
                        np.ones_like(Q) if member in subset else q_independent
                    )
                    for member in group.members
                },
            )
        )
    # No common-cause shock: every member fails only independently. With
    # independent causes (by rate, or as basic events), no cause has
    # struck: its own probability keeps its precision where one less the
    # shocks' would not.
    outcomes.append(
        (
            (
                group.model._no_shock(group.members, Q, R)
                if group.model.shocks == "independent"
                else 1.0 - total_shock
            ),
            {member: r_independent for member in group.members},
            {member: q_independent for member in group.members},
        )
    )
    return outcomes


def _first_member(group, p, q) -> Tuple[np.ndarray, np.ndarray]:
    """A group's members' probabilities of working and of failing (its
    first member's: the members of a group are alike)."""
    first = group.members[0]
    R = np.atleast_1d(np.asarray(p[first], dtype=float))
    Q = (
        1.0 - R
        if q is None
        else np.atleast_1d(np.asarray(q[first], dtype=float))
    )
    return R, Q


def group_outcomes(groups, p, q=None, check=None) -> List[List[Outcome]]:
    """Each common-cause group's outcomes (see ``_outcomes_of``) at the
    nodes' probabilities of working ``p`` and of failing ``q`` (each
    node's own ``ff``, so that a small one keeps its precision; else one
    less ``p``). ``check(index, group, Q)`` sees each group's members'
    probability of failing (the diagrams warn past ``ccf.VALIDITY``)."""
    out = []
    for index, group in enumerate(groups):
        R, Q = _first_member(group, p, q)
        if check is not None:
            check(index, group, Q)
        out.append(_outcomes_of(group, Q, R))
    return out


def expected_product(
    nodes, failing: Dict[Hashable, Any], outcomes: Sequence[List[Outcome]]
) -> Any:
    """The probability that every node of ``nodes`` has failed, the groups'
    members as their groups' outcomes have them, and the others from
    ``failing``: a product over the groups (they are independent) of each
    one's sum over its outcomes."""
    nodes = set(nodes)
    value: Any = 1.0
    for group in outcomes:
        members = nodes.intersection(group[0][1])
        if not members:
            continue
        total: Any = 0.0
        for weight, _, failed in group:
            term = weight
            for member in members:
                term = term * failed[member]
            total = total + term
        value = value * total
        nodes -= members
    for node in nodes:
        value = value * failing[node]
    return value


def outcome_count(group) -> int:
    """How many outcomes a group's shocks have (see ``_outcomes_of``)."""
    sets = group.model._cause_sets(group.members)
    if group.model._exclusive(group.members):  # one at most fires
        return len(sets) + 1
    unions: set = {frozenset()}
    for struck in sets:
        unions |= {union | struck for union in unions}
    return len(unions)


def combined(kind: int, R: list, Q: list, k: int = 0) -> Tuple[Any, Any]:
    """A module's probabilities of working and of failing from its
    members', in the order ``Decomposition._forward`` sums them."""
    if kind == SERIES:
        # Fails at the first failed member: q1 + r1 q2 + r1 r2 q3...
        works, fails = R[0], Q[0]
        for r, q in zip(R[1:], Q[1:]):
            fails = fails + works * q
            works = works * r
        return works, fails
    if kind == PARALLEL:
        # Works at the first working member: r1 + q1 r2 + q1 q2 r3...
        works, fails = R[0], Q[0]
        for r, q in zip(R[1:], Q[1:]):
            works = works + fails * r
            fails = fails * q
        return works, fails
    # below[j]: the probability that exactly j of the members so far work
    # (j < k); above: that at least k do.
    below: list = [1.0] + [0.0] * (k - 1)
    above: Any = 0.0
    for r, q in zip(R, Q):
        above = above + below[k - 1] * r
        for j in range(k - 1, 0, -1):
            below[j] = below[j] * q + below[j - 1] * r
        below[0] = below[0] * q
    return above, sum(below)


class Conditioned:
    """A decomposition's terms with each common-cause group's owner: the
    smallest term holding its members (``SYSTEM`` where only the core joins
    them). ``groups`` are the groups' members (none for a group written
    out as shocks). Built once per structure and set of groups."""

    def __init__(
        self, decomposition: Decomposition, groups: Sequence[Sequence]
    ):
        self.decomposition = decomposition
        terms = decomposition.terms
        self.terms = terms
        self.position = {
            term[1]: i for i, term in enumerate(terms) if term[0] == NODE
        }
        parent: List[Optional[int]] = [None] * len(terms)
        below: List[frozenset] = []
        for i, term in enumerate(terms):
            if term[0] == NODE:
                below.append(frozenset([term[1]]))
                continue
            for c in term[1]:
                parent[c] = i
            below.append(frozenset().union(*(below[c] for c in term[1])))
        self.parent = parent
        #: The nodes under each term.
        self.below = below
        #: The core's terms (none without a core).
        self.core_terms = (
            []
            if decomposition.root is not None
            else [i for i, up in enumerate(parent) if up is None]
        )
        #: Each owner's groups (by their index).
        self.owned: Dict[int, List[int]] = {}
        for g, members in enumerate(groups):
            positions = [
                self.position[m] for m in members if m in self.position
            ]
            if not positions:
                continue  # in no minimal path set: it never matters
            self.owned.setdefault(self._smallest(positions), []).append(g)
        #: Each group's members.
        self.members = [frozenset(members) for members in groups]

    def _ancestors(self, i: int) -> List[int]:
        chain = [i]
        while self.parent[chain[-1]] is not None:
            chain.append(self.parent[chain[-1]])  # type: ignore[arg-type]
        return chain

    def _smallest(self, positions: Sequence[int]) -> int:
        """The smallest term holding every one of ``positions``, or
        ``SYSTEM`` if none does."""
        common = self._ancestors(positions[0])
        for i in positions[1:]:
            others = set(self._ancestors(i))
            common = [a for a in common if a in others]
            if not common:
                return SYSTEM
        return common[0]


class Plan:
    """How a structure's common-cause groups are worked out: each group
    conditioned on within its owner (see ``Conditioned``), but where an
    owner's groups have more than ``COMBINATIONS`` outcomes between them,
    written out as shock events (``expanded``: each such group's causes,
    as the members each fails), in the structure ``expand`` builds with
    them (None where it cannot be worked out: the groups are then
    conditioned on). ``conditioned`` is over the structure the groups are
    evaluated on, and ``base`` over the original one, every group
    conditioned on (for probabilities at which exclusive causes have no
    independent equivalent)."""

    def __init__(
        self,
        decomposition: Decomposition,
        groups: Sequence[Any],
        expand: Optional[Callable[[dict], Optional[Decomposition]]] = None,
    ):
        members = [tuple(group.members) for group in groups]
        self.base = self.conditioned = Conditioned(decomposition, members)
        self.expanded: Dict[int, List[frozenset]] = {}
        if expand is None:
            return
        chosen: Dict[int, List[frozenset]] = {}
        for owned in self.conditioned.owned.values():
            count = math.prod(outcome_count(groups[g]) for g in owned)
            if count <= COMBINATIONS:
                continue
            for g in owned:
                chosen[g] = groups[g].model._cause_sets(groups[g].members)
        if not chosen:
            return
        structure = expand(chosen)
        if structure is None:
            return
        self.expanded = chosen
        self.conditioned = Conditioned(
            structure,
            [() if g in chosen else m for g, m in enumerate(members)],
        )

    @staticmethod
    def key(groups) -> tuple:
        """What a plan depends on of the groups: their members, and the
        members their causes fail."""
        return tuple(
            (
                tuple(group.members),
                tuple(group.model._cause_sets(group.members)),
                group.model._exclusive(group.members),
            )
            for group in groups
        )


class Evaluation:
    """The system's probabilities at given node probabilities, with the
    common-cause groups as ``plan`` works them out (see ``Plan``)."""

    def __init__(self, plan: Plan, groups, p, q, shape, check=None):
        """``p`` and ``q``: the nodes' probabilities of working and of
        failing (1-d arrays of length ``shape``); ``check(index, group,
        Q)`` sees each group's members' probability of failing."""
        self.shape = shape
        #: Each group's outcomes, conditioned on (None for one written out).
        self.outcomes: List[Optional[List[Outcome]]] = []
        #: Each written-out group's split into causes (see ``ccf.Causes``).
        self.causes: Dict[int, Any] = {}
        #: Each node's group, by its index.
        self.group_of: Dict[Hashable, int] = {
            member: g
            for g, group in enumerate(groups)
            for member in group.members
        }
        self.plan = plan
        self._from_models(plan, groups, p, q, check)
        self._settle()

    def _from_models(self, plan: Plan, groups, p, q, check) -> None:
        """The groups' outcomes, or their causes written out, from their
        models (see ``Plan``)."""
        splits: Dict[int, Any] = {}
        for g, group in enumerate(groups):
            R, Q = _first_member(group, p, q)
            if check is not None:
                check(g, group, Q)
            if g in plan.expanded:
                splits[g] = group.model._fired(group.members, Q, R)
        # Exclusive causes with no independent equivalent at these
        # probabilities: every group is conditioned on.
        written = all(split is not None for split in splits.values())
        self.c = plan.conditioned if written else plan.base
        p, q = dict(p), dict(q)
        for g, group in enumerate(groups):
            if not written or g not in splits:
                R, Q = _first_member(group, p, q)
                self.outcomes.append(_outcomes_of(group, Q, R))
                continue
            self.outcomes.append(None)
            self.causes[g] = splits[g]
            q_independent, r_independent, fired = splits[g]
            for member in group.members:
                p[member], q[member] = r_independent, q_independent
            for k, (_, fires, holds) in enumerate(fired):
                p[Shock(g, k)], q[Shock(g, k)] = holds, fires
        self.p, self.q = p, q

    def _settle(self) -> None:
        """Every term's probabilities with nothing held."""
        c = self.c
        terms = c.terms
        # Every term's probabilities with nothing held, each owner's summed
        # over its groups' outcomes. A term between a member and its owner
        # takes the member as if alone, which is never used: its owner is
        # always worked out over the outcomes, which hold every member.
        R_: list = [None] * len(terms)
        Q_: list = [None] * len(terms)
        self.R, self.Q = R_, Q_
        for i, term in enumerate(terms):
            if i in c.owned:
                R_[i], Q_[i] = self._mix(i, {}, {})
            elif term[0] == NODE:
                R_[i], Q_[i] = self.p[term[1]], self.q[term[1]]
            else:
                R_[i], Q_[i] = self._combine(i, R_, Q_)
        self._fv: Optional[list] = None

    # -- frames --------------------------------------------------------------

    def _shape(self, frame):
        """The shape of a frame's values: the evaluation's own points
        (``frame`` None), or with an axis for each set of combinations
        laid out after them (see ``Tabled``)."""
        return self.shape if frame is None else frame

    @staticmethod
    def _lift(value, frame):
        """A value from the evaluation's points (or a frame around
        ``frame``) in ``frame``: with an axis of length 1 for each set of
        combinations it does not vary over."""
        if frame is None:
            return value
        value = np.asarray(value)
        if value.ndim == 0:
            return value
        return value.reshape(value.shape + (1,) * (len(frame) - value.ndim))

    # -- the terms' probabilities ------------------------------------------

    def _combine(self, i: int, R, Q) -> Tuple[Any, Any]:
        term = self.c.terms[i]
        children = term[1]
        return combined(
            term[0],
            [R[c] for c in children],
            [Q[c] for c in children],
            term[2] if term[0] == KOON else 0,
        )

    def _overrides(self, owner: int, fixed: dict):
        """Each combination of the outcomes of the groups ``owner`` owns
        (but those whose outcome ``fixed`` gives, holding their members):
        its probability, and ``fixed`` with their members' probabilities
        given it."""
        groups = [
            self.outcomes[g]
            for g in self.c.owned[owner]
            if not self.c.members[g] <= fixed.keys()
        ]
        for combination in product(*groups):  # type: ignore[arg-type]
            weight: Any = 1.0
            given = dict(fixed)
            for outcome_weight, works, fails in combination:
                weight = weight * outcome_weight
                for member in works:
                    given[member] = (works[member], fails[member])
            yield weight, given

    def _order(self, top: int, keys) -> List[int]:
        """The terms under ``top`` (with it) whose probabilities change with
        the nodes ``keys``, children before parents; an owner under ``top``
        is one of them as a whole (see ``_mix``)."""
        order: List[int] = []
        stack: List[Tuple[int, bool]] = [(top, False)]
        while stack:
            i, ready = stack.pop()
            if ready:
                order.append(i)
                continue
            stack.append((i, True))
            term = self.c.terms[i]
            if term[0] == NODE or (i != top and i in self.c.owned):
                continue
            for c in term[1]:
                if self.c.below[c] & keys:
                    stack.append((c, False))
        return order

    def _pass(
        self, top: int, fixed: dict, hold: dict, frame=None
    ) -> Tuple[Any, Any]:
        """``top``'s probabilities with the nodes ``fixed`` at the given
        probabilities and those in ``hold`` working (True) or failed, at
        the points of ``frame``; ``top`` itself is not summed
        over its own groups' outcomes."""
        keys = fixed.keys() | hold.keys()
        R: Dict[int, Any] = {}
        Q: Dict[int, Any] = {}
        lift = self._lift
        for i in self._order(top, keys):
            term = self.c.terms[i]
            if i != top and i in self.c.owned:
                R[i], Q[i] = self._mix(i, fixed, hold, frame)
            elif term[0] == NODE:
                R[i], Q[i] = self._node(term[1], fixed, hold, frame)
            else:
                children = term[1]
                R[i], Q[i] = combined(
                    term[0],
                    [
                        R[c] if c in R else lift(self.R[c], frame)
                        for c in children
                    ],
                    [
                        Q[c] if c in Q else lift(self.Q[c], frame)
                        for c in children
                    ],
                    term[2] if term[0] == KOON else 0,
                )
        return R[top], Q[top]

    def _node(self, node, fixed: dict, hold: dict, frame=None):
        if node in fixed:
            works, fails = fixed[node]
        else:
            works = self._lift(self.p[node], frame)
            fails = self._lift(self.q[node], frame)
        if node not in hold:
            return works, fails
        ones = np.ones(self._shape(frame))
        return (ones, ones * 0.0) if hold[node] else (ones * 0.0, ones)

    def _mix(
        self, owner: int, fixed: dict, hold: dict, frame=None
    ) -> Tuple[Any, Any]:
        """``owner``'s probabilities, summed over its groups' outcomes."""
        works: Any = 0.0
        fails: Any = 0.0
        for weight, given in self._overrides(owner, fixed):
            r, q = self._unmixed(owner, given, hold, frame)
            works = works + weight * r
            fails = fails + weight * q
        return works, fails

    def _unmixed(
        self, owner: int, fixed: dict, hold: dict, frame=None
    ) -> Tuple[Any, Any]:
        """``owner``'s probabilities with ``fixed`` and ``hold``, not summed
        over its own groups' outcomes: the core's for ``SYSTEM``."""
        if owner != SYSTEM:
            return self._pass(owner, fixed, hold, frame)
        d = self.c.decomposition
        size = self._shape(frame)
        R, Q = self._top_terms(fixed, hold, frame)
        return (
            d._core_value(R, Q, False, size),
            d._core_value(R, Q, True, size),
        )

    # -- the system --------------------------------------------------------

    def system(self, fixed=None, hold=None, frame=None) -> Tuple[Any, Any]:
        """The probabilities that the system works and that it fails: given
        the outcomes of the groups whose members ``fixed`` holds (each
        member's probabilities of working and of failing in it), and the
        nodes of ``hold`` working (True) or failed (False); at the points
        of ``frame`` (see ``_shape``)."""
        fixed = {} if fixed is None else fixed
        hold = {} if hold is None else hold
        d = self.c.decomposition
        if d.always_works:
            one = np.ones(self._shape(frame))
            return one, one * 0.0
        top = SYSTEM if d.root is None else d.root
        if top != SYSTEM and not hold and not fixed:
            return self._lift(self.R[top], frame), self._lift(
                self.Q[top], frame
            )
        if top in self.c.owned:
            return self._mix(top, fixed, hold, frame)
        return self._unmixed(top, fixed, hold, frame)

    def _top_terms(self, fixed: dict, hold: dict, frame=None):
        """Every term's probabilities, those of the core's terms with
        ``fixed`` and ``hold`` (at the frame's points)."""
        R, Q = list(self.R), list(self.Q)
        keys = fixed.keys() | hold.keys()
        for t in self.c.core_terms:
            if self.c.below[t] & keys:
                if t in self.c.owned:
                    R[t], Q[t] = self._mix(t, fixed, hold, frame)
                else:
                    R[t], Q[t] = self._pass(t, fixed, hold, frame)
            elif frame is not None:
                R[t], Q[t] = self._lift(R[t], frame), self._lift(Q[t], frame)
        return R, Q

    # -- the importance measures -------------------------------------------

    def joints(self, node) -> Tuple[Dict[str, Any], bool]:
        """What the importance measures need of ``node``: the probabilities
        that it works and that it has failed (``works``, ``fails``), and
        that the system works and fails with it working (``up_ok``,
        ``down_ok``) and with it failed (``up_bad``, ``down_bad``); and
        whether those four are joint probabilities (for a group member,
        whose state says something of its group's shocks: summed over
        them, each weighed by its chance of the state) or given the node's
        state (for one outside the groups, held)."""
        g = self.group_of.get(node)
        if g is None:
            up_1, down_1 = self.system(hold={node: True})
            up_0, down_0 = self.system(hold={node: False})
            return (
                {
                    "works": self.p[node],
                    "fails": self.q[node],
                    "up_ok": up_1,
                    "down_ok": down_1,
                    "up_bad": up_0,
                    "down_bad": down_0,
                },
                False,
            )
        if g in self.causes:
            return self._written_out(node, g), True
        out: Dict[str, Any] = dict.fromkeys(
            ("works", "fails", "up_ok", "down_ok", "up_bad", "down_bad"), 0.0
        )
        for weight, given, failing in self.outcomes[g]:  # type: ignore
            fixed = {m: (given[m], failing[m]) for m in given}
            on, off = given[node], failing[node]
            up_1, down_1 = self.system(fixed, {node: True})
            up_0, down_0 = self.system(fixed, {node: False})
            out["works"] = out["works"] + weight * on
            out["fails"] = out["fails"] + weight * off
            out["up_ok"] = out["up_ok"] + weight * on * up_1
            out["down_ok"] = out["down_ok"] + weight * on * down_1
            out["up_bad"] = out["up_bad"] + weight * off * up_0
            out["down_bad"] = out["down_bad"] + weight * off * down_0
        return out, True

    def _written_out(self, node, g: int) -> Dict[str, Any]:
        """``joints`` for a member of a group written out as shocks: it
        works when neither its own causes nor any shock striking it has
        fired, and has failed through the first of them that has (its own,
        then the shocks in turn), so that each probability is a sum of
        products."""
        q_own, r_own, fired = self.causes[g]
        strikes = [
            (Shock(g, k), fires, holds)
            for k, (struck, fires, holds) in enumerate(fired)
            if node in struck
        ]
        held: Dict[Hashable, bool] = {node: True}
        works: Any = r_own
        for shock, _, holds in strikes:
            held[shock] = True
            works = works * holds
        up_1, down_1 = self.system(hold=held)
        up_0, down_0 = self.system(hold={node: False})
        fails: Any = q_own
        up_bad: Any = q_own * up_0
        down_bad: Any = q_own * down_0
        before: Any = r_own  # its own causes, and the shocks so far, held
        hold: Dict[Hashable, bool] = {node: True}
        for shock, fires, holds in strikes:
            up, down = self.system(hold={**hold, shock: False})
            fails = fails + before * fires
            up_bad = up_bad + before * fires * up
            down_bad = down_bad + before * fires * down
            before = before * holds
            hold[shock] = True
        return {
            "works": works,
            "fails": fails,
            "up_ok": works * up_1,
            "down_ok": works * down_1,
            "up_bad": up_bad,
            "down_bad": down_bad,
        }

    # -- Fussell-Vesely ----------------------------------------------------

    def failed_cut_sets(self) -> Dict[Hashable, Any]:
        """For each node, the probability that it has failed with some
        minimal cut set containing it (every node of the set failed), as
        ``Decomposition.failed_cut_sets`` gives it, the groups conditioned
        on within their owners (for a plan with no group written out).
        Within a term, the probability for a node of a member is the
        member's times the probability that the term's other members have
        failed as the term needs (see ``Decomposition.failed_cut_sets``),
        the members being independent given the outcomes of the groups the
        term or one above it owns."""
        assert not self.causes
        d = self.c.decomposition
        if d.always_works:
            return {}
        fv = self._fv_terms()
        if d.root is not None:
            return dict(fv[d.root])
        if SYSTEM not in self.c.owned:
            return self._core_shares(self.R, self.Q, fv)
        return self._fv_mix(SYSTEM, {})

    def _fv_unmixed(self, owner: int, fixed: dict, frame=None) -> dict:
        """``owner``'s probabilities for its nodes (see ``failed_cut_sets``)
        with ``fixed``, not summed over its own groups' outcomes."""
        if owner != SYSTEM:
            return self._fv_pass(owner, fixed, False, frame)[2]
        keys = fixed.keys()
        R, Q = list(self.R), list(self.Q)
        shares = list(self._fv_terms())
        for t in self.c.core_terms:
            if self.c.below[t] & keys:
                R[t], Q[t], shares[t] = self._fv_pass(t, fixed, True, frame)
            elif frame is not None:
                R[t], Q[t] = self._lift(R[t], frame), self._lift(Q[t], frame)
                shares[t] = self._lift_shares(shares[t], frame)
        return self._core_shares(R, Q, shares, frame)

    def _lift_shares(self, shares: dict, frame) -> dict:
        if frame is None:
            return shares
        return {node: self._lift(v, frame) for node, v in shares.items()}

    def _core_shares(self, R, Q, fv, frame=None) -> Dict[Hashable, Any]:
        """Each node's probability through the core's minimal cut sets:
        its term's, times the probability that for some cut set of the
        core containing the term every other term in it has failed."""
        d = self.c.decomposition
        terms, steps, roots = d._core_cut_plan()
        shape = self._shape(frame)
        values: list = [np.zeros(shape), np.ones(shape)]
        for pivot, failed, works in steps:
            values.append(Q[pivot] * values[failed] + R[pivot] * values[works])
        out: Dict[Hashable, Any] = {}
        for t, root in zip(terms, roots):
            for node, value in fv[t].items():
                out[node] = value * values[root]
        return out

    def _fv_terms(self) -> list:
        """Each term's probabilities for its nodes (see ``failed_cut_sets``)
        with nothing held, each owner's summed over its outcomes."""
        if self._fv is not None:
            return self._fv
        fv: list = [None] * len(self.c.terms)
        self._fv = fv
        for i, term in enumerate(self.c.terms):
            if i in self.c.owned:
                fv[i] = self._fv_mix(i, {})
            elif term[0] == NODE:
                fv[i] = {term[1]: self.Q[i]}
            else:
                fv[i] = self._fv_combine(i, self.R, self.Q, fv)
        return fv

    def _fv_combine(self, i: int, R, Q, fv) -> Dict[Hashable, Any]:
        term = self.c.terms[i]
        children = term[1]
        if term[0] == SERIES:
            shares: list = [1.0] * len(children)
        elif term[0] == PARALLEL:
            shares = _products_of_others([Q[c] for c in children])
        else:
            shares = _koon_cut_shares(
                [R[c] for c in children], [Q[c] for c in children], term[2]
            )
        out: Dict[Hashable, Any] = {}
        for c, share in zip(children, shares):
            for node, value in fv[c].items():
                out[node] = value * share
        return out

    def _fv_mix(
        self, owner: int, fixed: dict, frame=None
    ) -> Dict[Hashable, Any]:
        out: Dict[Hashable, Any] = {}
        for weight, given in self._overrides(owner, fixed):
            for node, value in self._fv_unmixed(owner, given, frame).items():
                out[node] = out.get(node, 0.0) + weight * value
        return out

    def _fv_pass(self, top: int, fixed: dict, mix_top: bool, frame=None):
        """``top``'s probabilities of working and of failing, and for its
        nodes (see ``failed_cut_sets``), with ``fixed``; summed over its
        own groups' outcomes with ``mix_top``."""
        if mix_top and top in self.c.owned:
            works, fails = self._mix(top, fixed, {}, frame)
            return works, fails, self._fv_mix(top, fixed, frame)
        R: Dict[int, Any] = {}
        Q: Dict[int, Any] = {}
        fv: Dict[int, Any] = {}
        base = self._fv_terms()
        lift = self._lift
        for i in self._order(top, fixed.keys()):
            term = self.c.terms[i]
            if i != top and i in self.c.owned:
                R[i], Q[i] = self._mix(i, fixed, {}, frame)
                fv[i] = self._fv_mix(i, fixed, frame)
            elif term[0] == NODE:
                R[i], Q[i] = self._node(term[1], fixed, {}, frame)
                fv[i] = {term[1]: Q[i]}
            else:
                children = term[1]
                Rc = {
                    c: R[c] if c in R else lift(self.R[c], frame)
                    for c in children
                }
                Qc = {
                    c: Q[c] if c in Q else lift(self.Q[c], frame)
                    for c in children
                }
                fvc = {
                    c: fv[c] if c in fv else self._lift_shares(base[c], frame)
                    for c in children
                }
                R[i], Q[i] = combined(
                    term[0],
                    [Rc[c] for c in children],
                    [Qc[c] for c in children],
                    term[2] if term[0] == KOON else 0,
                )
                fv[i] = self._fv_combine(i, Rc, Qc, fvc)
        return R[top], Q[top], fv[top]


#: The most points (each a base point and a combination of the states of
#: the groups conditioned on together there) a ``Tabled`` evaluation works
#: out in one pass of an owner (#218): past it, it refuses.
POINTS = 1 << 25

#: How many such points are worked out at once.
CHUNK = 1 << 17


class _Table:
    """A group's members' joint states, as ``Tabled`` takes them: each
    combination of the members up or down with a probability at some point
    (``works`` and ``fails`` one row each, 1 or 0 for each member), and its
    probability at each point (``probabilities``, a column each)."""

    def __init__(self, states):
        probabilities = np.asarray(states.probabilities, dtype=float)
        down = np.asarray(states.down, dtype=bool)
        kept = np.flatnonzero(np.any(probabilities > 0.0, axis=0))
        self.members = tuple(states.members)
        self.fails = down[kept].astype(float)
        self.works = 1.0 - self.fails
        self.probabilities = np.ascontiguousarray(probabilities[:, kept])

    def marginal(self, k: int) -> np.ndarray:
        """The ``k``-th member's probability of working at each point."""
        return self.probabilities @ self.works[:, k]


class Tabled(Evaluation):
    """``Evaluation`` with each common-cause group's members' joint states
    given as tables (a repairable system's, from its groups' chains),
    not from its model's outcomes: each combination of the members up or
    down, and its probability at each point (``tables``, each group's
    ``_ccf_chain.GroupStates``). Given its combination the members are up
    or down for certain, and independent of every other node, so each
    group is conditioned on within its owner as before (see
    ``Conditioned``); but where ``Evaluation`` sums an owner over its
    groups' outcomes one at a time, here they are points of their own:
    each point of the owner's repeated for each combination of its groups'
    states with a probability there, worked out together, ``CHUNK`` at a
    time, and summed back. So the memory taken stays bounded, and the time
    is that of the points: a sum over the owners where the groups' states
    are in separate modules, a product only where they meet. Before
    anything is worked out, the most points an owner can need is checked
    against ``POINTS`` (see ``_check``)."""

    def __init__(self, plan: Plan, groups, p, q, shape, tables):
        self.shape = shape
        self.plan = plan
        self.c = plan.base
        self.causes: Dict[int, Any] = {}
        self.outcomes = []
        self.groups = list(groups)
        self.tables = [_Table(states) for states in tables]
        self.group_of = {
            member: g
            for g, table in enumerate(self.tables)
            for member in table.members
        }
        self.p, self.q = dict(p), dict(q)
        self._joints: Dict[int, dict] = {}
        self._check()
        self._settle()

    # -- the size ----------------------------------------------------------

    def _count(self, owner: int) -> int:
        """How many combinations of its groups' states ``owner`` sums
        over."""
        return math.prod(
            len(self.tables[g].works) for g in self.c.owned[owner]
        )

    def _check(self) -> None:
        """Refuse before anything is worked out if an owner could need
        more than ``POINTS`` points at once: its own combinations, at each
        point of every owner above it (whose groups' members may lie in
        it, so that it is worked out again for each of theirs)."""
        owned = self.c.owned
        worst, where = 0, None
        for owner in owned:
            count = self._count(owner)
            chain = [] if owner == SYSTEM else self.c._ancestors(owner)[1:]
            for above in chain:
                if above in owned:
                    count *= self._count(above)
            if owner != SYSTEM and SYSTEM in owned:
                count *= self._count(SYSTEM)
            if count > worst:
                worst, where = count, owner
        if self.shape * worst <= POINTS:
            return
        members = [
            list(self.groups[g].members)
            for g in sorted(
                {g for o in owned for g in owned[o]}
                if where is None
                else set(owned[where])
            )
        ]
        raise NotImplementedError(
            f"The common-cause groups {members} meet in one part of the "
            "structure, whose exact values sum over every combination of "
            f"their members' states: {worst:,} of them at each of "
            f"{self.shape:,} times, more than the {POINTS:,} worked out "
            "exactly. Simulate the system with availability() or cost(), "
            "which take the groups in."
        )

    # -- an owner's combinations as points of their own ----------------------

    def _frames(self, owner: int, fixed: dict, frame):
        """The combinations of ``owner``'s groups' states (but those whose
        members ``fixed`` holds) as a new last axis of ``frame``, ``CHUNK``
        values at a time: for each chunk, the combinations' probabilities
        at each point, ``fixed`` with the groups' members up or down in
        each, and the chunk's frame. With no group left, ``fixed`` and the
        frame itself."""
        groups = [
            g
            for g in self.c.owned[owner]
            if not set(self.tables[g].members) <= fixed.keys()
        ]
        if not groups:
            yield None, fixed, frame
            return
        outer = (self.shape,) if frame is None else tuple(frame)
        tables = [self.tables[g] for g in groups]
        counts = [len(table.works) for table in tables]
        total = math.prod(counts)
        step = max(1, CHUNK // math.prod(outer))
        # A point's probability varies only along the first axis (the
        # points themselves) and the new one.
        spread = (self.shape,) + (1,) * (len(outer) - 1)
        for start in range(0, total, step):
            columns = np.arange(start, min(start + step, total))
            width = len(columns)
            inner = outer + (width,)
            # Each group's combination at each column (the last group's
            # changing fastest), then their probabilities in group order.
            digits = []
            rest = columns
            for count in reversed(counts):
                digits.append(rest % count)
                rest = rest // count
            digits.reverse()
            weights: Any = None
            for table, digit in zip(tables, digits):
                chance = table.probabilities[:, digit].reshape(
                    spread + (width,)
                )
                weights = chance if weights is None else weights * chance
            given = {
                member: (self._lift(w, inner), self._lift(f, inner))
                for member, (w, f) in fixed.items()
            }
            row = (1,) * len(outer) + (width,)
            for table, digit in zip(tables, digits):
                for k, member in enumerate(table.members):
                    given[member] = (
                        table.works[digit, k].reshape(row),
                        table.fails[digit, k].reshape(row),
                    )
            yield weights, given, inner

    def _mix(self, owner: int, fixed: dict, hold: dict, frame=None):
        works: Any = 0.0
        fails: Any = 0.0
        for weights, given, inner in self._frames(owner, fixed, frame):
            r, q = self._unmixed(owner, given, hold, inner)
            if weights is None:
                return r, q
            works = works + (weights * r).sum(axis=-1)
            fails = fails + (weights * q).sum(axis=-1)
        return works, fails

    def _fv_mix(self, owner: int, fixed: dict, frame=None) -> dict:
        out: Dict[Hashable, Any] = {}
        for weights, given, inner in self._frames(owner, fixed, frame):
            values = self._fv_unmixed(owner, given, inner)
            if weights is None:
                return values
            for node, value in values.items():
                out[node] = out.get(node, 0.0) + (weights * value).sum(axis=-1)
        return out

    # -- what the measures need --------------------------------------------

    def marginal(self, member) -> np.ndarray:
        """A group member's probability of working at each point."""
        table = self.tables[self.group_of[member]]
        return table.marginal(table.members.index(member))

    def joints(self, node) -> Tuple[Dict[str, Any], bool]:
        """As ``Evaluation.joints``: a member's are worked out for its
        whole group at once (see ``_group_joints``)."""
        g = self.group_of.get(node)
        if g is None:
            return super().joints(node)
        if g not in self._joints:
            self._joints[g] = self._group_joints(g)
        return self._joints[g][node], True

    def _group_joints(self, g: int) -> Dict[Hashable, Dict[str, Any]]:
        """``joints`` for each member of group ``g``: the system worked out
        once with the group's members in each of their combinations (a last
        axis, see ``_frames``), and summed over them for each member up
        and down."""
        table = self.tables[g]
        count = len(table.works)
        frame = (self.shape, count)
        given = {
            member: (
                table.works[:, k].reshape(1, count),
                table.fails[:, k].reshape(1, count),
            )
            for k, member in enumerate(table.members)
        }
        up, down = self.system(given, {}, frame)
        weights = table.probabilities

        def total(value) -> np.ndarray:
            return np.broadcast_to((weights * value).sum(axis=-1), self.shape)

        out: Dict[Hashable, Dict[str, Any]] = {}
        for k, member in enumerate(table.members):
            on, off = given[member]
            out[member] = {
                "works": total(on),
                "fails": total(off),
                "up_ok": total(on * up),
                "down_ok": total(on * down),
                "up_bad": total(off * up),
                "down_bad": total(off * down),
            }
        return out

    def combinations(self, g: int, down: np.ndarray) -> Tuple[Any, Any]:
        """The system's probabilities of working and of failing at each
        point (rows) with group ``g``'s members in each combination of
        ``down`` (columns: for each, whether each member is down), the
        other groups' as their tables have them."""
        down = np.asarray(down, dtype=bool)
        count = len(down)
        frame = (self.shape, count)
        given = {
            member: (
                np.where(down[:, k], 0.0, 1.0).reshape(1, count),
                np.where(down[:, k], 1.0, 0.0).reshape(1, count),
            )
            for k, member in enumerate(self.tables[g].members)
        }
        up, fails = self.system(given, {}, frame)
        return (
            np.array(np.broadcast_to(up, frame)),
            np.array(np.broadcast_to(fails, frame)),
        )

    def outcomes_of_tables(self) -> List[List[Outcome]]:
        """Each group's combinations as outcomes (see
        ``expected_product``)."""
        return [
            [
                (
                    table.probabilities[:, c],
                    dict(zip(table.members, table.works[c])),
                    dict(zip(table.members, table.fails[c])),
                )
                for c in range(len(table.works))
            ]
            for table in self.tables
        ]
