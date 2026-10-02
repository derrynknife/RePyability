"""The Shannon (pivotal) decomposition at the heart of the exact engine.

Given a family of sets of elements (e.g. the minimal path sets of an RBD's
components), the probability that at least one set has all of its elements
active is worked out by conditioning on one element at a time,

    P(S) = p_e * P(S | e active) + (1 - p_e) * P(S | e inactive),

and solving each distinct sub-problem once. The decomposition depends only
on the sets, so it is recorded once as a *plan* and replayed for any
probabilities (see ``_shannon_plan``). The same plan gives the derivative
with respect to each element's probability, and the minimal cut sets.
"""

from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np

# Value slots 0 and 1 of a Shannon plan hold the constant 0 and 1 arrays.
_ZERO, _ONE = 0, 1


def _shannon_plan(sets: Iterable[frozenset]) -> tuple[list, int]:
    """Record the Shannon decomposition of the probability that at least
    one of ``sets`` is satisfied as a replayable plan.

    The decomposition -- which element to pivot on, and which sub-problems
    recur -- depends only on the sets, not on the probabilities, so it can be
    worked out once and replayed for any probabilities. Returns ``(steps,
    root)``: step ``i`` fills value slot ``i + 2`` with
    ``p[pivot] * value[active] + (1 - p[pivot]) * value[inactive]``, and
    ``root`` is the slot holding the answer.

    The decomposition is a depth-first recursion (the active branch, then
    the inactive one, then the step itself), run on an explicit stack so
    that systems with a thousand or more components do not reach Python's
    recursion limit.
    """
    sets = [frozenset(s) for s in sets]
    slots: Dict[frozenset, int] = {}
    steps: list[tuple[Any, int, int]] = []

    def known(state: frozenset) -> Optional[int]:
        # state is a frozenset of frozensets: the sets still to be satisfied,
        # with already-active elements removed.
        if not state:
            # No set can be satisfied any more -> probability 0.
            return _ZERO
        if frozenset() in state:
            # A set has had all its elements satisfied -> probability 1.
            return _ONE
        return slots.get(state)

    def split(state: frozenset) -> list:
        # Pivot on the element appearing in the most sets, which tends to
        # collapse the problem (and the memo table) fastest.
        counts: Dict[Any, int] = {}
        for s in state:
            for element in s:
                counts[element] = counts.get(element, 0) + 1
        pivot = max(counts, key=lambda e: counts[e])

        # Pivot active: it satisfies its requirement, so drop it from every
        # set that contained it (other sets are unaffected).
        state_active = frozenset(s - {pivot} for s in state)
        # Pivot inactive: any set needing it can never be satisfied -> drop it.
        state_inactive = frozenset(s for s in state if pivot not in s)
        # [state, pivot, (branches still to solve), their solved slots]
        return [state, pivot, [state_active, state_inactive], []]

    root_state = frozenset(sets)
    root = known(root_state)
    stack = [] if root is not None else [split(root_state)]
    while stack:
        state, pivot, branches, solved = stack[-1]
        if len(solved) < 2:
            branch = branches[len(solved)]
            slot = known(branch)
            if slot is None:
                stack.append(split(branch))
            else:
                solved.append(slot)
            continue
        steps.append((pivot, solved[0], solved[1]))
        slots[state] = len(steps) + 1
        stack.pop()
        if stack:
            stack[-1][3].append(slots[state])
        else:
            root = slots[state]
    assert root is not None
    return steps, root


def _union_plan(
    families: Sequence[Iterable[frozenset]],
) -> tuple[list, list]:
    """One Shannon decomposition, in the format of :func:`_shannon_plan`'s,
    of the probabilities that at least one set of each of ``families`` is
    satisfied: ``(steps, roots)``, with ``roots[i]`` the slot holding the
    ``i``-th family's.

    The families share their sub-problems, which are solved once. Each
    sub-problem is kept minimal (a set holding another adds nothing to the
    union, so it is dropped), its sets as bitmasks, so that equal ones are
    found however they arose. With many overlapping families (the cut sets
    through each term of a meshed core, for its Fussell-Vesely importance)
    this is many times faster than decomposing each on its own."""
    elements = list(
        dict.fromkeys(e for family in families for s in family for e in s)
    )
    bit = {e: 1 << i for i, e in enumerate(elements)}
    slots: Dict[frozenset, int] = {}
    steps: list[tuple[Any, int, int]] = []

    def minimal(masks: Iterable[int]) -> frozenset:
        kept: list = []
        for m in sorted(set(masks), key=lambda m: bin(m).count("1")):
            if not any(k & m == k for k in kept):
                kept.append(m)
        return frozenset(kept)

    def known(state: frozenset) -> Optional[int]:
        if not state:
            return _ZERO
        if 0 in state:
            return _ONE
        return slots.get(state)

    def split(state: frozenset) -> list:
        # Pivot on the element in the most sets, as _shannon_plan does.
        counts: Dict[int, int] = {}
        for m in state:
            while m:
                low = m & -m
                counts[low] = counts.get(low, 0) + 1
                m ^= low
        pivot = max(counts, key=lambda b: (counts[b], -b))
        active = minimal(m & ~pivot for m in state)
        inactive = frozenset(m for m in state if not m & pivot)
        return [state, pivot, [active, inactive], []]

    roots: list = []
    for family in families:
        state = minimal(
            sum(bit[e] for e in s) for s in (frozenset(x) for x in family)
        )
        root = known(state)
        stack = [] if root is not None else [split(state)]
        while stack:
            current, pivot, branches, solved = stack[-1]
            if len(solved) < 2:
                branch = branches[len(solved)]
                slot = known(branch)
                if slot is None:
                    stack.append(split(branch))
                else:
                    solved.append(slot)
                continue
            steps.append((elements[pivot.bit_length() - 1], *solved))
            slots[current] = len(steps) + 1
            stack.pop()
            if stack:
                stack[-1][3].append(slots[current])
            else:
                root = slots[current]
        roots.append(root)
    return steps, roots


def _evaluate_shannon_plan(
    plan: tuple[list, int],
    element_probabilities: Dict[Any, np.ndarray],
    array_shape,
) -> np.ndarray:
    """Replay a :func:`_shannon_plan` for the given probabilities."""
    steps, root = plan
    values = [np.zeros(array_shape), np.ones(array_shape)]
    for pivot, active, inactive in steps:
        p = element_probabilities[pivot]
        values.append(p * values[active] + (1 - p) * values[inactive])
    return values[root]


def _shannon_value_and_gradient(
    plan: tuple[list, int],
    probabilities: Union[Dict[Any, Any], Sequence[Any]],
    complements: Union[Dict[Any, Any], Sequence[Any]],
    terminals: tuple[float, float] = (0.0, 1.0),
) -> tuple[Any, Dict[Any, Any]]:
    """A plan's value for the element probabilities (single values, or
    arrays of one shape), and its derivative with respect to each element's
    probability (the element's Birnbaum importance), by one forward and one
    reverse pass. ``complements`` holds each element's ``1 - p``, computed
    without cancellation, so the value keeps its full relative precision
    however small it is.

    ``terminals`` are the values of the plan's two constant slots: with
    ``(1.0, 0.0)`` in place of the default ``(0.0, 1.0)`` the plan gives the
    complement, the probability that no set is satisfied, just as precisely
    (and its derivative, the negated Birnbaum importance)."""
    steps, root = plan
    values = list(terminals)
    for pivot, active, inactive in steps:
        values.append(
            probabilities[pivot] * values[active]
            + complements[pivot] * values[inactive]
        )
    # None: a slot the root does not reach (or a constant, which needs none).
    adjoints: list = [None] * len(values)
    adjoints[root] = 1.0
    gradient: Dict[Any, Any] = {}
    for index in range(len(steps) - 1, -1, -1):
        adjoint = adjoints[index + 2]
        if adjoint is None:
            continue
        pivot, active, inactive = steps[index]
        change = adjoint * (values[active] - values[inactive])
        gradient[pivot] = (
            gradient[pivot] + change if pivot in gradient else change
        )
        for slot, weight in (
            (active, probabilities[pivot]),
            (inactive, complements[pivot]),
        ):
            if slot > _ONE:
                share = adjoint * weight
                adjoints[slot] = (
                    share if adjoints[slot] is None else adjoints[slot] + share
                )
    return values[root], gradient


def _minimal_cut_sets(plan: tuple[list, int]) -> set[frozenset]:
    """The minimal cut sets of the structure a :func:`_shannon_plan` was
    built from: the minimal transversals of its sets (see
    ``minimal_cut_sets_from_path_sets`` in ``rbd.py``)."""
    steps, root = plan
    # Cut sets as bitmasks (one bit per component), so each union and subset
    # test is a single integer operation.
    components = list(dict.fromkeys(pivot for pivot, _, _ in steps))
    bit = {component: 1 << i for i, component in enumerate(components)}
    # Value slots as in the plan: 0 never works (the empty set is a cut),
    # 1 always works (nothing is a cut), then one per step.
    cuts: list[list[int]] = [[0], []]
    for pivot, active, inactive in steps:
        spare_pivot = cuts[active]
        cuts.append(
            spare_pivot
            + [
                bit[pivot] | rest
                for rest in cuts[inactive]
                if not any(cut & rest == cut for cut in spare_pivot)
            ]
        )
    return {
        frozenset(c for c in components if cut & bit[c]) for cut in cuts[root]
    }
