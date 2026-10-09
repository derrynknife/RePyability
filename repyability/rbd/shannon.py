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


def replay(pivots, actives, inactives, root, works, fails, values):
    """A plan's value: slot ``i + 2`` of ``values`` is ``works[p] *
    values[active] + fails[p] * values[inactive]`` for step ``i``'s pivot
    row ``p`` and branches (``pivots``, ``actives``, ``inactives``), slots 0
    and 1 holding the values of failing and working; the value at
    ``root``. ``works`` and ``fails`` hold each pivot's probabilities of
    working and failing (by row); ``values`` has room for a value per slot
    (``room``), and keeps them for ``replay_gradient``.

    One function for every replay: run as Python, its values are numbers
    or arrays of any shape (a column of probabilities each element);
    compiled (``_bdd_kernel``), numbers, a column at a time."""
    slot = 2
    for p, a, b in zip(pivots, actives, inactives):
        values[slot] = works[p] * values[a] + fails[p] * values[b]
        slot += 1
    return values[root]


def room(steps: int, fail: Any, work: Any) -> list:
    """Room for ``replay``'s values, slots 0 and 1 holding ``fail`` and
    ``work``."""
    return [fail, work] + [None] * steps


def replay_gradient(
    pivots,
    actives,
    inactives,
    root,
    works,
    fails,
    values,
    one,
    adjoints,
    has_adjoint,
    gradient,
    reached,
):
    """The derivative of ``replay``'s value (whose slots ``values`` holds)
    with respect to each row's probability of working, by a reverse pass:
    into ``gradient`` (room for a value per row), with whether each row has
    one in ``reached`` (a row the pass never reaches has none). ``one`` is
    the root's adjoint, 1 (in the values' form); ``adjoints`` has room for
    a value per slot. ``has_adjoint`` and ``reached`` start all False.
    Each adjoint and derivative starts at its first share, not at 0 (which
    would turn a -0.0 into 0.0). Run as Python or compiled, as
    ``replay``."""
    adjoints[root] = one
    has_adjoint[root] = True
    for s in range(len(pivots) - 1, -1, -1):
        if not has_adjoint[s + 2]:
            continue
        p, a, b = pivots[s], actives[s], inactives[s]
        adjoint = adjoints[s + 2]
        # (A derivative or adjoint is its own once it has its first share,
        # so the rest are added in place.)
        if reached[p]:
            gradient[p] += adjoint * (values[a] - values[b])
        else:
            gradient[p] = adjoint * (values[a] - values[b])
        reached[p] = True
        if a > _ONE:
            if has_adjoint[a]:
                adjoints[a] += adjoint * works[p]
            else:
                adjoints[a] = adjoint * works[p]
            has_adjoint[a] = True
        if b > _ONE:
            if has_adjoint[b]:
                adjoints[b] += adjoint * fails[p]
            else:
                adjoints[b] = adjoint * fails[p]
            has_adjoint[b] = True


def value_and_gradient(
    pivots, actives, inactives, root, works, fails, fail, work, rows
) -> tuple:
    """``replay``'s value and ``replay_gradient``'s derivatives, in
    Python: the value, a list by row, and whether each row has one."""
    values = room(len(pivots), fail, work)
    value = replay(pivots, actives, inactives, root, works, fails, values)
    slots = len(values)
    gradient: list = [None] * rows
    reached = [False] * rows
    replay_gradient(
        pivots,
        actives,
        inactives,
        root,
        works,
        fails,
        values,
        1.0,
        [None] * slots,
        [False] * slots,
        gradient,
        reached,
    )
    return value, gradient, reached


def _plan_rows(plan: tuple[list, int]) -> tuple:
    """A plan's pivots, numbered by their first appearance, as ``replay``
    takes them: each step's pivot row, active and inactive slots; and the
    pivots in row order."""
    steps, _ = plan
    names = list(dict.fromkeys(pivot for pivot, _, _ in steps))
    row = {name: j for j, name in enumerate(names)}
    return (
        [row[pivot] for pivot, _, _ in steps],
        [a for _, a, _ in steps],
        [b for _, _, b in steps],
        names,
    )


def _evaluate_shannon_plan(
    plan: tuple[list, int],
    element_probabilities: Dict[Any, np.ndarray],
    array_shape,
) -> np.ndarray:
    """Replay a :func:`_shannon_plan` for the given probabilities."""
    pivots, actives, inactives, names = _plan_rows(plan)
    works = [element_probabilities[name] for name in names]
    return replay(
        pivots,
        actives,
        inactives,
        plan[1],
        works,
        [1 - p for p in works],
        room(len(pivots), np.zeros(array_shape), np.ones(array_shape)),
    )


def _shannon_value_and_gradient(
    plan: tuple[list, int],
    probabilities: Union[Dict[Any, Any], Sequence[Any]],
    complements: Union[Dict[Any, Any], Sequence[Any]],
    terminals: tuple[float, float] = (0.0, 1.0),
) -> tuple[Any, Dict[Any, Any]]:
    """A plan's value for the element probabilities (single values, or
    arrays of one shape), and its derivative with respect to each element's
    probability (the element's Birnbaum importance), by one forward and one
    reverse pass (``replay_gradient``). ``complements`` holds each
    element's ``1 - p``, computed without cancellation, so the value keeps
    its full relative precision however small it is.

    ``terminals`` are the values of the plan's two constant slots: with
    ``(1.0, 0.0)`` in place of the default ``(0.0, 1.0)`` the plan gives the
    complement, the probability that no set is satisfied, just as precisely
    (and its derivative, the negated Birnbaum importance)."""
    pivots, actives, inactives, names = _plan_rows(plan)
    value, gradient, reached = value_and_gradient(
        pivots,
        actives,
        inactives,
        plan[1],
        [probabilities[name] for name in names],
        [complements[name] for name in names],
        terminals[0],
        terminals[1],
        len(names),
    )
    return value, {
        name: gradient[j] for j, name in enumerate(names) if reached[j]
    }


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
        # The cut sets with the pivot working, and the pivot with each with
        # it failed that holds none of those. The structure is coherent,
        # so a cut set with the pivot working is one with it failed, and
        # holds a minimal one: a minimal one with it failed holds one with
        # it working only by being it, which a lookup finds (#172).
        spare_pivot = cuts[active]
        known = set(spare_pivot)
        cuts.append(
            spare_pivot
            + [
                bit[pivot] | rest
                for rest in cuts[inactive]
                if rest not in known
            ]
        )
    return {
        frozenset(c for c in components if cut & bit[c]) for cut in cuts[root]
    }
