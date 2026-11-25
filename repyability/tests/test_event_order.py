"""The order of a ``RepairableRBD`` simulation's events at one instant
(``_events._EventQueue``).

Events at one time are taken by class (failures, then planned stops, then
restorations), then a shared cause's or a shutdown's before the nodes', in
the order first queued, then by the node's place in the diagram. Here: the
Python queue and the compiled engine's heap (``_kernel._push``, ``_pop``)
against that order written out plainly, for random batches of equal times
queued in a shuffled order; the crews' chain, which copies the order a
strike's members fail in, is checked against the simulation in
``test_crews_ccf``."""

import random

import numpy as np
import pytest

from repyability.rbd import _events
from repyability.rbd._events import Event

#: Each kind of event, as (status, preventive, inspection, its class).
KINDS = [
    (False, False, False, _events.FAILURE),
    (False, True, False, _events.STOP),
    (False, False, True, _events.STOP),
    (True, False, False, _events.RESTORATION),
    (True, True, False, _events.RESTORATION),
    (True, False, True, _events.RESTORATION),
]


class Other:
    """A cause or a shutdown: no node of the diagram."""


def batches(seed: int, n_nodes: int = 7, size: int = 60):
    """Batches of events at a few equal times, each node at most once per
    class and time (a node has one event that is taken)."""
    rng = random.Random(seed)
    nodes: list = [f"n{k}" for k in range(n_nodes)]
    others: list = [Other(), Other()]
    for _ in range(40):
        seen: set = set()
        events: list = []
        while len(events) < size:
            t = float(rng.randrange(4))
            status, preventive, inspection, tie = rng.choice(KINDS)
            node = rng.choice(nodes + others)
            if (t, tie, id(node)) in seen:
                continue
            seen.add((t, tie, id(node)))
            events.append(Event(t, node, status, preventive, inspection))
        yield nodes, others, events


def expected(nodes, events):
    """The events in the stated order, written out plainly: the others in
    the order first queued, then the nodes in the diagram's."""
    first: dict = {}
    for event in events:
        if event.component not in nodes:
            first.setdefault(id(event.component), len(first))

    def key(event):
        node = event.component
        place = (
            (1, nodes.index(node)) if node in nodes else (0, first[id(node)])
        )
        return (event.time, _events.tie_class(event), place)

    return sorted(events, key=key)


def test_the_queue_takes_ties_in_the_stated_order():
    for nodes, _, events in batches(20261125):
        queue = _events._EventQueue(nodes)
        for event in events:
            queue.put(event)
        out = [queue.get() for _ in range(len(events))]
        # (Events compare by time alone: compare them as objects.)
        assert [id(e) for e in out] == [id(e) for e in expected(nodes, events)]


def test_the_class_of_each_kind():
    for status, preventive, inspection, tie in KINDS:
        event = Event(1.0, "a", status, preventive, inspection)
        assert _events.tie_class(event) == tie


def drain(queue):
    return [queue.get() for _ in range(queue.qsize())]


def test_a_group_s_event_takes_the_class_it_is_given():
    queue = _events._EventQueue(["a", "b"])
    restored = Event(1.0, "a", True)
    group = Event(1.0, "b", True)  # its status is the group's state
    failed = Event(1.0, "a", False)
    queue.put(restored)
    queue.put(group, _events.FAILURE)
    queue.put(failed)
    assert [id(e) for e in drain(queue)] == [
        id(failed),
        id(group),
        id(restored),
    ]


#: The compiled engine's kinds of event, and their classes.
KERNEL_KINDS = {
    0: _events.FAILURE,  # _FAIL
    1: _events.RESTORATION,  # _RESTORE
    2: _events.STOP,  # _PM_START
    3: _events.RESTORATION,  # _PM_END
    4: _events.RESTORATION,  # _PM_IN_PLACE
    5: _events.RESTORATION,  # _TEST_UP
    6: _events.STOP,  # _TEST_DOWN
    7: _events.STOP,  # _TEST_MISSED
    8: _events.FAILURE,  # _GROUP: a unit failing
    10: _events.RESTORATION,  # _GROUP_UP: a unit's repair done
}


def test_the_compiled_heap_takes_ties_in_the_stated_order():
    pytest.importorskip("numba")
    from repyability.rbd import _kernel

    kinds = sorted(KERNEL_KINDS)
    assert [_kernel._tie_class(k) for k in kinds] == [
        KERNEL_KINDS[k] for k in kinds
    ]
    rng = np.random.default_rng(20261125)
    for _ in range(40):
        size = 80
        times = rng.integers(0, 4, size).astype(float)
        nodes = rng.integers(0, 9, size)
        states = rng.choice(kinds, size)
        heap = (
            np.empty(size),
            np.empty(size, np.int64),
            np.empty(size, np.int64),
            np.empty(size, np.int64),
        )
        n = 0
        for k in range(size):
            n = _kernel._push(heap, n, times[k], nodes[k], states[k], k)
        out = []
        while n > 0:
            t, c, state, tag, n = _kernel._pop(heap, n)
            out.append((t, KERNEL_KINDS[state], c))
        assert out == sorted(
            (times[k], KERNEL_KINDS[states[k]], nodes[k]) for k in range(size)
        )
