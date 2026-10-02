"""The compiled event loop of ``RepairableRBD`` simulations (numba).

``_simulate`` is ``RepairableRBD._replicate`` for components that are
plain ``NonRepairable`` units or standby groups of them, under age or
block replacement, inspected for hidden failures or neither, and with as
many repair crews as needed or fewer (#155), operation for operation, over
arrays: the same heap
(``heapq``'s algorithm, so equal times come out in the same order), the
same arithmetic in the same order, and the draws read from the same keyed
streams (see ``_streams``). Each simulation's results go to its own row
of the output arrays, so the rows can be worked out in any order, on any
number of threads, and a simulation that runs out of draws (or of room for
its system's changes) is simply run again once there is more.
``_compiled`` prepares the arrays and adds the rows to the tally in order.

Importing this module imports numba, which compiles the loop on first use
(or loads it from numba's cache).
"""

import numba
import numpy as np
from numba import njit

# The kinds of term of the structure (see ``modular``).
_NODE, _SERIES, _PARALLEL = 0, 1, 2

# The kinds of event: a failure, a restoration, preventive maintenance
# (#155): an outage starting and ending, and a renewal in place (in zero
# time, of a working unit); and a test of a unit with hidden failures: one
# that leaves it up (in zero time, or ending), one that takes it down (off
# line, or finding its failure), and one that misses its failure. An
# event's kind is its state code in the heap.
_FAIL, _RESTORE, _PM_START, _PM_END, _PM_IN_PLACE = 0, 1, 2, 3, 4
_TEST_UP, _TEST_DOWN, _TEST_MISSED = 5, 6, 7
# A standby group's next event, as its entry in the heap (an entry an
# earlier one has superseded is skipped: its tag is not the group's); and a
# unit's repair waiting for a repair crew.
_GROUP, _UNIT_JOB = 8, 9
# A standby group's units' own events (``_StandbyGroup``): a unit failing
# in operation, a spare failing in standby, a repair done.
_UNIT_FAILS, _SPARE_FAILS, _UNIT_REPAIRED = 0, 1, 2


@njit(cache=True, inline="always")
def _push(heap, size, t, c, state, tag):
    """``heapq.heappush`` of ``(t, c, state, tag)``, ordered by ``t``
    alone; ``heap`` is the arrays of the four."""
    times, nodes, states, tags = heap
    pos = size
    while pos > 0:
        parent = (pos - 1) >> 1
        if t < times[parent]:
            times[pos] = times[parent]
            nodes[pos] = nodes[parent]
            states[pos] = states[parent]
            tags[pos] = tags[parent]
            pos = parent
        else:
            break
    times[pos] = t
    nodes[pos] = c
    states[pos] = state
    tags[pos] = tag
    return size + 1


@njit(cache=True, inline="always")
def _pop(heap, size):
    """``heapq.heappop``: the first item, and the heap's new size."""
    times, nodes, states, tags = heap
    t0, c0, state0, tag0 = times[0], nodes[0], states[0], tags[0]
    size -= 1
    if size > 0:
        t, c, state, tag = times[size], nodes[size], states[size], tags[size]
        # Move the smaller child up until reaching a leaf, then the last
        # item up from there (heapq's _siftup and _siftdown).
        pos = 0
        child = 1
        while child < size:
            right = child + 1
            if right < size and not times[child] < times[right]:
                child = right
            times[pos] = times[child]
            nodes[pos] = nodes[child]
            states[pos] = states[child]
            tags[pos] = tags[child]
            pos = child
            child = 2 * pos + 1
        while pos > 0:
            parent = (pos - 1) >> 1
            if t < times[parent]:
                times[pos] = times[parent]
                nodes[pos] = nodes[parent]
                states[pos] = states[parent]
                tags[pos] = tags[parent]
                pos = parent
            else:
                break
        times[pos] = t
        nodes[pos] = c
        states[pos] = state
        tags[pos] = tag
    return t0, c0, state0, tag0, size


@njit(cache=True, inline="always")
def _works(status, structure, value):
    """The structure function: 1 if the system works with the components'
    ``status``, else 0 (see ``Decomposition.structure_function``)."""
    (
        kind,
        k,
        node,
        child_start,
        child_end,
        children,
        root,
        always,
        core_start,
        core_end,
        core_members,
    ) = structure
    if always:
        return 1
    for i in range(kind.size):
        term = kind[i]
        if term == _NODE:
            value[i] = status[node[i]]
        elif term == _SERIES:
            works = 1
            for j in range(child_start[i], child_end[i]):
                if value[children[j]] == 0:
                    works = 0
                    break
            value[i] = works
        elif term == _PARALLEL:
            works = 0
            for j in range(child_start[i], child_end[i]):
                if value[children[j]] != 0:
                    works = 1
                    break
            value[i] = works
        else:
            count = 0
            for j in range(child_start[i], child_end[i]):
                if value[children[j]] != 0:
                    count += 1
            value[i] = 1 if count >= k[i] else 0
    if root >= 0:
        return value[root]
    for p in range(core_start.size):
        works = 1
        for j in range(core_start[p], core_end[p]):
            if value[core_members[j]] == 0:
                works = 0
                break
        if works:
            return 1
    return 0


@njit(cache=True, inline="always")
def _keep(kept, value, count, down, c, state, working_paths):
    """Bring the structure up to date with component ``c`` now ``state``
    (see ``_compiled._kept``): each term it stands for, and up the tree as
    far as the change goes, then the core's path sets of a top term that
    changed. The number of the core's path sets that work, after."""
    (
        kind,
        need,
        parent,
        node_term_start,
        node_term_end,
        node_terms,
        term_path_start,
        term_path_end,
        term_paths,
        _,
        _,
        _,
        _,
    ) = kept
    for e in range(node_term_start[c], node_term_end[c]):
        term = node_terms[e]
        value[term] = state
        works = state
        while True:
            above = parent[term]
            if above < 0:
                break
            members = count[above] + (1 if works else -1)
            count[above] = members
            if kind[above] == _PARALLEL:
                now = 1 if members > 0 else 0
            else:
                # A series term needs all its members, a vote k of them.
                now = 1 if members >= need[above] else 0
            if now == value[above]:
                term = -1
                break
            value[above] = now
            works = now
            term = above
        if term >= 0:
            # A top term changed: the core's path sets it is in.
            for q in range(term_path_start[term], term_path_end[term]):
                p = term_paths[q]
                if works:
                    down[p] -= 1
                    if down[p] == 0:
                        working_paths += 1
                else:
                    if down[p] == 0:
                        working_paths -= 1
                    down[p] += 1
    return working_paths


@njit(cache=True, inline="always")
def _renewal(c, t, maintenance, fail, lives, limit, flat, base):
    """``RepairableRBD._renewal`` for a unit under age or block replacement
    put into service as new at ``t``: its life drawn, and its next event,
    its failure or its maintenance, whichever is due first (a failure at
    the same time first). Returns a status code (a stream's number + 1 if
    it ran out of draws), the event's time and its kind."""
    policy, interval, duration = maintenance[0], maintenance[1], maintenance[2]
    s = fail[c]
    k = lives[c]
    if k >= limit[s]:
        return s + 1, 0.0, _FAIL
    lives[c] = k + 1
    life = flat[base[s] + k]
    if policy[c] == 1:
        due = t + interval[c]
    else:
        # The next multiple of the interval after t.
        due = (np.floor(t / interval[c]) + 1.0) * interval[c]
        if not due > t:
            due = due + interval[c]
    if due < t:
        due = t
    if t + life <= due:
        return 0, t + life, _FAIL
    if duration[c] < 0:
        return 0, due, _PM_IN_PLACE
    return 0, due, _PM_START


@njit(cache=True, inline="always")
def _test_due(t, interval, offset):
    """``_Inspection.due``: the first test after ``t``."""
    if offset == 0.0:
        k = np.floor(t / interval) + 1.0
        due = k * interval
        if due > t:
            return due
        return (k + 1.0) * interval
    k = np.floor((t - offset) / interval) + 1.0
    due = offset + k * interval
    if due > t:
        return due
    return offset + (k + 1.0) * interval


@njit(cache=True, inline="always")
def _test_finding(t, interval, offset):
    """``_Inspection.finds``: the first test at or after ``t``."""
    if offset == 0.0:
        k = np.ceil(t / interval)
        due = k * interval
        if due >= t:
            return due
        return (k + 1.0) * interval
    k = np.ceil((t - offset) / interval)
    due = offset + k * interval
    if due >= t:
        return due
    return offset + (k + 1.0) * interval


@njit(cache=True, inline="always")
def _full_test(t, partial, interval, offset, per_full):
    """``_Inspection.is_full``: whether the test at ``t`` finds every
    failure (``round`` rounds half to even, as ``np.rint`` does)."""
    if not partial:
        return True
    k = int(np.rint((t - offset) / interval))
    return k % per_full == 0


@njit(cache=True, inline="always")
def _tested_next(c, t, upkeep, pending):
    """``RepairableRBD._inspected_next``: a working unit's next event, its
    hidden failure or its next test, whichever comes first (a failure at
    the same time first)."""
    interval, offset, test_time = upkeep[5], upkeep[6], upkeep[10]
    failure = pending[c]
    due = _test_due(t, interval[c], offset[c])
    if failure <= due:
        return failure, _FAIL
    if test_time[c] < 0:
        return due, _TEST_UP
    return due, _TEST_DOWN


@njit(cache=True, inline="always")
def _before(rank, due, order, other_rank, other_due, other_order):
    """Whether a waiting job is started before another (``_Crews``'
    order): of higher priority (lower rank), then due first, then queued
    first."""
    if rank != other_rank:
        return rank < other_rank
    if due != other_due:
        return due < other_due
    return order < other_order


@njit(cache=True, inline="always")
def _wait(queue, size, rank, due, order, c, done, kind):
    """Queue component ``c``'s job, due at ``due`` and ending at ``done``
    (an event of ``kind``) if started at once, until a repair crew is free
    (``_Crews.request``); the queue's new size."""
    ranks, dues, orders, nodes, ends, kinds = queue
    pos = size
    while pos > 0:
        parent = (pos - 1) >> 1
        if not _before(
            rank, due, order, ranks[parent], dues[parent], orders[parent]
        ):
            break
        ranks[pos] = ranks[parent]
        dues[pos] = dues[parent]
        orders[pos] = orders[parent]
        nodes[pos] = nodes[parent]
        ends[pos] = ends[parent]
        kinds[pos] = kinds[parent]
        pos = parent
    ranks[pos] = rank
    dues[pos] = due
    orders[pos] = order
    nodes[pos] = c
    ends[pos] = done
    kinds[pos] = kind
    return size + 1


@njit(cache=True, inline="always")
def _start_waiting(queue, size):
    """The waiting job a free crew starts (``_Crews.release``): its
    component, when it fell due, when and how it ends if started then, and
    the queue's new size."""
    ranks, dues, orders, nodes, ends, kinds = queue
    c0, due0, done0, kind0 = nodes[0], dues[0], ends[0], kinds[0]
    size -= 1
    if size > 0:
        rank, due, order = ranks[size], dues[size], orders[size]
        c, done, kind = nodes[size], ends[size], kinds[size]
        pos = 0
        child = 1
        while child < size:
            right = child + 1
            if right < size and _before(
                ranks[right],
                dues[right],
                orders[right],
                ranks[child],
                dues[child],
                orders[child],
            ):
                child = right
            if not _before(
                ranks[child], dues[child], orders[child], rank, due, order
            ):
                break
            ranks[pos] = ranks[child]
            dues[pos] = dues[child]
            orders[pos] = orders[child]
            nodes[pos] = nodes[child]
            ends[pos] = ends[child]
            kinds[pos] = kinds[child]
            pos = child
            child = 2 * pos + 1
        ranks[pos] = rank
        dues[pos] = due
        orders[pos] = order
        nodes[pos] = c
        ends[pos] = done
        kinds[pos] = kind
    return c0, due0, done0, kind0, size


@njit(cache=True, inline="always")
def _release(
    key, t, t_end, n, crew, holding, queue, pending, plan, heap, size
):
    """``_Crews.release`` and ``RepairableRBD._crew_started``: job ``key``
    (a component's, or ``n`` plus a standby unit's) is done at ``t``, and
    its crew starts the next waiting job, if any, which ends as late as it
    waited: a component's end is queued (a unit off line for a test not
    ageing meanwhile), a unit's repair queued by its group. The heap's new
    size, or -1 if it has no room."""
    holding[key] = 0
    if crew[1] == 0:
        crew[0] += 1
        return size
    w, due, done, kind, waiting = _start_waiting(queue, crew[1])
    crew[1] = waiting
    holding[w] = 1
    wait = t - due
    ends = done + wait
    if w < n:
        if not np.isnan(pending[w]):
            pending[w] = pending[w] + wait
        if ends < t_end:
            if size == heap[0].size:
                return -1
            size = _push(heap, size, ends, w, kind, 0)
        return size
    group_node, group_first, group_count, unit_group, units, groups, tags = (
        plan
    )
    u = w - n
    g = unit_group[u]
    _unit_event(u, g, ends, _UNIT_REPAIRED, units, groups)
    if groups[4][g] < 0 or ends < groups[5][g]:
        return _arm(
            g,
            group_node[g],
            t_end,
            group_first,
            group_count,
            units,
            groups,
            heap,
            size,
            tags,
        )
    return size


@njit(cache=True, inline="always")
def _draw(s, counter, i, limit, flat, base):
    """The next draw of stream ``s``, the ``counter[i]``-th: 0 and the
    draw, or the stream's code (``s + 1``) if the simulation has none
    left."""
    k = counter[i]
    if k >= limit[s]:
        return s + 1, 0.0
    counter[i] = k + 1
    return 0, flat[base[s] + k]


@njit(cache=True, inline="always")
def _unit_event(u, g, time, kind, units, groups):
    """``_StandbyGroup._queue``: unit ``u``'s next event (a unit has one at
    a time; a new one supersedes it), ordered after those of its group
    queued before."""
    times, orders, kinds, valid = units[2], units[3], units[4], units[5]
    order = groups[0]
    times[u] = time
    orders[u] = order[g]
    kinds[u] = kind
    valid[u] = 1
    order[g] += 1


@njit(cache=True, inline="always")
def _operate(u, g, t, units, groups):
    """``_StandbyGroup._operate``: unit ``u`` operates from ``t``."""
    life, since, operating = units[0], units[1], units[6]
    operating[u] = 1
    groups[1][g] += 1
    since[u] = t
    _unit_event(u, g, t + life[u], _UNIT_FAILS, units, groups)


@njit(cache=True, inline="always")
def _stand_by(u, g, t, dormancy, first, units, groups):
    """``_StandbyGroup._wait``: unit ``u`` joins the spares at ``t``."""
    spares_n, spares = groups[2], groups[3]
    spares[first[g] + spares_n[g]] = u
    spares_n[g] += 1
    units[1][u] = t
    if dormancy[g] > 0.0:
        _unit_event(
            u, g, t + units[0][u] / dormancy[g], _SPARE_FAILS, units, groups
        )


@njit(cache=True, inline="always")
def _group_next(g, first, count, units):
    """The unit of group ``g`` whose event is next (its earliest, of those
    the first queued), or -1 if none has one."""
    times, orders, valid = units[2], units[3], units[5]
    best = -1
    for u in range(first[g], first[g] + count[g]):
        if valid[u] and (
            best < 0
            or times[u] < times[best]
            or (times[u] == times[best] and orders[u] < orders[best])
        ):
            best = u
    return best


@njit(cache=True, inline="always")
def _arm(g, c, t_end, first, count, units, groups, heap, size, tags):
    """``_StandbyGroup._arm``: queue group ``g``'s next event in the heap,
    as node ``c``'s, under a new tag (an earlier entry there is left, and
    skipped); none if it falls at or after the end. The heap's new size,
    or -1 if it has no room."""
    best = _group_next(g, first, count, units)
    entry_tag, entry_time = groups[4], groups[5]
    if best < 0 or units[2][best] >= t_end:
        entry_tag[g] = -1
        return size
    if size == heap[0].size:
        return -1
    tags[0] += 1
    entry_tag[g] = tags[0]
    entry_time[g] = units[2][best]
    return _push(heap, size, units[2][best], c, _GROUP, tags[0])


@njit(cache=True)
def truth_table(n, structure):
    """``_works`` for every state of ``n`` components: entry ``mask`` is
    whether the system works when component ``c`` is up exactly if bit
    ``c`` of ``mask`` is set."""
    table = np.empty(1 << n, np.int8)
    status = np.empty(n, np.int8)
    value = np.empty(structure[0].size, np.int8)
    for mask in range(1 << n):
        for c in range(n):
            status[c] = (mask >> c) & 1
        table[mask] = _works(status, structure, value)
    return table


@njit(cache=True, inline="always")
def _up_kind(kind):
    """Whether an event of ``kind`` leaves its component up."""
    return (
        kind == _RESTORE
        or kind == _PM_END
        or kind == _PM_IN_PLACE
        or kind == _TEST_UP
    )


@njit(cache=True)
def _first(c, upkeep, fail, counters, pending, limit, flat, base):
    """Component ``c``'s first event, new at 0 (``RepairableRBD.
    initialize_event_queue``): its failure, maintenance or test. A status
    code (a stream's number + 1 if it ran out of draws), the event's time
    and its kind."""
    lives = counters[0]
    if upkeep[0][c]:
        return _renewal(c, 0.0, upkeep, fail, lives, limit, flat, base)
    code, life = _draw(fail[c], lives, c, limit, flat, base)
    if code != 0:
        return code, 0.0, _FAIL
    if upkeep[5][c] > 0.0:
        pending[c] = 0.0 + life
        t, kind = _tested_next(c, 0.0, upkeep, pending)
        return 0, t, kind
    return 0, life, _FAIL


@njit(cache=True)
def _crew_follow(
    c,
    t,
    up_now,
    t_next,
    kind_next,
    t_end,
    crews,
    rank,
    pending,
    plan,
    heap,
    size,
):
    """The repair crews (``RepairableRBD._crew_follow_up``) after
    component ``c``'s event at ``t``: a crew that has finished starts the
    next waiting job; a job that falls due takes a free crew or waits. The
    heap's new size (-1 if it has no room) and whether the next event
    waits for a crew."""
    crew, holding, queue = crews
    n = pending.size
    if up_now:
        if holding[c]:
            size = _release(
                c, t, t_end, n, crew, holding, queue, pending, plan, heap, size
            )
        return size, False
    if _up_kind(kind_next):
        if crew[0] > 0:
            crew[0] -= 1
            holding[c] = 1
            return size, False
        crew[1] = _wait(
            queue, crew[1], rank[c], t, crew[2], c, t_next, kind_next
        )
        crew[2] += 1
        return size, True
    return size, False


@njit(cache=True)
def _group_step(
    g, c, t, t_end, limited, crews, rank, pending, standby, draws, heap, size
):
    """``_StandbyGroup.advance``: standby group ``g`` (node ``c``) takes its
    next event at ``t`` -- a unit failing, or one repaired -- and queues its
    next in the heap. A status code, whether it is up after it, whether a
    unit failed (charged as a repair) and the heap's new size."""
    (
        group_node,
        group_first,
        group_count,
        group_k,
        dormancy,
        switching,
        switch,
        unit_fail,
        unit_repair,
        unit_group,
        units,
        groups,
        tags,
    ) = standby
    limit, flat, base = draws
    crew, holding, queue = crews
    n = pending.size
    groups[4][g] = -1
    u = _group_next(g, group_first, group_count, units)
    units[5][u] = 0
    unit_kind = units[4][u]
    broken = False
    if unit_kind == _UNIT_REPAIRED:
        if limited:
            plan = (
                group_node,
                group_first,
                group_count,
                unit_group,
                units,
                groups,
                tags,
            )
            size = _release(
                n + u,
                t,
                t_end,
                n,
                crew,
                holding,
                queue,
                pending,
                plan,
                heap,
                size,
            )
            if size < 0:
                return -1, False, False, size
        code, life = _draw(unit_fail[u], units[7], u, limit, flat, base)
        if code != 0:
            return code, False, False, size
        units[0][u] = life
        if groups[1][g] < group_k[g]:
            _operate(u, g, t, units, groups)
        else:
            _stand_by(u, g, t, dormancy, group_first, units, groups)
    else:
        if unit_kind == _SPARE_FAILS:
            # Out of the spares.
            j = group_first[g]
            while groups[3][j] != u:
                j += 1
            end = group_first[g] + groups[2][g] - 1
            while j < end:
                groups[3][j] = groups[3][j + 1]
                j += 1
            groups[2][g] -= 1
        else:
            units[6][u] = 0
            groups[1][g] -= 1
        broken = True
        # Its repair, drawn now: a job for a crew.
        code, span = _draw(unit_repair[u], units[8], u, limit, flat, base)
        if code != 0:
            return code, False, False, size
        if not limited or crew[0] > 0:
            if limited:
                crew[0] -= 1
                holding[n + u] = 1
            _unit_event(u, g, t + span, _UNIT_REPAIRED, units, groups)
        else:
            crew[1] = _wait(
                queue,
                crew[1],
                rank[n + u],
                t,
                crew[2],
                n + u,
                t + span,
                _UNIT_JOB,
            )
            crew[2] += 1
        if unit_kind == _UNIT_FAILS and groups[2][g] > 0:
            # The spare that has waited longest is switched in, if the
            # switch works.
            spare = groups[3][group_first[g]]
            switched = switching[g] >= 1.0
            if not switched and switching[g] > 0.0:
                code, uniform = _draw(
                    switch[g], groups[6], g, limit, flat, base
                )
                if code != 0:
                    return code, False, False, size
                switched = uniform < switching[g]
            if switched:
                end = group_first[g] + groups[2][g] - 1
                for j in range(group_first[g], end):
                    groups[3][j] = groups[3][j + 1]
                groups[2][g] -= 1
                # It has used up its life at the dormant rate so far (all
                # of it, at most).
                units[5][spare] = 0
                used = dormancy[g] * (t - units[1][spare])
                units[0][spare] = max(units[0][spare] - used, 0.0)
                _operate(spare, g, t, units, groups)
    size = _arm(
        g,
        c,
        t_end,
        group_first,
        group_count,
        units,
        groups,
        heap,
        size,
        tags,
    )
    if size < 0:
        return -1, False, False, size
    return 0, groups[1][g] == group_k[g], broken, size


@njit(cache=True)
def _start_group(g, c, t_end, standby, draws, heap, size):
    """``_StandbyGroup.__init__``: group ``g``'s units, new, the first k
    operating, and its first event queued as node ``c``'s. A status code
    and the heap's new size."""
    (
        _,
        group_first,
        group_count,
        group_k,
        dormancy,
        _,
        _,
        unit_fail,
        _,
        _,
        units,
        groups,
        tags,
    ) = standby
    limit, flat, base = draws
    for u in range(group_first[g], group_first[g] + group_count[g]):
        code, life = _draw(unit_fail[u], units[7], u, limit, flat, base)
        if code != 0:
            return code, size
        units[0][u] = life
        if u - group_first[g] < group_k[g]:
            _operate(u, g, 0.0, units, groups)
        else:
            _stand_by(u, g, 0.0, dormancy, group_first, units, groups)
    size = _arm(
        g,
        c,
        t_end,
        group_first,
        group_count,
        units,
        groups,
        heap,
        size,
        tags,
    )
    if size < 0:
        return -1, size
    return 0, size


@njit(cache=True, inline="always")
def _level_view(L, state):
    """Nested RBD (level) ``L``'s heap and repair crews (its free crews,
    which components hold one and the jobs waiting), from ``state``."""
    heaps, crew2, holding, queue2 = state[3], state[5], state[7], state[6]
    heap = (heaps[0][L], heaps[1][L], heaps[2][L], heaps[3][L])
    queue = (
        queue2[0][L],
        queue2[1][L],
        queue2[2][L],
        queue2[3][L],
        queue2[4][L],
        queue2[5][L],
    )
    return heap, (crew2[L], holding, queue)


@njit(cache=True)
def _advance(root, t_end, upkeep, fail, repair, draws, state):
    """``RepairableRBD.next_event`` for nested RBD ``root`` (a level of its
    own): take its events in time order until its system changes, and
    return that change: a status code, its time, the new state and whether
    it is a planned outage. With none before the end, the end and the
    state it is left in. A nested RBD of its own steps the same way when
    its next change is wanted, so the levels on the way down wait on a
    stack (numba's cache keeps no recursive functions). A component's next
    event is written out as in ``_simulate``, for the same reason."""
    (
        status,
        counters,
        pending,
        _,
        sizes,
        _,
        _,
        _,
        standby,
        level_up,
        level_mask,
        stack,
        awaiting,
        pending_change,
    ) = state
    level_crews, rank = upkeep[14], upkeep[15]
    group_of = upkeep[16]
    child_level, level_start = upkeep[27], upkeep[29]
    level_table_start, level_table = upkeep[31], upkeep[32]
    changed_, new_, time_, planned_ = pending_change
    policy, duration = upkeep[0], upkeep[2]
    tested, test_offset, coverage, partial = upkeep[5:9]
    per_full, test_time, test_draw = upkeep[9:12]
    lives, repairs, maintained, timed, finding = counters
    limit, flat, base = draws
    plan = (
        standby[0],
        standby[1],
        standby[2],
        standby[9],
        standby[10],
        standby[11],
        standby[12],
    )
    entry_tag = standby[11][4]
    stack[0] = root
    depth = 1
    returned = False
    rt = 0.0
    rs = 0
    rp = False
    L = root
    heap, crews = _level_view(L, state)
    while depth > 0:
        if stack[depth - 1] != L:
            L = stack[depth - 1]
            heap, crews = _level_view(L, state)
        limited = level_crews[L] >= 0
        size = sizes[L]
        if returned:
            # The change of the nested RBD whose event L was taking: its
            # next event, queued; and L's own change, if that event made
            # one.
            returned = False
            c = awaiting[L]
            awaiting[L] = -1
            if rs:
                kind_next = _RESTORE
            elif rp:
                kind_next = _PM_START
            else:
                kind_next = _FAIL
            if rt < t_end:
                if size == heap[0].size:
                    return -1, 0.0, 0, False
                size = _push(heap, size, rt, c, kind_next, 0)
                sizes[L] = size
            if changed_[L]:
                changed_[L] = 0
                level_up[L] = new_[L]
                rt = time_[L]
                rs = new_[L]
                rp = planned_[L] != 0
                depth -= 1
                returned = True
                continue
        if size == 0:
            # No change before the end: its queue is used up.
            rt = t_end
            rs = level_up[L]
            rp = False
            depth -= 1
            returned = True
            continue
        t, c, kind_now, tag_now, size = _pop(heap, size)
        sizes[L] = size
        bit = 1 << (c - level_start[L])
        if kind_now == _GROUP:
            g = group_of[c]
            if tag_now != entry_tag[g]:
                continue  # superseded by an earlier one
            code, now, _, size = _group_step(
                g,
                c,
                t,
                t_end,
                limited,
                crews,
                rank,
                pending,
                standby,
                draws,
                heap,
                size,
            )
            sizes[L] = size
            if code != 0:
                return code, 0.0, 0, False
            state_now = 1 if now else 0
            if state_now == status[c]:
                continue
            status[c] = state_now
            if state_now:
                level_mask[L] |= bit
            else:
                level_mask[L] &= ~bit
            if state_now != level_up[L]:
                new = level_table[level_table_start[L] + level_mask[L]]
                if new != level_up[L]:
                    level_up[L] = new
                    rt = t
                    rs = new
                    rp = False
                    depth -= 1
                    returned = True
            continue
        up_now = _up_kind(kind_now)
        state_now = 1 if up_now else 0
        status[c] = state_now
        if state_now:
            level_mask[L] |= bit
        else:
            level_mask[L] &= ~bit
        changed = False
        new = level_up[L]
        if state_now != level_up[L]:
            new = level_table[level_table_start[L] + level_mask[L]]
            changed = new != level_up[L]
        planned = (
            changed and new == 0 and _PM_START <= kind_now <= _TEST_MISSED
        )
        child = child_level[c]
        if child >= 0:
            # A nested RBD's next change, from its own events.
            awaiting[L] = c
            changed_[L] = 1 if changed else 0
            new_[L] = new
            time_[L] = t
            planned_[L] = 1 if planned else 0
            stack[depth] = child
            depth += 1
            continue
        # The component's next event (RepairableRBD._follow_up), as in
        # _simulate.
        if tested[c] > 0.0:
            if kind_now == _FAIL:
                pending[c] = np.nan
                t_next = _test_finding(t, tested[c], test_offset[c])
                kind_next = _TEST_DOWN
                if not _full_test(
                    t_next, partial[c], tested[c], test_offset[c], per_full[c]
                ):
                    code, uniform = _draw(
                        test_draw[c], finding, c, limit, flat, base
                    )
                    if code != 0:
                        return code, 0.0, 0, False
                    if not uniform < coverage[c]:
                        kind_next = _TEST_MISSED
            elif kind_now == _TEST_MISSED:
                t_next = _test_due(t, tested[c], test_offset[c])
                kind_next = _TEST_MISSED
                if _full_test(
                    t_next, partial[c], tested[c], test_offset[c], per_full[c]
                ):
                    kind_next = _TEST_DOWN
            elif kind_now == _TEST_UP:
                t_next, kind_next = _tested_next(c, t, upkeep, pending)
            elif kind_now == _TEST_DOWN:
                span = 0.0
                if test_time[c] >= 0:
                    code, span = _draw(
                        test_time[c], timed, c, limit, flat, base
                    )
                    if code != 0:
                        return code, 0.0, 0, False
                if np.isnan(pending[c]):
                    code, took = _draw(
                        repair[c], repairs, c, limit, flat, base
                    )
                    if code != 0:
                        return code, 0.0, 0, False
                    t_next = t + span + took
                    kind_next = _RESTORE
                else:
                    pending[c] = pending[c] + span
                    t_next = t + span
                    kind_next = _TEST_UP
            else:
                code, life = _draw(fail[c], lives, c, limit, flat, base)
                if code != 0:
                    return code, 0.0, 0, False
                pending[c] = t + life
                t_next, kind_next = _tested_next(c, t, upkeep, pending)
        elif kind_now == _FAIL:
            code, took = _draw(repair[c], repairs, c, limit, flat, base)
            if code != 0:
                return code, 0.0, 0, False
            t_next = t + took
            kind_next = _RESTORE
        elif kind_now == _PM_START:
            code, took = _draw(duration[c], maintained, c, limit, flat, base)
            if code != 0:
                return code, 0.0, 0, False
            t_next = t + took
            kind_next = _PM_END
        elif policy[c]:
            code, t_next, kind_next = _renewal(
                c, t, upkeep, fail, lives, limit, flat, base
            )
            if code != 0:
                return code, 0.0, 0, False
        else:
            code, life = _draw(fail[c], lives, c, limit, flat, base)
            if code != 0:
                return code, 0.0, 0, False
            t_next = t + life
            kind_next = _FAIL
        waits = False
        if limited:
            size, waits = _crew_follow(
                c,
                t,
                up_now,
                t_next,
                kind_next,
                t_end,
                crews,
                rank,
                pending,
                plan,
                heap,
                size,
            )
            sizes[L] = size
            if size < 0:
                return -1, 0.0, 0, False
        if not waits and t_next < t_end:
            if size == heap[0].size:
                return -1, 0.0, 0, False
            size = _push(heap, size, t_next, c, kind_next, 0)
            sizes[L] = size
        if changed:
            level_up[L] = new
            rt = t
            rs = new
            rp = planned
            depth -= 1
            returned = True
    return 0, rt, rs, rp


@njit(cache=True)
def _start_level(L, t_end, upkeep, fail, repair, draws, state):
    """``RepairableRBD.initialize_event_queue`` for nested RBD (level)
    ``L``, from new: its components' first events in order -- a nested
    RBD's first change, once it has started (its levels start before the
    levels they are in, see ``_simulate``) -- then its standby groups' and
    its system's state. A status code."""
    status, counters, pending = state[0], state[1], state[2]
    sizes, standby, level_up, level_mask = (
        state[4],
        state[8],
        state[9],
        state[10],
    )
    limit, flat, base = draws
    group_of = upkeep[16]
    child_level, _, level_start, level_stop, level_table_start = upkeep[27:32]
    level_table = upkeep[32]
    heap, _ = _level_view(L, state)
    size = 0
    for c in range(level_start[L], level_stop[L]):
        if group_of[c] >= 0:
            continue
        child = child_level[c]
        if child >= 0:
            status[c] = level_up[child]
            code, t, now, planned = _advance(
                child, t_end, upkeep, fail, repair, draws, state
            )
            if now:
                kind = _RESTORE
            elif planned:
                kind = _PM_START
            else:
                kind = _FAIL
        else:
            code, t, kind = _first(
                c, upkeep, fail, counters, pending, limit, flat, base
            )
        if code != 0:
            return code
        if t < t_end:
            size = _push(heap, size, t, c, kind, 0)
    for c in range(level_start[L], level_stop[L]):
        g = group_of[c]
        if g >= 0:
            code, size = _start_group(g, c, t_end, standby, draws, heap, size)
            if code != 0:
                return code
    sizes[L] = size
    mask = 0
    for c in range(level_start[L], level_stop[L]):
        if status[c]:
            mask |= 1 << (c - level_start[L])
    level_mask[L] = mask
    level_up[L] = level_table[level_table_start[L] + mask]
    return 0


@njit(cache=True)
def _simulate(
    todo, lo, hi, first, t_end, system, structure, kept, draws, out, upkeep
):
    """Simulations ``todo[lo:hi]`` (each into row ``r - first``). Whether
    the system works is looked up in the table of every state, or, without
    one, kept up to date as components change (``kept``, see
    ``_compiled._kept``). ``upkeep`` is the components' preventive
    maintenance and inspections, the repair crews, the standby groups and
    the nested RBDs (see ``_compiled._System.upkeep``): the components of
    nested RBDs come after the system's own, each nested RBD a level with
    its own heap and crews, stepped to its next change by ``_advance``.
    The system's own events are taken here, written out rather than
    through the helpers ``_advance`` shares: their arguments' reference
    counts would cost twice the time."""
    (
        start,
        active,
        fail,
        repair,
        slot_start,
        slot_end,
        slot_category,
        slot_stream,
        slot_value,
        cost_index,
        has_rate,
        rate,
        has_costs,
        system_rate,
        initial_up,
        table,
    ) = system
    flat, offsets, rows, first_block, columns = draws
    tabled = table.size > 0
    (
        uptime_out,
        node_out,
        counts_out,
        system_out,
        change_count,
        change_times,
        change_deltas,
        cost_out,
        category_out,
        node_cost_out,
        status_out,
    ) = out
    # The system's own components, and every level's.
    n = node_out.shape[1]
    n_all = start.size
    streams = columns.size
    room = change_times.shape[1]
    base = np.empty(streams, np.int64)
    limit = np.empty(streams, np.int64)
    status = np.empty(n_all, np.int8)
    lives = np.empty(n_all, np.int64)
    repairs = np.empty(n_all, np.int64)
    charged = np.empty(slot_category.size, np.int64)
    policy, _, duration, pm_charge, pm_amount = upkeep[:5]
    (
        tested,
        test_offset,
        coverage,
        partial,
        per_full,
        test_time,
        test_draw,
        test_charge,
        test_amount,
    ) = upkeep[5:14]
    # The repair crews (-1 for as many as needed) and each component's rank
    # in their queue, and which components hold one, the jobs waiting for
    # one (see ``_wait``) and how many were ever queued.
    level_crews, rank = upkeep[14], upkeep[15]
    crews = level_crews[0]
    # The standby groups (#155; see ``_compiled._System.upkeep``).
    (
        group_of,
        group_node,
        group_first,
        group_count,
        group_k,
        dormancy,
        switching,
        switch,
        unit_fail,
        unit_repair,
        unit_group,
    ) = upkeep[16:27]
    # The nested RBDs: each nested node's level, each level's components,
    # and the nested levels, innermost first.
    child_level, level_start, level_stop = upkeep[27], upkeep[29], upkeep[30]
    level_order = upkeep[33]
    levels = level_start.size
    m = unit_fail.size
    # Each unit's life left (as of when it last started operating or
    # waiting), that time, its next event's time, order and kind and
    # whether it has one, whether it operates, and its draws of lives and
    # of repairs.
    units = (
        np.empty(m),
        np.empty(m),
        np.empty(m),
        np.empty(m, np.int64),
        np.empty(m, np.int8),
        np.empty(m, np.int8),
        np.empty(m, np.int8),
        np.empty(m, np.int64),
        np.empty(m, np.int64),
    )
    # Each group's events queued so far, units operating, spares (in the
    # order they joined, from the group's first unit's place), entry in
    # the heap (its tag, -1 for none, and time) and draws of switches; and
    # the last tag given.
    g_count = group_count.size
    groups = (
        np.empty(g_count, np.int64),
        np.empty(g_count, np.int64),
        np.empty(g_count, np.int64),
        np.empty(m, np.int64),
        np.empty(g_count, np.int64),
        np.empty(g_count),
        np.empty(g_count, np.int64),
    )
    tags = np.zeros(1, np.int64)
    plan = (
        group_node,
        group_first,
        group_count,
        unit_group,
        units,
        groups,
        tags,
    )
    # Each level's free crews, jobs waiting and jobs ever queued; which
    # components (and units) hold a crew; and each level's jobs waiting
    # (see ``_wait``).
    crew2 = np.empty((levels, 3), np.int64)
    holding = np.empty(n_all + m, np.int8)
    queue2 = (
        np.empty((levels, n_all + m)),
        np.empty((levels, n_all + m)),
        np.empty((levels, n_all + m), np.int64),
        np.empty((levels, n_all + m), np.int64),
        np.empty((levels, n_all + m)),
        np.empty((levels, n_all + m), np.int8),
    )
    maintained = np.empty(n_all, np.int64)
    pm_charged = np.empty(n_all, np.int64)
    # Each unit with hidden failures: when it is due to fail (NaN once it
    # has failed), and its draws of test times, of tests' finding and of
    # inspection charges.
    pending = np.empty(n_all)
    timed = np.empty(n_all, np.int64)
    finding = np.empty(n_all, np.int64)
    test_charged = np.empty(n_all, np.int64)
    counters = (lives, repairs, maintained, timed, finding)
    stream_draws = (limit, flat, base)
    standby = (
        group_node,
        group_first,
        group_count,
        group_k,
        dormancy,
        switching,
        switch,
        unit_fail,
        unit_repair,
        unit_group,
        units,
        groups,
        tags,
    )
    # Each level's heap: one event a component, and room for standby
    # groups' superseded entries (with the room for the system's changes,
    # which grows when either runs out).
    widest = 0
    for level in range(levels):
        widest = max(widest, level_stop[level] - level_start[level])
    cap = widest + 1 + room
    heaps = (
        np.empty((levels, cap)),
        np.empty((levels, cap), np.int64),
        np.empty((levels, cap), np.int8),
        np.empty((levels, cap), np.int64),
    )
    sizes = np.zeros(levels, np.int64)
    # Each level's system state and its components' states as bits; and
    # the levels stepping to their next change (see ``_advance``): their
    # stack, the nested node whose event each is taking, and whether that
    # event changed it, to what, when and if planned.
    level_up = np.empty(levels, np.int8)
    level_mask = np.zeros(levels, np.int64)
    stack = np.empty(levels, np.int64)
    awaiting = np.empty(levels, np.int64)
    pending_change = (
        np.zeros(levels, np.int8),
        np.zeros(levels, np.int8),
        np.zeros(levels),
        np.zeros(levels, np.int8),
    )
    shared = (
        status,
        counters,
        pending,
        heaps,
        sizes,
        crew2,
        queue2,
        holding,
        standby,
        level_up,
        level_mask,
        stack,
        awaiting,
        pending_change,
    )
    # The system's own heap and crews.
    heap, (crew, _, queue) = _level_view(0, shared)
    last = np.empty(n)
    up_at = np.empty(n)
    down_at = np.empty(n)
    value0, count0, down0, working_paths0 = kept[9:]
    root, always = structure[6], structure[7]
    value = np.empty(value0.size, np.int8)
    count = np.empty(count0.size, np.int32)
    down = np.empty(down0.size, np.int32)
    for i in range(lo, hi):
        r = todo[i]
        row = r - first
        # Where this simulation's draws are: its column of each stream.
        for s in range(streams):
            width = columns[s]
            block = r // width - first_block[s]
            limit[s] = rows[s, block]
            base[s] = offsets[s, block] + (r % width) * limit[s]
        node_out[row] = 0.0
        counts_out[row] = 0
        category_out[row] = 0.0
        node_cost_out[row] = 0.0
        status[:] = start
        # Which components are up, as bits, for the truth table (kept only
        # with a table: shifting by 64 bits or more is undefined).
        mask = 0
        if tabled:
            for c in range(n):
                if start[c]:
                    mask |= 1 << c
        else:
            value[:] = value0
            count[:] = count0
            down[:] = down0
        working_paths = working_paths0
        lives[:] = 0
        repairs[:] = 0
        charged[:] = 0
        maintained[:] = 0
        pm_charged[:] = 0
        pending[:] = np.nan
        timed[:] = 0
        finding[:] = 0
        test_charged[:] = 0
        for level in range(levels):
            crew2[level, 0] = level_crews[level]
            crew2[level, 1] = 0
            crew2[level, 2] = 0
        holding[:] = 0
        sizes[:] = 0
        level_up[:] = 1
        awaiting[:] = -1
        pending_change[0][:] = 0
        units[5][:] = 0
        units[6][:] = 0
        units[7][:] = 0
        units[8][:] = 0
        groups[0][:] = 0
        groups[1][:] = 0
        groups[2][:] = 0
        groups[4][:] = -1
        groups[6][:] = 0
        tags[0] = 0
        last[:] = 0.0
        up_at[:] = 0.0
        down_at[:] = 0.0
        failures = 0
        restorations = 0
        planned = 0
        changes = 0
        rep_cost = 0.0
        code = 0
        size = 0
        # The nested RBDs start first, innermost first: each starts from
        # new on its own, and only its first change is wanted where it is.
        for j in range(level_order.size):
            code = _start_level(
                level_order[j],
                t_end,
                upkeep,
                fail,
                repair,
                stream_draws,
                shared,
            )
            if code != 0:
                break
        # Each component's first failure (or maintenance, or test, or a
        # nested RBD's first change), in the components' order.
        for c in range(n):
            if code != 0:
                break
            if active[c] and group_of[c] < 0:
                kind_next = _FAIL
                if child_level[c] >= 0:
                    status[c] = level_up[child_level[c]]
                    code, t, now, was_planned = _advance(
                        child_level[c],
                        t_end,
                        upkeep,
                        fail,
                        repair,
                        stream_draws,
                        shared,
                    )
                    if code != 0:
                        break
                    if now:
                        kind_next = _RESTORE
                    elif was_planned:
                        kind_next = _PM_START
                elif policy[c]:
                    code, t, kind_next = _renewal(
                        c, 0.0, upkeep, fail, lives, limit, flat, base
                    )
                    if code != 0:
                        break
                elif tested[c] > 0.0:
                    s = fail[c]
                    k = lives[c]
                    if k >= limit[s]:
                        code = s + 1
                        break
                    lives[c] = k + 1
                    pending[c] = 0.0 + flat[base[s] + k]
                    t, kind_next = _tested_next(c, 0.0, upkeep, pending)
                else:
                    s = fail[c]
                    k = lives[c]
                    if k >= limit[s]:
                        code = s + 1
                        break
                    lives[c] = k + 1
                    t = flat[base[s] + k]
                if t < t_end:
                    size = _push(heap, size, t, c, kind_next, 0)
        # Each standby group's units, new, the first k operating, and the
        # group's first event (``_StandbyGroup.__init__``).
        for c in range(n):
            if code != 0:
                break
            g = group_of[c]
            if not active[c] or g < 0:
                continue
            for u in range(group_first[g], group_first[g] + group_count[g]):
                code, life = _draw(
                    unit_fail[u], units[7], u, limit, flat, base
                )
                if code != 0:
                    break
                units[0][u] = life
                if u - group_first[g] < group_k[g]:
                    _operate(u, g, 0.0, units, groups)
                else:
                    _stand_by(u, g, 0.0, dormancy, group_first, units, groups)
            if code != 0:
                break
            size = _arm(
                g,
                c,
                t_end,
                group_first,
                group_count,
                units,
                groups,
                heap,
                size,
                tags,
            )
            if size < 0:
                code = -1
                break
        up = initial_up
        system_up = 0.0
        system_down = 0.0
        since = 0.0
        while code == 0 and size > 0:
            t, c, kind_now, tag_now, size = _pop(heap, size)
            grouped = False
            if kind_now == _GROUP:
                # A standby group's own event (RepairableRBD._StandbyGroup.
                # advance): a unit failure, charged as a repair, or a
                # repair's end.
                g = group_of[c]
                if tag_now != groups[4][g]:
                    continue  # superseded by an earlier one
                groups[4][g] = -1
                u = _group_next(g, group_first, group_count, units)
                units[5][u] = 0
                unit_kind = units[4][u]
                broken = False
                if unit_kind == _UNIT_REPAIRED:
                    if crews >= 0:
                        size = _release(
                            n_all + u,
                            t,
                            t_end,
                            n_all,
                            crew,
                            holding,
                            queue,
                            pending,
                            plan,
                            heap,
                            size,
                        )
                        if size < 0:
                            code = -1
                            break
                    code, life = _draw(
                        unit_fail[u], units[7], u, limit, flat, base
                    )
                    if code != 0:
                        break
                    units[0][u] = life
                    if groups[1][g] < group_k[g]:
                        _operate(u, g, t, units, groups)
                    else:
                        _stand_by(
                            u, g, t, dormancy, group_first, units, groups
                        )
                else:
                    if unit_kind == _SPARE_FAILS:
                        # Out of the spares.
                        j = group_first[g]
                        while groups[3][j] != u:
                            j += 1
                        end = group_first[g] + groups[2][g] - 1
                        while j < end:
                            groups[3][j] = groups[3][j + 1]
                            j += 1
                        groups[2][g] -= 1
                    else:
                        units[6][u] = 0
                        groups[1][g] -= 1
                    broken = True
                    # Its repair, drawn now: a job for a crew.
                    code, span = _draw(
                        unit_repair[u], units[8], u, limit, flat, base
                    )
                    if code != 0:
                        break
                    if crews < 0 or crew[0] > 0:
                        if crews >= 0:
                            crew[0] -= 1
                            holding[n_all + u] = 1
                        _unit_event(
                            u, g, t + span, _UNIT_REPAIRED, units, groups
                        )
                    else:
                        crew[1] = _wait(
                            queue,
                            crew[1],
                            rank[n_all + u],
                            t,
                            crew[2],
                            n_all + u,
                            t + span,
                            _UNIT_JOB,
                        )
                        crew[2] += 1
                    if unit_kind == _UNIT_FAILS and groups[2][g] > 0:
                        # The spare that has waited longest is switched in,
                        # if the switch works.
                        spare = groups[3][group_first[g]]
                        switched = switching[g] >= 1.0
                        if not switched and switching[g] > 0.0:
                            code, uniform = _draw(
                                switch[g], groups[6], g, limit, flat, base
                            )
                            if code != 0:
                                break
                            switched = uniform < switching[g]
                        if switched:
                            end = group_first[g] + groups[2][g] - 1
                            for j in range(group_first[g], end):
                                groups[3][j] = groups[3][j + 1]
                            groups[2][g] -= 1
                            # It has used up its life at the dormant rate so
                            # far (all of it, at most).
                            units[5][spare] = 0
                            used = dormancy[g] * (t - units[1][spare])
                            units[0][spare] = max(units[0][spare] - used, 0.0)
                            _operate(spare, g, t, units, groups)
                size = _arm(
                    g,
                    c,
                    t_end,
                    group_first,
                    group_count,
                    units,
                    groups,
                    heap,
                    size,
                    tags,
                )
                if size < 0:
                    code = -1
                    break
                if broken:
                    for q in range(slot_start[c], slot_end[c]):
                        s = slot_stream[q]
                        if s >= 0:
                            k = charged[q]
                            if k >= limit[s]:
                                code = s + 1
                                break
                            charged[q] = k + 1
                            charge = flat[base[s] + k]
                        else:
                            charge = slot_value[q]
                        rep_cost += charge
                        category_out[row, slot_category[q]] += charge
                        node_cost_out[row, cost_index[c]] += charge
                    if code != 0:
                        break
                up_now = groups[1][g] == group_k[g]
                if up_now == status[c]:
                    continue
                grouped = True
            else:
                up_now = (
                    kind_now == _RESTORE
                    or kind_now == _PM_END
                    or kind_now == _PM_IN_PLACE
                    or kind_now == _TEST_UP
                )
            if not grouped and kind_now >= _PM_START and up_now == status[c]:
                # No change of state: maintenance in zero time of a working
                # unit (renewed in place), a test in zero time of a working
                # unit, or a test of a failed one, which finds its failure
                # (charged its repair now) or misses it.
                if kind_now <= _PM_IN_PLACE:
                    q = pm_charge[c]
                    if q > -2:
                        if q == -1:
                            charge = pm_amount[c]
                        else:
                            k = pm_charged[c]
                            if k >= limit[q]:
                                code = q + 1
                                break
                            pm_charged[c] = k + 1
                            charge = flat[base[q] + k]
                        rep_cost += charge
                        category_out[row, 2] += charge
                        node_cost_out[row, cost_index[c]] += charge
                else:
                    q = test_charge[c]
                    if q > -2:
                        if q == -1:
                            charge = test_amount[c]
                        else:
                            k = test_charged[c]
                            if k >= limit[q]:
                                code = q + 1
                                break
                            test_charged[c] = k + 1
                            charge = flat[base[q] + k]
                        rep_cost += charge
                        category_out[row, 3] += charge
                        node_cost_out[row, cost_index[c]] += charge
                    if not up_now and kind_now != _TEST_MISSED:
                        # The failure found: its corrective charges, summed
                        # and then added, as the Python loop adds them.
                        found_cost = 0.0
                        for q in range(slot_start[c], slot_end[c]):
                            s = slot_stream[q]
                            if s >= 0:
                                k = charged[q]
                                if k >= limit[s]:
                                    code = s + 1
                                    break
                                charged[q] = k + 1
                                charge = flat[base[s] + k]
                            else:
                                charge = slot_value[q]
                            category_out[row, slot_category[q]] += charge
                            node_cost_out[row, cost_index[c]] += charge
                            found_cost += charge
                        if code != 0:
                            break
                        rep_cost += found_cost
            else:
                state = 1 if up_now else 0
                if up:
                    system_up_t = system_up + (t - since)
                    system_down_t = system_down
                else:
                    system_up_t = system_up
                    system_down_t = system_down + (t - since)
                if status[c]:
                    node_out[row, c, 0] += t - last[c]
                    node_out[row, c, 1] += system_up_t - up_at[c]
                else:
                    node_out[row, c, 2] += system_down_t - down_at[c]
                last[c] = t
                up_at[c] = system_up_t
                down_at[c] = system_down_t
                status[c] = state
                if tabled:
                    if state:
                        mask |= 1 << c
                    else:
                        mask &= ~(1 << c)
                else:
                    working_paths = _keep(
                        kept, value, count, down, c, state, working_paths
                    )
                if state:
                    counts_out[row, 2, c] += 1
                elif kind_now == _PM_START:
                    # A planned outage: charged the preventive cost.
                    q = pm_charge[c]
                    if q > -2:
                        if q == -1:
                            charge = pm_amount[c]
                        else:
                            k = pm_charged[c]
                            if k >= limit[q]:
                                code = q + 1
                                break
                            pm_charged[c] = k + 1
                            charge = flat[base[q] + k]
                        rep_cost += charge
                        category_out[row, 2] += charge
                        node_cost_out[row, cost_index[c]] += charge
                elif kind_now == _TEST_DOWN:
                    # A test that takes a working unit off line.
                    q = test_charge[c]
                    if q > -2:
                        if q == -1:
                            charge = test_amount[c]
                        else:
                            k = test_charged[c]
                            if k >= limit[q]:
                                code = q + 1
                                break
                            test_charged[c] = k + 1
                            charge = flat[base[q] + k]
                        rep_cost += charge
                        category_out[row, 3] += charge
                        node_cost_out[row, cost_index[c]] += charge
                else:
                    counts_out[row, 0, c] += 1
                    # A hidden failure is charged when a test finds it, a
                    # standby group's at each unit's.
                    if tested[c] == 0.0 and not grouped:
                        for q in range(slot_start[c], slot_end[c]):
                            s = slot_stream[q]
                            if s >= 0:
                                k = charged[q]
                                if k >= limit[s]:
                                    code = s + 1
                                    break
                                charged[q] = k + 1
                                charge = flat[base[s] + k]
                            else:
                                charge = slot_value[q]
                            rep_cost += charge
                            category_out[row, slot_category[q]] += charge
                            node_cost_out[row, cost_index[c]] += charge
                        if code != 0:
                            break
                if tabled:
                    works = table[mask]
                elif always:
                    works = 1
                elif root >= 0:
                    works = value[root]
                else:
                    works = 1 if working_paths > 0 else 0
                if state != up and works != up:
                    system_up = system_up_t
                    system_down = system_down_t
                    since = t
                    up = 1 - up
                    if changes == room:
                        code = -1
                        break
                    change_times[row, changes] = t
                    if up:
                        change_deltas[row, changes] = 1
                        restorations += 1
                        counts_out[row, 3, c] += 1
                    else:
                        change_deltas[row, changes] = -1
                        if _PM_START <= kind_now <= _TEST_MISSED:
                            planned += 1
                        else:
                            failures += 1
                            counts_out[row, 1, c] += 1
                    changes += 1
                if grouped:
                    continue  # the group has queued its next event
            # The next event: a nested RBD's next change, from its own
            # events (no crew works on a nested RBD); or the component's.
            child = child_level[c]
            if child >= 0:
                code, t_next, now, was_planned = _advance(
                    child, t_end, upkeep, fail, repair, stream_draws, shared
                )
                if code != 0:
                    break
                if t_next < t_end:
                    if now:
                        kind_next = _RESTORE
                    elif was_planned:
                        kind_next = _PM_START
                    else:
                        kind_next = _FAIL
                    if size == heap[0].size:
                        code = -1
                        break
                    size = _push(heap, size, t_next, c, kind_next, 0)
                continue
            if tested[c] > 0.0:
                # A unit with hidden failures (RepairableRBD.
                # _inspected_follow_up).
                if kind_now == _FAIL:
                    # Failed, unseen: found by the first test at or after
                    # its failure, unless that one can miss it and does.
                    pending[c] = np.nan
                    t_next = _test_finding(t, tested[c], test_offset[c])
                    kind_next = _TEST_DOWN
                    if not _full_test(
                        t_next,
                        partial[c],
                        tested[c],
                        test_offset[c],
                        per_full[c],
                    ):
                        s = test_draw[c]
                        k = finding[c]
                        if k >= limit[s]:
                            code = s + 1
                            break
                        finding[c] = k + 1
                        if not flat[base[s] + k] < coverage[c]:
                            kind_next = _TEST_MISSED
                elif kind_now == _TEST_MISSED:
                    # Missed: so is the next test, unless it is a full one.
                    t_next = _test_due(t, tested[c], test_offset[c])
                    kind_next = _TEST_MISSED
                    if _full_test(
                        t_next,
                        partial[c],
                        tested[c],
                        test_offset[c],
                        per_full[c],
                    ):
                        kind_next = _TEST_DOWN
                elif kind_now == _TEST_UP:
                    t_next, kind_next = _tested_next(c, t, upkeep, pending)
                elif kind_now == _TEST_DOWN:
                    # A test that takes time: of a failed unit, repaired
                    # once it is done; of a working one, off line (and not
                    # ageing) until then.
                    span = 0.0
                    if test_time[c] >= 0:
                        s = test_time[c]
                        k = timed[c]
                        if k >= limit[s]:
                            code = s + 1
                            break
                        timed[c] = k + 1
                        span = flat[base[s] + k]
                    if np.isnan(pending[c]):
                        s = repair[c]
                        k = repairs[c]
                        if k >= limit[s]:
                            code = s + 1
                            break
                        repairs[c] = k + 1
                        t_next = t + span + flat[base[s] + k]
                        kind_next = _RESTORE
                    else:
                        pending[c] = pending[c] + span
                        t_next = t + span
                        kind_next = _TEST_UP
                else:
                    # Repaired: as new, from t.
                    s = fail[c]
                    k = lives[c]
                    if k >= limit[s]:
                        code = s + 1
                        break
                    lives[c] = k + 1
                    pending[c] = t + flat[base[s] + k]
                    t_next, kind_next = _tested_next(c, t, upkeep, pending)
            elif kind_now == _FAIL:
                s = repair[c]
                k = repairs[c]
                if k >= limit[s]:
                    code = s + 1
                    break
                repairs[c] = k + 1
                t_next = t + flat[base[s] + k]
                kind_next = _RESTORE
            elif kind_now == _PM_START:
                s = duration[c]
                k = maintained[c]
                if k >= limit[s]:
                    code = s + 1
                    break
                maintained[c] = k + 1
                t_next = t + flat[base[s] + k]
                kind_next = _PM_END
            elif policy[c]:
                code, t_next, kind_next = _renewal(
                    c, t, upkeep, fail, lives, limit, flat, base
                )
                if code != 0:
                    break
            else:
                s = fail[c]
                k = lives[c]
                if k >= limit[s]:
                    code = s + 1
                    break
                lives[c] = k + 1
                t_next = t + flat[base[s] + k]
                kind_next = _FAIL
            if crews >= 0:
                # The repair crews (RepairableRBD._crew_follow_up).
                if up_now:
                    if holding[c]:
                        # Its job done, its crew starts the next waiting
                        # one, if any.
                        size = _release(
                            c,
                            t,
                            t_end,
                            n_all,
                            crew,
                            holding,
                            queue,
                            pending,
                            plan,
                            heap,
                            size,
                        )
                        if size < 0:
                            code = -1
                            break
                elif (
                    kind_next == _RESTORE
                    or kind_next == _PM_END
                    or kind_next == _PM_IN_PLACE
                    or kind_next == _TEST_UP
                ):
                    # A job falls due: started by a free crew, or waiting.
                    if crew[0] > 0:
                        crew[0] -= 1
                        holding[c] = 1
                    else:
                        crew[1] = _wait(
                            queue,
                            crew[1],
                            rank[c],
                            t,
                            crew[2],
                            c,
                            t_next,
                            kind_next,
                        )
                        crew[2] += 1
                        continue
            if t_next < t_end:
                if size == heap[0].size:
                    code = -1
                    break
                size = _push(heap, size, t_next, c, kind_next, 0)
        status_out[row] = code
        if code != 0:
            continue
        if up:
            system_up += t_end - since
        else:
            system_down += t_end - since
        for c in range(n):
            if status[c]:
                node_out[row, c, 0] += t_end - last[c]
                node_out[row, c, 1] += system_up - up_at[c]
            else:
                node_out[row, c, 2] += system_down - down_at[c]
        for c in range(n):
            if has_rate[c]:
                charge = rate[c] * (t_end - node_out[row, c, 0])
                rep_cost += charge
                category_out[row, 4] += charge
                node_cost_out[row, cost_index[c]] += charge
        if has_costs:
            charge = system_rate * (t_end - system_up)
            rep_cost += charge
            category_out[row, 5] += charge
        uptime_out[row] = system_up
        cost_out[row] = rep_cost
        system_out[row, 0] = failures
        system_out[row, 1] = restorations
        system_out[row, 2] = planned
        change_count[row] = changes


@njit(cache=True, nogil=True)
def run_serial(
    todo, first, t_end, system, structure, kept, draws, out, upkeep
):
    """Simulations ``todo``, one after another (without holding the GIL,
    so that threads can run several at once)."""
    _simulate(
        todo,
        0,
        todo.size,
        first,
        t_end,
        system,
        structure,
        kept,
        draws,
        out,
        upkeep,
    )


def run_parallel(
    pool,
    todo,
    first,
    t_end,
    system,
    structure,
    kept,
    draws,
    out,
    chunks,
    upkeep,
):
    """Simulations ``todo``, in ``chunks`` pieces on the threads of
    ``pool``: the one compiled loop of ``run_serial``, which numba
    compiles once (a loop over ``prange`` would be compiled again, as
    long). Each simulation writes its own row, so the pieces run in any
    order."""
    m = todo.size
    pieces = [
        pool.submit(
            run_serial,
            todo[j * m // chunks : (j + 1) * m // chunks],
            first,
            t_end,
            system,
            structure,
            kept,
            draws,
            out,
            upkeep,
        )
        for j in range(chunks)
    ]
    for piece in pieces:
        piece.result()


def threads(jobs) -> int:
    """The threads a run with ``n_jobs`` resolved to ``jobs`` uses (numba's
    own limit at most)."""
    if jobs is None or jobs <= 1:
        return 1
    return int(min(jobs, numba.config.NUMBA_NUM_THREADS))


def used() -> bool:
    """Whether the loop has been compiled (or loaded) in this process."""
    return bool(run_serial.signatures)
