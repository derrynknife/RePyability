"""The compiled event loop of ``RepairableRBD`` simulations (numba).

``_simulate`` is ``RepairableRBD._replicate`` for components that are
plain ``NonRepairable`` units, under age or block replacement, inspected
for hidden failures or neither, and with as many repair crews as needed or
fewer (#155), operation for operation, over arrays: the same heap
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
from numba import njit, prange

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


@njit(cache=True, inline="always")
def _push(times, nodes, states, size, t, c, state):
    """``heapq.heappush`` of ``(t, c, state)``, ordered by ``t`` alone."""
    pos = size
    while pos > 0:
        parent = (pos - 1) >> 1
        if t < times[parent]:
            times[pos] = times[parent]
            nodes[pos] = nodes[parent]
            states[pos] = states[parent]
            pos = parent
        else:
            break
    times[pos] = t
    nodes[pos] = c
    states[pos] = state
    return size + 1


@njit(cache=True, inline="always")
def _pop(times, nodes, states, size):
    """``heapq.heappop``: the first item, and the heap's new size."""
    t0, c0, state0 = times[0], nodes[0], states[0]
    size -= 1
    if size > 0:
        t, c, state = times[size], nodes[size], states[size]
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
            pos = child
            child = 2 * pos + 1
        while pos > 0:
            parent = (pos - 1) >> 1
            if t < times[parent]:
                times[pos] = times[parent]
                nodes[pos] = nodes[parent]
                states[pos] = states[parent]
                pos = parent
            else:
                break
        times[pos] = t
        nodes[pos] = c
        states[pos] = state
    return t0, c0, state0, size


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


@njit(cache=True)
def _simulate(
    todo, lo, hi, first, t_end, system, structure, kept, draws, out, upkeep
):
    """Simulations ``todo[lo:hi]`` (each into row ``r - first``). Whether
    the system works is looked up in the table of every state, or, without
    one, kept up to date as components change (``kept``, see
    ``_compiled._kept``). ``upkeep`` is the components' preventive
    maintenance and inspections, and the repair crews (see
    ``_compiled._System.upkeep``)."""
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
    n = start.size
    streams = columns.size
    room = change_times.shape[1]
    base = np.empty(streams, np.int64)
    limit = np.empty(streams, np.int64)
    status = np.empty(n, np.int8)
    heap_times = np.empty(n + 1)
    heap_nodes = np.empty(n + 1, np.int64)
    heap_states = np.empty(n + 1, np.int8)
    lives = np.empty(n, np.int64)
    repairs = np.empty(n, np.int64)
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
    crews, rank = upkeep[14], upkeep[15]
    holding = np.empty(n, np.int8)
    queue = (
        np.empty(n),
        np.empty(n),
        np.empty(n, np.int64),
        np.empty(n, np.int64),
        np.empty(n),
        np.empty(n, np.int8),
    )
    maintained = np.empty(n, np.int64)
    pm_charged = np.empty(n, np.int64)
    # Each unit with hidden failures: when it is due to fail (NaN once it
    # has failed), and its draws of test times, of tests' finding and of
    # inspection charges.
    pending = np.empty(n)
    timed = np.empty(n, np.int64)
    finding = np.empty(n, np.int64)
    test_charged = np.empty(n, np.int64)
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
        free = crews
        holding[:] = 0
        waiting = 0
        queued = 0
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
        # Each component's first failure (or maintenance, or test), in the
        # components' order.
        for c in range(n):
            if active[c]:
                kind_next = _FAIL
                if policy[c]:
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
                    size = _push(
                        heap_times,
                        heap_nodes,
                        heap_states,
                        size,
                        t,
                        c,
                        kind_next,
                    )
        up = initial_up
        system_up = 0.0
        system_down = 0.0
        since = 0.0
        while code == 0 and size > 0:
            t, c, kind_now, size = _pop(
                heap_times, heap_nodes, heap_states, size
            )
            up_now = (
                kind_now == _RESTORE
                or kind_now == _PM_END
                or kind_now == _PM_IN_PLACE
                or kind_now == _TEST_UP
            )
            if kind_now >= _PM_START and up_now == status[c]:
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
                    # A hidden failure is charged when a test finds it.
                    if tested[c] == 0.0:
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
                        if kind_now >= _PM_START:
                            planned += 1
                        else:
                            failures += 1
                            counts_out[row, 1, c] += 1
                    changes += 1
            # The next event.
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
                        # one, if any, ending as late as it waited.
                        holding[c] = 0
                        if waiting == 0:
                            free += 1
                        else:
                            w, due, done, kind_done, waiting = _start_waiting(
                                queue, waiting
                            )
                            holding[w] = 1
                            wait = t - due
                            ends = done + wait
                            # Off line for a test, a unit does not age while
                            # it waits.
                            if not np.isnan(pending[w]):
                                pending[w] = pending[w] + wait
                            if ends < t_end:
                                size = _push(
                                    heap_times,
                                    heap_nodes,
                                    heap_states,
                                    size,
                                    ends,
                                    w,
                                    kind_done,
                                )
                elif (
                    kind_next == _RESTORE
                    or kind_next == _PM_END
                    or kind_next == _PM_IN_PLACE
                    or kind_next == _TEST_UP
                ):
                    # A job falls due: started by a free crew, or waiting.
                    if free > 0:
                        free -= 1
                        holding[c] = 1
                    else:
                        waiting = _wait(
                            queue,
                            waiting,
                            rank[c],
                            t,
                            queued,
                            c,
                            t_next,
                            kind_next,
                        )
                        queued += 1
                        continue
            if t_next < t_end:
                size = _push(
                    heap_times,
                    heap_nodes,
                    heap_states,
                    size,
                    t_next,
                    c,
                    kind_next,
                )
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


@njit(cache=True)
def run_serial(
    todo, first, t_end, system, structure, kept, draws, out, upkeep
):
    """Simulations ``todo``, one after another."""
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


@njit(cache=True, parallel=True)
def run_parallel(
    todo, first, t_end, system, structure, kept, draws, out, chunks, upkeep
):
    """Simulations ``todo``, ``chunks`` of them at a time on numba's
    threads."""
    m = todo.size
    for j in prange(chunks):
        _simulate(
            todo,
            j * m // chunks,
            (j + 1) * m // chunks,
            first,
            t_end,
            system,
            structure,
            kept,
            draws,
            out,
            upkeep,
        )


def threads(jobs) -> int:
    """The threads a run with ``n_jobs`` resolved to ``jobs`` uses (numba's
    own limit at most)."""
    if jobs is None or jobs <= 1:
        return 1
    return int(min(jobs, numba.config.NUMBA_NUM_THREADS))


def set_threads(count: int) -> None:
    numba.set_num_threads(count)


def get_threads() -> int:
    return int(numba.get_num_threads())


def used() -> bool:
    """Whether the loop has been compiled (or loaded) in this process."""
    return bool(run_serial.signatures or run_parallel.signatures)
