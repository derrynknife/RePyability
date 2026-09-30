"""The compiled event loop of ``RepairableRBD`` simulations (numba).

``_simulate`` is ``RepairableRBD._replicate`` for components that are
plain ``NonRepairable`` units, operation for operation, over arrays: the
same heap (``heapq``'s algorithm, so equal times come out in the same
order), the same arithmetic in the same order, and the draws read from the
same keyed streams (see ``_streams``). Each simulation's results go to its
own row of the output arrays, so the rows can be worked out in any order,
on any number of threads, and a simulation that runs out of draws (or of
room for its system's changes) is simply run again once there is more.
``_compiled`` prepares the arrays and adds the rows to the tally in order.

Importing this module imports numba, which compiles the loop on first use
(or loads it from numba's cache).
"""

import numba
import numpy as np
from numba import njit, prange

# The kinds of term of the structure (see ``modular``).
_NODE, _SERIES, _PARALLEL = 0, 1, 2


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
def _simulate(todo, lo, hi, first, t_end, system, structure, draws, out):
    """Simulations ``todo[lo:hi]`` (each into row ``r - first``)."""
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
    last = np.empty(n)
    up_at = np.empty(n)
    down_at = np.empty(n)
    value = np.empty(structure[0].size, np.int8)
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
        # Which components are up, as bits, for the truth table.
        mask = 0
        for c in range(n):
            if start[c]:
                mask |= 1 << c
        lives[:] = 0
        repairs[:] = 0
        charged[:] = 0
        last[:] = 0.0
        up_at[:] = 0.0
        down_at[:] = 0.0
        failures = 0
        restorations = 0
        changes = 0
        rep_cost = 0.0
        code = 0
        size = 0
        # Each component's first failure, in the components' order.
        for c in range(n):
            if active[c]:
                s = fail[c]
                k = lives[c]
                if k >= limit[s]:
                    code = s + 1
                    break
                lives[c] = k + 1
                t = flat[base[s] + k]
                if t < t_end:
                    size = _push(
                        heap_times, heap_nodes, heap_states, size, t, c, 0
                    )
        up = initial_up
        system_up = 0.0
        system_down = 0.0
        since = 0.0
        while code == 0 and size > 0:
            t, c, state, size = _pop(heap_times, heap_nodes, heap_states, size)
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
            if state:
                mask |= 1 << c
            else:
                mask &= ~(1 << c)
            if state:
                counts_out[row, 2, c] += 1
            else:
                counts_out[row, 0, c] += 1
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
            if (
                state != up
                and (
                    table[mask] if tabled else _works(status, structure, value)
                )
                != up
            ):
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
                    failures += 1
                    counts_out[row, 1, c] += 1
                changes += 1
            # The next event: a failure after a restoration, a restoration
            # after a failure.
            if state:
                s = fail[c]
                k = lives[c]
                lives[c] = k + 1
                next_state = 0
            else:
                s = repair[c]
                k = repairs[c]
                repairs[c] = k + 1
                next_state = 1
            if k >= limit[s]:
                code = s + 1
                break
            t_next = t + flat[base[s] + k]
            if t_next < t_end:
                size = _push(
                    heap_times,
                    heap_nodes,
                    heap_states,
                    size,
                    t_next,
                    c,
                    next_state,
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
        system_out[row, 2] = 0
        change_count[row] = changes


@njit(cache=True)
def run_serial(todo, first, t_end, system, structure, draws, out):
    """Simulations ``todo``, one after another."""
    _simulate(todo, 0, todo.size, first, t_end, system, structure, draws, out)


@njit(cache=True, parallel=True)
def run_parallel(todo, first, t_end, system, structure, draws, out, chunks):
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
            draws,
            out,
        )


def threads(jobs) -> int:
    """The threads a run with ``n_jobs`` resolved to ``jobs`` uses (numba's
    own limit at most)."""
    if jobs is None or jobs <= 1:
        return 1
    return int(min(jobs, numba.config.NUMBA_NUM_THREADS))


def set_threads(count: int) -> None:
    numba.set_num_threads(count)


def used() -> bool:
    """Whether the loop has been compiled (or loaded) in this process."""
    return bool(run_serial.signatures or run_parallel.signatures)
