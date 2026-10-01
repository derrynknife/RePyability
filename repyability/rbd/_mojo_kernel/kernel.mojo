"""The compiled event loop of ``RepairableRBD`` simulations, in Mojo.

``_simulate`` is ``_kernel._simulate`` (numba), and so
``RepairableRBD._replicate``, operation for operation: the same heap
(``heapq``'s algorithm, so equal times come out in the same order), the same
arithmetic in the same order, and the draws read from the same keyed
streams. Its results are the same to the last bit.

Two things differ in how, not in what, it works out:

- whether the system works is kept up to date as components change, rather
  than worked out afresh: each term of the structure (see ``modular``)
  keeps how many of its members work, and a change goes up the tree only as
  far as it changes a term; a core keeps how many members of each of its
  path sets are down, and how many path sets work. An event costs the depth
  of the tree at most, rather than the size of the structure;
- every array is read through a raw pointer, laid out by ``_mojo``, and the
  draws where the streams' blocks hold them, rather than copied together.

The loop releases the GIL, so ``_mojo`` runs it on several Python threads at
once, each on a share of the batch.
"""

from std.os import abort
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from std.python.bindings import PythonModuleBuilder

comptime F64P = Pointer[Float64, MutAnyOrigin]
comptime I64P = Pointer[Int64, MutAnyOrigin]
comptime I32P = Pointer[Int32, MutAnyOrigin]
comptime I8P = Pointer[Int8, MutAnyOrigin]

# The kinds of term of the structure (see ``modular``).
comptime NODE = 0
comptime SERIES = 1
comptime PARALLEL = 2

# The positions of the run's arrays in the address table ``_mojo`` passes
# (``_mojo.ADDRESSES``, which ``layout`` lets it check).
comptime A_START = 0
comptime A_ACTIVE = 1
comptime A_FAIL = 2
comptime A_REPAIR = 3
comptime A_SLOT_START = 4
comptime A_SLOT_END = 5
comptime A_SLOT_CATEGORY = 6
comptime A_SLOT_STREAM = 7
comptime A_SLOT_VALUE = 8
comptime A_COST_INDEX = 9
comptime A_HAS_RATE = 10
comptime A_RATE = 11
comptime A_BLOCKS_AT = 12
comptime A_ROWS = 13
comptime A_FIRST_BLOCK = 14
comptime A_COLUMNS = 15
comptime A_UPTIME = 16
comptime A_NODE = 17
comptime A_COUNTS = 18
comptime A_SYSTEM = 19
comptime A_CHANGE_COUNT = 20
comptime A_CHANGE_TIMES = 21
comptime A_CHANGE_DELTAS = 22
comptime A_COST = 23
comptime A_CATEGORY = 24
comptime A_NODE_COST = 25
comptime A_STATUS = 26
comptime A_TODO = 27
comptime A_TERM_KIND = 28
comptime A_TERM_NEED = 29
comptime A_TERM_PARENT = 30
comptime A_NODE_TERM_START = 31
comptime A_NODE_TERM_END = 32
comptime A_NODE_TERMS = 33
comptime A_TERM_PATH_START = 34
comptime A_TERM_PATH_END = 35
comptime A_TERM_PATHS = 36
comptime A_VALUE0 = 37
comptime A_COUNT0 = 38
comptime A_DOWN0 = 39
comptime ADDRESSES = 40

# The positions of the run's sizes in the table of sizes.
comptime S_N = 0
comptime S_STREAMS = 1
comptime S_ROOM = 2
comptime S_SLOTS = 3
comptime S_BLOCKS = 4
comptime S_FIRST = 5
comptime S_HAS_COSTS = 6
comptime S_INITIAL_UP = 7
comptime S_TERMS = 8
comptime S_PATHS = 9
comptime S_ROOT = 10
comptime S_ALWAYS = 11
comptime S_COSTED = 12
comptime S_WORKING_PATHS0 = 13
comptime SIZES = 14


@always_inline
def f64(table: I64P, i: Int) -> F64P:
    return F64P(unsafe_from_address=Int(table[unsafe_offset=i]))


@always_inline
def i64(table: I64P, i: Int) -> I64P:
    return I64P(unsafe_from_address=Int(table[unsafe_offset=i]))


@always_inline
def i32(table: I64P, i: Int) -> I32P:
    return I32P(unsafe_from_address=Int(table[unsafe_offset=i]))


@always_inline
def i8(table: I64P, i: Int) -> I8P:
    return I8P(unsafe_from_address=Int(table[unsafe_offset=i]))


@always_inline
def draw(base: I64P, s: Int, k: Int64) -> Float64:
    """Draw ``k`` of stream ``s``: ``base`` holds the address of the
    simulation's draws in each stream's block, read where they are."""
    return F64P(unsafe_from_address=Int(base[unsafe_offset=s] + 8 * k))[]


@always_inline
def push(
    times: F64P,
    nodes: I32P,
    states: I8P,
    size: Int,
    t: Float64,
    c: Int,
    state: Int8,
) -> Int:
    """``heapq.heappush`` of ``(t, c, state)``, ordered by ``t`` alone."""
    var pos = size
    while pos > 0:
        var parent = (pos - 1) >> 1
        var pt = times[unsafe_offset=parent]
        if t < pt:
            times[unsafe_offset=pos] = pt
            nodes[unsafe_offset=pos] = nodes[unsafe_offset=parent]
            states[unsafe_offset=pos] = states[unsafe_offset=parent]
            pos = parent
        else:
            break
    times[unsafe_offset=pos] = t
    nodes[unsafe_offset=pos] = Int32(c)
    states[unsafe_offset=pos] = state
    return size + 1


@always_inline
def pop(times: F64P, nodes: I32P, states: I8P, size_in: Int) -> Int:
    """``heapq.heappop`` (the caller has read the first item first): the
    heap's new size."""
    var size = size_in - 1
    if size > 0:
        var t = times[unsafe_offset=size]
        var c = nodes[unsafe_offset=size]
        var state = states[unsafe_offset=size]
        # Move the smaller child up until reaching a leaf, then the last
        # item up from there (heapq's _siftup and _siftdown).
        var pos = 0
        var child = 1
        while child < size:
            var right = child + 1
            if right < size and not (
                times[unsafe_offset=child] < times[unsafe_offset=right]
            ):
                child = right
            times[unsafe_offset=pos] = times[unsafe_offset=child]
            nodes[unsafe_offset=pos] = nodes[unsafe_offset=child]
            states[unsafe_offset=pos] = states[unsafe_offset=child]
            pos = child
            child = 2 * pos + 1
        while pos > 0:
            var parent = (pos - 1) >> 1
            var pt = times[unsafe_offset=parent]
            if t < pt:
                times[unsafe_offset=pos] = pt
                nodes[unsafe_offset=pos] = nodes[unsafe_offset=parent]
                states[unsafe_offset=pos] = states[unsafe_offset=parent]
                pos = parent
            else:
                break
        times[unsafe_offset=pos] = t
        nodes[unsafe_offset=pos] = c
        states[unsafe_offset=pos] = state
    return size


def simulate(
    addresses: I64P,
    sizes: I64P,
    t_end: Float64,
    system_rate: Float64,
    lo: Int,
    hi: Int,
    scratch_at: Int,
):
    """Simulations ``todo[lo:hi]`` (each into row ``r - first``), with the
    working memory at ``scratch_at``."""
    var start = i8(addresses, A_START)
    var active = i8(addresses, A_ACTIVE)
    var fail = i64(addresses, A_FAIL)
    var repair = i64(addresses, A_REPAIR)
    var slot_start = i64(addresses, A_SLOT_START)
    var slot_end = i64(addresses, A_SLOT_END)
    var slot_category = i64(addresses, A_SLOT_CATEGORY)
    var slot_stream = i64(addresses, A_SLOT_STREAM)
    var slot_value = f64(addresses, A_SLOT_VALUE)
    var cost_index = i64(addresses, A_COST_INDEX)
    var has_rate = i8(addresses, A_HAS_RATE)
    var rate = f64(addresses, A_RATE)
    var blocks_at = i64(addresses, A_BLOCKS_AT)
    var rows = i64(addresses, A_ROWS)
    var first_block = i64(addresses, A_FIRST_BLOCK)
    var columns = i64(addresses, A_COLUMNS)
    var uptime_out = f64(addresses, A_UPTIME)
    var node_out = f64(addresses, A_NODE)
    var counts_out = i64(addresses, A_COUNTS)
    var system_out = i64(addresses, A_SYSTEM)
    var change_count = i64(addresses, A_CHANGE_COUNT)
    var change_times = f64(addresses, A_CHANGE_TIMES)
    var change_deltas = i8(addresses, A_CHANGE_DELTAS)
    var cost_out = f64(addresses, A_COST)
    var category_out = f64(addresses, A_CATEGORY)
    var node_cost_out = f64(addresses, A_NODE_COST)
    var status_out = i64(addresses, A_STATUS)
    var todo = i64(addresses, A_TODO)
    var term_kind = i8(addresses, A_TERM_KIND)
    var term_need = i32(addresses, A_TERM_NEED)
    var term_parent = i32(addresses, A_TERM_PARENT)
    var node_term_start = i32(addresses, A_NODE_TERM_START)
    var node_term_end = i32(addresses, A_NODE_TERM_END)
    var node_terms = i32(addresses, A_NODE_TERMS)
    var term_path_start = i32(addresses, A_TERM_PATH_START)
    var term_path_end = i32(addresses, A_TERM_PATH_END)
    var term_paths = i32(addresses, A_TERM_PATHS)
    var value0 = i8(addresses, A_VALUE0)
    var count0 = i32(addresses, A_COUNT0)
    var down0 = i32(addresses, A_DOWN0)

    var n = Int(sizes[unsafe_offset=S_N])
    var streams = Int(sizes[unsafe_offset=S_STREAMS])
    var room = Int(sizes[unsafe_offset=S_ROOM])
    var slots = Int(sizes[unsafe_offset=S_SLOTS])
    var blocks = Int(sizes[unsafe_offset=S_BLOCKS])
    var first = Int(sizes[unsafe_offset=S_FIRST])
    var has_costs = sizes[unsafe_offset=S_HAS_COSTS] != 0
    var initial_up = Int8(sizes[unsafe_offset=S_INITIAL_UP])
    var terms = Int(sizes[unsafe_offset=S_TERMS])
    var paths = Int(sizes[unsafe_offset=S_PATHS])
    var root = Int(sizes[unsafe_offset=S_ROOT])
    var always = sizes[unsafe_offset=S_ALWAYS] != 0
    var costed = Int(sizes[unsafe_offset=S_COSTED])
    var working_paths0 = Int(sizes[unsafe_offset=S_WORKING_PATHS0])

    # Working memory, carved out of the thread's scratch (8-byte items
    # first, then 4-byte, then 1-byte, so each is aligned).
    var at = scratch_at
    var base = I64P(unsafe_from_address=at)
    at += 8 * streams
    var limit = I64P(unsafe_from_address=at)
    at += 8 * streams
    var heap_times = F64P(unsafe_from_address=at)
    at += 8 * (n + 1)
    var lives = I64P(unsafe_from_address=at)
    at += 8 * n
    var repairs = I64P(unsafe_from_address=at)
    at += 8 * n
    var charged = I64P(unsafe_from_address=at)
    at += 8 * slots
    var last = F64P(unsafe_from_address=at)
    at += 8 * n
    var up_at = F64P(unsafe_from_address=at)
    at += 8 * n
    var down_at = F64P(unsafe_from_address=at)
    at += 8 * n
    var heap_nodes = I32P(unsafe_from_address=at)
    at += 4 * (n + 1)
    var count = I32P(unsafe_from_address=at)
    at += 4 * terms
    var down = I32P(unsafe_from_address=at)
    at += 4 * paths
    var status = I8P(unsafe_from_address=at)
    at += n
    var heap_states = I8P(unsafe_from_address=at)
    at += n + 1
    var value = I8P(unsafe_from_address=at)

    for i in range(lo, hi):
        var r = Int(todo[unsafe_offset=i])
        var row = r - first
        var nb = row * 3 * n
        var cb = row * 4 * n
        var gb = row * 6
        var kb = row * costed
        var tb = row * room
        # Where this simulation's draws are: its column of each stream.
        for s in range(streams):
            var width = Int(columns[unsafe_offset=s])
            var block = r // width - Int(first_block[unsafe_offset=s])
            var l = rows[unsafe_offset=s * blocks + block]
            limit[unsafe_offset=s] = l
            base[unsafe_offset=s] = (
                blocks_at[unsafe_offset=s * blocks + block]
                + Int64(r % width) * l * 8
            )
        for j in range(3 * n):
            node_out[unsafe_offset=nb + j] = 0.0
        for j in range(4 * n):
            counts_out[unsafe_offset=cb + j] = 0
        for j in range(6):
            category_out[unsafe_offset=gb + j] = 0.0
        for j in range(costed):
            node_cost_out[unsafe_offset=kb + j] = 0.0
        for c in range(n):
            status[unsafe_offset=c] = start[unsafe_offset=c]
            lives[unsafe_offset=c] = 0
            repairs[unsafe_offset=c] = 0
            last[unsafe_offset=c] = 0.0
            up_at[unsafe_offset=c] = 0.0
            down_at[unsafe_offset=c] = 0.0
        for q in range(slots):
            charged[unsafe_offset=q] = 0
        for j in range(terms):
            value[unsafe_offset=j] = value0[unsafe_offset=j]
            count[unsafe_offset=j] = count0[unsafe_offset=j]
        for j in range(paths):
            down[unsafe_offset=j] = down0[unsafe_offset=j]
        var working_paths = working_paths0
        var failures: Int64 = 0
        var restorations: Int64 = 0
        var changes = 0
        var rep_cost: Float64 = 0.0
        var code: Int64 = 0
        var size = 0
        # Each component's first failure, in the components' order.
        for c in range(n):
            if active[unsafe_offset=c] != 0:
                var s = Int(fail[unsafe_offset=c])
                var k = lives[unsafe_offset=c]
                if k >= limit[unsafe_offset=s]:
                    code = Int64(s + 1)
                    break
                lives[unsafe_offset=c] = k + 1
                var t = draw(base, s, k)
                if t < t_end:
                    size = push(
                        heap_times, heap_nodes, heap_states, size, t, c, 0
                    )
        var up = initial_up
        var system_up: Float64 = 0.0
        var system_down: Float64 = 0.0
        var since: Float64 = 0.0
        while code == 0 and size > 0:
            var t = heap_times[unsafe_offset=0]
            var c = Int(heap_nodes[unsafe_offset=0])
            var state = heap_states[unsafe_offset=0]
            size = pop(heap_times, heap_nodes, heap_states, size)
            var system_up_t: Float64
            var system_down_t: Float64
            if up != 0:
                system_up_t = system_up + (t - since)
                system_down_t = system_down
            else:
                system_up_t = system_up
                system_down_t = system_down + (t - since)
            var cell = nb + 3 * c
            if status[unsafe_offset=c] != 0:
                node_out[unsafe_offset=cell + 0] += t - last[unsafe_offset=c]
                node_out[unsafe_offset=cell + 1] += (
                    system_up_t - up_at[unsafe_offset=c]
                )
            else:
                node_out[unsafe_offset=cell + 2] += (
                    system_down_t - down_at[unsafe_offset=c]
                )
            last[unsafe_offset=c] = t
            up_at[unsafe_offset=c] = system_up_t
            down_at[unsafe_offset=c] = system_down_t
            status[unsafe_offset=c] = state
            # Each term the component stands for, and up the tree as far as
            # the change goes.
            for e in range(
                Int(node_term_start[unsafe_offset=c]),
                Int(node_term_end[unsafe_offset=c]),
            ):
                var term = Int(node_terms[unsafe_offset=e])
                value[unsafe_offset=term] = state
                var works = state
                while True:
                    var parent = Int(term_parent[unsafe_offset=term])
                    if parent < 0:
                        break
                    var members = count[unsafe_offset=parent]
                    if works != 0:
                        members += 1
                    else:
                        members -= 1
                    count[unsafe_offset=parent] = members
                    var kind = Int(term_kind[unsafe_offset=parent])
                    var need = term_need[unsafe_offset=parent]
                    var now: Int8
                    if kind == PARALLEL:
                        now = 1 if members > 0 else 0
                    else:
                        # A series term needs all its members, a vote k.
                        now = 1 if members >= need else 0
                    if now == value[unsafe_offset=parent]:
                        term = -1
                        break
                    value[unsafe_offset=parent] = now
                    works = now
                    term = parent
                # A term at the top of the tree changed: the core's path
                # sets it belongs to.
                if term >= 0:
                    for q in range(
                        Int(term_path_start[unsafe_offset=term]),
                        Int(term_path_end[unsafe_offset=term]),
                    ):
                        var p = Int(term_paths[unsafe_offset=q])
                        var d = down[unsafe_offset=p]
                        if works != 0:
                            d -= 1
                            if d == 0:
                                working_paths += 1
                        else:
                            if d == 0:
                                working_paths -= 1
                            d += 1
                        down[unsafe_offset=p] = d
            if state != 0:
                counts_out[unsafe_offset=cb + 2 * n + c] += 1
            else:
                counts_out[unsafe_offset=cb + c] += 1
                for q in range(
                    Int(slot_start[unsafe_offset=c]),
                    Int(slot_end[unsafe_offset=c]),
                ):
                    var s = Int(slot_stream[unsafe_offset=q])
                    var charge: Float64
                    if s >= 0:
                        var k = charged[unsafe_offset=q]
                        if k >= limit[unsafe_offset=s]:
                            code = Int64(s + 1)
                            break
                        charged[unsafe_offset=q] = k + 1
                        charge = draw(base, s, k)
                    else:
                        charge = slot_value[unsafe_offset=q]
                    rep_cost += charge
                    category_out[
                        unsafe_offset=gb + Int(slot_category[unsafe_offset=q])
                    ] += charge
                    node_cost_out[
                        unsafe_offset=kb + Int(cost_index[unsafe_offset=c])
                    ] += charge
                if code != 0:
                    break
            if state != up:
                var system: Int8
                if always:
                    system = 1
                elif root >= 0:
                    system = value[unsafe_offset=root]
                else:
                    system = 1 if working_paths > 0 else 0
                if system != up:
                    system_up = system_up_t
                    system_down = system_down_t
                    since = t
                    up = 1 - up
                    if changes == room:
                        code = -1
                        break
                    change_times[unsafe_offset=tb + changes] = t
                    if up != 0:
                        change_deltas[unsafe_offset=tb + changes] = 1
                        restorations += 1
                        counts_out[unsafe_offset=cb + 3 * n + c] += 1
                    else:
                        change_deltas[unsafe_offset=tb + changes] = -1
                        failures += 1
                        counts_out[unsafe_offset=cb + n + c] += 1
                    changes += 1
            # The next event: a failure after a restoration, a restoration
            # after a failure.
            var s: Int
            var k: Int64
            var next_state: Int8
            if state != 0:
                s = Int(fail[unsafe_offset=c])
                k = lives[unsafe_offset=c]
                lives[unsafe_offset=c] = k + 1
                next_state = 0
            else:
                s = Int(repair[unsafe_offset=c])
                k = repairs[unsafe_offset=c]
                repairs[unsafe_offset=c] = k + 1
                next_state = 1
            if k >= limit[unsafe_offset=s]:
                code = Int64(s + 1)
                break
            var t_next = t + draw(base, s, k)
            if t_next < t_end:
                size = push(
                    heap_times,
                    heap_nodes,
                    heap_states,
                    size,
                    t_next,
                    c,
                    next_state,
                )
        status_out[unsafe_offset=row] = code
        if code != 0:
            continue
        if up != 0:
            system_up += t_end - since
        else:
            system_down += t_end - since
        for c in range(n):
            var cell = nb + 3 * c
            if status[unsafe_offset=c] != 0:
                node_out[unsafe_offset=cell + 0] += (
                    t_end - last[unsafe_offset=c]
                )
                node_out[unsafe_offset=cell + 1] += (
                    system_up - up_at[unsafe_offset=c]
                )
            else:
                node_out[unsafe_offset=cell + 2] += (
                    system_down - down_at[unsafe_offset=c]
                )
        for c in range(n):
            if has_rate[unsafe_offset=c] != 0:
                var charge = rate[unsafe_offset=c] * (
                    t_end - node_out[unsafe_offset=nb + 3 * c]
                )
                rep_cost += charge
                category_out[unsafe_offset=gb + 4] += charge
                node_cost_out[
                    unsafe_offset=kb + Int(cost_index[unsafe_offset=c])
                ] += charge
        if has_costs:
            var charge = system_rate * (t_end - system_up)
            rep_cost += charge
            category_out[unsafe_offset=gb + 5] += charge
        uptime_out[unsafe_offset=row] = system_up
        cost_out[unsafe_offset=row] = rep_cost
        system_out[unsafe_offset=3 * row] = failures
        system_out[unsafe_offset=3 * row + 1] = restorations
        system_out[unsafe_offset=3 * row + 2] = 0
        change_count[unsafe_offset=row] = Int64(changes)


def run(
    addresses: PythonObject,
    sizes: PythonObject,
    t_end: PythonObject,
    system_rate: PythonObject,
    lo: PythonObject,
    hi: PythonObject,
    scratch: PythonObject,
) raises -> PythonObject:
    """Simulations ``todo[lo:hi]``, without the GIL: ``addresses`` and
    ``sizes`` are the addresses of the tables of the run's arrays and
    sizes, and ``scratch`` that of the calling thread's working memory."""
    var table = I64P(unsafe_from_address=Int(py=addresses))
    var numbers = I64P(unsafe_from_address=Int(py=sizes))
    var end = Float64(py=t_end)
    var charge = Float64(py=system_rate)
    var first = Int(py=lo)
    var stop = Int(py=hi)
    var memory = Int(py=scratch)
    var python = Python()
    with GILReleased(python):
        simulate(table, numbers, end, charge, first, stop, memory)
    return PythonObject(None)


def layout(_unused: PythonObject) raises -> PythonObject:
    """The table sizes this build expects, for ``_mojo`` to check."""
    return Python.tuple(PythonObject(ADDRESSES), PythonObject(SIZES))


@export
def PyInit_kernel() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("kernel")
        module.def_function[run]("run")
        module.def_function[layout]("layout")
        return module.finalize()
    except e:
        abort(String("Could not initialise the Mojo kernel: ", e))
