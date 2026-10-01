"""RePyability's batched structure function and Weibull sampler, in Mojo.

A prototype for issue #116. ``lifetimes`` draws a NonRepairableRBD's system
lifetimes from a block of uniforms, as ``NonRepairableRBD.random`` does:
each component's lifetime from its column (a Weibull quantile), then the
diagram's structure as a program of min/max steps over them (see
``kernels.py``, which builds the program from RePyability's own module
tree).

The rows are worked in blocks of ``BLOCK``: each step of the program runs
over a block's rows at once, so its values stay in cache and its min/max
loops vectorise. The quantile is computed one of three ways (``math``):

- 0: Mojo's own ``log1p`` and ``**``, one row at a time;
- 1: the same, eight rows at a time (SIMD);
- 2: the C library's ``log1p`` and ``pow`` (the functions Python's ``math``
  and Numba call), one row at a time.
"""

from std.ffi import external_call
from std.math import log1p
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.runtime import initialize_runtime

from max.algorithm import parallelize

# The program's steps (kind), as kernels.py writes them.
comptime MIN = 0
comptime MAX = 1
comptime KTH = 2  # the k-th largest of the children
comptime STEP = 3  # a decision-diagram step: max(min(a, b), c)

# Rows a block, and SIMD lanes.
comptime BLOCK = 128
comptime LANES = 8
# The most children a k-out-of-n step may have.
comptime MAX_KTH = 64

comptime F64Ptr = Pointer[Float64, MutAnyOrigin]
comptime I64Ptr = Pointer[Int64, MutAnyOrigin]


@export
def PyInit_rbd_kernel() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("rbd_kernel")
        m.def_function[lifetimes](
            "lifetimes",
            docstring="System lifetimes from uniforms (see kernels.py).",
        )
        m.def_function[weibull]("weibull", docstring="Weibull quantiles.")
        m.def_function[nothing]("nothing", docstring="Calls nothing.")
        return m.finalize()
    except e:
        abort(String("error creating the rbd_kernel module: ", e))


def nothing(x: PythonObject) raises -> PythonObject:
    """For timing a call from Python."""
    return x


def _quantile(u: Float64, alpha: Float64, inv_beta: Float64, math: Int) -> Float64:
    """A Weibull quantile, as surpyval's: alpha * (-log1p(-u)) ** (1 / beta)."""
    if math == 2:
        var e = -external_call["log1p", Float64](-u)
        return alpha * external_call["pow", Float64](e, inv_beta)
    var e = -log1p(-u)
    return alpha * e**inv_beta


def weibull(
    u_addr: PythonObject,
    out_addr: PythonObject,
    n: PythonObject,
    alpha: PythonObject,
    beta: PythonObject,
    math: PythonObject,
) raises -> PythonObject:
    """``n`` Weibull quantiles of the uniforms at ``u_addr``, into
    ``out_addr``, by ``math`` (0: Mojo, scalar; 1: Mojo, SIMD; 2: libm)."""
    var u = u_addr.unsafe_get_as_pointer[DType.float64]()
    var out = out_addr.unsafe_get_as_pointer[DType.float64]()
    var count = Int(py=n)
    var a = Float64(py=alpha)
    var ib = 1.0 / Float64(py=beta)
    var how = Int(py=math)
    var i = 0
    if how == 1:
        while i + LANES <= count:
            var x = u.unsafe_load[width=LANES](i)
            var e = -log1p(-x)
            out.unsafe_store(i, a * e ** SIMD[DType.float64, LANES](ib))
            i += LANES
    while i < count:
        out.unsafe_store(i, _quantile(u.unsafe_load(i), a, ib, how))
        i += 1
    return PythonObject(None)


def _rows(
    u: F64Ptr,
    width: Int,
    first: Int,
    rows: Int,
    alpha: F64Ptr,
    inv_beta: F64Ptr,
    gamma: F64Ptr,
    kind: I64Ptr,
    karg: I64Ptr,
    start: I64Ptr,
    children: I64Ptr,
    n_ops: Int,
    result: Int,
    vals: F64Ptr,
    dest: F64Ptr,
    math: Int,
):
    """Rows ``first`` to ``first + rows - 1`` (at most BLOCK of them):
    ``vals`` is the block's scratch, slot by slot."""
    # The components' lifetimes, column by column.
    for c in range(width):
        var a = alpha.unsafe_load(c)
        var ib = inv_beta.unsafe_load(c)
        var g = gamma.unsafe_load(c)
        var slot = vals.unsafe_offset(c * BLOCK)
        var r = 0
        if math == 1:
            var gather = SIMD[DType.float64, LANES](0.0)
            while r + LANES <= rows:
                comptime for lane in range(LANES):
                    gather[lane] = u.unsafe_load((first + r + lane) * width + c)
                var e = -log1p(-gather)
                slot.unsafe_store(
                    r, a * e ** SIMD[DType.float64, LANES](ib) + g
                )
                r += LANES
        while r < rows:
            var x = u.unsafe_load((first + r) * width + c)
            slot.unsafe_store(r, _quantile(x, a, ib, math) + g)
            r += 1
    # The steps, each over the block's rows.
    var base = width + 3
    var buffer = SIMD[DType.float64, MAX_KTH](0.0)
    for j in range(n_ops):
        var target = vals.unsafe_offset((base + j) * BLOCK)
        var lo = Int(start.unsafe_load(j))
        var hi = Int(start.unsafe_load(j + 1))
        var how = Int(kind.unsafe_load(j))
        if how == MIN or how == MAX:
            var first_child = vals.unsafe_offset(
                Int(children.unsafe_load(lo)) * BLOCK
            )
            for r in range(0, rows, LANES):
                target.unsafe_store(r, first_child.unsafe_load[width=LANES](r))
            for i in range(lo + 1, hi):
                var child = vals.unsafe_offset(
                    Int(children.unsafe_load(i)) * BLOCK
                )
                for r in range(0, rows, LANES):
                    var x = target.unsafe_load[width=LANES](r)
                    var y = child.unsafe_load[width=LANES](r)
                    if how == MIN:
                        target.unsafe_store(r, min(x, y))
                    else:
                        target.unsafe_store(r, max(x, y))
        elif how == STEP:
            var a = vals.unsafe_offset(Int(children.unsafe_load(lo)) * BLOCK)
            var b = vals.unsafe_offset(
                Int(children.unsafe_load(lo + 1)) * BLOCK
            )
            var c = vals.unsafe_offset(
                Int(children.unsafe_load(lo + 2)) * BLOCK
            )
            for r in range(0, rows, LANES):
                var x = a.unsafe_load[width=LANES](r)
                var y = b.unsafe_load[width=LANES](r)
                var z = c.unsafe_load[width=LANES](r)
                target.unsafe_store(r, max(min(x, y), z))
        else:
            # KTH: the k-th largest of the children, row by row.
            var m = hi - lo
            var k = Int(karg.unsafe_load(j))
            for r in range(rows):
                for i in range(m):
                    buffer[i] = vals.unsafe_load(
                        Int(children.unsafe_load(lo + i)) * BLOCK + r
                    )
                # Selection: move the k largest to the front.
                for p in range(k):
                    var best = p
                    for q in range(p + 1, m):
                        if buffer[q] > buffer[best]:
                            best = q
                    var keep = buffer[p]
                    buffer[p] = buffer[best]
                    buffer[best] = keep
                target.unsafe_store(r, buffer[k - 1])
    var answer = vals.unsafe_offset(result * BLOCK)
    for r in range(rows):
        dest.unsafe_store(first + r, answer.unsafe_load(r))


def lifetimes(
    u_addr: PythonObject,
    rows: PythonObject,
    width: PythonObject,
    params: PythonObject,
    program: PythonObject,
    sizes: PythonObject,
    out_addr: PythonObject,
    threads: PythonObject,
    math: PythonObject,
) raises -> PythonObject:
    """System lifetimes of ``rows`` rows of ``width`` uniforms.

    ``params`` holds the addresses of the columns' alpha, 1 / beta and
    gamma; ``program`` those of the steps' kind, k, children start and
    children; ``sizes`` the number of steps and the result's slot.
    """
    initialize_runtime()
    var u = u_addr.unsafe_get_as_pointer[DType.float64]()
    var out = out_addr.unsafe_get_as_pointer[DType.float64]()
    var alpha = params[0].unsafe_get_as_pointer[DType.float64]()
    var inv_beta = params[1].unsafe_get_as_pointer[DType.float64]()
    var gamma = params[2].unsafe_get_as_pointer[DType.float64]()
    var kind = program[0].unsafe_get_as_pointer[DType.int64]()
    var karg = program[1].unsafe_get_as_pointer[DType.int64]()
    var start = program[2].unsafe_get_as_pointer[DType.int64]()
    var children = program[3].unsafe_get_as_pointer[DType.int64]()
    var n_ops = Int(py=sizes[0])
    var result = Int(py=sizes[1])
    var n = Int(py=rows)
    var w = Int(py=width)
    var how = Int(py=math)
    var workers = max(1, Int(py=threads))
    var slots = w + 3 + n_ops
    var blocks = (n + BLOCK - 1) // BLOCK
    var per_worker = (blocks + workers - 1) // workers

    def work(worker: Int) {imm}:
        var scratch = List[Float64](length=slots * BLOCK, fill=0.0)
        var vals = F64Ptr(unsafe_from_address=Int(scratch.unsafe_ptr()))
        for r in range(BLOCK):
            vals.unsafe_store(w * BLOCK + r, -Float64.MAX - Float64.MAX)
            vals.unsafe_store((w + 1) * BLOCK + r, Float64.MAX + Float64.MAX)
            vals.unsafe_store((w + 2) * BLOCK + r, 0.0)
        var block = worker * per_worker
        var last = min(blocks, block + per_worker)
        while block < last:
            var first = block * BLOCK
            _rows(
                u,
                w,
                first,
                min(BLOCK, n - first),
                alpha,
                inv_beta,
                gamma,
                kind,
                karg,
                start,
                children,
                n_ops,
                result,
                vals,
                out,
                how,
            )
            block += 1
        _ = scratch^

    if workers == 1:
        work(0)
    else:
        parallelize(work, workers)
    return PythonObject(None)
