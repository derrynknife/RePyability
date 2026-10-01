"""The Mojo engine of ``RepairableRBD`` simulations: the compiled loop of
``_mojo_kernel/kernel.mojo``, run by ``_compiled.Runner`` as it runs the
numba one, on the same arrays, for the same results to the last bit.

Mojo is an optional dependency (``pip install "repyability[mojo]"``). The
kernel is compiled from its source the first time it is used (some seconds)
and kept, named by a hash of the source, next to it in ``__mojocache__``,
or, where the package cannot be written to, in the user's cache folder; later
processes load it at once.

The loop keeps whether the system works up to date as components change
(see the kernel's docstring), so here the structure is laid out for that:
each term's parent, what it needs of its members, the terms each component
stands for and the core's path sets each top term belongs to, and every
term's state with the components as they start.

The loop releases the GIL: a run on several threads gives each thread of
the runner's pool a share of the batch.
"""

import functools
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

_SOURCE_DIR = Path(__file__).parent / "_mojo_kernel"
_SOURCE = _SOURCE_DIR / "kernel.mojo"
_MODULE = "repyability.rbd._mojo_kernel.kernel"
_lock = threading.Lock()
_kernel: Any = None
#: Why the kernel could not be compiled or loaded in this process, if it
#: could not (it is not tried again).
_failure: Optional[str] = None

# The kinds of term of the structure (see ``modular``).
_NODE, _SERIES, _PARALLEL = 0, 1, 2
#: The tasks a batch on threads is cut into, per thread: the threads take
#: them from the pool in turn, so that they finish together however long
#: each simulation takes.
TASKS_PER_THREAD = 4


def available() -> bool:
    """Whether Mojo is installed (without importing or compiling
    anything)."""
    return importlib.util.find_spec("mojo") is not None or bool(
        shutil.which("mojo")
    )


def require() -> None:
    """Raise an ImportError unless Mojo is installed."""
    if not available():
        raise ImportError(
            "engine='mojo' needs Mojo, an optional dependency: install it "
            "with pip install 'repyability[mojo]'."
        )


def failed() -> bool:
    """Whether the kernel failed to compile or load in this process."""
    return _failure is not None


@functools.lru_cache(maxsize=None)
def _digest() -> str:
    """What the compiled kernel depends on: its source, Mojo's and Python's
    versions, and the CPU (Mojo compiles for the host's features, so a
    cache shared between machines keeps one kernel for each kind)."""
    import platform
    from importlib import metadata

    try:
        mojo = metadata.version("mojo")
    except metadata.PackageNotFoundError:
        mojo = "?"
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as info:
            cpu += next(
                (
                    line
                    for line in info
                    if line.startswith(("flags", "Features"))
                ),
                "",
            )
    except OSError:
        pass
    digest = hashlib.sha256(_SOURCE.read_bytes())
    for part in (mojo, sys.version, platform.machine(), cpu):
        digest.update(b"\0" + part.encode())
    return digest.hexdigest()[:16]


def _target() -> Path:
    return _cache_dir() / f"kernel.hash-{_digest()}.so"


def cached() -> bool:
    """Whether the kernel is compiled already (and so loads at once)."""
    return used() or _target().is_file()


def _cache_dir() -> Path:
    """Where the compiled kernel is kept: next to its source if that can be
    written to, else the user's cache folder."""
    here = _SOURCE_DIR / "__mojocache__"
    if os.access(here if here.is_dir() else _SOURCE_DIR, os.W_OK):
        return here
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return Path(root) / "repyability" / "mojo"


def _compile(target: Path) -> None:
    """Build the kernel into ``target`` (through a temporary file, so that
    another process never loads half a library)."""
    try:
        from mojo.run import subprocess_run_mojo

        def run(args):
            return subprocess_run_mojo(args, capture_output=True)

    except ImportError:  # a mojo on the PATH, without its Python package

        def run(args):
            return subprocess.run(
                ["mojo"] + args, capture_output=True, check=False
            )

    partial = target.with_suffix(f".{os.getpid()}.part.so")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        result = run(
            ["build", str(_SOURCE), "--emit", "shared-lib", "-o", str(partial)]
        )
    except (OSError, RuntimeError) as error:  # no mojo, or nowhere to write
        raise ImportError(
            f"Could not compile the Mojo simulation kernel: {error}"
        ) from error
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise ImportError(
            "Could not compile the Mojo simulation kernel:\n"
            + result.stderr.decode(errors="replace")
        )
    os.replace(partial, target)


def load() -> Any:
    """The compiled kernel, compiled first if need be."""
    global _kernel, _failure
    if _kernel is not None:
        return _kernel
    with _lock:
        if _failure is not None:
            raise ImportError(_failure)
        if _kernel is None:
            require()
            try:
                _kernel = _load()
            except ImportError as error:
                _failure = str(error)
                raise
    return _kernel


def _load() -> Any:
    """Compile the kernel if it is not in the cache, and load it."""
    target = _target()
    if not target.is_file():
        _compile(target)
    spec = importlib.util.spec_from_file_location(_MODULE, str(target))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except OSError as error:  # the Mojo runtime libraries not found
        raise ImportError(
            f"Could not load the Mojo kernel: {error}"
        ) from error
    if tuple(module.layout(None)) != (len(ADDRESSES), len(SIZES)):
        raise ImportError(
            "The compiled Mojo kernel does not match this version of "
            f"RePyability: delete {target} to rebuild it."
        )
    sys.modules[_MODULE] = module
    return module


def used() -> bool:
    """Whether the kernel is loaded in this process."""
    return _kernel is not None


#: The arrays the kernel reads and writes, in the order of its address
#: table (keep in step with the kernel's ``A_`` positions).
ADDRESSES = (
    "start",
    "active",
    "fail",
    "repair",
    "slot_start",
    "slot_end",
    "slot_category",
    "slot_stream",
    "slot_value",
    "cost_index",
    "has_rate",
    "rate",
    "blocks_at",
    "rows",
    "first_block",
    "columns",
    "uptime",
    "node",
    "counts",
    "system",
    "change_count",
    "change_times",
    "change_deltas",
    "cost",
    "category",
    "node_cost",
    "status",
    "todo",
    "term_kind",
    "term_need",
    "term_parent",
    "node_term_start",
    "node_term_end",
    "node_terms",
    "term_path_start",
    "term_path_end",
    "term_paths",
    "value0",
    "count0",
    "down0",
)
#: The sizes, in the order of its table of sizes (the kernel's ``S_``).
SIZES = (
    "n",
    "streams",
    "room",
    "slots",
    "blocks",
    "first",
    "has_costs",
    "initial_up",
    "terms",
    "paths",
    "root",
    "always",
    "costed",
    "working_paths0",
)


class Structure:
    """The structure function laid out to be kept up to date (see the
    module docstring), for components that start as ``start``."""

    def __init__(self, structure: tuple, start: np.ndarray):
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
        terms, n = kind.size, start.size
        parent = np.full(terms, -1, np.int32)
        need = np.zeros(terms, np.int32)
        for i in range(terms):
            members = children[child_start[i] : child_end[i]]
            parent[members] = i
            if kind[i] == _SERIES:
                need[i] = members.size
            elif kind[i] not in (_NODE, _PARALLEL):
                need[i] = k[i]
        # The terms each component stands for (a repeated node, more than
        # one).
        owners: list = [[] for _ in range(n)]
        for i in range(terms):
            if kind[i] == _NODE:
                owners[node[i]].append(i)
        self.node_term_start, self.node_term_end, self.node_terms = _csr(
            owners
        )
        # The core's path sets each term belongs to.
        paths = core_start.size
        belongs: list = [[] for _ in range(terms)]
        for p in range(paths):
            for j in range(core_start[p], core_end[p]):
                belongs[core_members[j]].append(p)
        self.term_path_start, self.term_path_end, self.term_paths = _csr(
            belongs
        )
        # Every term's state, with the components as they start.
        value = np.zeros(terms, np.int8)
        count = np.zeros(terms, np.int32)
        for i in range(terms):
            if kind[i] == _NODE:
                value[i] = start[node[i]]
                continue
            members = children[child_start[i] : child_end[i]]
            count[i] = int(np.count_nonzero(value[members]))
            if kind[i] == _PARALLEL:
                value[i] = count[i] > 0
            else:
                value[i] = count[i] >= need[i]
        down = np.array(
            [
                np.count_nonzero(
                    value[core_members[core_start[p] : core_end[p]]] == 0
                )
                for p in range(paths)
            ],
            np.int32,
        )
        self.term_kind = kind.astype(np.int8)
        self.term_need = need
        self.term_parent = parent
        self.value0 = value
        self.count0 = count
        self.down0 = down
        self.terms = terms
        self.paths = paths
        self.root = int(root)
        self.always = int(always)
        self.working_paths0 = int(np.count_nonzero(down == 0))


def _csr(groups) -> tuple:
    """Groups of numbers as starts, ends and the numbers one after another
    (as 32-bit integers)."""
    sizes = np.array([len(group) for group in groups], np.int64)
    ends = np.cumsum(sizes)
    members = [member for group in groups for member in group]
    return (
        (ends - sizes).astype(np.int32),
        ends.astype(np.int32),
        np.array(members, np.int32),
    )


def _scratch_bytes(n: int, streams: int, slots: int, s: Structure) -> int:
    return (
        8 * (2 * streams + (n + 1) + 2 * n + slots + 3 * n)
        + 4 * ((n + 1) + s.terms + s.paths)
        + (n + (n + 1) + s.terms)
        + 64
    )


class Prepared:
    """A run's system for the kernel: its arrays, the structure laid out to
    be kept up to date, and working memory for each thread."""

    def __init__(self, system: tuple, structure: tuple):
        (
            self.start,
            self.active,
            self.fail,
            self.repair,
            self.slot_start,
            self.slot_end,
            self.slot_category,
            self.slot_stream,
            self.slot_value,
            self.cost_index,
            self.has_rate,
            self.rate,
            has_costs,
            system_rate,
            initial_up,
            _,
        ) = system
        self.has_costs = int(has_costs)
        self.system_rate = float(system_rate)
        self.initial_up = int(initial_up)
        self.structure = Structure(structure, self.start)
        self._scratch: dict = {}
        self._fixed: Optional[dict] = None

    def scratch(self, index: int, streams: int) -> np.ndarray:
        """Thread ``index``'s working memory."""
        size = _scratch_bytes(
            self.start.size, streams, self.slot_stream.size, self.structure
        )
        memory = self._scratch.get(index)
        if memory is None or memory.nbytes < size:
            memory = self._scratch[index] = np.zeros(size // 8 + 1)
        return memory

    def tables(self, todo, first, draws, out) -> tuple:
        """The tables of addresses and sizes for a batch (the scratch's
        address is filled in for each thread)."""
        blocks_at, rows, first_block, columns = draws
        s = self.structure
        if self._fixed is None:
            self._fixed = {
                name: _address(getattr(self, name)) for name in ADDRESSES[:12]
            }
            self._fixed.update(
                (name, _address(getattr(s, name))) for name in ADDRESSES[28:40]
            )
        (
            uptime,
            node,
            counts,
            system,
            change_count,
            change_times,
            change_deltas,
            cost,
            category,
            node_cost,
            status,
        ) = out
        arrays = dict(self._fixed)
        for name, array, dtype in (
            ("blocks_at", blocks_at, np.int64),
            ("rows", rows, np.int64),
            ("first_block", first_block, np.int64),
            ("columns", columns, np.int64),
            ("uptime", uptime, np.float64),
            ("node", node, np.float64),
            ("counts", counts, np.int64),
            ("system", system, np.int64),
            ("change_count", change_count, np.int64),
            ("change_times", change_times, np.float64),
            ("change_deltas", change_deltas, np.int8),
            ("cost", cost, np.float64),
            ("category", category, np.float64),
            ("node_cost", node_cost, np.float64),
            ("status", status, np.int64),
            ("todo", todo, np.int64),
        ):
            arrays[name] = _address(array, dtype)
        addresses = np.array([arrays[name] for name in ADDRESSES], np.int64)
        sizes = np.array(
            [
                self.start.size,
                columns.size,
                change_times.shape[1],
                self.slot_stream.size,
                blocks_at.shape[1],
                first,
                self.has_costs,
                self.initial_up,
                s.terms,
                s.paths,
                s.root,
                s.always,
                node_cost.shape[1],
                s.working_paths0,
            ],
            np.int64,
        )
        return addresses, sizes


def _address(array: np.ndarray, dtype=None) -> int:
    """The address of ``array``'s data, which must be C-contiguous (and of
    ``dtype``, when given) for the kernel to read it."""
    if not array.flags.c_contiguous or (
        dtype is not None and array.dtype != dtype
    ):
        raise ValueError("The Mojo kernel needs contiguous arrays.")
    return int(array.ctypes.data)


def prepare(system: tuple, structure: tuple) -> Prepared:
    load()
    return Prepared(system, structure)


def run(
    prepared: Prepared,
    todo,
    first,
    t_end,
    draws,
    out,
    pool=None,
    threads: int = 1,
) -> None:
    """Simulations ``todo`` into ``out`` (rows ``r - first``): on the calling
    thread, or shared among ``threads`` threads of ``pool``."""
    kernel = load()
    addresses, sizes = prepared.tables(todo, first, draws, out)
    streams = draws[3].size
    m = todo.size
    t_end = float(t_end)
    rate = prepared.system_rate
    if pool is None or threads <= 1 or m < 2:
        scratch = prepared.scratch(0, streams)
        kernel.run(
            addresses.ctypes.data,
            sizes.ctypes.data,
            t_end,
            rate,
            0,
            m,
            scratch.ctypes.data,
        )
        return
    tasks = min(m, TASKS_PER_THREAD * threads)
    local = threading.local()
    counter = iter(range(threads))

    def task(j: int) -> None:
        index = getattr(local, "index", None)
        if index is None:
            index = local.index = next(counter)
        scratch = prepared.scratch(index, streams)
        kernel.run(
            addresses.ctypes.data,
            sizes.ctypes.data,
            t_end,
            rate,
            j * m // tasks,
            (j + 1) * m // tasks,
            scratch.ctypes.data,
        )

    list(pool.map(task, range(tasks)))
