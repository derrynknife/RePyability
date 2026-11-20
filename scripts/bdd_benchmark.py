"""Measure what a meshed core's decision diagram costs to build
(``repyability/rbd/bdd.py``): for grids, layered meshes (a chain of
bridges made wider) and random acyclic diagrams of growing width, the
frontier cost (``bdd._cost``) of each order ``bdd.order`` chooses from,
and for the search the states it interns, the steps of the plan it makes,
its time, its peak memory and whether it passed ``bdd.STEP_LIMIT``.

    python scripts/bdd_benchmark.py            # the table
    python scripts/bdd_benchmark.py --quick    # the small diagrams only
    python scripts/bdd_benchmark.py --limit 100_000_000

The peak is ``tracemalloc``'s over the Python search (whose states are
Python's own objects), which makes it several times slower; the time is of
a run without it. With numba installed the compiled search is timed too:
its memory is not Python's, so its peak is read from the growth of the
process's resident memory (which only ever grows: a lower bound after the
first large diagram). A measurement, not a test: nothing here is asserted.
"""

import argparse
import resource
import time
import tracemalloc
from typing import Callable, Dict, List, Tuple

import numpy as np

from repyability import RBD
from repyability.rbd import _compiled, bdd
from repyability.rbd.modular import decompose


def grid(rows: int, cols: int) -> list:
    """``rows`` lines of ``cols`` nodes, each node also feeding the one
    below it."""
    edges = [("s", (r, 0)) for r in range(rows)]
    edges += [((r, cols - 1), "t") for r in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
    return edges


def mesh(width: int, layers: int) -> list:
    """``layers`` layers of ``width`` nodes, each node feeding the nodes
    beside its place in the next layer: a chain of bridges, made wider."""
    edges = [("s", (0, j)) for j in range(width)]
    edges += [((layers - 1, j), "t") for j in range(width)]
    for i in range(layers - 1):
        for j in range(width):
            for d in (-1, 0, 1):
                if 0 <= j + d < width:
                    edges.append(((i, j), (i + 1, j + d)))
    return edges


def random_dag(n: int, p: float, seed: int = 1) -> list:
    """``n`` nodes, each joined to each later one with probability
    ``p``."""
    rng = np.random.default_rng(seed)
    edges: set = {
        (i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p
    }
    for i in range(n):
        if not any(a == i for a, _ in edges):
            edges.add((i, "t"))
        if not any(b == i for _, b in edges):
            edges.add(("s", i))
    return sorted(edges, key=str)


def diagrams(quick: bool) -> List[Tuple[str, list]]:
    out = [
        ("grid 6x12", grid(6, 12)),
        ("grid 10x20", grid(10, 20)),
        ("mesh 6x12", mesh(6, 12)),
        ("mesh 8x12", mesh(8, 12)),
        ("random 35, 0.2", random_dag(35, 0.2)),
        ("random 45, 0.2", random_dag(45, 0.2)),
    ]
    if not quick:
        out += [
            ("grid 14x14", grid(14, 14)),
            ("grid 18x18", grid(18, 18)),
            ("mesh 10x12", mesh(10, 12)),
            ("mesh 12x12", mesh(12, 12)),
            ("mesh 14x12", mesh(14, 12)),
            ("random 60, 0.2", random_dag(60, 0.2)),
            ("random 70, 0.2", random_dag(70, 0.2)),
            ("random 80, 0.2", random_dag(80, 0.2)),
            ("random 60, 0.4", random_dag(60, 0.4)),
        ]
    return out


class _Core(Exception):
    """Carries the core ``bdd.build`` was about to be given."""


def core_of(edges: list) -> tuple:
    """The core of the diagram of ``edges``, as ``bdd.build``'s arguments
    but for the order: ``(vertices, pred, succ, k, source, sink,
    variable)``."""
    real = bdd.build

    def stop(*args):
        raise _Core(args)

    bdd.build = stop
    try:
        decompose(RBD(edges).G, "s", "t", core="bdd")
    except _Core as found:
        return found.args[0]
    finally:
        bdd.build = real
    raise ValueError("The diagram has no core.")


def orders(core: tuple) -> Dict[str, list]:
    """Each order ``bdd.order`` chooses from, by name."""
    sequence, pred, succ, _, source, sink, _ = core
    vertices = sorted(sequence)
    return {
        name: candidate(vertices, pred, succ, source, sink)
        for name, candidate in bdd.CANDIDATES.items()
    }


def searched(core: tuple, limit: int) -> dict:
    """The Python search of a core: its states, the plan's steps, its
    time, its peak memory, and whether it passed ``limit``."""
    interned: List[dict] = []
    real = bdd.search

    def spy(*args):
        interned.append(args[22])
        return real(*args)

    out: dict = {}
    kept = bdd.STEP_LIMIT, bdd.COMPILED
    bdd.STEP_LIMIT, bdd.COMPILED, bdd.search = limit, False, spy
    try:
        for traced in (False, True):
            if traced:
                tracemalloc.start()
            began = time.perf_counter()
            try:
                steps = len(bdd.build(*core)[0])
                out["too large"] = False
            except bdd.TooLarge:
                steps = None
                out["too large"] = True
            if traced:
                out["peak MB"] = tracemalloc.get_traced_memory()[1] / 2**20
                tracemalloc.stop()
            else:
                out["seconds"] = time.perf_counter() - began
            out["plan steps"] = steps
            out["states"] = len(interned[-1])
    finally:
        bdd.STEP_LIMIT, bdd.COMPILED = kept
        bdd.search = real
    return out


def compiled(core: tuple, limit: int) -> dict:
    """The compiled search of a core: its time (compiled already) and the
    growth of the process's resident memory."""
    kept = bdd.STEP_LIMIT, bdd.COMPILED
    bdd.STEP_LIMIT, bdd.COMPILED = limit, True

    def run() -> None:
        try:
            bdd.build(*core)
        except bdd.TooLarge:
            pass

    try:
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        began = time.perf_counter()
        run()
        seconds = time.perf_counter() - began
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    finally:
        bdd.STEP_LIMIT, bdd.COMPILED = kept
    return {"seconds": seconds, "grown MB": (after - before) / 2**10}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--limit", type=int, default=bdd.STEP_LIMIT)
    options = parser.parse_args()
    numba = _compiled.available()
    if numba:
        compiled(core_of(grid(3, 4)), options.limit)  # compile it first
    names = list(bdd.CANDIDATES)
    head = ["diagram", "nodes", "edges"]
    head += [f"log2 cost, {name}" for name in names]
    head += ["order", "states", "plan steps", "seconds", "peak MB", "limit"]
    if numba:
        head += ["compiled seconds", "compiled grown MB"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for label, edges in diagrams(options.quick):
        core = core_of(edges)
        sequence, pred, succ, _, source, sink, _ = core
        cost: Callable = lambda o: bdd._cost(o, pred, succ, source, sink)
        candidates = orders(core)
        chosen = min(names, key=lambda name: cost(candidates[name]))
        assert candidates[chosen] == list(sequence)
        got = searched(core, options.limit)
        row = [label, str(len(sequence)), str(len(edges))]
        row += [f"{np.log2(cost(candidates[name])):.1f}" for name in names]
        row += [
            chosen,
            f"{got['states']:,}",
            "-" if got["too large"] else f"{got['plan steps']:,}",
            f"{got['seconds']:.2f}",
            f"{got['peak MB']:.1f}",
            "passed" if got["too large"] else "",
        ]
        if numba:
            fast = compiled(core, options.limit)
            row += [f"{fast['seconds']:.2f}", f"{fast['grown MB']:.0f}"]
        print("| " + " | ".join(row) + " |", flush=True)


if __name__ == "__main__":
    main()
