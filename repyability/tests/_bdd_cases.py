"""What ``bdd_records.json`` records of the decision diagrams (#207), and
how: each core ``bdd.build`` is given while decomposing a set of diagrams
(random ones, ones with components drawn in several places, meshed ones),
by label (``cores``), as plain lists (``canonical``); a digest of each
core's plan (``digest``); the steps a core's search takes
(``steps_taken``); and the bits of the values and gradients replayed on
two diagrams (``replay_values``). Run this module as a script to
re-record them: only after a deliberate change of the plans."""

import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Iterator, Tuple

import numpy as np
from surpyval import FixedEventProbability

from repyability import RBD, NonRepairableRBD
from repyability.rbd import bdd, modular
from repyability.rbd.modular import decompose
from repyability.tests.test_bdd_core import bridges
from repyability.tests.test_rbd_modular import random_diagram


def grid(rows, cols):
    """A meshed grid, from the input to the output, with links downward."""

    def name(r, c):
        return f"n{r}_{c}"

    edges = [("s", name(r, 0)) for r in range(rows)]
    edges += [(name(r, cols - 1), "t") for r in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((name(r, c), name(r, c + 1)))
            if r + 1 < rows:
                edges.append((name(r, c), name(r + 1, c)))
    return edges


def digest(plan: object) -> str:
    """A plan's record: a digest of its steps and root, as written."""
    return hashlib.sha256(repr(plan).encode()).hexdigest()[:24]


def canonical(sequence, pred, succ, k, source, sink, variable) -> dict:
    """A core as ``bdd.build`` takes it, as plain lists (sets and dicts in
    order), each variable by its ``repr``: what the record keeps."""
    every = [source, *sequence, sink]
    return {
        "vertices": every,
        "succ": [sorted(succ.get(v, ()), key=str) for v in every],
        "k": [k.get(v, 1) for v in every],
        "variable": [repr(variable[v]) for v in sequence],
    }


def rebuilt(core: dict) -> tuple:
    """``build``'s arguments for a core ``canonical`` kept."""
    source, *sequence, sink = core["vertices"]
    succ = {v: list(w) for v, w in zip(core["vertices"], core["succ"])}
    pred: dict = {v: [] for v in core["vertices"]}
    for v, ws in succ.items():
        for w in ws:
            pred[w].append(v)
    k = dict(zip(core["vertices"], core["k"]))
    variable = dict(zip(sequence, core["variable"]))
    return sequence, pred, succ, k, source, sink, variable


def _shared(rng, edges):
    """Fixed probabilities for each node, one or two of them drawn in
    another's place."""
    names = sorted({n for e in edges for n in e} - {"s", "t"}, key=str)
    models = {
        n: FixedEventProbability.from_params(float(rng.uniform(0.05, 0.5)))
        for n in names
    }
    for _ in range(int(rng.integers(1, 3))):
        a, b = rng.choice(len(names), 2, replace=False)
        if not isinstance(models[names[a]], str):
            models[names[b]] = names[a]
    return models


def _diagrams() -> Iterator[Tuple[str, Callable[[], object]]]:
    """Each diagram, by label, as a function that decomposes it."""
    for seed in range(250):
        rng = np.random.default_rng(seed)
        edges, k = random_diagram(rng, int(rng.integers(3, 12)))

        def plain(edges=edges, k=k):
            rbd = RBD(edges, k=k, on_infeasible_rbd="ignore")
            if rbd.structure_check["is_valid"]:
                decompose(rbd.G, "s", "t", core="bdd")

        yield f"random {seed}", plain
    for seed in range(190):
        rng = np.random.default_rng(10_000 + seed)
        edges, k = random_diagram(rng, int(rng.integers(4, 11)))
        models = _shared(rng, edges)

        def shared(edges=edges, k=k, models=models):
            rbd = NonRepairableRBD(edges, models, k=k)
            decompose(rbd.G, "s", "t", rbd._component_aliases(), core="bdd")

        yield f"shared {seed}", shared
    for label, edges in [
        ("bridges 12", bridges(12)),
        ("grid 4x9", grid(4, 9)),
        ("grid 7x7", grid(7, 7)),
        ("grid 10x20", grid(10, 20)),
    ]:

        def meshed(edges=edges):
            decompose(RBD(edges).G, "s", "t", core="bdd")

        yield label, meshed


def cores() -> Dict[str, dict]:
    """Each core ``bdd.build`` is given, by its diagram's label and a
    digest of the core, as ``canonical`` keeps it."""
    found: Dict[str, dict] = {}
    real = bdd.build
    label = [""]

    def spy(*args):
        found[f"{label[0]}, {digest(canonical(*args))}"] = canonical(*args)
        return real(*args)

    bdd.build = spy
    try:
        for name, run in _diagrams():
            label[0] = name
            try:
                run()
            except ValueError:
                continue
    finally:
        bdd.build = real
    return found


def steps_taken(args: tuple) -> int:
    """The steps the search takes on a core (``build``'s arguments): the
    least ``STEP_LIMIT`` it builds within, bisected."""
    limit = bdd.STEP_LIMIT
    try:
        low, high = 0, 1
        while True:
            bdd.STEP_LIMIT = high
            try:
                bdd.build(*args)
                break
            except bdd.TooLarge:
                low, high = high, 2 * high
        while high - low > 1:
            middle = (low + high) // 2
            bdd.STEP_LIMIT = middle
            try:
                bdd.build(*args)
                high = middle
            except bdd.TooLarge:
                low = middle
        return high
    finally:
        bdd.STEP_LIMIT = limit


def _bits(values) -> list:
    return np.asarray(values, float).ravel().view(np.uint64).tolist()


def replay_values() -> Dict[str, str]:
    """A digest of the bits of each value and gradient replayed on two
    diagrams for probabilities of each shape (one node sure to work), by
    diagram and shape, replayed as ``modular.COMPILED_STEPS`` says."""
    out = {}
    for name, edges in [("bridges 6", bridges(6)), ("grid 5x6", grid(5, 6))]:
        rbd = RBD(edges)
        graph = decompose(rbd.G, "s", "t", core="bdd")
        for shape in [None, (7,), (2, 3), (40,)]:
            rng = np.random.default_rng(len(edges))
            nodes = sorted(rbd.nodes, key=str)
            size = () if shape is None else shape
            p = {n: rng.uniform(0.0, 1.0, size) for n in nodes}
            p = {n: float(v) if shape is None else v for n, v in p.items()}
            p[nodes[0]] = 1.0 if shape is None else np.ones(shape)
            q = {n: 1.0 - v for n, v in p.items()}
            works, fails = graph.probabilities(p, q, shape=shape)
            w, f, g = graph.value_and_gradient(p, q, shape=shape)
            got = {
                "works": _bits(works),
                "fails": _bits(fails),
                "w": _bits(w),
                "f": _bits(f),
                "gradient": {str(n): _bits(g[n]) for n in sorted(g, key=str)},
            }
            out[f"{name} {shape}"] = digest(json.dumps(got, sort_keys=True))
    return out


RECORDS = Path(__file__).with_name("bdd_records.json")

#: The cores whose steps are recorded: a meshed one, and those with
#: components drawn in several places.
STEPPED = ("shared", "grid 4x9")


def record() -> None:
    """Re-record ``bdd_records.json`` with the search and replay in
    Python."""
    compiled, steps_cut = bdd.COMPILED, modular.COMPILED_STEPS
    bdd.COMPILED, modular.COMPILED_STEPS = False, 10**9
    try:
        kept = cores()
        plans, steps = {}, {}
        for label, core in kept.items():
            args = rebuilt(core)
            plan = bdd.build(*args)
            plans[label] = digest(plan)
            if label.startswith(STEPPED) and (plan[0] or plan[1] > 1):
                steps[label] = steps_taken(args)
        values = replay_values()
    finally:
        bdd.COMPILED, modular.COMPILED_STEPS = compiled, steps_cut
    RECORDS.write_text(
        json.dumps(
            {"cores": kept, "plans": plans, "steps": steps, "values": values},
            indent=0,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    record()
