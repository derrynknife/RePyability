"""Shards (#152): a simulation run as plain data, run anywhere.

A ``RepairableRBD``'s run of ``N`` simulations is ``N`` independent pieces
of work: simulation ``i`` draws from streams seeded by the run's entropy and
``i`` alone (see ``_streams``), and the run's totals are kept exactly
(#151), so pieces of a run put together in any order give the run's result
to the last bit. A *shard* is a range of a run's simulations as plain data,
JSON: the system as ``RepairableRBD.to_dict`` gives it, the run's entropy
(drawn once, by whoever plans the run, so every shard draws the same
streams) and settings, and the range. ``run_shard`` runs one anywhere, on
any engine: it takes the shard's bytes and gives back its *partial*, the
simulations' totals as the bytes of a NumPy ``.npz`` file, read without
pickle (``SimulationChunk.to_npz``).

So any executor can run a large simulation: ``concurrent.futures``,
``multiprocessing``, Ray, Dask, a serverless function, or a batch system
through the command line::

    python -m repyability.rbd.shards < shard.json > partial.npz

``RepairableRBD.shards`` plans a run's shards, and
``RepairableRBD.availability_from_chunks`` puts their partials together;
``RepairableRBD.availability(..., shard_map=...)`` does both, through any
``map``, in rounds when it runs to a tolerance.
"""

import json
import sys
from typing import Any, Dict

#: What a shard's JSON is.
KIND = "RePyabilityShard"


def run_shard(shard) -> bytes:
    """Run a shard (see the module docstring): its simulations of its run,
    by the engine it names, here.

    Parameters
    ----------
    shard : bytes or str
        The shard, as ``RepairableRBD.shards`` made it: its JSON.

    Returns
    -------
    bytes
        Its partial: the simulations' totals, as a ``SimulationChunk``'s
        ``to_npz`` bytes, for ``RepairableRBD.availability_from_chunks``.

    Raises
    ------
    ValueError
        If ``shard`` is not a shard, or was made by another version of
        RePyability (which can simulate differently).

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD, run_shard
    >>> E = surv.Exponential.from_params
    >>> rbd = RepairableRBD(
    ...     [("s", "a"), ("a", "t")],
    ...     {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
    ... )
    >>> shards = rbd.shards(100.0, 2048, seed=3, size=1024)
    >>> partials = [run_shard(shard) for shard in reversed(shards)]
    >>> merged = rbd.availability_from_chunks(partials, mc_samples=2048)
    >>> whole = rbd.availability(100.0, mc_samples=2048, seed=3)
    >>> merged.system_uptime == whole.system_uptime
    True
    """
    from repyability.rbd._runs import _states_from_key
    from repyability.rbd.repairable_rbd import RepairableRBD
    from repyability.rbd.serialisation import _node_name

    from . import _runs

    data = _shard(shard)
    rbd = RepairableRBD.from_dict(data["system"])
    if not isinstance(rbd, RepairableRBD):
        raise ValueError("This shard's system is not a RepairableRBD.")
    if data.get("histories"):
        return _modules_partial(rbd, data)
    working = {_node_name(node) for node in data["working_nodes"]}
    broken = {_node_name(node) for node in data["broken_nodes"]}
    chunk = _runs._chunk(
        rbd,
        float(data["t_simulation"]),
        int(data["start"]),
        int(data["stop"]),
        int(data["entropy"]),
        working,
        broken,
        data["method"],
        bool(data["antithetic"]),
        data["demand"],
        data["engine"],
        None,
        False,
        _states_from_key(rbd, data["state"]),
        data["curve_points"],
    )
    return chunk.to_npz()


def _modules_partial(rbd, data: Dict[str, Any]) -> bytes:
    """A conditional run's shard of its modules (#189): their histories
    and costs in the shard's simulations (see
    ``_conditional.partial_bytes``), ``rbd`` being the modules' own
    diagram (``RepairableRBD._modules_rbd``)."""
    from repyability.rbd import _conditional, _timeline_runs
    from repyability.rbd._runs import _states_from_key

    start, stop = int(data["start"]), int(data["stop"])
    histories, tally = _timeline_runs.with_costs(
        rbd,
        float(data["t_simulation"]),
        stop - start,
        None,
        bool(data["antithetic"]),
        data["engine"],
        None,
        start,
        _states_from_key(rbd, data["state"]),
        entropy=int(data["entropy"]),
    )
    nodes = list(rbd.components)
    return _conditional.partial_bytes(
        [histories[node] for node in nodes],
        tally,
        start,
        stop,
        nodes,
        rbd.has_costs,
    )


def _shard(shard) -> Dict[str, Any]:
    """A shard's data, from its JSON (bytes or text), made by this version
    of RePyability."""
    from repyability._version import __version__

    try:
        data = json.loads(shard)
    except (TypeError, ValueError):
        data = None
    if not isinstance(data, dict) or data.get("kind") != KIND:
        raise ValueError(
            "This is not a shard: make shards with RepairableRBD.shards."
        )
    made_by = data.get("repyability_version")
    if made_by != __version__:
        raise ValueError(
            f"This shard was made by RePyability {made_by}, and this is "
            f"{__version__}: run it with the version that made it, as "
            "another can simulate it differently."
        )
    return data


def main(argv=None) -> int:
    """Run the shard read from standard input (or the file named first),
    and write its partial to standard output (or the file named second):
    ``python -m repyability.rbd.shards [shard.json [partial.npz]]``."""
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) > 2 or any(a in ("-h", "--help") for a in args):
        print(main.__doc__, file=sys.stderr)
        return 0 if args and args[0] in ("-h", "--help") else 2
    if args:
        with open(args[0], "rb") as source:
            shard = source.read()
    else:
        shard = sys.stdin.buffer.read()
    partial = run_shard(shard)
    if len(args) == 2:
        with open(args[1], "wb") as target:
            target.write(partial)
    else:
        sys.stdout.buffer.write(partial)
        sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover (run as a command)
    raise SystemExit(main())
