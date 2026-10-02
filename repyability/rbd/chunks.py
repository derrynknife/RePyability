"""Part of a repairable system's simulation run, to run anywhere and merge
(#114).

``RepairableRBD.availability(t, mc_samples=N, seed=s)`` runs simulations
``0`` to ``N - 1`` of the run that ``t``, ``s`` and its other settings
define. Each simulation draws from streams of its own (see ``_streams``),
so simulation ``i`` comes out the same wherever, whenever and in whatever
company it runs: simulations ``a`` to ``b`` can run on one machine and the
rest on others. ``RepairableRBD.simulate_chunk`` runs such a range and
returns a ``SimulationChunk``, the simulations' totals, which saves to JSON
and merges with the run's other chunks; ``RepairableRBD.
availability_from_chunks`` turns them into the run's result.
"""

import copy
import io
import json
from typing import Any, Dict, Iterable, List, Tuple, Union
from zipfile import BadZipFile

import numpy as np

from repyability._version import __version__

#: The per-simulation values (and the curve's changes or counts) a chunk
#: keeps as arrays in its ``to_npz`` form, rather than in its JSON header.
_ARRAYS = {
    "uptimes": np.float64,
    "cost_samples": np.float64,
    "delivered": np.float64,
    "changes": np.float64,
    "deltas": np.int64,
    "binned": np.int64,
}


def _canonical(nodes) -> List[str]:
    """Node names in an order and form that survive JSON."""
    from repyability.rbd.serialisation import _node_name

    return sorted(repr(_node_name(node)) for node in nodes)


class SimulationChunk:
    """Simulations of a repairable system's run, by their positions in it.

    Made by a ``RepairableRBD``'s
    [`simulate_chunk`][repyability.RepairableRBD.simulate_chunk] for one
    range of simulations; merged with the run's other chunks by ``merge``;
    saved with ``to_dict`` or ``to_json`` and loaded with ``from_dict`` or
    ``from_json``; and turned into the run's
    [`AvailabilityResult`][repyability.AvailabilityResult] by its
    ``availability_from_chunks``.

    Parameters
    ----------
    ranges : iterable of (int, int)
        The simulations held, as ``(start, stop)`` ranges, in order.
    settings : dict
        The run they belong to (see ``settings`` below).
    tally : object
        Their totals, as the simulation keeps them.

    Attributes
    ----------
    ranges : list[tuple[int, int]]
        The simulations held, as ``(start, stop)`` ranges (``stop``
        excluded), in order.
    settings : dict
        The run they belong to: ``"t_simulation"``, ``"entropy"`` (the
        number the seed gives the streams), ``"method"``,
        ``"working_nodes"`` and ``"broken_nodes"``, ``"antithetic"``,
        ``"demand"``, ``"fingerprint"``, a hash of the system saved as
        JSON (with the RePyability version), or None if it cannot be saved,
        ``"state"``, the components' states at the start as JSON, or None
        for new, and ``"curve_points"``, the grid the curve is counted on,
        or None for every change (see ``RepairableRBD.availability``).
    """

    def __init__(self, ranges, settings: Dict[str, Any], tally) -> None:
        self.ranges: List[Tuple[int, int]] = [
            (int(a), int(b)) for a, b in ranges
        ]
        self.settings = dict(settings)
        self._tally = tally

    def __eq__(self, other) -> bool:
        """Whether ``other`` is a chunk of the same simulations of the same
        run, with the same totals."""
        if not isinstance(other, SimulationChunk):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    @property
    def n_simulations(self) -> int:
        """How many simulations the chunk holds."""
        return sum(stop - start for start, stop in self.ranges)

    def __repr__(self) -> str:
        spans = ", ".join(f"{a}-{b - 1}" for a, b in self.ranges)
        return (
            f"SimulationChunk(simulations {spans} of a run over "
            f"{self.settings['t_simulation']:g})"
        )

    @staticmethod
    def merge(
        chunks: Iterable[Union["SimulationChunk", dict]],
    ) -> "SimulationChunk":
        """One chunk of the simulations of ``chunks``, which must be of the
        same run (the same system, window, seed and settings) and hold
        different simulations. They are put in order of their first
        simulations, and each must end before the next begins.

        Parameters
        ----------
        chunks : iterable of SimulationChunk or dict
            The chunks, or their ``to_dict`` data.

        Returns
        -------
        SimulationChunk
            Their simulations, in order. The chunks given are unchanged.

        Raises
        ------
        ValueError
            If there are none, they belong to different runs, or their
            simulations overlap or interleave.
        """
        items = [SimulationChunk._load(chunk) for chunk in chunks]
        if not items:
            raise ValueError("There are no chunks to merge.")
        first = items[0]
        key = _run_key(first.settings)
        for chunk in items[1:]:
            if _run_key(chunk.settings) != key:
                raise ValueError(
                    "The chunks are of different runs (their system, window, "
                    "seed or settings differ), so they cannot be merged."
                )
        items.sort(key=lambda chunk: chunk.ranges[0][0])
        for before, after in zip(items, items[1:]):
            if before.ranges[-1][1] > after.ranges[0][0]:
                raise ValueError(
                    f"Simulations {before.ranges} and {after.ranges} "
                    "overlap or interleave: each chunk's simulations must "
                    "come before the next's."
                )
        tally = copy.deepcopy(items[0]._tally)
        ranges = list(items[0].ranges)
        for chunk in items[1:]:
            tally.merge(chunk._tally)
            for start, stop in chunk.ranges:
                if ranges[-1][1] == start:
                    ranges[-1] = (ranges[-1][0], stop)
                else:
                    ranges.append((start, stop))
        return SimulationChunk(ranges, first.settings, tally)

    def to_dict(self) -> dict:
        """The chunk as JSON data (``from_dict`` loads it).

        Returns
        -------
        dict
            Its ranges, run settings and totals, with the RePyability
            version that made it.
        """
        return {
            "kind": "SimulationChunk",
            "repyability_version": __version__,
            "ranges": [list(r) for r in self.ranges],
            "settings": self.settings,
            "totals": self._tally.to_dict(),
        }

    @staticmethod
    def _load(chunk) -> "SimulationChunk":
        """A chunk, from itself, its ``to_dict`` data or its ``to_npz``
        bytes."""
        if isinstance(chunk, SimulationChunk):
            return chunk
        if isinstance(chunk, (bytes, bytearray, memoryview)):
            return SimulationChunk.from_npz(chunk)
        return SimulationChunk.from_dict(chunk)

    def to_npz(self) -> bytes:
        """The chunk as the bytes of a NumPy ``.npz`` file (``from_npz``
        loads it): its per-simulation values and the curve's changes (or
        counts) as arrays, and the rest, the ranges, settings and exact
        totals, as JSON in an array of its own. Compact, and read without
        pickle, so a worker can send it back to any coordinator (see
        ``repyability.rbd.shards``).

        Returns
        -------
        bytes
            The ``.npz`` file's bytes.
        """
        data = self.to_dict()
        totals = data["totals"]
        arrays = {}
        for name, kind in _ARRAYS.items():
            values = totals.pop(name, None)
            if values is not None:
                arrays[name] = np.asarray(values, dtype=kind)
        header = json.dumps(data).encode()
        buffer = io.BytesIO()
        np.savez(
            buffer, header=np.frombuffer(header, dtype=np.uint8), **arrays
        )
        return buffer.getvalue()

    @classmethod
    def from_npz(cls, data) -> "SimulationChunk":
        """The chunk ``to_npz`` saved.

        Parameters
        ----------
        data : bytes
            ``to_npz``'s output.

        Returns
        -------
        SimulationChunk
            The chunk.

        Raises
        ------
        ValueError
            If ``data`` is not a saved chunk.
        """
        try:
            with np.load(io.BytesIO(bytes(data)), allow_pickle=False) as npz:
                saved = json.loads(npz["header"].tobytes().decode())
                totals = saved["totals"]
                for name in _ARRAYS:
                    if name in npz.files:
                        totals[name] = npz[name].tolist()
        except (OSError, EOFError, ValueError, KeyError, BadZipFile) as error:
            raise ValueError(
                f"This is not a saved SimulationChunk: {error}"
            ) from None
        totals.setdefault("binned", None)
        return cls.from_dict(saved)

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationChunk":
        """The chunk ``to_dict`` saved.

        Parameters
        ----------
        data : dict
            ``to_dict``'s output (or its JSON, parsed).

        Returns
        -------
        SimulationChunk
            The chunk.

        Raises
        ------
        ValueError
            If ``data`` is not a saved chunk.
        """
        from repyability.rbd.repairable_rbd import _Tally
        from repyability.rbd.serialisation import _node_name

        if not isinstance(data, dict) or data.get("kind") != (
            "SimulationChunk"
        ):
            raise ValueError("This is not a saved SimulationChunk.")
        settings = dict(data["settings"])
        for name in ("working_nodes", "broken_nodes"):
            settings[name] = [_node_name(n) for n in settings[name]]
        return cls(
            [tuple(r) for r in data["ranges"]],
            settings,
            _Tally.from_dict(data["totals"]),
        )

    def to_json(self, fp=None) -> Union[str, None]:
        """The chunk as a JSON document (``from_json`` loads it): returned,
        or written to ``fp``.

        Parameters
        ----------
        fp : str, os.PathLike or file, optional
            A path, or a file opened for writing, to write the document to;
            by default None: it is returned.

        Returns
        -------
        str or None
            ``to_dict``'s output, as JSON, or None once written to ``fp``.
        """
        from repyability.utils.json_io import write_json

        return write_json(json.dumps(self.to_dict()), fp)

    @classmethod
    def from_json(cls, text) -> "SimulationChunk":
        """The chunk ``to_json`` saved.

        Parameters
        ----------
        text : str, os.PathLike or file
            ``to_json``'s output: its text, a path to a file holding it, or
            a file opened for reading.

        Returns
        -------
        SimulationChunk
            The chunk.
        """
        from repyability.utils.json_io import read_json

        return cls.from_dict(json.loads(read_json(text)))


def _run_key(settings: Dict[str, Any]) -> tuple:
    """What two chunks of one run share (node sets in a JSON-proof form)."""
    return (
        float(settings["t_simulation"]),
        int(settings["entropy"]),
        settings["method"],
        tuple(_canonical(settings["working_nodes"])),
        tuple(_canonical(settings["broken_nodes"])),
        bool(settings["antithetic"]),
        None if settings["demand"] is None else float(settings["demand"]),
        settings["fingerprint"],
        settings.get("state"),
        settings.get("curve_points"),
    )
