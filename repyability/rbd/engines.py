"""Simulation engines from other packages.

A ``RepairableRBD`` simulation runs in Python, or compiled with numba (see
``_compiled``). Another package can add an engine of its own: an object
registered under the ``repyability.engines`` entry point group (or with
:func:`register`), which ``engine=<its name>`` then runs, and which
``engine="auto"`` prefers to numba when its ``priority`` is higher.

An engine is an object with:

- ``name`` (str): what ``engine=`` takes. Not ``"auto"``, ``"python"`` or
  ``"numba"``.
- ``api`` (int): the version of this interface it was written for,
  :data:`API`. An engine written for another is left out, with a warning.
- ``priority`` (int): ``engine="auto"`` tries the engines from the highest
  priority down; numba's is 0.
- ``available()``: whether it can run here, without loading anything
  costly.
- ``worthwhile(draws)``: whether a run expected to take about ``draws``
  failure and repair draws is long enough to repay loading it (always, once
  loaded).
- ``load()``: get ready to run (compile, say); raises ``ImportError`` if it
  cannot. ``engine="auto"`` then warns and runs on the next engine, and
  does not try this one again in the process.
- ``runner(rbd, plan, tally, progress, working, broken, method, jobs)``: an
  object that runs simulations ``start`` to ``stop`` of the run when called
  with them, adding them to ``tally`` in order, and has a ``close()``. Its
  arguments are those ``_compiled.Runner`` takes, the run's own objects:
  the engine runs what ``_compiled.unsupported`` allows, reading the
  run's streams (``plan``), and must give the same results as the Python
  engine to the last bit.

The run's objects are RePyability's own, and change with it: :data:`API` is
raised when they change in a way an engine must follow.
"""

import warnings
from typing import Any, Dict, List, Optional

#: The version of the engine interface.
API = 1
#: The entry point group engines are registered under.
GROUP = "repyability.engines"
#: The names the built-in engines (and ``engine="auto"``) take.
RESERVED = ("auto", "python", "numba")

_engines: Optional[Dict[str, Any]] = None
#: The engines that could not be loaded in this process, and why.
_failed: Dict[str, str] = {}


def _check(engine: Any, origin: str) -> bool:
    """Whether ``engine`` can be registered (with a warning if not)."""
    name = getattr(engine, "name", None)
    problem = None
    if not isinstance(name, str) or name in RESERVED:
        problem = f"its name {name!r} is not one it can take"
    elif getattr(engine, "api", None) != API:
        problem = (
            f"it was written for version {getattr(engine, 'api', None)!r} "
            f"of the engine interface, not {API}"
        )
    elif not all(
        callable(getattr(engine, method, None))
        for method in ("available", "worthwhile", "load", "runner")
    ):
        problem = "it lacks a method of the engine interface"
    if problem is not None:
        warnings.warn(
            f"The simulation engine from {origin} is left out: {problem}.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False
    return True


def _discover() -> Dict[str, Any]:
    """The engines registered under the entry point group."""
    from importlib import metadata

    found: Dict[str, Any] = {}
    for entry in metadata.entry_points(group=GROUP):
        try:
            engine = entry.load()
        except Exception as error:  # a broken plugin must not break runs
            warnings.warn(
                f"The simulation engine {entry.name!r} could not be "
                f"imported: {error}",
                RuntimeWarning,
                stacklevel=3,
            )
            continue
        if _check(engine, f"{entry.value!r}"):
            found[engine.name] = engine
    return found


def registered() -> Dict[str, Any]:
    """The engines other packages add, by name (found the first time it is
    asked)."""
    global _engines
    if _engines is None:
        _engines = _discover()
    return _engines


def register(engine: Any) -> None:
    """Add an engine (as an entry point would)."""
    if _check(engine, repr(engine)):
        registered()[engine.name] = engine
        _failed.pop(engine.name, None)


def unregister(name: str) -> None:
    """Remove the engine named ``name``, if there is one."""
    registered().pop(name, None)
    _failed.pop(name, None)


def get(name: str) -> Optional[Any]:
    """The registered engine named ``name``, or None."""
    return registered().get(name)


def usable(engine: Any) -> bool:
    """Whether ``engine`` can run here and has not failed to load."""
    return engine.name not in _failed and bool(engine.available())


def by_priority(minimum: Optional[int] = None) -> List[Any]:
    """The usable engines, highest priority first (those of at least
    ``minimum``, if given)."""
    engines = [
        engine
        for engine in registered().values()
        if usable(engine) and (minimum is None or engine.priority >= minimum)
    ]
    return sorted(engines, key=lambda engine: -engine.priority)


def load(engine: Any) -> None:
    """Load ``engine``, remembering a failure (so that it is not tried
    again) and raising it."""
    if engine.name in _failed:
        raise ImportError(_failed[engine.name])
    try:
        engine.load()
    except ImportError as error:
        _failed[engine.name] = str(error)
        raise
