import functools
import sys
import threading
from contextlib import contextmanager

import numpy as np

from repyability.utils.checks import seed as check_seed

#: Held while a simulation runs, in whichever thread (#216). The event loop
#: keeps a run's state on the diagram (and its nested diagrams and
#: components), and draws that cannot be streamed come from numpy's global
#: RNG, which a seeded run seeds and restores, so two runs at once would
#: cross. Simulations called from several threads run one at a time, each
#: as it would alone: the Python loop holds the GIL anyway, and ``n_jobs``
#: runs on processes, each with a lock of its own, or (numba's loop) on
#: threads that do not take it. Re-entrant: a run may start others.
SIMULATIONS = threading.RLock()


def outside_level() -> int:
    """The ``stacklevel`` at which a warning raised by the caller of this
    function points at the first frame outside the package (its tests
    count as outside): through however many of its own calls and wrappers
    the warning was reached."""
    level, frame = 2, sys._getframe(2)
    while frame.f_back is not None:
        name = frame.f_globals.get("__name__", "")
        if not name.startswith("repyability.") or name.startswith(
            "repyability.tests"
        ):
            break
        frame, level = frame.f_back, level + 1
    return level


@contextmanager
def numpy_seed(seed):
    """Temporarily seed numpy's global RNG, restoring the previous state on
    exit.

    surpyval's ``.random()`` draws from numpy's *global* RNG unless given a
    ``random_state``, and the simulations draw from that global stream (their
    batched draws replay it), so reproducible Monte-Carlo simulations are
    obtained by seeding it. This context manager seeds it for the duration of
    a simulation and restores the caller's RNG state afterwards, so calling a
    simulation with ``seed=...`` is reproducible *without* disturbing the
    surrounding program's random stream. ``seed=None`` is a no-op (i.e. the
    simulation stays non-reproducible, using whatever global state exists).
    Either way it holds ``SIMULATIONS``, so that another thread's draws
    cannot come between a seeded simulation's.

    Parameters
    ----------
    seed : int or None
        The seed to apply, or None to leave the global RNG untouched.
    """
    seed = check_seed(seed)
    with SIMULATIONS:
        if seed is None:
            yield
            return
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            yield
        finally:
            np.random.set_state(state)


#: The parameters that take node names (#225).
NODE_ARGUMENTS = ("working_nodes", "broken_nodes", "nodes")


def node_names(func):
    """``func`` (a method) with a bare string given to one of its
    ``NODE_ARGUMENTS`` taken as one node's name (#225), not as its
    characters: ``working_nodes="belt"`` is ``["belt"]``, where iterating
    it gave ``{"b", "e", "l", "t"}``, and an error that changed from run to
    run with the strings' hashes. ``func`` itself where it has none of
    them."""
    import inspect

    try:
        names = list(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        return func
    places = {
        name: names.index(name) - 1  # in the arguments after ``self``
        for name in NODE_ARGUMENTS
        if name in names
    }
    if not places or getattr(func, "node_names", False):
        return func

    @functools.wraps(func)
    def wrap(obj, *args, **kwargs):
        listed = None
        for name, place in places.items():
            if 0 <= place < len(args) and isinstance(args[place], str):
                listed = list(args) if listed is None else listed
                listed[place] = [args[place]]
            elif isinstance(kwargs.get(name), str):
                kwargs[name] = [kwargs[name]]
        return func(obj, *(args if listed is None else listed), **kwargs)

    wrap.node_names = True  # type: ignore[attr-defined]
    return wrap


def check_probability(func):
    """Checks the target probability is between 0 and 1."""

    @functools.wraps(func)
    def wrap(obj, target: float, *args, **kwargs):
        if target > 1:
            raise ValueError("target cannot be above 1.")
        elif target < 0:
            raise ValueError("target cannot be below 0.")
        else:
            return func(obj, target, *args, **kwargs)

    return wrap


def conditional_survival(model, x, X, *args, **kwargs):
    """Conditional survival from any model exposing ``sf``.

    Returns the probability of surviving a *further* ``x`` given the item has
    already survived to ``X``:

    .. math::
        R(x \\mid X) = \\frac{R(X + x)}{R(X)}

    Parameters
    ----------
    model : object
        Anything with an ``sf(x, ...)`` method (a distribution, a standby
        arrangement, an RBD, ...).
    x : array_like or scalar
        The further duration(s) at which conditional survival is evaluated.
    X : array_like or scalar
        The age(s) the item is known to have survived to.

    Returns
    -------
    float or numpy.ndarray
        The conditional survival probability, clipped to ``[0, 1]``; where the
        item has all but surely failed by ``X`` (``R(X) ≈ 0``) it is ``0``.
        A float if both ``x`` and ``X`` are scalars, otherwise an array.
    """
    scalar_in = np.ndim(x) == 0 and np.ndim(X) == 0
    x = np.atleast_1d(np.asarray(x, dtype=float))
    X = np.atleast_1d(np.asarray(X, dtype=float))
    denom = np.asarray(model.sf(X, *args, **kwargs), dtype=float)
    numer = np.asarray(model.sf(x + X, *args, **kwargs), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer / denom
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, 1.0)
    return out.item() if scalar_in else out
