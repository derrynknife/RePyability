import functools
from contextlib import contextmanager

import numpy as np


@contextmanager
def numpy_seed(seed):
    """Temporarily seed numpy's global RNG, restoring the previous state on
    exit.

    surpyval's ``.random()`` draws from numpy's *global* RNG and exposes no
    seed argument, so reproducible Monte-Carlo simulations are obtained by
    seeding that global RNG. This context manager seeds it for the duration of
    a simulation and restores the caller's RNG state afterwards, so calling a
    simulation with ``seed=...`` is reproducible *without* disturbing the
    surrounding program's random stream. ``seed=None`` is a no-op (i.e. the
    simulation stays non-reproducible, using whatever global state exists).

    Parameters
    ----------
    seed : int or None
        The seed to apply, or None to leave the global RNG untouched.
    """
    if seed is None:
        yield
        return
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


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
