import numpy as np


def check_probability(func):
    """checks probability is between 0 and 1"""

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
    numpy.ndarray
        The conditional survival probability, clipped to ``[0, 1]``; where the
        item has all but surely failed by ``X`` (``R(X) ≈ 0``) it is ``0``.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    X = np.atleast_1d(np.asarray(X, dtype=float))
    denom = np.asarray(model.sf(X, *args, **kwargs), dtype=float)
    numer = np.asarray(model.sf(x + X, *args, **kwargs), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer / denom
    out = np.where(np.isfinite(out), out, 0.0)
    return np.clip(out, 0.0, 1.0)
