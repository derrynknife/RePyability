"""Checks of the arguments the public API shares."""

import math
from numbers import Real

import numpy as np


def is_whole(value) -> bool:
    """Whether ``value`` is a whole number: ``3`` or ``3.0``, but not
    ``True``."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, np.integer)):
        return True
    return isinstance(value, (float, np.floating)) and bool(
        float(value).is_integer()
    )


def whole_number(value, name: str, minimum: int = 1) -> int:
    """``value`` as an ``int``, if it is a whole number (see ``is_whole``)
    of at least ``minimum``; else a ``ValueError`` that names it."""
    if not is_whole(value) or value < minimum:
        raise ValueError(
            f"{name} must be a whole number, at least {minimum}, got "
            f"{value!r}."
        )
    return int(value)


#: The ways the structure is evaluated: through the minimal path sets or the
#: minimal cut sets (both give the same values).
_STRUCTURE_METHODS = {"p": "p", "paths": "p", "c": "c", "cuts": "c"}


def structure_method(method) -> str:
    """``"p"`` or ``"c"``, from a structure ``method``: ``"p"`` or
    ``"paths"`` for the minimal path sets, ``"c"`` or ``"cuts"`` for the
    minimal cut sets (#179); else a ``ValueError``."""
    try:
        return _STRUCTURE_METHODS[method]
    except (KeyError, TypeError):
        raise ValueError(
            "`method` must be 'p' (or 'paths') or 'c' (or 'cuts'), got "
            f"{method!r}."
        ) from None


def simulation_options(method: str, given: dict) -> None:
    """Refuse the simulation options in ``given`` (those passed: not None
    or False) to ``method``, whose answer is exact unless
    ``method="simulate"``: they would do nothing."""
    passed = [
        name
        for name, value in given.items()
        if value is not None and value is not False
    ]
    if passed:
        raise TypeError(
            f"{method} takes {', '.join(passed)} only with "
            "method='simulate': its answer is otherwise exact."
        )


def is_number(value) -> bool:
    """Whether ``value`` is a real number: an int or a float (numpy's
    too, or any ``numbers.Real``), but not a bool, nor text that reads as
    a number (#233)."""
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, Real):
        return True
    return (
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and value.dtype.kind in "iuf"
    )


def number_or_nan(value) -> float:
    """``value`` as a float if it is a real number (see ``is_number``),
    else NaN: for a check that then refuses it in its own words. Text such
    as ``"8760"``, which ``float`` would read, and booleans, which it would
    read as 1 and 0, are not numbers here (#233)."""
    return float(value) if is_number(value) else math.nan


def real_array(value, name: str) -> np.ndarray:
    """``value`` (a number or an array of them) as a float array, or a
    TypeError naming ``name``: text, which numpy would read as a number
    (``"8760"``), and booleans, which it would read as 1 and 0, are not
    numbers here (#233)."""
    array = np.asarray(value)
    kind = array.dtype.kind
    if kind == "O" and not any(
        isinstance(item, (str, bytes, bool, np.bool_))
        for item in array.ravel()
    ):
        return array.astype(float)
    if kind not in "iuf":
        raise TypeError(f"{name} must be numbers, got {value!r}.")
    return array.astype(float)


#: The seeds every Monte-Carlo method takes, in words.
SEEDS = (
    "a whole number from 0 to 2**32 - 1 (or a list of them), or None for "
    "an unseeded run"
)


def seed(value):
    """``value``, a seed, if it is one (see ``SEEDS``), else a TypeError
    or ValueError saying what one is (#232). A numpy ``Generator`` is no
    seed: the simulations draw from streams of their own, which one number
    seeds."""
    if value is None:
        return None
    if isinstance(value, np.random.Generator):
        raise TypeError(
            f"seed must be {SEEDS}, got a numpy Generator: the simulations "
            "draw from streams of their own, which one number seeds. Give "
            "one drawn from it, seed=int(rng.integers(2**32))."
        )
    whole = np.asarray(value) if isinstance(value, (list, tuple)) else value
    if isinstance(whole, np.ndarray):
        if (
            whole.size == 0
            or whole.dtype.kind not in "iu"
            or bool(np.any(whole < 0))
        ):
            raise ValueError(f"seed must be {SEEDS}, got {value!r}.")
        return value
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"seed must be {SEEDS}, got {value!r}.")
    if value < 0:
        raise ValueError(f"seed must be {SEEDS}, got {value!r}.")
    return value


def unfitted_distribution(model) -> "str | None":
    """The name of the surpyval distribution ``model`` is, where it is
    the distribution itself (``surv.Weibull``) rather than a model of it
    (fitted, or made with ``from_params``); None otherwise (#233)."""
    if not type(model).__module__.startswith("surpyval"):
        return None
    if not (hasattr(model, "fit") and hasattr(model, "from_params")):
        return None
    if hasattr(model, "params"):
        return None
    name = getattr(model, "name", None)
    return str(name) if name else type(model).__name__.rstrip("_")


def no_distribution(model, what: str) -> None:
    """Raise if ``model``, given as ``what``, is a surpyval distribution
    itself rather than a model of it, saying how to make one (#233)."""
    name = unfitted_distribution(model)
    if name is not None:
        raise TypeError(
            f"{what} is surpyval's {name} distribution itself, not a model "
            f"of it: fit it to data (surv.{name}.fit(times)) or give its "
            f"parameters (surv.{name}.from_params([...]))."
        )
