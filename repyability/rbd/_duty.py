"""Duty cycles: a node whose life is in operating time, run part of the
time.

A node that operates a fraction ``d`` of the calendar ages only then, so
its life in calendar time is its operating life over ``d``: ``R_c(t) =
R(d t)``. A surpyval parametric distribution of a time stays one under that
change of scale, with its parameters moved (``_SCALED``), so the node keeps
its model's kind and every exact method and simulation takes it as it would
the model fitted in calendar time.
"""

import math
from typing import Callable, Dict, List

from ._model_utils import distribution_name, is_fixed_probability


def _scale_first(p: List[float], d: float) -> List[float]:
    """A scale parameter first (a characteristic life), shapes after."""
    return [p[0] / d, *p[1:]]


def _location_scale(p: List[float], d: float) -> List[float]:
    """A location and a scale, both times."""
    return [p[0] / d, p[1] / d]


#: How each surpyval distribution's parameters move when its time is
#: divided by ``d`` (``R_c(t) = R(d t)``).
_SCALED: Dict[str, Callable[[List[float], float], List[float]]] = {
    "Exponential": lambda p, d: [p[0] * d],
    "Weibull": _scale_first,
    "ExpoWeibull": _scale_first,
    "LogLogistic": _scale_first,
    "Rayleigh": _scale_first,
    # The log of a time.
    "LogNormal": lambda p, d: [p[0] - math.log(d), p[1]],
    "Galton": lambda p, d: [p[0] - math.log(d), p[1]],
    # A shape and a rate.
    "Gamma": lambda p, d: [p[0], p[1] * d],
    "Normal": _location_scale,
    "Gauss": _location_scale,
    "Gumbel": _location_scale,
    "GumbelLEV": _location_scale,
    "Logistic": _location_scale,
    "Uniform": _location_scale,
}


def duty_fraction(node, value) -> float:
    """A node's duty, checked: the fraction of the time it operates, in
    ``(0, 1]``."""
    from repyability.utils.checks import number_or_nan

    d = number_or_nan(value)
    if not 0.0 < d <= 1.0:
        raise ValueError(
            f"The duty of node {node!r} must be the fraction of the time it "
            f"operates, in (0, 1], got {value!r}."
        )
    return d


def on_calendar(node, model, d: float):
    """``model``, a life in operating time, as the life on the calendar of
    a node that operates the fraction ``d`` of the time; raise for a model
    that is not a surpyval parametric distribution of a time."""
    if d == 1.0:
        return model
    name = distribution_name(model)
    move = _SCALED.get(name) if name is not None else None
    if move is None or is_fixed_probability(model):
        raise ValueError(
            f"Node {node!r} is given a duty, which moves a surpyval "
            "parametric distribution of a time onto the calendar, but its "
            f"life is {name or type(model).__name__}: give the life it has "
            "on the calendar instead."
        )
    import surpyval

    params = [float(p) for p in model.params]
    extras = dict(getattr(model, "extras", {}))
    if extras.get("gamma"):
        # An offset is a time too.
        extras["gamma"] = float(extras["gamma"]) / d
    assert name is not None  # a distribution of time (see _SCALED)
    return getattr(surpyval, name).from_params(move(params, d), **extras)
