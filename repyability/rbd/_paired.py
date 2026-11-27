"""Comparing two designs (``compare``): the exact difference where both
values are worked out, and otherwise the mean of paired differences, each
pair simulated with common random numbers.

Each element (a diagram's node, a tree's event, a network's link or node,
a mission's component) draws its uniforms from a stream of its own, keyed
by the comparison's key, its name and the uniform's place in its draw
(``uniforms``), so an element of the same name in both designs draws the
same numbers in both: the same life where its model is the same, and the
same quantile of its own model where it is not. A
``NonRepairableRBD``, a ``FaultTree`` (through its diagram, or its own
draws with dynamic gates), a ``Network`` and a ``PhasedMission`` all draw
from these streams, so their comparisons pair the same way.
"""

import zlib
from typing import Hashable, Optional

import numpy as np

from . import _montecarlo as montecarlo
from .results import ConfidenceInterval

#: What ``method`` may be.
METHODS = (None, "exact", "simulate")


def check(method, mc_samples, confidence) -> None:
    """Refuse a ``method``, ``mc_samples`` or ``confidence`` that is not
    valid."""
    if method not in METHODS:
        raise ValueError(
            "method must be None (exact where both values are), 'exact' "
            f"or 'simulate', got {method!r}."
        )
    montecarlo.check_count(mc_samples, False, "mc_samples")
    montecarlo.check_confidence(confidence)


def key(seed) -> int:
    """The comparison's key, from its ``seed``."""
    return int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0])


def uniforms(key: int, label: str, j: int, n: int) -> np.ndarray:
    """``n`` uniforms of the ``j``-th draw of the stream ``label`` (an
    element's is ``repr`` of its name, see ``element``)."""
    crc = zlib.crc32(label.encode())
    return np.random.default_rng([key, crc, j]).random(n)


def element(key: int, name: Hashable, j: int, n: int) -> np.ndarray:
    """``n`` uniforms of the element ``name``'s ``j``-th draw."""
    return uniforms(key, repr(name), j, n)


def exact(difference: float, confidence: float) -> ConfidenceInterval:
    """The exact ``difference``, with no error."""
    return ConfidenceInterval(
        estimate=difference,
        lower=difference,
        upper=difference,
        confidence=confidence,
        standard_error=0.0,
        n_samples=0,
        method="exact",
    )


def simulated(
    differences: np.ndarray,
    confidence: float,
    clip: Optional[float] = None,
) -> ConfidenceInterval:
    """The mean of the paired ``differences``, with its standard error
    and a normal interval (within ``[-clip, clip]`` where given: a
    difference of probabilities lies within ``[-1, 1]``)."""
    estimate = float(np.mean(differences))
    standard_error = montecarlo.standard_error(differences, False)
    half = montecarlo.z_value(confidence) * standard_error
    lower, upper = estimate - half, estimate + half
    if clip is not None:
        lower, upper = max(lower, -clip), min(upper, clip)
    return ConfidenceInterval(
        estimate=estimate,
        lower=lower,
        upper=upper,
        confidence=confidence,
        standard_error=standard_error,
        n_samples=len(differences),
        method="simulated",
    )


def same_kind(this, other, convert: str) -> None:
    """Refuse to compare ``this`` with ``other`` of another kind, naming
    the conversion (``convert``) that makes them one."""
    if not isinstance(other, type(this)):
        raise TypeError(
            f"{type(this).__name__}.compare takes another "
            f"{type(this).__name__}, got {type(other).__name__}: {convert}."
        )
