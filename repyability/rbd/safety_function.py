"""A safety function: voted channels with proof tests, diagnostics,
repair and common cause, and the results a functional-safety study asks
of it (IEC 61508, IEC 61511): the average probability of failure on demand
(PFDavg), the average frequency of a dangerous failure (PFH), the
spurious-trip rate and the safety integrity level (SIL) each falls in.

``SafetyFunction`` is a builder. It draws two ``RepairableRBD``s and adds
no mathematics below them:

- the **dangerous** diagram, in which each channel is its undetected
  dangerous failures (hidden, found at the proof test) in series with its
  detected ones (revealed, repaired), and the channels are voted;
- the **safe** diagram, of the channels' safe failures under the dual
  vote: M-out-of-N channels acting trips the process, so the function
  stays clear of a spurious trip while N - M + 1 of them have not failed
  safe.

The PFDavg is the dangerous diagram's ``mean_unavailability()``, the PFH
its ``system_failure_frequency()``, and the spurious-trip rate the safe
diagram's ``system_failure_frequency()``.
"""

import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import numpy as np

from repyability._version import __version__
from repyability.rbd import _long_run, routes
from repyability.rbd import uncertainty as _uncertainty
from repyability.rbd._common import _constant_rate
from repyability.rbd._model_utils import MODEL_ERRORS
from repyability.rbd.ccf import BetaFactor, CCFGroup
from repyability.rbd.helper_classes import PerfectReliability
from repyability.rbd.results import (
    PFDBreakdown,
    SafetyFunctionResult,
    SILBand,
    SimplifiedSafetyFormulas,
    UncertaintyResult,
)
from repyability.rbd.routes import AnalysisRoute, AnalysisRoutes
from repyability.utils.checks import (
    is_number,
    no_distribution,
    number_or_nan,
    one_of,
    whole_number,
)
from repyability.utils.deprecation import refuse_removed_names
from repyability.utils.json_io import read_json, write_json

#: The demand modes of IEC 61508-4, and the measure each one's SIL is
#: read from.
MODES = ("low_demand", "high_demand", "continuous")
_MEASURE = {
    "low_demand": "pfd_avg",
    "high_demand": "pfh",
    "continuous": "pfh",
}

#: The lower edge of each SIL's band (IEC 61508-1 Tables 2 and 3: "at
#: least 1e-5 and less than 1e-4" is SIL 4): the PFDavg's, and the PFH's
#: per hour. A band's upper edge is the next one's lower edge.
_SIL_EDGES = {
    "pfd_avg": {4: 1e-5, 3: 1e-4, 2: 1e-3, 1: 1e-2, 0: 1e-1},
    "pfh": {4: 1e-9, 3: 1e-8, 2: 1e-7, 1: 1e-6, 0: 1e-5},
}

#: The votes IEC 61508-6 Annex B gives simplified formulas for (B.3.2.2
#: and B.3.3.2; its 1oo2D apart), as (M, N).
_ANNEX_B_VOTES = ((1, 1), (1, 2), (2, 2), (2, 3), (1, 3))


def sil_band(value, mode: str, hours_per_unit: float = 1.0) -> SILBand:
    """The safety integrity level a PFDavg or a PFH falls in, by IEC
    61508-1 Tables 2 and 3.

    Each band takes its lower edge and not its upper one: a PFDavg of
    exactly ``1e-3`` is SIL 2, and anything below it down to ``1e-4`` is
    SIL 3. The tables stop at SIL 4, so a value below its band is still
    SIL 4, and a value at or above SIL 1's upper edge has no SIL (0).

    Parameters
    ----------
    value : float
        The PFDavg (``mode="low_demand"``), a probability, or the PFH
        (``"high_demand"`` or ``"continuous"``), per unit of time.
    mode : {"low_demand", "high_demand", "continuous"}
        The function's demand mode, which says which table applies. It is
        never guessed.
    hours_per_unit : float, optional
        For a PFH: how many hours one unit of its time is (8760 for a
        rate per year). The table's bands are per hour. By default 1, a
        PFH per hour.

    Returns
    -------
    SILBand
        The level, its band's edges, the value's margin to each, and for
        a PFDavg the risk reduction factor.

    Raises
    ------
    ValueError
        If ``mode`` is not one of the three, or ``value`` is not a
        number of at least 0, or ``hours_per_unit`` is not positive.

    Examples
    --------
    >>> from repyability import sil_band
    >>> band = sil_band(4.4e-3, "low_demand")
    >>> band.sil, band.lower, band.upper
    (2, 0.001, 0.01)
    >>> round(band.risk_reduction_factor)
    227
    >>> sil_band(1e-3, "low_demand").sil  # a band takes its lower edge
    2
    >>> sil_band(5e-4, "high_demand", hours_per_unit=8760).sil  # per year
    3
    """
    one_of("mode", mode, MODES)
    amount = number_or_nan(value)
    if not amount >= 0.0:
        raise ValueError(
            f"value must be a number of at least 0, got {value!r}."
        )
    hours = number_or_nan(hours_per_unit)
    if not (hours > 0.0 and math.isfinite(hours)):
        raise ValueError(
            "hours_per_unit must be a positive, finite number, got "
            f"{hours_per_unit!r}."
        )
    measure = _MEASURE[mode]
    edges = _SIL_EDGES[measure]
    judged = amount / hours if measure == "pfh" else amount
    sil = next(
        (level for level in (4, 3, 2, 1) if judged < edges[level - 1]), 0
    )
    lower = edges[sil]
    upper = edges[sil - 1] if sil else math.inf
    return SILBand(
        sil=sil,
        mode=mode,
        measure=measure,
        value=judged,
        lower=lower,
        upper=upper,
        margin_to_lower=judged / lower,
        margin_to_upper=math.inf if judged == 0.0 else upper / judged,
        risk_reduction_factor=(
            None
            if measure == "pfh"
            else (math.inf if judged == 0.0 else 1.0 / judged)
        ),
    )


def _rate(value, name: str) -> float:
    rate = number_or_nan(value)
    if not (rate >= 0.0 and math.isfinite(rate)):
        raise ValueError(
            f"{name} must be a failure rate, a finite number of at least "
            f"0, got {value!r}."
        )
    return rate


def _fraction(value, name: str) -> float:
    fraction = number_or_nan(value)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}.")
    return fraction


def _repair_model(value, name: str):
    """A repair as a ``RepairableRBD`` spec's ``"repairability"``, and
    its mean: a mean time (an exponential time of that mean; 0 is
    instant), ``"instant"``, or a surpyval model as it is."""
    import surpyval as surv

    if isinstance(value, str):
        if value == "instant":
            return "instant", 0.0
        raise ValueError(
            f"{name} must be a mean time, 'instant' or a surpyval model, "
            f"got {value!r}."
        )
    if is_number(value):
        mean = float(value)
        if not (mean >= 0.0 and math.isfinite(mean)):
            raise ValueError(
                f"{name} must be a mean time of at least 0, 'instant' or "
                f"a surpyval model, got {value!r}."
            )
        if mean == 0.0:
            return "instant", 0.0
        return surv.Exponential.from_params([1.0 / mean]), mean
    no_distribution(value, name)
    try:
        return value, float(value.mean())
    except (AttributeError, *MODEL_ERRORS):
        raise ValueError(
            f"{name} must be a mean time of at least 0, 'instant' or a "
            f"surpyval model, got {value!r}."
        ) from None


@dataclass(frozen=True)
class SafetyChannel:
    """One channel of a safety function, as plain data: its failure rates
    by kind, its repair and its proof test.

    A channel's failures are split two ways, as IEC 61508 splits them.
    **Dangerous** failures stop the channel acting on a demand; **safe**
    ones make it act with no demand (a spurious trip signal). A dangerous
    failure is **detected** where the diagnostics reveal it at once, and
    is repaired; it is **undetected** where only the proof test finds
    it. The rates are per unit of the function's time (per hour, where
    the PFH is to be read against the standard's bands as it is).

    Parameters
    ----------
    dangerous_undetected : float or surpyval model
        The rate of dangerous failures only the proof test finds, or a
        life model for them (a fitted Weibull, say), which the tested
        unit's model takes as it is.
    dangerous_detected : float, optional
        The rate of dangerous failures the diagnostics reveal. By
        default 0.
    safe : float, optional
        The rate of safe failures. By default 0.
    repair : float, "instant" or surpyval model
        The repair of a revealed failure (a detected dangerous one, or a
        safe one): its mean time (the standard's MTTR; taken as an
        exponential time, and 0 as instant), or a model of its time.
    proof_test_interval : float
        The time between proof tests (the standard's T1).
    proof_test_coverage : float, optional
        The chance a proof test finds an undetected failure that is
        there. Below 1, the failures it misses wait for ``full_test``.
        By default 1.
    full_test : float, optional
        The interval of the test that finds everything (an overhaul), a
        whole multiple of ``proof_test_interval``; needed with a
        coverage below 1.
    proof_test_offset : float, optional
        The time of the first proof test, less than the interval, for
        channels tested at different times (staggered). By default 0:
        the channel is tested at every multiple of the interval.
    undetected_repair : float, "instant" or surpyval model, optional
        The repair of a failure a proof test finds (the standard's MRT),
        given as ``repair`` is. By default the same as ``repair``.

    Raises
    ------
    ValueError
        If a rate is negative, a coverage is outside ``[0, 1]``, or a
        repair is none of the three.

    Examples
    --------
    A transmitter failing dangerously five times in a million hours, 60%
    of it revealed by its diagnostics, proof-tested yearly:

    >>> from repyability import SafetyChannel
    >>> channel = SafetyChannel.from_rate(
    ...     1e-5, dangerous_fraction=0.5, diagnostic_coverage=0.6,
    ...     repair=8.0, proof_test_interval=8760.0,
    ... )
    >>> f"{channel.dangerous_undetected:.1e}, {channel.dangerous_detected:.1e}"
    '2.0e-06, 3.0e-06'
    >>> round(channel.safe_failure_fraction, 6)
    0.8
    """

    dangerous_undetected: Any
    dangerous_detected: float = 0.0
    safe: float = 0.0
    repair: Any = None
    proof_test_interval: Any = None
    proof_test_coverage: float = 1.0
    full_test: Optional[float] = None
    proof_test_offset: float = 0.0
    undetected_repair: Any = None

    def __post_init__(self):
        if is_number(self.dangerous_undetected):
            _rate(self.dangerous_undetected, "dangerous_undetected")
        else:
            no_distribution(self.dangerous_undetected, "dangerous_undetected")
            if not callable(getattr(self.dangerous_undetected, "sf", None)):
                raise ValueError(
                    "dangerous_undetected must be a failure rate (a "
                    "number of at least 0) or a surpyval model, got "
                    f"{self.dangerous_undetected!r}."
                )
        _rate(self.dangerous_detected, "dangerous_detected")
        _rate(self.safe, "safe")
        if self.repair is None:
            raise ValueError(
                "A SafetyChannel needs its repair: a mean time to "
                "restoration (repair=8.0), 'instant' or a surpyval model."
            )
        _repair_model(self.repair, "repair")
        if self.undetected_repair is not None:
            _repair_model(self.undetected_repair, "undetected_repair")
        interval = number_or_nan(self.proof_test_interval)
        if not (interval > 0.0 and math.isfinite(interval)):
            raise ValueError(
                "A SafetyChannel needs its proof_test_interval, a "
                f"positive, finite time, got {self.proof_test_interval!r}."
            )
        _fraction(self.proof_test_coverage, "proof_test_coverage")
        offset = number_or_nan(self.proof_test_offset)
        if not 0.0 <= offset < interval:
            raise ValueError(
                "proof_test_offset must be a time of at least 0 and less "
                "than the proof_test_interval, got "
                f"{self.proof_test_offset!r}."
            )

    @classmethod
    def from_rate(
        cls,
        rate: float,
        dangerous_fraction: float = 0.5,
        diagnostic_coverage: float = 0.0,
        **options,
    ) -> "SafetyChannel":
        """A channel from its total failure rate, split into dangerous and
        safe, and the dangerous part into detected and undetected.

        Parameters
        ----------
        rate : float
            The channel's failure rate, of every kind.
        dangerous_fraction : float, optional
            The share of the failures that are dangerous; the rest are
            safe. By default 0.5, as IEC 61508-6's tables take it.
        diagnostic_coverage : float, optional
            The share of the dangerous failures the diagnostics detect
            (the standard's DC). By default 0.
        **options
            The channel's other arguments (``repair``,
            ``proof_test_interval``, ...).

        Returns
        -------
        SafetyChannel
            With ``dangerous_undetected = rate * dangerous_fraction *
            (1 - diagnostic_coverage)``, ``dangerous_detected = rate *
            dangerous_fraction * diagnostic_coverage`` and ``safe = rate
            * (1 - dangerous_fraction)``.
        """
        total = _rate(rate, "rate")
        dangerous = total * _fraction(dangerous_fraction, "dangerous_fraction")
        coverage = _fraction(diagnostic_coverage, "diagnostic_coverage")
        return cls(
            dangerous_undetected=dangerous * (1.0 - coverage),
            dangerous_detected=dangerous * coverage,
            safe=total - dangerous,
            **options,
        )

    @property
    def undetected_rate(self) -> Optional[float]:
        """The rate of undetected dangerous failures, where it is
        constant (a number, or a model of a constant rate); else None."""
        if is_number(self.dangerous_undetected):
            return float(self.dangerous_undetected)
        return _constant_rate(self.dangerous_undetected)

    @property
    def diagnostic_coverage(self) -> float:
        """The share of the dangerous failures that are detected (DC):
        ``dangerous_detected`` over the dangerous rate. NaN where the
        undetected failures' rate is not constant, or nothing fails
        dangerously."""
        undetected = self.undetected_rate
        if undetected is None:
            return math.nan
        dangerous = undetected + float(self.dangerous_detected)
        if dangerous == 0.0:
            return math.nan
        return float(self.dangerous_detected) / dangerous

    @property
    def safe_failure_fraction(self) -> float:
        """The safe failure fraction (SFF): the safe and the detected
        dangerous rates over the total rate. NaN where the undetected
        failures' rate is not constant, or the channel never fails."""
        undetected = self.undetected_rate
        if undetected is None:
            return math.nan
        revealed = float(self.safe) + float(self.dangerous_detected)
        if revealed + undetected == 0.0:
            return math.nan
        return revealed / (revealed + undetected)

    def _life(self):
        """The undetected failures' life, or None where there are none."""
        import surpyval as surv

        if not is_number(self.dangerous_undetected):
            return self.dangerous_undetected
        rate = float(self.dangerous_undetected)
        return surv.Exponential.from_params([rate]) if rate > 0.0 else None

    def _repairs(self) -> Tuple[Tuple[Any, float], Tuple[Any, float]]:
        """The revealed failures' repair and the proof test's, each as a
        spec's ``"repairability"`` with its mean."""
        revealed = _repair_model(self.repair, "repair")
        if self.undetected_repair is None:
            return revealed, revealed
        return revealed, _repair_model(
            self.undetected_repair, "undetected_repair"
        )


@dataclass(frozen=True)
class _Stage:
    """One vote of a function: its channels by name, how many must act,
    and its common-cause fractions."""

    channels: Dict[Hashable, SafetyChannel]
    k: int
    beta: float
    beta_detected: float
    beta_safe: float
    detected_trips: bool

    def detected(self, channel: SafetyChannel) -> float:
        """The channel's rate of detected failures that leave it
        dangerous: none where a detected failure trips it."""
        return (
            0.0 if self.detected_trips else float(channel.dangerous_detected)
        )

    def tripping(self, channel: SafetyChannel) -> float:
        """The channel's rate of failures that make it vote to trip."""
        extra = (
            float(channel.dangerous_detected) if self.detected_trips else 0.0
        )
        return float(channel.safe) + extra


@dataclass(frozen=True)
class _Series:
    """Parts in series: the function acts when every part does, and
    trips spuriously when any part does."""

    parts: Tuple[Any, ...]


@dataclass(frozen=True)
class _Voted:
    """A vote of parts (each a vote, a vote of votes or a series): it acts
    when ``k`` of them act, and trips spuriously when ``k`` of them trip,
    with a shared cause over every channel beneath it by kind."""

    parts: Tuple[Any, ...]
    k: int
    beta: float
    beta_detected: float
    beta_safe: float


_Part = Union[_Stage, _Series, _Voted]


def _leaves(part: _Part) -> List[_Stage]:
    """The votes of channels in ``part``, in order."""
    if isinstance(part, _Stage):
        return [part]
    return [stage for inner in part.parts for stage in _leaves(inner)]


def _with_channels(part: _Part, channels: Dict[Hashable, Any]) -> _Part:
    """``part`` with the channels named in ``channels`` replaced."""
    if isinstance(part, _Stage):
        return dataclasses.replace(
            part,
            channels={
                name: channels.get(name, channel)
                for name, channel in part.channels.items()
            },
        )
    return dataclasses.replace(
        part, parts=tuple(_with_channels(p, channels) for p in part.parts)
    )


def _described(part: _Part, inside: bool = False) -> str:
    if isinstance(part, _Stage):
        return f"{part.k}oo{len(part.channels)}"
    if isinstance(part, _Voted):
        inner = ", ".join(_described(p, True) for p in part.parts)
        return f"{part.k}oo{len(part.parts)}({inner})"
    inner = ", ".join(_described(p, True) for p in part.parts)
    return f"[{inner}]" if inside and len(part.parts) > 1 else inner


def _shared_groups(part: _Part) -> List[List[Hashable]]:
    """The channels each shared cause spans, by name (a vote's with a
    beta, of any kind)."""
    out = []
    if isinstance(part, _Stage):
        if len(part.channels) > 1 and (
            part.beta or part.beta_detected or part.beta_safe
        ):
            out.append(list(part.channels))
        return out
    if isinstance(part, _Voted) and (
        part.beta or part.beta_detected or part.beta_safe
    ):
        out.append([name for s in _leaves(part) for name in s.channels])
    for inner in part.parts:
        out.extend(_shared_groups(inner))
    return out


def _parts(method: str, functions, what: str) -> List[_Series]:
    """The structures of ``functions``, checked: at least one, each a
    ``SafetyFunction``, no channel's name in two."""
    if not functions:
        raise ValueError(f"{method} needs at least one SafetyFunction.")
    seen: set = set()
    out = []
    for function in functions:
        if not isinstance(function, SafetyFunction):
            raise ValueError(
                f"{method} takes SafetyFunctions, got {function!r}."
            )
        shared = seen & set(function.channels)
        if shared:
            joined = "put in series" if what == "series" else "voted on"
            raise ValueError(
                f"The functions {joined} must name their channels "
                f"differently; {sorted(map(repr, shared))} is in more "
                "than one."
            )
        seen |= set(function.channels)
        out.append(function._structure)
    return out


def _shared_over(part: "_Voted", name: str) -> None:
    """Refuse a shared cause ``name`` over a vote of votes whose channels
    differ, or one of which shares a cause of that kind already."""
    stages = _leaves(part)
    channels = [c for stage in stages for c in stage.channels.values()]
    names = [n for stage in stages for n in stage.channels]
    if any(c != channels[0] for c in channels[1:]) or any(
        s.detected_trips != stages[0].detected_trips for s in stages
    ):
        raise ValueError(
            f"{name} needs the channels beneath the vote alike, as a beta "
            "factor shares a fraction of one rate among them, and "
            f"channels {names!r} differ. Give every channel the same "
            f"SafetyChannel (and detected_trips), or {name}=0."
        )

    def already(inner) -> bool:
        if isinstance(inner, _Stage):
            return len(inner.channels) > 1 and getattr(inner, name) > 0.0
        if isinstance(inner, _Voted) and getattr(inner, name) > 0.0:
            return True
        return any(already(p) for p in inner.parts)

    if any(already(inner) for inner in part.parts):
        raise ValueError(
            f"{name} on a vote of votes shares a cause over every channel "
            "beneath it, and some of them share one of that kind already "
            "(a channel is in one shared cause of each kind). Give the "
            f"{name} on the outer vote alone, or on the inner ones alone."
        )


_IN, _OUT = ("function", "input"), ("function", "output")


def _alike(stage: _Stage, what: str, name: str) -> None:
    channels = list(stage.channels.values())
    if any(channel != channels[0] for channel in channels[1:]):
        raise ValueError(
            f"{name} needs the vote's channels alike, as a beta factor "
            f"shares a fraction of one {what} rate among them, and "
            f"channels {list(stage.channels)!r} differ. Give every "
            f"channel of the vote the same SafetyChannel, or {name}=0."
        )


@refuse_removed_names
class SafetyFunction:
    """A safety function: channels voted M-out-of-N, with proof tests,
    diagnostics, repair and common cause.

    One call (``results``) gives its PFDavg, its PFH, its spurious-trip
    rate and the SIL its demand mode reads. Each is worked out from one
    of two ``RepairableRBD``s the function draws, ``dangerous`` and
    ``safe`` (see below), by the methods those have: no rare-event
    formula stands between the channels and the answer. The standards'
    simplified formulas are ``simplified_formulas()``, to set beside
    them. See [safety functions](../guide/safety-functions.md).

    A function of several votes in series (sensors, a logic solver,
    final elements) is ``SafetyFunction.series`` of one function each.

    Parameters
    ----------
    channels : dict
        The vote's channels, a [`SafetyChannel`][repyability.SafetyChannel]
        by name.
    k : int
        How many of the channels must act for the function to (the M of
        MooN): 1 for 1oo2, 2 for 2oo3.
    beta : float, optional
        The fraction of each channel's undetected dangerous failures
        that come from a cause shared by every channel of the vote (a
        [`BetaFactor`][repyability.BetaFactor] by rate). By default 0.
    beta_detected : float, optional
        The same for the detected dangerous failures (the standard's
        beta D). By default 0.
    beta_safe : float, optional
        The same for the safe failures. By default 0.
    detected_trips : bool, optional
        Whether a channel's detected dangerous failure makes it vote to
        trip. By default False: the channel stays dangerous until it is
        repaired, counting towards the PFDavg and the PFH. With True the
        detected failures are safe ones: they leave the dangerous
        diagram and join the safe failures in the spurious-trip rate.

    Attributes
    ----------
    dangerous : RepairableRBD
        The diagram of the dangerous failures: each channel's undetected
        failures, a component ``(name, "DU")`` with the proof test as its
        ``"inspection"``, in series with its detected ones, ``(name,
        "DD")``, the channels voted ``k``-out-of-N. Every ``RepairableRBD``
        analysis is open on it (the PFD over time, importances,
        ``optimal_inspection_intervals``, simulation).
    safe : RepairableRBD or None
        The diagram of the safe failures, components ``(name, "S")``,
        voted (N - k + 1)-out-of-N: the function trips spuriously when
        ``k`` channels have failed safe together. None where no channel
        can fail safe.

    Raises
    ------
    ValueError
        If ``k`` is not a whole number from 1 to the number of channels,
        a beta is outside ``[0, 1]``, or a beta is given for channels
        that are not alike.

    Examples
    --------
    Two transmitters voted 1oo2, each with a dangerous undetected rate of
    2 per million hours, proof-tested yearly, with a 10% common cause:

    >>> from repyability import SafetyChannel, SafetyFunction
    >>> channel = SafetyChannel(
    ...     2e-6, dangerous_detected=3e-6, safe=5e-6,
    ...     repair=8.0, proof_test_interval=8760.0,
    ... )
    >>> function = SafetyFunction({"a": channel, "b": channel}, k=1, beta=0.1)
    >>> result = function.results("low_demand")
    >>> f"{result.pfd_avg:.3e}", result.sil.sil
    ('9.585e-04', 3)
    >>> f"{result.pfh:.3e}"
    '2.746e-07'
    >>> f"{result.spurious_trip_rate:.3e}"
    '9.999e-06'
    """

    def __init__(
        self,
        channels: Dict[Hashable, SafetyChannel],
        k: int,
        beta: float = 0.0,
        beta_detected: float = 0.0,
        beta_safe: float = 0.0,
        detected_trips: bool = False,
    ):
        try:
            channels = dict(channels)
        except (TypeError, ValueError):
            raise ValueError(
                "channels must be a dict of SafetyChannel by name, got "
                f"{channels!r}."
            ) from None
        if not channels:
            raise ValueError("A SafetyFunction needs at least one channel.")
        for name, channel in channels.items():
            if not isinstance(channel, SafetyChannel):
                raise ValueError(
                    f"Channel {name!r} must be a SafetyChannel, got "
                    f"{channel!r}."
                )
        k = whole_number(k, "k")
        if k > len(channels):
            raise ValueError(
                f"k must be at most the number of channels, "
                f"{len(channels)}, got {k}."
            )
        if not isinstance(detected_trips, bool):
            raise ValueError(
                "detected_trips must be True or False, got "
                f"{detected_trips!r}."
            )
        stage = _Stage(
            channels,
            k,
            _fraction(beta, "beta"),
            _fraction(beta_detected, "beta_detected"),
            _fraction(beta_safe, "beta_safe"),
            detected_trips,
        )
        self._build(_Series((stage,)))

    @classmethod
    def series(cls, *functions: "SafetyFunction") -> "SafetyFunction":
        """A function of several votes in series: it acts when every
        one of ``functions`` does (sensors, logic and final elements,
        say), and trips spuriously when any of them does.

        The votes are drawn into one dangerous and one safe diagram, so
        the results are the whole function's own: votes proof-tested at
        the same times are not treated as independent on average.

        Parameters
        ----------
        *functions : SafetyFunction
            The votes, their channels named differently.

        Returns
        -------
        SafetyFunction
            The function of them all.

        Raises
        ------
        ValueError
            If none is given, one is not a ``SafetyFunction``, or two
            share a channel's name.
        """
        parts = _parts("series", functions, "series")
        out = cls.__new__(cls)
        out._build(
            _Series(
                tuple(
                    inner
                    for part in parts
                    for inner in (
                        part.parts if isinstance(part, _Series) else (part,)
                    )
                )
            )
        )
        return out

    @classmethod
    def vote(
        cls,
        k: int,
        *functions: "SafetyFunction",
        beta: float = 0.0,
        beta_detected: float = 0.0,
        beta_safe: float = 0.0,
    ) -> "SafetyFunction":
        """A vote of votes: the function acts when ``k`` of ``functions``
        act, and trips spuriously when ``k`` of them trip.

        A 2oo4 arrangement wired as 1oo2 of 2oo2 is ``vote(1, pair_a,
        pair_b)``, each pair a 2oo2 ``SafetyFunction``. A function voted
        on may itself be a vote of votes or votes in series (a sensor
        vote and its own logic, say). Both diagrams are drawn whole, so
        the results are the function's own.

        Parameters
        ----------
        k : int
            How many of ``functions`` must act, from 1 to their number.
        *functions : SafetyFunction
            What is voted on, their channels named differently.
        beta : float, optional
            The fraction of each channel's undetected dangerous failures
            that come from a cause shared by every channel beneath the
            vote (a [`BetaFactor`][repyability.BetaFactor] by rate).
            Needs the channels alike, and none of them in a shared cause
            of that kind already. By default 0.
        beta_detected : float, optional
            The same for the detected dangerous failures. By default 0.
        beta_safe : float, optional
            The same for the safe failures. By default 0.

        Returns
        -------
        SafetyFunction
            The function of the vote.

        Raises
        ------
        ValueError
            If ``k`` is not a whole number from 1 to the number of
            functions, one is not a ``SafetyFunction``, two share a
            channel's name, or a beta is given for channels that are not
            alike or already share a cause of its kind.

        Examples
        --------
        Two pairs of transmitters, each pair voting 2oo2, the pairs 1oo2:

        >>> from repyability import SafetyChannel, SafetyFunction
        >>> unit = SafetyChannel(
        ...     2e-6, dangerous_detected=3e-6, safe=5e-6,
        ...     repair=8.0, proof_test_interval=8760.0,
        ... )
        >>> pair_a = SafetyFunction({"a1": unit, "a2": unit}, k=2)
        >>> pair_b = SafetyFunction({"b1": unit, "b2": unit}, k=2)
        >>> function = SafetyFunction.vote(1, pair_a, pair_b, beta=0.05)
        >>> function
        SafetyFunction(1oo2(2oo2, 2oo2); channels=['a1', 'a2', 'b1', 'b2'])
        >>> f"{function.pfd_avg():.3e}"
        '8.007e-04'
        """
        parts = _parts("vote", functions, "vote")
        k = whole_number(k, "k")
        if k > len(parts):
            raise ValueError(
                f"k must be at most the number of functions voted on, "
                f"{len(parts)}, got {k}."
            )
        part = _Voted(
            tuple(
                (
                    p.parts[0]
                    if isinstance(p, _Series) and len(p.parts) == 1
                    else p
                )
                for p in parts
            ),
            k,
            _fraction(beta, "beta"),
            _fraction(beta_detected, "beta_detected"),
            _fraction(beta_safe, "beta_safe"),
        )
        for name, value in (
            ("beta", part.beta),
            ("beta_detected", part.beta_detected),
            ("beta_safe", part.beta_safe),
        ):
            if value > 0.0:
                _shared_over(part, name)
        out = cls.__new__(cls)
        out._build(_Series((part,)))
        return out

    def _build(self, structure: _Series) -> None:
        from repyability.rbd.repairable_rbd import RepairableRBD

        self._structure = structure
        stages = _leaves(structure)
        self._stages = stages
        self.channels: Dict[Hashable, SafetyChannel] = {
            name: channel
            for stage in stages
            for name, channel in stage.channels.items()
        }
        self._undetected: set = set()
        edges: List[tuple] = []
        components: Dict[Any, Any] = {}
        votes: Dict[Any, int] = {}
        groups: List[CCFGroup] = []
        safe_edges: List[tuple] = []
        safe_components: Dict[Any, Any] = {}
        safe_votes: Dict[Any, int] = {}
        safe_groups: List[CCFGroup] = []
        count = iter(range(1 << 30))
        # The nodes of each kind a stage draws, for the shared causes of
        # the votes above it.
        kinds: Dict[int, Dict[str, list]] = {}

        def voted(vote, k, n):
            votes[vote] = k
            safe_votes[vote] = n - k + 1
            components[vote] = PerfectReliability
            safe_components[vote] = PerfectReliability

        def shared(nodes, beta, tripping):
            (safe_groups if tripping else groups).append(
                CCFGroup(nodes, BetaFactor(beta, basis="rate"))
            )

        def draw(part, before):
            if isinstance(part, _Series):
                for inner in part.parts:
                    before = draw(inner, before)
                return before
            if isinstance(part, _Voted):
                ends = [draw(inner, before) for inner in part.parts]
                vote = ("vote", next(count))
                for end in ends:
                    edges.append((end, vote))
                    safe_edges.append((end, vote))
                voted(vote, part.k, len(part.parts))
                beneath = [kinds[id(stage)] for stage in _leaves(part)]
                for beta, kind in (
                    (part.beta, "undetected"),
                    (part.beta_detected, "detected"),
                    (part.beta_safe, "tripping"),
                ):
                    nodes = [node for nodes in beneath for node in nodes[kind]]
                    if beta > 0.0 and len(nodes) > 1:
                        shared(nodes, beta, kind == "tripping")
                return vote
            return stage_drawn(part, before)

        def stage_drawn(stage, before):
            vote = ("vote", next(count))
            undetected, detected, tripping = [], [], []
            for name, channel in stage.channels.items():
                (repair, _), (test_repair, _) = channel._repairs()
                chain = []
                life = channel._life()
                if life is not None:
                    inspection = {
                        "interval": float(channel.proof_test_interval),
                        "coverage": float(channel.proof_test_coverage),
                    }
                    if channel.full_test is not None:
                        inspection["full_test"] = channel.full_test
                    if channel.proof_test_offset:
                        inspection["offset"] = float(channel.proof_test_offset)
                    components[(name, "DU")] = {
                        "reliability": life,
                        "repairability": test_repair,
                        "inspection": inspection,
                    }
                    chain.append((name, "DU"))
                    undetected.append((name, "DU"))
                if stage.detected(channel) > 0.0:
                    components[(name, "DD")] = _revealed(
                        stage.detected(channel), repair
                    )
                    chain.append((name, "DD"))
                    detected.append((name, "DD"))
                if not chain:
                    components[(name, "DU")] = PerfectReliability
                    chain.append((name, "DU"))
                edges.append((before, chain[0]))
                edges.extend(zip(chain, chain[1:]))
                edges.append((chain[-1], vote))
                safe = (name, "S")
                if stage.tripping(channel) > 0.0:
                    safe_components[safe] = _revealed(
                        stage.tripping(channel), repair
                    )
                    tripping.append(safe)
                else:
                    safe_components[safe] = PerfectReliability
                safe_edges.extend([(before, safe), (safe, vote)])
            n = len(stage.channels)
            for nodes, beta, what, name in (
                (undetected, stage.beta, "undetected dangerous", "beta"),
                (
                    detected,
                    stage.beta_detected,
                    "detected dangerous",
                    "beta_detected",
                ),
                (tripping, stage.beta_safe, "safe", "beta_safe"),
            ):
                if beta > 0.0 and n > 1 and nodes:
                    _alike(stage, what, name)
                    shared(nodes, beta, nodes is tripping)
            self._undetected |= set(undetected)
            kinds[id(stage)] = {
                "undetected": undetected,
                "detected": detected,
                "tripping": tripping,
            }
            voted(vote, stage.k, n)
            return vote

        end = draw(structure, _IN)
        edges.append((end, _OUT))
        safe_edges.append((end, _OUT))
        self.dangerous = RepairableRBD(
            edges, components, k=votes, ccf_groups=groups or None
        )
        self.safe: Optional[Any] = None
        if any(
            stage.tripping(channel) > 0.0
            for stage in stages
            for channel in stage.channels.values()
        ):
            self.safe = RepairableRBD(
                safe_edges,
                safe_components,
                k=safe_votes,
                ccf_groups=safe_groups or None,
            )

    def __repr__(self) -> str:
        return (
            f"SafetyFunction({_described(self._structure)}; "
            f"channels={list(self.channels)!r})"
        )

    def pfd_avg(self) -> float:
        """The average probability of failure on demand (PFDavg): the
        long-run fraction of time the function could not act on a
        demand.

        The dangerous diagram's
        [`mean_unavailability`][repyability.RepairableRBD.mean_unavailability]:
        the probability that the function is down, at each time of the
        proof tests' period, averaged over it. It is exact (or numerical
        to a stated accuracy) and holds whatever the rates and the
        intervals: no ``lambda * T << 1`` is assumed.

        Returns
        -------
        float
            The PFDavg, in ``[0, 1]``.

        Raises
        ------
        NotImplementedError
            As ``RepairableRBD.mean_unavailability`` does; see
            ``analysis_routes()``.
        """
        return float(self.dangerous.mean_unavailability())

    def pfd_breakdown(self) -> PFDBreakdown:
        """Where the PFDavg comes from: each channel's undetected
        failures' share (``(name, "DU")``), its detected failures'
        (``(name, "DD")``), and each common cause's, adding up to it.

        The dangerous diagram's
        [`pfd_breakdown`][repyability.RepairableRBD.pfd_breakdown].

        Returns
        -------
        PFDBreakdown
            The PFDavg, its independent and common-cause parts, and
            every share.
        """
        return self.dangerous.pfd_breakdown()

    def pfh(self, undetected_only: bool = False) -> float:
        """The average frequency of a dangerous failure of the function
        (PFH): how often, per unit time in the long run, it goes from
        able to act to unable.

        The dangerous diagram's
        [`system_failure_frequency`][repyability.RepairableRBD.system_failure_frequency]:
        at each time of the proof tests' period, each channel's failure
        rate times the probability that its failure takes the function
        down, averaged over the period.

        Parameters
        ----------
        undetected_only : bool, optional
            Count only the function's failures that an undetected
            failure completes, as IEC 61508-6 Annex B's formulas do (they
            take a detected failure to be answered by bringing the
            equipment to a safe state). By default False: every
            dangerous failure of the function counts, whichever kind of
            channel failure completes it.

        Returns
        -------
        float
            Dangerous failures of the function per unit time.

        Raises
        ------
        NotImplementedError
            As ``RepairableRBD.system_failure_frequency`` does; see
            ``analysis_routes()``.
        """
        if not isinstance(undetected_only, bool):
            raise ValueError(
                "undetected_only must be True or False, got "
                f"{undetected_only!r}."
            )
        if not undetected_only:
            return float(self.dangerous.system_failure_frequency())
        terms, _ = _long_run._outage_terms(self.dangerous)
        total = 0.0
        for cause, term in terms:
            if cause in self._undetected or (
                isinstance(cause, tuple)
                and cause not in self.dangerous.components
                and all(member in self._undetected for member in cause)
            ):
                total += term
        return float(total)

    def spurious_trip_rate(self) -> float:
        """How often the function trips with no demand, per unit time in
        the long run.

        The safe diagram's
        [`system_failure_frequency`][repyability.RepairableRBD.system_failure_frequency]:
        a vote of ``k``-out-of-N trips when ``k`` of its channels have
        failed safe and are not yet repaired, and a function of several
        votes when any of them does. With ``detected_trips=True`` a
        detected dangerous failure counts as a safe one.

        Returns
        -------
        float
            Spurious trips per unit time; 0 where no channel can fail
            safe.
        """
        if self.safe is None:
            return 0.0
        return float(self.safe.system_failure_frequency())

    def safe_failure_fraction(self) -> Dict[Hashable, float]:
        """Each channel's safe failure fraction (SFF): its safe and
        detected dangerous rates over its total rate.

        A number only. The architectural constraints it is read against
        (IEC 61508-2's routes 1H and 2H, with the hardware fault
        tolerance) are a judgement of the records behind the rates, and
        are not made here.

        Returns
        -------
        dict
            The fraction by channel; NaN for a channel whose undetected
            failures have no constant rate.
        """
        return {
            name: channel.safe_failure_fraction
            for name, channel in self.channels.items()
        }

    def sil(self, mode: str, hours_per_unit: float = 1.0) -> SILBand:
        """The safety integrity level the function's hardware failures
        reach in a demand mode, by IEC 61508-1 Tables 2 and 3.

        Parameters
        ----------
        mode : {"low_demand", "high_demand", "continuous"}
            The demand mode. In low demand the level is read from the
            PFDavg; in high demand and continuous mode from the PFH
            (every dangerous failure of the function, ``pfh()``).
        hours_per_unit : float, optional
            How many hours one unit of the channels' time is, for the
            PFH's bands, which are per hour. By default 1: rates per
            hour.

        Returns
        -------
        SILBand
            See [`sil_band`][repyability.sil_band].
        """
        one_of("mode", mode, MODES)
        value = self.pfd_avg() if _MEASURE[mode] == "pfd_avg" else self.pfh()
        return sil_band(value, mode, hours_per_unit)

    def results(
        self, mode: str, hours_per_unit: float = 1.0
    ) -> SafetyFunctionResult:
        """The function's results in one call: PFDavg, PFH, spurious-trip
        rate, the SIL of its demand mode, and where the PFDavg comes
        from.

        Parameters
        ----------
        mode : {"low_demand", "high_demand", "continuous"}
            The demand mode the SIL is read in, as for ``sil``.
        hours_per_unit : float, optional
            As for ``sil``.

        Returns
        -------
        SafetyFunctionResult
            Every result, each as its own method gives it.
        """
        one_of("mode", mode, MODES)
        pfd_avg, pfh = self.pfd_avg(), self.pfh()
        trips = self.spurious_trip_rate()
        return SafetyFunctionResult(
            mode=mode,
            pfd_avg=pfd_avg,
            pfh=pfh,
            pfh_undetected=self.pfh(undetected_only=True),
            spurious_trip_rate=trips,
            mean_time_to_spurious_trip=(
                math.inf if trips == 0.0 else 1.0 / trips
            ),
            sil=sil_band(
                pfd_avg if _MEASURE[mode] == "pfd_avg" else pfh,
                mode,
                hours_per_unit,
            ),
            pfd_breakdown=self.pfd_breakdown(),
            safe_failure_fraction=self.safe_failure_fraction(),
            diagnostic_coverage={
                name: channel.diagnostic_coverage
                for name, channel in self.channels.items()
            },
        )

    def _simplified_refusal(self) -> Optional[str]:
        """Why the simplified formulas do not apply, or None."""
        use = (
            " The exact values (pfd_avg, pfh, spurious_trip_rate) hold "
            "for it."
        )
        if any(isinstance(p, _Voted) for p in self._structure.parts):
            return (
                "The simplified formulas are for one vote of channels, "
                "and this function votes on votes." + use
            )
        if len(self._stages) > 1:
            return (
                "The simplified formulas are for one vote, and this "
                "function has several in series. Take each vote's own "
                "simplified_formulas()." + use
            )
        stage = self._stages[0]
        vote = (stage.k, len(stage.channels))
        if vote not in _ANNEX_B_VOTES:
            return (
                f"IEC 61508-6 Annex B gives no simplified formula for a "
                f"{vote[0]}oo{vote[1]} vote (it gives 1oo1, 1oo2, 2oo2, "
                "2oo3 and 1oo3)." + use
            )
        channels = list(stage.channels.values())
        if any(channel != channels[0] for channel in channels[1:]):
            return "The simplified formulas take the channels alike." + use
        channel = channels[0]
        if channel.undetected_rate is None:
            return (
                "The simplified formulas take constant failure rates, "
                "and the undetected failures' life has none." + use
            )
        return None

    def simplified_formulas(self) -> SimplifiedSafetyFormulas:
        """The simplified formulas of IEC 61508-6 Annex B (B.3.2.2 for
        the PFDavg, B.3.3.2 for the PFH) for this vote, and the
        rare-event spurious-trip rate, to set beside the exact values.

        For 1oo1, 1oo2, 2oo2, 2oo3 and 1oo3 votes of channels alike, with
        constant rates. With ``lambda_D = lambda_DU + lambda_DD``, ``r =
        N - M + 1`` failures taking the vote down, and the down times

        ```text
        t_j = (lambda_DU / lambda_D) (T1 / (j + 1) + MRT)
              + (lambda_DD / lambda_D) MTTR        (t_1 is t_CE, t_2 t_GE)
        ```

        the annex's formulas are, for ``r = 1``, ``PFDavg = N lambda_D
        t_1`` and ``PFH = N lambda_DU``, and otherwise, with ``lambda_i
        = (1 - beta_D) lambda_DD + (1 - beta) lambda_DU``,

        ```text
        PFDavg = N!/(M-1)! lambda_i^r t_1 ... t_r
                 + beta_D lambda_DD MTTR + beta lambda_DU (T1/2 + MRT)
        PFH    = N!/(M-1)! lambda_i^(r-1) (1 - beta) lambda_DU t_1 ... t_(r-1)
                 + beta lambda_DU
        ```

        They are first-order approximations: they take ``lambda_D * T1``
        small, multiply average down times where the exact value
        averages a product, leave the common cause out of a vote one
        failure takes down, and count in the PFH only the failures an
        undetected failure completes (compare ``pfh(undetected_only=
        True)``). The spurious-trip rate is ``C(N, M) M lambda_S^M
        MTTR^(M-1)``, the safe failures' independent part, plus ``beta_S
        lambda_S`` where ``M > 1``. See [safety
        functions](../guide/safety-functions.md) for how far each lies
        from the exact value, and for a published worked example they
        reproduce.

        A proof test that can miss (``proof_test_coverage`` PTC below 1,
        everything found every ``full_test`` T2) takes ``PTC T1 + (1 -
        PTC) T2`` in place of ``T1`` in the down times and the common
        cause, as B.3.2.5 does for the PFDavg; the PFH takes the same
        down times.

        Returns
        -------
        SimplifiedSafetyFormulas
            The formulas' PFDavg, PFH and spurious-trip rate, and the
            down times ``t_1 ... t_r``.

        Raises
        ------
        NotImplementedError
            For a function of several votes, a vote the annex has no
            formula for, a vote of votes, channels that differ, or a
            life with no constant rate.
        """
        refusal = self._simplified_refusal()
        if refusal is not None:
            raise NotImplementedError(refusal)
        stage = self._stages[0]
        channel = next(iter(stage.channels.values()))
        m, n = stage.k, len(stage.channels)
        r = n - m + 1
        (_, mttr), (_, mrt) = channel._repairs()
        undetected = float(channel.undetected_rate or 0.0)
        detected = stage.detected(channel)
        dangerous = undetected + detected
        # IEC 61508-6 B.3.2.5: a failure the proof test finds waits on
        # average as one of an interval T1, and one it misses as one of
        # the full test's T2, so the down times and the common cause take
        # PTC T1 + (1 - PTC) T2 where they take T1.
        coverage = float(channel.proof_test_coverage)
        interval = float(channel.proof_test_interval)
        if coverage < 1.0 and channel.full_test is not None:
            # (The diagrams refuse a coverage below 1 with no full test.)
            interval = coverage * interval + (1.0 - coverage) * float(
                channel.full_test
            )
        if dangerous > 0.0:
            times = tuple(
                (undetected / dangerous) * (interval / (j + 1) + mrt)
                + (detected / dangerous) * mttr
                for j in range(1, r + 1)
            )
        else:
            times = tuple(0.0 for _ in range(r))
        orders = math.factorial(n) / math.factorial(m - 1)
        if r == 1:
            pfd_avg = n * dangerous * times[0]
            pfh = n * undetected
        else:
            own_undetected = (1.0 - stage.beta) * undetected
            own = (1.0 - stage.beta_detected) * detected + own_undetected
            pfd_avg = (
                orders * own**r * math.prod(times)
                + stage.beta_detected * detected * mttr
                + stage.beta * undetected * (interval / 2.0 + mrt)
            )
            pfh = (
                orders
                * own ** (r - 1)
                * own_undetected
                * math.prod(times[:-1])
                + stage.beta * undetected
            )
        safe = stage.tripping(channel)
        own_safe = (1.0 - stage.beta_safe) * safe if m > 1 else safe
        trips = math.comb(n, m) * m * own_safe**m * mttr ** (m - 1)
        if m > 1:
            trips += stage.beta_safe * safe
        return SimplifiedSafetyFormulas(
            pfd_avg=pfd_avg,
            pfh=pfh,
            spurious_trip_rate=trips,
            down_times=times,
        )

    # -- parameter uncertainty ------------------------------------------

    def measure_uncertainty(
        self,
        measure: str,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
    ) -> UncertaintyResult:
        """A result of the function over plausible values of its channels'
        failure rates (parameter uncertainty).

        A rate is estimated from records, so it is uncertain: each of the
        ``n_draws`` draws gives every uncertain rate a plausible value,
        and the function, rebuilt with them, its ``measure``, worked out
        as its own method works it out. The spread of the draws, and
        their percentiles (``interval``), say how well it is known.

        ``uncertainty`` maps a channel's name, or a tuple of names of one
        population (channels whose rates are one estimate, drawn alike in
        each draw), to ``{rate: distribution}``, ``rate`` one of
        ``"dangerous_undetected"``, ``"dangerous_detected"`` and
        ``"safe"``, the distribution anything with a quantile function
        (surpyval's ``qf`` or scipy.stats' ``ppf``) of values of at least
        0: a gamma from a count of failures over a time in service, say.
        Where ``dangerous_undetected`` is a surpyval model, it is drawn as
        a diagram's node is (``"fit"``, ``{parameter: distribution}`` or
        a list of models; see
        [`mean_availability_uncertainty`][repyability.RepairableRBD.mean_availability_uncertainty]),
        and ``"fit"`` alone stands for ``{"dangerous_undetected":
        "fit"}``. Channels sharing a cause (a beta) must be drawn as one
        population, or not at all, as a beta factor takes them alike.

        Parameters
        ----------
        measure : {"pfd_avg", "pfh", "pfh_undetected", "spurious_trip_rate"}
            The result drawn: ``pfd_avg()``, ``pfh()``,
            ``pfh(undetected_only=True)`` or ``spurious_trip_rate()``.
        uncertainty : dict, optional
            ``{channel or tuple of channels: uncertainty}`` (see above).
            By default, every channel whose undetected failures' life is
            a surpyval fit with a parameter covariance or a Bayesian fit,
            drawn ``"fit"``, once for all the channels holding that fit.
        n_draws : int, optional
            The number of draws, by default 1000: one evaluation each.
        seed : int, optional
            Seed for the draws.
        sampling : str, optional
            ``"random"`` (the default), or ``"sobol"``: the draws from the
            points of a scrambled Sobol sequence.

        Returns
        -------
        UncertaintyResult
            The measure of every draw (``samples``), its ``nominal`` value
            with the channels as given, its summaries, and what the band
            is (``band``: ``"given"`` for distributions given,
            ``"confidence"`` or ``"credible"`` for fits, or ``"mixed"``).

        Raises
        ------
        ValueError
            If ``measure`` is none of the four, ``uncertainty`` names an
            unknown channel or rate, a population whose channels hold
            different rates, or channels sharing a cause apart; if a
            distribution has no quantile function or gives a value below
            0; if ``n_draws`` or ``sampling`` is invalid; or, as
            ``routes.Refused``, if no ``uncertainty`` is given and no
            channel's life is a fit.

        Examples
        --------
        A 1oo2 vote whose undetected failures were 3 in 1.5 million
        channel-hours: the rate's gamma distribution (shape 3, rate 1.5e6
        hours) for both channels, one population:

        >>> from scipy import stats
        >>> from repyability import SafetyChannel, SafetyFunction
        >>> unit = SafetyChannel(
        ...     2e-6, dangerous_detected=3e-6, safe=5e-6,
        ...     repair=8.0, proof_test_interval=8760.0,
        ... )
        >>> function = SafetyFunction({"a": unit, "b": unit}, k=1, beta=0.1)
        >>> rate = stats.gamma(3, scale=1 / 1.5e6)
        >>> result = function.measure_uncertainty(
        ...     "pfd_avg", {("a", "b"): {"dangerous_undetected": rate}},
        ...     n_draws=50, seed=1,
        ... )
        >>> f"{result.nominal:.3e}", result.band
        ('9.585e-04', 'given')
        >>> lower, upper = result.interval(0.9)
        >>> f"{lower:.1e}", f"{upper:.1e}"
        ('3.1e-04', '2.3e-03')
        """
        one_of("measure", measure, _MEASURES)
        sources = self._uncertain_rates(uncertainty)

        def draw(n: int, source) -> List[Dict[Hashable, Any]]:
            drawn: List[Dict[Hashable, Dict[str, Any]]] = [
                {} for _ in range(n)
            ]
            for names, field, spec, label in sources:
                value = getattr(self.channels[names[0]], field)
                if is_number(value):
                    values = list(
                        _uncertainty._quantiles(spec, n, source, label, field)
                    )
                    if not all(v >= 0.0 and math.isfinite(v) for v in values):
                        raise ValueError(
                            f"{label}: the distribution for {field!r} gives "
                            "values below 0 or not finite; a rate's "
                            "distribution must give values of at least 0."
                        )
                else:
                    values = _uncertainty.draw_models(
                        value, spec, n, source, label
                    )
                for i in range(n):
                    for name in names:
                        drawn[i].setdefault(name, {})[field] = values[i]
            return [
                {
                    name: dataclasses.replace(self.channels[name], **fields)
                    for name, fields in row.items()
                }
                for row in drawn
            ]

        rows = _uncertainty.draw_with(
            draw, _uncertainty.check_draws(n_draws), seed, sampling
        )
        samples = np.array(
            [self._with(row)._measure(measure) for row in rows], dtype=float
        )
        return UncertaintyResult(
            samples=samples,
            nominal=self._measure(measure),
            n_draws=len(samples),
            band=_uncertainty.band(
                _uncertainty.input_band(
                    getattr(self.channels[names[0]], field), spec
                )
                for names, field, spec, _ in sources
            ),
        )

    def _measure(self, measure: str) -> float:
        if measure == "pfd_avg":
            return self.pfd_avg()
        if measure == "pfh":
            return self.pfh()
        if measure == "pfh_undetected":
            return self.pfh(undetected_only=True)
        return self.spurious_trip_rate()

    def _with(self, channels: Dict[Hashable, Any]) -> "SafetyFunction":
        """This function with the channels named in ``channels``
        replaced."""
        out = type(self).__new__(type(self))
        structure = _with_channels(self._structure, channels)
        assert isinstance(structure, _Series)
        out._build(structure)
        return out

    def _uncertain_rates(self, uncertainty) -> List[tuple]:
        """``uncertainty``, checked: ``(names, rate, spec, label)`` for
        each rate drawn."""
        if uncertainty is None:
            fits: Dict[int, List[Hashable]] = {}
            for name, channel in self.channels.items():
                life = channel.dangerous_undetected
                if not is_number(life) and _uncertainty.is_fit(life):
                    fits.setdefault(id(life), []).append(name)
            if not fits:
                raise routes.Refused(
                    "measure_uncertainty needs what is uncertain: no "
                    "channel's undetected failures' life is a fit with a "
                    "parameter covariance or a Bayesian fit. Give "
                    "uncertainty={channel: {rate: distribution}}, e.g. "
                    "{'a': {'dangerous_undetected': "
                    "scipy.stats.gamma(3, scale=1 / 1.5e6)}}."
                )
            uncertainty = {tuple(names): "fit" for names in fits.values()}
        if not isinstance(uncertainty, dict):
            raise ValueError(
                "uncertainty must be a dict of {channel or tuple of "
                f"channels: {{rate: distribution}}}}, got {uncertainty!r}."
            )
        out = []
        drawn: Dict[Hashable, Any] = {}
        for key, spec in uncertainty.items():
            names = (
                (key,)
                if key in self.channels
                else tuple(key) if isinstance(key, tuple) else (key,)
            )
            unknown = [n for n in names if n not in self.channels]
            if unknown or not names:
                raise ValueError(
                    f"uncertainty names {key!r}, which is not a channel "
                    f"or a tuple of channels; the channels are "
                    f"{list(self.channels)!r}."
                )
            if isinstance(spec, str) and spec == "fit":
                spec = {"dangerous_undetected": "fit"}
            if not isinstance(spec, dict) or not spec:
                raise ValueError(
                    f"The uncertainty of {key!r} must be {{rate: "
                    "distribution}, the rate one of "
                    f"{list(_RATES)!r}, or 'fit', got {spec!r}."
                )
            for field, given in spec.items():
                if field not in _RATES:
                    raise ValueError(
                        f"The uncertainty of {key!r} names {field!r}, "
                        f"which is not a rate; the rates are "
                        f"{list(_RATES)!r}."
                    )
                values = [getattr(self.channels[n], field) for n in names]
                if any(v is not values[0] and v != values[0] for v in values):
                    raise ValueError(
                        f"The channels {list(names)!r} hold different "
                        f"{field} rates, so they are not one population: "
                        "draw each on its own."
                    )
                for name in names:
                    if drawn.setdefault((name, field), key) is not key:
                        raise ValueError(
                            f"The {field} rate of channel {name!r} is "
                            "drawn twice; give it under one key."
                        )
                out.append((names, field, given, f"Channel {key!r}"))
        for group in _shared_groups(self._structure):
            keys = {
                (field, drawn.get((name, field)))
                for name in group
                for field in _RATES
            }
            fields = {field for field, _ in keys}
            if len(keys) != len(fields):
                raise ValueError(
                    f"The channels {group!r} share a cause (a beta), which "
                    "takes them alike, so draw their rates together, as "
                    f"one population: uncertainty={{{tuple(group)!r}: "
                    "...}."
                )
        return out

    # -- saving ---------------------------------------------------------

    def to_dict(self) -> dict:
        """The function as a JSON-friendly dict (see ``from_dict``): its
        votes, how they are joined, and each channel once, its models in
        surpyval's own format, as a diagram saves them (see
        [`RBD.to_dict`][repyability.RBD.to_dict]).

        Channels that are alike are saved once and loaded as one
        ``SafetyChannel``, so a vote with a beta is rebuilt with its
        channels alike.

        Returns
        -------
        dict
            The ``repyability_version`` that wrote it, the ``type``
            (``"SafetyFunction"``), the ``channels`` (each saved once) and
            the ``function``: a ``"vote"`` of channels, a ``"vote_of"``
            functions or a ``"series"``, nested.

        Raises
        ------
        NotImplementedError
            If a model cannot be saved: one that is not surpyval's (nor a
            RePyability node model).

        Examples
        --------
        >>> from repyability import SafetyChannel, SafetyFunction
        >>> unit = SafetyChannel(
        ...     2e-6, dangerous_detected=3e-6, safe=5e-6,
        ...     repair=8.0, proof_test_interval=8760.0,
        ... )
        >>> function = SafetyFunction({"a": unit, "b": unit}, k=1, beta=0.1)
        >>> d = function.to_dict()
        >>> d["type"], len(d["channels"]), d["function"]["kind"]
        ('SafetyFunction', 1, 'series')
        >>> SafetyFunction.from_dict(d).pfd_avg() == function.pfd_avg()
        True
        """
        table: List[SafetyChannel] = []

        def index(channel: SafetyChannel) -> int:
            for i, known in enumerate(table):
                if known == channel:
                    return i
            table.append(channel)
            return len(table) - 1

        def saved(part) -> dict:
            if isinstance(part, _Stage):
                return {
                    "kind": "vote",
                    "k": part.k,
                    "channels": [
                        {"name": name, "channel": index(channel)}
                        for name, channel in part.channels.items()
                    ],
                    "beta": part.beta,
                    "beta_detected": part.beta_detected,
                    "beta_safe": part.beta_safe,
                    "detected_trips": part.detected_trips,
                }
            if isinstance(part, _Voted):
                return {
                    "kind": "vote_of",
                    "k": part.k,
                    "parts": [saved(p) for p in part.parts],
                    "beta": part.beta,
                    "beta_detected": part.beta_detected,
                    "beta_safe": part.beta_safe,
                }
            return {"kind": "series", "parts": [saved(p) for p in part.parts]}

        function = saved(self._structure)
        return {
            "repyability_version": __version__,
            "type": "SafetyFunction",
            "channels": [_channel_to_dict(channel) for channel in table],
            "function": function,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyFunction":
        """Rebuild a function from ``to_dict``'s output, by calling its
        constructors again (``SafetyFunction``, ``series`` and ``vote``),
        so it is checked again. Tuple names, which JSON turns into lists,
        are turned back.

        Parameters
        ----------
        d : dict
            As ``to_dict`` returns (possibly after a JSON round trip).

        Returns
        -------
        SafetyFunction
            The function.

        Raises
        ------
        ValueError
            If ``d`` is not a safety function's dict, a model kind is
            unknown, or a constructor rejects the stored inputs.
        """
        from repyability.rbd.serialisation import _node_name

        kind = d.get("type") if isinstance(d, dict) else None
        if kind != "SafetyFunction":
            raise ValueError(
                "SafetyFunction.from_dict takes a safety function's "
                'to_dict() document, whose "type" is "SafetyFunction", '
                f"got {kind!r}."
            )
        table = [_channel_from_dict(saved) for saved in d["channels"]]

        def loaded(part: dict) -> "SafetyFunction":
            what = part.get("kind")
            if what == "vote":
                return cls(
                    {
                        _node_name(entry["name"]): table[entry["channel"]]
                        for entry in part["channels"]
                    },
                    part["k"],
                    beta=part["beta"],
                    beta_detected=part["beta_detected"],
                    beta_safe=part["beta_safe"],
                    detected_trips=part["detected_trips"],
                )
            if what == "vote_of":
                return cls.vote(
                    part["k"],
                    *(loaded(p) for p in part["parts"]),
                    beta=part["beta"],
                    beta_detected=part["beta_detected"],
                    beta_safe=part["beta_safe"],
                )
            if what == "series":
                return cls.series(*(loaded(p) for p in part["parts"]))
            raise ValueError(
                f"Unknown part of a safety function {what!r}: expected "
                "'vote', 'vote_of' or 'series'."
            )

        return loaded(d["function"])

    def to_json(self, fp=None, **json_kwargs) -> Optional[str]:
        """The function as a JSON document (see ``to_dict``): returned,
        or written to ``fp``.

        Parameters
        ----------
        fp : str, os.PathLike or file, optional
            A path, or a file opened for writing, to write the document
            to; by default None: it is returned.
        **json_kwargs
            Passed to ``json.dumps``, e.g. ``indent=2``.

        Returns
        -------
        str or None
            The JSON document, or None once written to ``fp``.

        Raises
        ------
        NotImplementedError
            If a model cannot be saved (see ``to_dict``).
        TypeError
            If a channel's name cannot be encoded as JSON.
        """
        return write_json(json.dumps(self.to_dict(), **json_kwargs), fp)

    @classmethod
    def from_json(cls, s) -> "SafetyFunction":
        """Rebuild a function from ``to_json``'s output.

        Parameters
        ----------
        s : str, os.PathLike or file
            The JSON document: its text, a path to a file holding it, or
            a file opened for reading.

        Returns
        -------
        SafetyFunction
            The function.

        Raises
        ------
        ValueError
            If ``s`` is not valid JSON, or for the reasons ``from_dict``
            gives.
        """
        return cls.from_dict(json.loads(read_json(s)))

    def analysis_routes(self) -> AnalysisRoutes:
        """How each result of this function is computed, found without
        running it: exactly, numerically, by simulation, or not at all.

        The routes are the two diagrams' own: the PFDavg's is the
        dangerous diagram's ``mean_unavailability``'s, the PFH's its
        ``system_failure_frequency``'s, and the spurious-trip rate's the
        safe diagram's ``system_failure_frequency``'s.

        Returns
        -------
        AnalysisRoutes
            An [`AnalysisRoute`][repyability.AnalysisRoute] for each of
            ``pfd_avg``, ``pfd_breakdown``, ``pfh``,
            ``spurious_trip_rate``, ``safe_failure_fraction``, ``sil``,
            ``results``, ``measure_uncertainty`` and
            ``simplified_formulas``.
        """
        dangerous = self.dangerous.analysis_routes()
        pfd_avg = dangerous["mean_unavailability"]
        pfh = dangerous["system_failure_frequency"]
        if self.safe is None:
            trips = AnalysisRoute(
                routes.EXACT, "No channel can fail safe: the rate is 0."
            )
        else:
            trips = self.safe.analysis_routes()["system_failure_frequency"]
        breakdown = dangerous["pfd_breakdown"]
        refusal = self._simplified_refusal()
        return AnalysisRoutes(
            {
                "pfd_avg": pfd_avg,
                "pfd_breakdown": breakdown,
                "pfh": pfh,
                "spurious_trip_rate": trips,
                "safe_failure_fraction": AnalysisRoute(
                    routes.EXACT, "A ratio of the channels' rates."
                ),
                "sil": _worst(
                    "The band of the PFDavg or of the PFH, by the mode.",
                    pfd_avg,
                    pfh,
                ),
                "results": _worst(
                    "Each result as its own method gives it.",
                    pfd_avg,
                    pfh,
                    trips,
                    breakdown,
                ),
                "measure_uncertainty": _worst(
                    "Each draw's value as its method gives it.",
                    pfd_avg,
                    pfh,
                    trips,
                ),
                "simplified_formulas": (
                    AnalysisRoute(
                        routes.EXACT,
                        "IEC 61508-6 Annex B's closed formulas, which "
                        "are themselves first-order approximations.",
                    )
                    if refusal is None
                    else routes.refused(refusal)
                ),
            }
        )


#: The rates a channel's uncertainty may draw.
_RATES = ("dangerous_undetected", "dangerous_detected", "safe")
#: The results ``measure_uncertainty`` draws.
_MEASURES = ("pfd_avg", "pfh", "pfh_undetected", "spurious_trip_rate")
#: A channel's fields, as saved.
_FIELDS = tuple(field.name for field in dataclasses.fields(SafetyChannel))


def _value_to_dict(value):
    """A channel's field as saved: a number, ``"instant"`` or None as it
    is, a model in its saved form."""
    from repyability.rbd.serialisation import serialise_model

    if value is None or isinstance(value, str) or is_number(value):
        return value
    return {"model": serialise_model(value)}


def _channel_to_dict(channel: SafetyChannel) -> dict:
    return {
        field: _value_to_dict(getattr(channel, field)) for field in _FIELDS
    }


def _channel_from_dict(saved: dict) -> SafetyChannel:
    from repyability.rbd.serialisation import deserialise_model

    return SafetyChannel(
        **{
            field: (
                deserialise_model(value["model"])
                if isinstance(value, dict)
                else value
            )
            for field, value in saved.items()
            if field in _FIELDS
        }
    )


def _revealed(rate: float, repair) -> dict:
    """A component whose failures are revealed, at a constant rate."""
    import surpyval as surv

    return {
        "reliability": surv.Exponential.from_params([rate]),
        "repairability": repair,
    }


def _worst(reason: str, *parts: AnalysisRoute) -> AnalysisRoute:
    """The route of a result made of ``parts``: the first refusal among
    them, or the least exact of their routes."""
    for part in parts:
        if part.route == routes.REFUSED:
            return part
    order = (routes.EXACT, routes.NUMERICAL, routes.SIMULATED)
    worst = max(parts, key=lambda part: order.index(part.route))
    return AnalysisRoute(worst.route, f"{reason} {worst.reason}", worst.nodes)
