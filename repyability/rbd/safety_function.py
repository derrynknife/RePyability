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

import math
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

from repyability.rbd import _long_run, routes
from repyability.rbd._common import _constant_rate
from repyability.rbd._model_utils import MODEL_ERRORS
from repyability.rbd.ccf import BetaFactor, CCFGroup
from repyability.rbd.helper_classes import PerfectReliability
from repyability.rbd.results import (
    PFDBreakdown,
    SafetyFunctionResult,
    SILBand,
    SimplifiedSafetyFormulas,
)
from repyability.rbd.routes import AnalysisRoute, AnalysisRoutes
from repyability.utils.checks import (
    is_number,
    no_distribution,
    number_or_nan,
    one_of,
    whole_number,
)

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
        self._build([stage])

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
        if not functions:
            raise ValueError("series needs at least one SafetyFunction.")
        stages: List[_Stage] = []
        seen: set = set()
        for function in functions:
            if not isinstance(function, SafetyFunction):
                raise ValueError(
                    f"series takes SafetyFunctions, got {function!r}."
                )
            for stage in function._stages:
                shared = seen & set(stage.channels)
                if shared:
                    raise ValueError(
                        "The functions put in series must name their "
                        f"channels differently; {sorted(map(repr, shared))} "
                        "is in more than one."
                    )
                seen |= set(stage.channels)
                stages.append(stage)
        out = cls.__new__(cls)
        out._build(stages)
        return out

    def _build(self, stages: List[_Stage]) -> None:
        from repyability.rbd.repairable_rbd import RepairableRBD

        self._stages = list(stages)
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
        before: Tuple[Any, ...] = _IN
        for index, stage in enumerate(stages):
            vote = ("vote", index)
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
                    (safe_groups if nodes is tripping else groups).append(
                        CCFGroup(nodes, BetaFactor(beta, basis="rate"))
                    )
            self._undetected |= set(undetected)
            for held, count in (
                (votes, stage.k),
                (safe_votes, n - stage.k + 1),
            ):
                held[vote] = count
            components[vote] = PerfectReliability
            safe_components[vote] = PerfectReliability
            before = vote
        edges.append((before, _OUT))
        safe_edges.append((before, _OUT))
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
        votes = ", ".join(
            f"{stage.k}oo{len(stage.channels)}" for stage in self._stages
        )
        return f"SafetyFunction({votes}; channels={list(self.channels)!r})"

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
        if channel.proof_test_coverage < 1.0:
            return (
                "The simplified formulas here take a proof test that "
                "finds every failure (proof_test_coverage=1)." + use
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
        from the exact value.

        Returns
        -------
        SimplifiedSafetyFormulas
            The formulas' PFDavg, PFH and spurious-trip rate, and the
            down times ``t_1 ... t_r``.

        Raises
        ------
        NotImplementedError
            For a function of several votes, a vote the annex has no
            formula for, channels that differ, a life with no constant
            rate, or a proof test that can miss.
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
        interval = float(channel.proof_test_interval)
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
            ``results`` and ``simplified_formulas``.
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
