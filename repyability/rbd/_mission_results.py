"""The results of a phased mission taken as a whole (see
``PhasedMission.summary`` and ``PhasedMission.failure_drivers``)."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional

import numpy as np

from repyability.rbd.results import ConfidenceInterval, _ResultMapping


@dataclass(frozen=True)
class MissionDrivers(_ResultMapping):
    """Which blocks drove a mission's failures, phase by phase.

    Returned by
    [`PhasedMission.failure_drivers`][repyability.PhasedMission.failure_drivers].
    For each block and each phase, the fall in the phase's failure
    probability if the block never failed (the mission worked out again
    with the block held working throughout); its total over the phases is
    the fall in the mission's unreliability. A fall can be negative in a
    phase: a block that never fails can carry missions into a later phase
    that then fail there.

    Attributes
    ----------
    method : str
        ``"exact"`` or ``"simulated"`` (with common random numbers: the
        same missions played again with the block held).
    phases : list
        The phases' names, in order.
    falls : dict
        ``{block: {phase: fall}}``.
    totals : dict
        ``{block: total fall}``, the fall in the mission's unreliability.
    standard_errors : dict or None
        Simulated: each total's standard error (``None`` when exact).
    n_samples : int or None
        Simulated: the number of missions (``None`` when exact).
    """

    method: str
    phases: List[Hashable]
    falls: Dict[Hashable, Dict[Hashable, float]]
    totals: Dict[Hashable, float]
    standard_errors: Optional[Dict[Hashable, float]] = None
    n_samples: Optional[int] = None

    def top(self, phase: Hashable) -> Optional[Hashable]:
        """The block whose never failing would lower the failure
        probability of ``phase`` the most (None if none would).

        Parameters
        ----------
        phase : Hashable
            The phase's name.
        """
        best, fall = None, 0.0
        for block, falls in self.falls.items():
            if falls[phase] > fall:
                best, fall = block, falls[phase]
        return best

    def __repr__(self) -> str:
        ranked = sorted(self.totals.items(), key=lambda kv: -kv[1])[:3]
        shown = ", ".join(f"{block!r}: {total:.4g}" for block, total in ranked)
        return (
            f"MissionDrivers({self.method}, {len(self.totals)} block(s); "
            f"top: {shown})"
        )


@dataclass(frozen=True)
class MissionPhaseSummary(_ResultMapping):
    """One phase of a mission's summary (see
    [`MissionSummary`][repyability.MissionSummary]).

    Attributes
    ----------
    name : Hashable
        The phase's name.
    start : float
        When it starts, from the mission's start.
    end : float
        When it ends, from the mission's start.
    failure_probability : float
        The probability that the mission fails in it: gets through the
        earlier phases and not this one.
    share : float
        Its share of the mission's failures (``nan`` where the mission
        cannot fail).
    conditional_failure : float
        The probability that it fails given it was reached (``nan`` where
        it cannot be reached).
    reliability_at_end : float
        The probability that the mission gets through it.
    settings : dict
        Each block set in the phase: ``{block: {"state": ...,
        "covariates": ...}}``.
    takeovers : dict
        Each block on standby in the phase: the fraction of missions in
        which it was started in the phase (simulated).
    top_driver : Hashable or None
        The block whose never failing would lower the phase's failure
        probability the most (see ``MissionDrivers``).
    """

    name: Hashable
    start: float
    end: float
    failure_probability: float
    share: float
    conditional_failure: float
    reliability_at_end: float
    settings: Dict[Hashable, Dict[str, Any]]
    takeovers: Dict[Hashable, float]
    top_driver: Optional[Hashable]


@dataclass(frozen=True)
class MissionSummary(_ResultMapping):
    """A whole mission at once, from one run (see
    [`PhasedMission.summary`][repyability.PhasedMission.summary]).

    Attributes
    ----------
    method : str
        ``"exact"`` or ``"simulated"``.
    reason : str or None
        Why it was simulated (``None`` when exact).
    duration : float
        The mission's length.
    reliability : float
        The probability that the mission succeeds.
    unreliability : float
        The probability that it fails.
    interval : ConfidenceInterval or None
        Simulated: the reliability's confidence interval.
    times : numpy.ndarray
        A grid of times across the mission, every phase boundary
        included.
    reliability_over_time : numpy.ndarray
        The probability that the mission has not failed by each time.
    phases : list of MissionPhaseSummary
        Each phase.
    drivers : MissionDrivers
        Which blocks drove the failures, by phase.
    riskiest_phase : Hashable or None
        The phase the mission most often fails in (None if it cannot
        fail).
    n_samples : int or None
        Simulated: the number of missions.
    """

    method: str
    reason: Optional[str]
    duration: float
    reliability: float
    unreliability: float
    interval: Optional[ConfidenceInterval]
    times: np.ndarray
    reliability_over_time: np.ndarray
    phases: List[MissionPhaseSummary]
    drivers: MissionDrivers
    riskiest_phase: Optional[Hashable]
    n_samples: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"MissionSummary({self.method}, reliability "
            f"{self.reliability:.6g}, {len(self.phases)} phase(s), "
            f"riskiest {self.riskiest_phase!r}, {len(self.times)} times)"
        )
