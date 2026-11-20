"""Contract terms that pay on how a system performed, as plain data: an
:class:`AvailabilityContract` pays a bonus or charges a penalty in each
period by the system's availability in it (see
``RepairableRBD.npv_distribution``)."""

import dataclasses
import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from repyability.utils.checks import is_number


def _amount(value, name: str) -> float:
    if not is_number(value) or not (math.isfinite(value) and value >= 0.0):
        raise ValueError(
            f"{name} must be a finite number, 0 or more; got {value!r}."
        )
    return float(value)


@dataclasses.dataclass(frozen=True)
class AvailabilityContract:
    """An availability contract: in each period, a penalty for every point
    of availability below a target, and a bonus for every point above it.

    In plain terms: the system's availability in a period is the share of
    the period it was up. A point is one percentage point of it (0.01):
    a period at 93% against a target of 95% falls 2 points short, and
    costs ``2 * penalty_per_point``; at 96% it is 1 point over, and earns
    ``bonus_per_point``. Part points pay in proportion. ``cap`` limits
    what one period can charge or pay, either way. Each period's payment
    falls at the period's end. An expected value cannot price such terms,
    as they pay on thresholds; ``npv_distribution(contract=...)`` prices
    them run by run.

    Parameters
    ----------
    periods : float or sequence of float
        A number: the length of each period, the periods back to back from
        0 (the time after the last whole period within the horizon is not
        measured). A sequence: the periods' ends, increasing from more
        than 0, each period from the end of the one before (the first
        from 0); every end must be within the horizon valued. In the
        component models' unit.
    target : float
        The availability the contract asks for, from 0 to 1.
    penalty_per_point : float
        What each point below ``target`` costs in a period, 0 or more.
    bonus_per_point : float, optional
        What each point above ``target`` earns in a period, 0 or more, by
        default 0.
    cap : float, optional
        The most one period's penalty, or bonus, can be, more than 0. By
        default None: no limit.

    Raises
    ------
    ValueError
        If ``periods`` is not a positive length or an increasing sequence
        of positive ends, ``target`` is not from 0 to 1, a price is
        negative or not finite, or ``cap`` is not more than 0.

    Examples
    --------
    Monthly periods of 730 hours, a target of 95%, 1,000 for each point
    short and 400 for each point over, at most 10,000 a month:

    >>> from repyability import AvailabilityContract
    >>> terms = AvailabilityContract(
    ...     730.0, target=0.95, penalty_per_point=1000.0,
    ...     bonus_per_point=400.0, cap=10_000.0,
    ... )
    >>> [round(float(x), 6) for x in terms.payoff([0.93, 0.96, 0.5])]
    [-2000.0, 400.0, -10000.0]
    >>> terms.period_ends(2200.0)
    (730.0, 1460.0, 2190.0)
    """

    periods: Union[float, Tuple[float, ...]]
    target: float
    penalty_per_point: float
    bonus_per_point: float = 0.0
    cap: Optional[float] = None

    def __post_init__(self):
        periods: Any = self.periods
        if is_number(periods):
            if not (math.isfinite(periods) and periods > 0.0):
                raise ValueError(
                    "periods must be a period's length, a finite number "
                    f"more than 0, or the periods' ends; got {periods!r}."
                )
            periods = float(periods)
        else:
            try:
                ends = tuple(float(end) for end in periods)
            except (TypeError, ValueError):
                raise ValueError(
                    "periods must be a period's length or a sequence of "
                    f"the periods' ends; got {periods!r}."
                ) from None
            steps = np.diff((0.0,) + ends)
            if (
                not ends
                or not all(math.isfinite(end) for end in ends)
                or np.any(steps <= 0.0)
            ):
                raise ValueError(
                    "The periods' ends must be finite and increasing, the "
                    f"first more than 0; got {periods!r}."
                )
            periods = ends
        object.__setattr__(self, "periods", periods)
        target = self.target
        if not is_number(target) or not 0.0 <= target <= 1.0:
            raise ValueError(
                "target must be an availability from 0 to 1 (95% is "
                f"0.95); got {target!r}."
            )
        object.__setattr__(self, "target", float(target))
        for name in ("penalty_per_point", "bonus_per_point"):
            object.__setattr__(self, name, _amount(getattr(self, name), name))
        if self.cap is not None:
            cap = self.cap
            if not is_number(cap) or not cap > 0.0:
                raise ValueError(
                    "cap must be more than 0, or None for no limit; got "
                    f"{cap!r}."
                )
            object.__setattr__(self, "cap", float(cap))

    def period_ends(self, horizon: float) -> Tuple[float, ...]:
        """The ends of the periods measured over ``[0, horizon]``.

        Parameters
        ----------
        horizon : float
            The length of time valued.

        Returns
        -------
        tuple of float
            Each period's end, in order.

        Raises
        ------
        ValueError
            If a period's end given is after ``horizon``, or no whole
            period of the length given fits in it.
        """
        horizon = float(horizon)
        if isinstance(self.periods, tuple):
            if self.periods[-1] > horizon:
                raise ValueError(
                    f"The contract's last period ends at "
                    f"{self.periods[-1]:g}, after the horizon "
                    f"{horizon:g}: value a horizon that holds every "
                    "period, or leave the later periods out."
                )
            return self.periods
        count = math.floor(round(horizon / self.periods, 9))
        if count < 1:
            raise ValueError(
                f"No whole period of {self.periods:g} fits in the horizon "
                f"{horizon:g}: value a longer horizon, or give shorter "
                "periods."
            )
        return tuple(self.periods * k for k in range(1, count + 1))

    def payoff(self, availability) -> np.ndarray:
        """What a period with each availability pays: positive a bonus,
        negative a penalty.

        Parameters
        ----------
        availability : array_like
            Availabilities, each from 0 to 1.

        Returns
        -------
        numpy.ndarray
            The payments, of the same shape.
        """
        points = 100.0 * (np.asarray(availability, float) - self.target)
        paid = np.where(
            points >= 0.0,
            self.bonus_per_point * points,
            self.penalty_per_point * points,
        )
        if self.cap is not None:
            paid = np.clip(paid, -self.cap, self.cap)
        return paid + 0.0

    def to_dict(self) -> Dict[str, Any]:
        """The terms as plain data, ready for ``json.dumps``
        (``from_dict`` reads them back).

        Returns
        -------
        dict
            ``periods`` (a length, or a list of ends), ``target``,
            ``penalty_per_point``, ``bonus_per_point`` and ``cap``.
        """
        return {
            "periods": (
                list(self.periods)
                if isinstance(self.periods, tuple)
                else self.periods
            ),
            "target": self.target,
            "penalty_per_point": self.penalty_per_point,
            "bonus_per_point": self.bonus_per_point,
            "cap": self.cap,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AvailabilityContract":
        """The terms ``to_dict`` wrote.

        Parameters
        ----------
        data : dict
            The plain data.

        Returns
        -------
        AvailabilityContract
            The contract.
        """
        return cls(**data)
