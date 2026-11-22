"""What a system earns while it works, by stretches of model time: the
``value_rate`` of ``RepairableRBD.expected_npv`` as steps (UC-57 of the
maintenance README)."""

import math
from typing import Any, Dict, Tuple

import numpy as np

from repyability.utils import checks


class ValueSchedule:
    """The value a system earns per unit time while it works, in steps
    of model time: ``rates[k]`` from ``times[k]`` to ``times[k + 1]``
    (the last for the rest of the horizon). Plain data, so a valuation
    made with it can be saved and made again: a function of time is not
    taken.

    A steady growth of the value (a price that rises) is
    ``expected_npv``'s ``escalation={"value": g}``, which applies to a
    schedule as to a single rate.

    Parameters
    ----------
    times : array_like
        Where each step starts, in the models' unit, increasing, the
        first 0.
    rates : array_like
        The value per unit time while the system works in each step, at
        least 0, one for each of ``times``.

    Raises
    ------
    ValueError
        If ``times`` do not start at 0 and increase, or ``rates`` are not
        one finite number of at least 0 for each.

    Examples
    --------
    A plant that earns 10 an hour for its first year and 12 after:

    >>> from repyability import ValueSchedule
    >>> value = ValueSchedule([0.0, 8760.0], [10.0, 12.0])
    >>> value.rate_at([0.0, 9000.0]).tolist()
    [10.0, 12.0]
    >>> ValueSchedule.from_dict(value.to_dict()) == value
    True
    """

    def __init__(self, times, rates):
        if callable(times) or callable(rates):
            raise TypeError(
                "A ValueSchedule is steps of plain numbers, so that a "
                "valuation can be saved and made again: give the times "
                "the rates change and the rate from each, "
                "ValueSchedule(times, rates)."
            )
        starts = checks.real_array(times, "times").ravel().astype(float)
        values = checks.real_array(rates, "rates").ravel().astype(float)
        if not (
            starts.size
            and starts[0] == 0.0
            and np.all(np.isfinite(starts))
            and np.all(np.diff(starts) > 0.0)
        ):
            raise ValueError(
                "A ValueSchedule's times are where its steps start: "
                f"increasing, the first 0; got {times!r}."
            )
        if values.shape != starts.shape or not (
            np.all(np.isfinite(values)) and np.all(values >= 0.0)
        ):
            raise ValueError(
                "A ValueSchedule's rates are the value per unit time in "
                "each step: a finite number of at least 0 for each of its "
                f"{starts.size} times; got {rates!r}."
            )
        self._times: Tuple[float, ...] = tuple(starts.tolist())
        self._rates: Tuple[float, ...] = tuple(values.tolist())

    @property
    def times(self) -> Tuple[float, ...]:
        """Where each step starts."""
        return self._times

    @property
    def rates(self) -> Tuple[float, ...]:
        """The value per unit time while the system works in each step."""
        return self._rates

    def rate_at(self, t) -> np.ndarray:
        """The rate at times ``t`` (each step's from its start).

        Parameters
        ----------
        t : array_like
            Times, at least 0.

        Returns
        -------
        numpy.ndarray
            The rate at each.
        """
        x = np.asarray(t, dtype=float)
        k = np.searchsorted(np.asarray(self._times), x, side="right") - 1
        return np.asarray(self._rates)[np.maximum(k, 0)]

    def steps(self, stop: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The steps that start before ``stop``.

        Parameters
        ----------
        stop : float
            Where the steps are cut: the end of the time valued.

        Returns
        -------
        tuple
            The steps' starts, ends (the last ``stop``) and rates, as
            arrays.
        """
        times = np.asarray(self._times)
        keep = times < stop
        starts = times[keep]
        ends = np.append(starts[1:], stop)
        return starts, ends, np.asarray(self._rates)[keep]

    def to_dict(self) -> Dict[str, Any]:
        """The schedule as plain data, for ``json.dumps``.

        Returns
        -------
        dict
            ``{"times": [...], "rates": [...]}``.
        """
        return {"times": list(self._times), "rates": list(self._rates)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValueSchedule":
        """The schedule ``to_dict`` gave.

        Parameters
        ----------
        data : dict
            ``{"times": [...], "rates": [...]}``.

        Returns
        -------
        ValueSchedule
        """
        return cls(data["times"], data["rates"])

    def __eq__(self, other) -> bool:
        if not isinstance(other, ValueSchedule):
            return NotImplemented
        return self._times == other._times and self._rates == other._rates

    def __hash__(self) -> int:
        return hash((self._times, self._rates))

    def __repr__(self) -> str:
        return (
            f"ValueSchedule(times={list(self._times)}, "
            f"rates={list(self._rates)})"
        )


def value_of(value_rate) -> Any:
    """``expected_npv``'s ``value_rate`` checked: a number of at least 0,
    or a ``ValueSchedule``."""
    if isinstance(value_rate, ValueSchedule):
        return value_rate
    number = checks.number_or_nan(value_rate)
    if not (math.isfinite(number) and number >= 0.0):
        raise ValueError(
            "value_rate must be a finite number, at least 0 (the value "
            "earned per unit time while the system works), or a "
            f"ValueSchedule of steps; got {value_rate!r}."
        )
    return number
