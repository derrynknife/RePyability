"""A component that degrades through stages of lower capacity before it
fails: a node with several capacity levels over time (#98)."""

from typing import Any, Sequence, Tuple

import numpy as np

from ._model_utils import model_mean, never_fails
from .numerical_convolution import ConvolvedSurvival
from .results import CapacityDistribution
from .standby_node import StandbyModel, _ExponentialStandbySurvival


class DegradingNode(StandbyModel):
    """A component that degrades through stages before it fails.

    It runs in its first stage, at the first capacity, for a time drawn
    from the first stage's model; then in the second stage, at the second
    capacity, for a time drawn from the second's; and so on. It has failed
    once its last stage ends. A pump that runs at full output until it
    wears, then at half output until it fails, has two stages. The stages'
    times are independent, and each is a lifetime model, such as one fitted
    by surpyval to the times units spent in that stage.

    As an RBD node its reliability is the probability that its last stage
    has not ended: its lifetime is the sum of its stages' times, as a cold
    standby arrangement's is, so it is a
    [`StandbyModel`][repyability.StandbyModel] of the stage models and can
    be used wherever one can (a node of a ``NonRepairableRBD``, or the
    reliability of a ``RepairableRBD`` component). The capacity analysis
    (``capacity_distribution``) uses its stages' capacities, weighted by
    the probability of being in each stage, unless the RBD gives the node a
    capacity of its own.

    Parameters
    ----------
    stages : sequence of (float, model)
        The stages in order, each a ``(capacity, model)`` pair: the
        capacity the component has in the stage, a positive number
        (``inf`` for no limit), and a lifetime model for the time it spends
        there, exposing ``sf``, ``df``, ``random`` and ``mean`` (e.g. a
        surpyval distribution).

    Attributes
    ----------
    stages : tuple of (float, model)
        The stages, with the capacities as floats.
    capacities : tuple of float
        The stages' capacities, in order.

    Raises
    ------
    ValueError
        If there are no stages, a stage is not a ``(capacity, model)`` pair,
        or a capacity is not a positive number.

    Examples
    --------
    A pump at full output (100) for a Weibull time, then at half output
    for an exponential one:

    >>> import surpyval as surv
    >>> from repyability import DegradingNode
    >>> pump = DegradingNode([
    ...     (100, surv.Weibull.from_params([1000, 2])),
    ...     (50, surv.Exponential.from_params([1 / 500])),
    ... ])
    >>> round(pump.mean(), 1)  # 886.2 + 500
    1386.2
    >>> capacity = pump.capacity_distribution(1000)
    >>> capacity.levels.tolist()
    [0.0, 50.0, 100.0]
    >>> capacity.probabilities.round(4).tolist()
    [0.3152, 0.3169, 0.3679]
    >>> round(float(pump.sf(1000)), 4)  # in either stage
    0.6848
    """

    # The stage the component is in comes from the partial sums of the
    # stages' times.
    _partial_sums = True

    def __init__(self, stages: Sequence[Tuple[Any, Any]]):
        stages = list(stages)
        if not stages:
            raise ValueError("A DegradingNode needs at least one stage.")
        capacities = []
        models = []
        for i, stage in enumerate(stages):
            try:
                capacity, model = stage
            except (TypeError, ValueError):
                raise ValueError(
                    f"Stage {i} must be a (capacity, model) pair, got "
                    f"{stage!r}."
                ) from None
            if isinstance(capacity, (bool, np.bool_)) or not isinstance(
                capacity, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    f"The capacity of stage {i} must be a number, got "
                    f"{capacity!r}."
                )
            if not float(capacity) > 0.0:
                raise ValueError(
                    f"The capacity of stage {i} must be positive (inf for no "
                    f"limit), got {capacity!r}."
                )
            capacities.append(float(capacity))
            models.append(model)
        super().__init__(models)
        self.capacities = tuple(capacities)
        self.stages = tuple(zip(self.capacities, models))

    def stage_probabilities(self, x) -> np.ndarray:
        """The probability that the component is in each stage at time/s
        ``x``: one row per stage, in order (one value per stage for a
        scalar ``x``). The rest of the probability, ``ff(x)``, is that it
        has failed.

        Being in stage ``j`` at ``x`` means the first ``j - 1`` stages have
        ended by ``x`` and the ``j``-th has not, so its probability is the
        survival function of the sum of the first ``j`` stages' times less
        that of the first ``j - 1``. Those come from the same (exact or
        numerical) convolution as ``sf``, so the rows add up to ``sf(x)``.
        A time before 0 is taken as 0: a new component is in its first
        stage.

        Parameters
        ----------
        x : float or array_like
            The time(s).

        Returns
        -------
        numpy.ndarray
            Shape ``(n_stages,)`` for a scalar ``x``, else
            ``(n_stages, len(x))``.
        """
        scalar = np.ndim(x) == 0
        x = np.maximum(np.atleast_1d(np.asarray(x, dtype=float)), 0.0)
        partial = [self._partial_sf(j, x) for j in range(1, self.N + 1)]
        rows = [partial[0]] + [
            np.clip(partial[j] - partial[j - 1], 0.0, 1.0)
            for j in range(1, self.N)
        ]
        out = np.vstack(rows)
        return out[:, 0] if scalar else out

    def _partial_sf(self, j: int, x: np.ndarray) -> np.ndarray:
        """The survival function of the sum of the first ``j`` stages'
        times at ``x``: the first stage's own ``sf``, and after it the
        partial sums of the convolution behind ``sf`` (Erlang for identical
        exponential stages)."""
        if j == 1:
            return np.asarray(self.reliabilities[0].sf(x), dtype=float)
        survival = self._sf_model
        if isinstance(survival, _ExponentialStandbySurvival):
            from scipy.stats import gamma

            return gamma.sf(x, a=j, scale=1.0 / survival.rate)
        assert isinstance(survival, ConvolvedSurvival)
        return np.asarray(survival.partial_sf(j, x), dtype=float)

    def sf(self, x, *args, **kwargs):
        """Survival function (reliability): the probability that the last
        stage has not ended by time/s ``x``.

        With one stage, the stage model's own ``sf``; with several, that of
        the sum of the stages' times (see
        [`StandbyModel.sf`][repyability.StandbyModel.sf]).

        Parameters
        ----------
        x : array_like
            The time(s).
        *args, **kwargs
            Passed on, with ``x``, to the underlying survival model.

        Returns
        -------
        float or numpy.ndarray
            The probability of surviving beyond ``x``.
        """
        if self.N == 1:
            return self.reliabilities[0].sf(x, *args, **kwargs)
        return super().sf(x, *args, **kwargs)

    def ff(self, x, *args, **kwargs):
        """Cumulative failure probability, ``1 - sf(x)``.

        Parameters
        ----------
        x : array_like
            The time(s).
        *args, **kwargs
            Passed on, with ``x``, to the underlying survival model.

        Returns
        -------
        float or numpy.ndarray
            The probability that the last stage has ended by ``x``.
        """
        if self.N == 1:
            return self.reliabilities[0].ff(x, *args, **kwargs)
        return super().ff(x, *args, **kwargs)

    def mean(self, mc_samples=None, seed=None):
        """Mean lifetime (MTTF): the sum of the stages' mean times, exactly;
        infinite if a stage may never end.

        Parameters
        ----------
        mc_samples : int, optional
            Ignored: accepted so ``mean`` can be called as a
            ``StandbyModel``'s is.
        seed : int or None, optional
            Ignored, as ``mc_samples`` is.

        Returns
        -------
        float
            The mean lifetime.
        """
        return float(sum(model_mean(m) for m in self.reliabilities))

    def capacity_distribution(self, x) -> CapacityDistribution:
        """The distribution of the component's capacity at time/s ``x``: the
        capacity of the stage it is in, or 0 once it has failed.

        Parameters
        ----------
        x : float or array_like
            The time(s).

        Returns
        -------
        CapacityDistribution
            The capacities it can have (0 and each stage's, equal ones
            merged) and their probabilities: one per level for a scalar
            ``x``, and one row per level and one column per time for an
            array.
        """
        from .capacity import merged

        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        stages = self.stage_probabilities(x)
        failed = np.clip(1.0 - np.sum(stages, axis=0), 0.0, 1.0)
        levels, probabilities = merged(
            np.array((0.0,) + self.capacities), np.vstack([failed, stages])
        )
        return CapacityDistribution(
            levels, probabilities[:, 0] if scalar else probabilities
        )

    def stage_fractions(self) -> np.ndarray:
        """The long-run fraction of its working time the component spends
        in each stage, when it is renewed after each failure.

        Each renewal runs through the stages again, so by the
        renewal-reward theorem the fraction in stage ``j`` is its mean time
        over the mean lifetime. A stage that may never end (a model with
        units that never fail) catches the component for good sooner or
        later, and the fractions are then the probabilities of being caught
        in each stage.

        Returns
        -------
        numpy.ndarray
            One fraction per stage; they sum to 1.
        """
        ends = np.array([1.0 - never_fails(m) for m in self.reliabilities])
        if np.all(ends == 1.0):
            means = np.array([model_mean(m) for m in self.reliabilities])
            total = float(np.sum(means))
            if total > 0.0:
                return means / total
            # A component that fails at once spends no time in any stage.
            return np.eye(self.N)[0]
        before = np.concatenate([[1.0], np.cumprod(ends)[:-1]])
        caught = before * (1.0 - ends)
        return caught / np.sum(caught)
