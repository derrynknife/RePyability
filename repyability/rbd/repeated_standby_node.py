import numpy as np

from repyability.utils.wrappers import numpy_seed

from ._sampling import RowSampler, column, inverse_sampler
from .numerical_convolution import (
    ConvolvedSurvival,
    is_perfect_switching,
    switch_success_probs,
)


class RepeatedStandbyNode:
    def __init__(
        self,
        model,
        repeats,
        N=10_000,
        lower=-np.inf,
        switching_probability=1.0,
    ):
        # N and lower are kept for backwards compatibility (a Kaplan-Meier fit
        # was previously made here); they are no longer used.
        self.model = model
        self.repeats = repeats
        self.switching_probability = switching_probability

        # Repeated cold standby: the lifetime is the sum of `repeats`
        # independent copies of `model` (a mixture of partial sums under
        # imperfect switching). Its survival function is computed
        # deterministically by numerical convolution.
        self._sf_model = ConvolvedSurvival(
            [model] * repeats, switching_probability=switching_probability
        )

    def random(self, size, seed=None):
        # Sum of `repeats` independent draws from the base model. Under
        # imperfect switching a spare only contributes if every switch up to
        # and including its own has succeeded.
        with numpy_seed(seed):
            x_random = np.asarray(self.model.random(size), dtype=float)
            if is_perfect_switching(self.switching_probability):
                for _ in range(self.repeats - 1):
                    x_random = x_random + self.model.random(size)
            else:
                probs = switch_success_probs(
                    self.switching_probability, self.repeats
                )
                running = np.ones(size, dtype=bool)
                for p in probs:
                    running = running & (np.random.random(size) < p)
                    x_random = x_random + np.where(
                        running, self.model.random(size), 0.0
                    )
        return x_random

    def _row_sampler(self):
        """``random(1)`` as a :class:`~._sampling.RowSampler`, so an RBD with
        this node batches its draws; ``None`` unless the model's draws can be
        replayed. Columns follow the order ``random(1)`` draws in: one per
        copy, and under imperfect switching a switch draw before each
        spare's."""
        unit = inverse_sampler(self.model)
        if unit is None:
            return None

        if is_perfect_switching(self.switching_probability):

            def draw(u):
                x = column(u, 0, unit)
                for j in range(1, self.repeats):
                    x = x + column(u, j, unit)
                return x

            return RowSampler(self.repeats, draw)

        probs = switch_success_probs(self.switching_probability, self.repeats)

        def draw(u):
            x = column(u, 0, unit)
            running = np.ones(len(u), dtype=bool)
            for i, p in enumerate(probs):
                running = running & (u[:, 1 + 2 * i] < p)
                x = x + np.where(running, column(u, 2 + 2 * i, unit), 0.0)
            return x

        return RowSampler(1 + 2 * len(probs), draw)

    def mean(self, *args, **kwargs):
        # Exact, deterministic mean from the convolution (E[T] = integral of
        # the survival function).
        return self._sf_model.mean()

    def sf(self, *args, **kwargs):
        return self._sf_model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        return self._sf_model.ff(*args, **kwargs)

    def cs(self, x, X):
        """Conditional survival ``R(x | X) = sf(X + x) / sf(X)``."""
        from repyability.utils.wrappers import conditional_survival

        return conditional_survival(self, x, X)
