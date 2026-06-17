import numpy as np

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

    def random(self, size):
        # Sum of `repeats` independent draws from the base model. Under
        # imperfect switching a spare only contributes if every switch up to
        # and including its own has succeeded.
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

    def mean(self, *args, **kwargs):
        # Exact, deterministic mean from the convolution (E[T] = integral of
        # the survival function).
        return self._sf_model.mean()

    def sf(self, *args, **kwargs):
        return self._sf_model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        return self._sf_model.ff(*args, **kwargs)
