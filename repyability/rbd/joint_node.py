"""Joint node: one unit of a fitted joint model, given its readings, as a
block of a non-repairable diagram.

A surpyval joint model (``surpyval.WeibullJointPH``, or any
``surpyval.JointPH(dist, path_model)`` fitter) models a marker read
repeatedly on each unit (a vibration level, a wear measurement) together
with the unit's life. Its ``predict`` gives one unit's
``DynamicPrediction``: the unit's failure-age distribution given its
covariates, its readings so far and that it is still running at its age
now. A ``JointNode`` puts that unit into a diagram.

Time in a diagram is time from now, and a prediction's functions take ages
since new. The node is the prediction moved to now: at mission time ``t``
its reliability is ``pred.sf(age + t)``, which surpyval has already
conditioned on the unit surviving to ``age``; its lifetimes are
``pred.random() - age`` and its mean is the unit's mean residual life. The
age is the prediction's, so the node is not conditioned again on a
``NodeState`` age (see ``NonRepairableRBD.sf_given_state``).
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._sampling import RowSampler

#: Why a repairable diagram takes no joint node, for its messages.
NOT_REPAIRABLE = (
    "A component that is replaced (a NonRepairable, or a component of a "
    "RepairableRBD) takes no joint model as its life: the prediction is "
    "of the unit in service now, given its readings, while each unit that "
    "replaces it is new, without them, and every life of such a "
    "component is drawn from one surpyval distribution. Use a "
    "NonRepairableRBD for the mission from now (a JointNode), and give a "
    "replaced component a distribution fitted to the population's lives."
)


def _joint_classes() -> tuple:
    """surpyval's ``JointModel`` and ``DynamicPrediction``."""
    from surpyval.univariate.joint import DynamicPrediction, JointModel

    return JointModel, DynamicPrediction


def is_joint(model) -> bool:
    """Whether ``model`` is a joint model's: a ``JointNode``, or surpyval's
    ``JointModel`` or ``DynamicPrediction``."""
    return isinstance(model, (JointNode, *_joint_classes()))


def as_node(model):
    """``model``, but a surpyval ``DynamicPrediction`` as the
    ``JointNode`` of its unit (a diagram takes a prediction as a node)."""
    _, prediction = _joint_classes()
    if isinstance(model, prediction):
        return JointNode(model)
    return model


class JointNode:
    """One unit of a fitted joint model, given its readings, as a node.

    surpyval's joint models (``surpyval.WeibullJointPH``) model a marker
    read repeatedly on each unit together with the unit's life; a fitted
    model's ``predict(Z, markers, given)`` is one unit's failure-age
    distribution given its covariates, its readings so far and that it
    is still running at age ``given``. This node is that unit in a
    [`NonRepairableRBD`][repyability.NonRepairableRBD], from now: at time
    ``t`` from now its reliability is ``pred.sf(age + t)`` (already
    conditioned on the unit reaching ``age``), its lifetimes are
    ``pred.random() - age`` and its ``mean`` is the unit's mean residual
    life. A ``DynamicPrediction`` given as a diagram's node is taken as
    the ``JointNode`` of its unit.

    The node's age is the prediction's: a ``NodeState`` for it in
    ``sf_given_state`` and the other condition-based methods may say
    that it has failed, or give that same age, but not another, which
    would condition it again (predict again, with ``given`` the new age
    and the readings since). A
    [`RepairableRBD`][repyability.RepairableRBD] refuses it: the units
    that replace this one are new, without its readings. See the guide's
    joint-model page.

    Parameters
    ----------
    model : surpyval DynamicPrediction or JointModel
        The unit's prediction (``joint_model.predict(...)``), or the
        fitted joint model, with the unit's ``Z``, ``markers`` and
        ``age``.
    Z : dict, array_like or surpyval ItemLifetime, optional
        With a ``JointModel``: the unit's covariates (``{name: value}``,
        or a row in the order of the model's ``feature_names``), needed
        where the model has covariates; or an ``ItemLifetime`` still
        running, which holds the unit's covariates, readings and age.
    markers : dict, optional
        With a ``JointModel``: the unit's readings so far, ``{age:
        value}``.
    age : float, optional
        With a ``JointModel``: the unit's age now (surpyval's ``given``),
        at or after its last reading; 0 (new) by default, when it has no
        readings.

    Raises
    ------
    TypeError
        If ``model`` is neither, or ``Z``, ``markers`` or ``age`` is given
        with a prediction, which holds them.
    ValueError
        If the model's checks refuse the unit's inputs, or a covariate is
        missing (``nan``).

    Examples
    --------
    Fit a joint model to a small fleet in surpyval, and put a pump whose
    marker has risen into a diagram, 40 hours old:

    >>> import numpy as np
    >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
    >>> from repyability import JointNode, NonRepairableRBD
    >>> rng = np.random.default_rng(3)
    >>> items = []
    >>> for k in range(60):
    ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
    ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
    ...     end = min(life, 120.0)
    ...     ages = np.arange(0.0, end, 10.0)
    ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
    ...     outcome = "observed" if life < 120 else "right_censored"
    ...     items.append(
    ...         ItemLifetime(k, **{outcome: end},
    ...                      markers=dict(zip(ages, y)))
    ...     )
    >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
    >>> pred = model.predict(markers={0: 2.1, 20: 2.5, 40: 3.0}, given=40)
    >>> pump = JointNode(pred)
    >>> bool(pump.sf(20.0) == pred.sf(60.0))
    True
    >>> rbd = NonRepairableRBD([("in", "pump"), ("pump", "out")],
    ...                        {"pump": pump})
    >>> bool(rbd.sf(20.0) == pred.sf(60.0))
    True
    """

    def __init__(
        self,
        model: Any,
        Z: Any = None,
        markers: Optional[dict] = None,
        age: Optional[float] = None,
    ):
        joint_model, prediction = _joint_classes()
        if isinstance(model, prediction):
            given = [
                name
                for name, value in (
                    ("Z", Z),
                    ("markers", markers),
                    ("age", age),
                )
                if value is not None
            ]
            if given:
                raise TypeError(
                    "A JointNode of a DynamicPrediction takes the unit's "
                    "covariates, readings and age from the prediction: "
                    f"leave out {', '.join(given)} (or give the "
                    "JointModel with them)."
                )
            pred = model
        elif isinstance(model, joint_model):
            if markers and age is None:
                raise ValueError(
                    "Give the unit's age now, age=..., at or after its "
                    "last reading: the readings alone do not say it."
                )
            pred = model.predict(Z, markers, given=age)
        else:
            raise TypeError(
                "A JointNode takes a surpyval joint model's prediction for "
                "one unit (joint_model.predict(Z, markers, given)), or the "
                "fitted JointModel with the unit's Z, markers and age; got "
                f"a {type(model).__name__}."
            )
        if not np.all(np.isfinite(np.asarray(pred.covariates, dtype=float))):
            raise ValueError(
                "The unit's covariates are missing (nan), so its life has "
                "no prediction: give each of them."
            )
        self.prediction = pred
        self.model = pred.model
        self.age = float(pred.age)

    # -- Node reliability interface ---------------------------------------

    def _at(self, name: str, x: ArrayLike) -> Any:
        """The prediction's function ``name`` at the ages ``age + x`` (the
        times ``x`` from now), with the shape of ``x``."""
        t = np.asarray(x, dtype=float)
        out = np.asarray(getattr(self.prediction, name)(self.age + t), float)
        return out.reshape(t.shape)[()]

    def sf(self, x: ArrayLike) -> Any:
        """The probability that the unit survives a further ``x``:
        ``pred.sf(age + x)``.

        Parameters
        ----------
        x : array_like
            The time(s) from now.

        Returns
        -------
        float or numpy.ndarray
            With the shape of ``x``.
        """
        return self._at("sf", x)

    def ff(self, x: ArrayLike) -> Any:
        """The probability that the unit fails within a further ``x``:
        ``pred.ff(age + x)``, worked out in its own right.

        Parameters
        ----------
        x : array_like
            The time(s) from now.
        """
        return self._at("ff", x)

    def Hf(self, x: ArrayLike) -> Any:
        """The cumulative hazard over the next ``x``, ``-log sf(x)``.

        Parameters
        ----------
        x : array_like
            The time(s) from now.
        """
        return self._at("Hf", x)

    def hf(self, x: ArrayLike) -> Any:
        """The unit's hazard ``x`` from now, given its readings.

        Parameters
        ----------
        x : array_like
            The time(s) from now.
        """
        return self._at("hf", x)

    def df(self, x: ArrayLike) -> Any:
        """The density of the unit's life left at ``x`` from now.

        Parameters
        ----------
        x : array_like
            The time(s) from now.
        """
        return self._at("df", x)

    def qf(self, p: ArrayLike) -> Any:
        """The life left that the unit outlives with probability ``1 -
        p``: ``pred.qf(p) - age`` (``inf`` where the survival never falls
        to ``1 - p``).

        Parameters
        ----------
        p : array_like
            The probabilities of failing.
        """
        q = np.asarray(p, dtype=float)
        out = np.asarray(self.prediction.qf(q), dtype=float) - self.age
        return out.reshape(q.shape)[()]

    def _knots(self) -> np.ndarray:
        """Times marking the curve's scale, for its integral: its
        quantiles."""
        from ._point_availability import knots

        return knots(self)

    def mean(self) -> float:
        """The unit's mean residual life: the integral of ``sf`` from now,
        to a relative 1e-10 on pieces split at its quantiles; ``inf`` where
        its survival stays above 0 for ever (a marker path that falls
        away can leave a chance of never failing).

        Returns
        -------
        float
            The mean life left.
        """
        from ._mean_lifetime import mean_lifetime

        return float(mean_lifetime(self.sf, self._knots()))

    def _draw(self, u: np.ndarray) -> np.ndarray:
        """The lives left ``qf(u)`` of the uniforms ``u``."""
        return np.asarray(self.qf(np.asarray(u, dtype=float)), dtype=float)

    def random(self, size: int) -> np.ndarray:
        """Draw ``size`` lives left, from now: ``pred.qf(u) - age`` of
        uniforms ``u`` from numpy's global RNG, as ``pred.random`` draws
        failure ages. Seed it (``np.random.seed``) to reproduce the draws;
        an RBD's seeded methods do.

        Parameters
        ----------
        size : int
            The number of draws.

        Returns
        -------
        numpy.ndarray
            The ``size`` lives left, shape ``(size,)``.
        """
        return self._draw(np.random.uniform(size=size))

    def _row_sampler(self) -> RowSampler:
        """``random(1)`` as a :class:`~._sampling.RowSampler` (one uniform
        per draw), so an RBD with this node batches its draws."""
        return RowSampler(
            1, lambda u: self._draw(np.ascontiguousarray(u[:, 0]))
        )

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """The node as a JSON-friendly dict: the joint model's own
        ``to_dict()`` and the unit's inputs, its covariates (in the order
        of the model's ``feature_names``), its readings as ``[age,
        value]`` pairs and its age. ``from_dict`` rebuilds it, and an
        RBD's ``to_dict``/``to_json`` save a joint node so. A joint model
        on a path model of one's own does not save (surpyval raises
        ``NotImplementedError``).

        Returns
        -------
        dict
            ``{"model": ..., "covariates": [...], "markers": [[age,
            value], ...], "age": ...}``.
        """
        pred = self.prediction
        return {
            "model": self.model.to_dict(),
            "covariates": [float(z) for z in np.ravel(pred.covariates)],
            "markers": [[float(a), float(v)] for a, v in pred.markers.items()],
            "age": self.age,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JointNode":
        """Rebuild a node from :meth:`to_dict`'s dict.

        Parameters
        ----------
        d : dict
            A dict ``to_dict`` made, e.g. after a JSON round trip.

        Returns
        -------
        JointNode
            The node, its prediction made again from the saved model and
            the unit's inputs.

        Examples
        --------
        >>> import json
        >>> import numpy as np
        >>> from surpyval import ItemLifetime, SurpyvalData, WeibullJointPH
        >>> from repyability import JointNode
        >>> rng = np.random.default_rng(3)
        >>> items = []
        >>> for k in range(60):
        ...     start, rate = rng.normal(2.0, 0.4), rng.normal(0.02, 0.005)
        ...     life = 100 * rng.weibull(2.0) * np.exp(-0.9 * (start - 2))
        ...     end = min(life, 120.0)
        ...     ages = np.arange(0.0, end, 10.0)
        ...     y = start + rate * ages + rng.normal(0, 0.2, ages.size)
        ...     outcome = "observed" if life < 120 else "right_censored"
        ...     items.append(
        ...         ItemLifetime(k, **{outcome: end},
        ...                      markers=dict(zip(ages, y)))
        ...     )
        >>> model = WeibullJointPH.fit(SurpyvalData.from_items(items))
        >>> node = JointNode(model, markers={0: 2.1, 40: 3.0}, age=40)
        >>> back = JointNode.from_dict(json.loads(json.dumps(node.to_dict())))
        >>> bool(back.sf(20.0) == node.sf(20.0))
        True
        """
        import surpyval

        model = surpyval.from_dict(d["model"])
        covariates = list(d.get("covariates") or [])
        markers = {float(a): float(v) for a, v in d.get("markers") or []}
        return cls(
            model,
            Z=covariates if covariates else None,
            markers=markers or None,
            age=float(d["age"]),
        )

    def __repr__(self) -> str:
        return (
            f"JointNode(age={self.age:g}, readings="
            f"{len(self.prediction.markers)})"
        )
