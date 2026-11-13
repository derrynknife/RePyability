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
    markers : dict or surpyval Measurements, optional
        With a ``JointModel``: the unit's readings so far, ``{age:
        value}``; or a ``surpyval.Measurements`` holding them, with the
        unit's id ``i`` (and ``measurement``, where it holds several).
    age : float, optional
        With a ``JointModel``: the unit's age now (surpyval's ``given``),
        at or after its last reading; 0 (new) by default, when it has no
        readings.
    i : str, number or tuple, optional
        With ``markers`` a ``Measurements``: the unit's id in it, whose
        readings (``markers.for_unit(i)``) the prediction takes.
    measurement : str, optional
        With ``markers`` a ``Measurements`` of several measurements: the
        one the joint model was fitted to
        (``markers.with_measurement(measurement)``).

    Attributes
    ----------
    i : str, number, tuple or None
        The unit's id in the ``Measurements`` its readings came from
        (None for readings given as a dict, or a prediction).
    measurement : str or None
        The name of the measurement the readings are of, where they came
        from a ``Measurements`` that names it.
    marker_unit : str or None
        The unit the readings are in, the ``Measurements``' label for the
        measurement as it was typed (``"mm/s"``), or None where it has
        none. It is a label only: nothing is converted.

    Raises
    ------
    TypeError
        If ``model`` is neither, or ``Z``, ``markers`` or ``age`` is given
        with a prediction, which holds them, or ``i`` or ``measurement``
        without a ``Measurements``.
    ValueError
        If the model's checks refuse the unit's inputs, a covariate is
        missing (``nan``), or the ``Measurements`` has no readings of
        unit ``i``, no measurement ``measurement``, or several and no
        ``measurement`` named.

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

    The same pump's readings kept as surpyval ``Measurements``, with the
    fleet's other pumps and their unit, read by its id:

    >>> from surpyval import Measurements
    >>> readings = Measurements(
    ...     {"P-101": {0: 2.1, 20: 2.5, 40: 3.0}, "P-102": {0: 1.9}},
    ...     name="vibration", units="mm/s",
    ... )
    >>> node = JointNode(model, markers=readings, i="P-101", age=40)
    >>> bool(node.sf(20.0) == pump.sf(20.0))
    True
    >>> node.i, node.measurement, node.marker_unit
    ('P-101', 'vibration', 'mm/s')
    """

    def __init__(
        self,
        model: Any,
        Z: Any = None,
        markers: Any = None,
        age: Optional[float] = None,
        *,
        i: Any = None,
        measurement: Any = None,
    ):
        joint_model, prediction = _joint_classes()
        self.i: Any = None
        self.measurement: Any = None
        self.marker_unit: Optional[str] = None
        if isinstance(model, joint_model) and _is_measurements(markers):
            markers = self._unit_readings(markers, i, measurement) or None
        elif (i is not None or measurement is not None) and not isinstance(
            model, prediction
        ):
            given = [
                name
                for name, value in (("i", i), ("measurement", measurement))
                if value is not None
            ]
            raise TypeError(
                f"{' and '.join(given)} pick a unit's readings out of a "
                "surpyval Measurements given as markers=; these markers "
                "are not one (a dict {age: value} is the unit's readings "
                "already)."
            )
        if isinstance(model, prediction):
            given = [
                name
                for name, value in (
                    ("Z", Z),
                    ("markers", markers),
                    ("age", age),
                    ("i", i),
                    ("measurement", measurement),
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

    def _unit_readings(self, readings: Any, i: Any, name: Any) -> dict:
        """Unit ``i``'s readings ``{age: value}`` of the measurement
        ``name`` (the one there is, if None) in the surpyval
        ``Measurements`` ``readings``, keeping the unit's id, the
        measurement's name and its unit label on the node."""
        if i is None:
            raise ValueError(
                "Give the unit's id, i=..., to read its readings from the "
                f"Measurements (its units: {_listed(readings.ids)})."
            )
        if name is None:
            if len(readings.names) > 1:
                raise ValueError(
                    "The Measurements hold several measurements ("
                    f"{_listed(readings.names)}): name the one the joint "
                    "model was fitted to, measurement=..."
                )
            name = readings.names[0]
        elif name not in readings.names:
            raise ValueError(
                f"The Measurements have no measurement {name!r}; they "
                f"have {_listed(readings.names)}."
            )
        one = readings.with_measurement(name)
        try:
            unit_readings = one.for_unit(i)[name]
        except KeyError:
            raise ValueError(
                f"The measurement {name!r} has no readings of unit {i!r}; "
                f"its units are {_listed(one.ids)}."
            ) from None
        self.i = i
        self.measurement = name
        self.marker_unit = one.units.get(name)
        return dict(unit_readings)

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

        Readings taken from a ``Measurements`` keep the unit's id
        (``"i"``), the measurement's name (``"measurement"``) and its unit
        label (``"marker_unit"``), each where it has one.

        Returns
        -------
        dict
            ``{"model": ..., "covariates": [...], "markers": [[age,
            value], ...], "age": ...}``, and ``"i"``, ``"measurement"``
            and ``"marker_unit"`` where the node has them.
        """
        pred = self.prediction
        out = {
            "model": self.model.to_dict(),
            "covariates": [float(z) for z in np.ravel(pred.covariates)],
            "markers": [[float(a), float(v)] for a, v in pred.markers.items()],
            "age": self.age,
        }
        if self.i is not None:
            out["i"] = _saved_id(self.i)
        if self.measurement is not None:
            out["measurement"] = _saved_id(self.measurement)
        if self.marker_unit is not None:
            out["marker_unit"] = self.marker_unit
        return out

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
        node = cls(
            model,
            Z=covariates if covariates else None,
            markers=markers or None,
            age=float(d["age"]),
        )
        node.i = _loaded_id(d.get("i"))
        node.measurement = _loaded_id(d.get("measurement"))
        node.marker_unit = d.get("marker_unit")
        return node

    def __repr__(self) -> str:
        labels = "".join(
            f"{name}={value!r}, "
            for name, value in (
                ("i", self.i),
                ("measurement", self.measurement),
                ("marker_unit", self.marker_unit),
            )
            if value is not None
        )
        return (
            f"JointNode({labels}age={self.age:g}, readings="
            f"{len(self.prediction.markers)})"
        )


def _is_measurements(markers: Any) -> bool:
    """Whether ``markers`` is a surpyval ``Measurements``."""
    from surpyval import Measurements

    return isinstance(markers, Measurements)


def _listed(values: list) -> str:
    """``values`` for a message: each repr'd, at most ten."""
    shown = ", ".join(repr(v) for v in values[:10])
    return shown + (
        f" and {len(values) - 10} more" if len(values) > 10 else ""
    )


def _saved_id(value: Any) -> Any:
    """A unit's id or a measurement's name as JSON keeps it: a tuple as
    ``{"tuple": [...]}`` (as surpyval saves a dataset's ids), text or a
    number as it is."""
    if isinstance(value, tuple):
        return {"tuple": [_saved_id(v) for v in value]}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _loaded_id(value: Any) -> Any:
    """``_saved_id``'s value back."""
    if isinstance(value, dict) and set(value) == {"tuple"}:
        return tuple(_loaded_id(v) for v in value["tuple"])
    return value
