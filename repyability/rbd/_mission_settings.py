"""A mission drawn as one diagram, each phase setting its blocks' states,
models and covariates (see ``PhasedMission.from_diagram``).

A phase is a :class:`MissionPhase`: a name, a duration, and for some
blocks a :class:`BlockSetting` (a state, and the block's life model or the
covariates of a regression block in the phase), with the k-out-of-n votes
it needs. How a block behaves in a phase is read through one function,
:func:`behaviour`, which takes its setting: a new field of
``BlockSetting`` is one more branch there.

Every block has its own running clock (its age) and a budget of
cumulative hazard, an ``Exp(1)`` draw. While it runs on the model ``M``
from age ``a`` to ``a + d`` it spends ``H_M(a, a + d)``; on standby,
idle, it neither ages nor spends; guaranteed, it ages but spends nothing.
It fails when it has spent its budget. A block's life on its running clock
(``Life``) is one model per phase (:func:`phase_model`): the phase's
model, or for a regression block given covariates, its node's regression
model at them (a model of the phase like any other), or else its node's
model. From one phase's model to the next its age is carried
(``Life.carry``): kept, with the hazard left to spend, or, between one
fitted regression model at two covariate vectors of a family whose
covariates rescale time, rescaled with them (:func:`_time_ratio`). Where
its running clock is the mission's (it is never on standby) the phases'
models make one life on that clock, :class:`PhasedLife`; where it is not,
the models are taken phase by phase as the block runs.
"""

import dataclasses
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd._sampling import RowSampler
from repyability.utils.checks import (
    is_number,
    no_distribution,
    one_of,
    real_array,
)

#: Running, ageing and able to fail (the default).
WORKING = "working"
#: Cold standby: idle (not ageing, unable to fail) until the phase's
#: structure needs it, then running, ageing and able to fail for the rest
#: of the phase.
STANDBY = "standby"
#: Running and ageing, but unable to fail in the phase.
GUARANTEED = "guaranteed"
#: Taken as working by the phase's structure, whatever its condition; it
#: runs, ages and can fail all the same.
NOT_NEEDED = "not_needed"
#: The states a block can be in during a phase.
STATES = (WORKING, STANDBY, GUARANTEED, NOT_NEEDED)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BlockSetting:
    """How one block of the diagram behaves in one phase of a mission.

    Made by keyword; a bare state name stands for ``BlockSetting(state=...)``
    in a [`MissionPhase`][repyability.MissionPhase]'s ``blocks``. See the
    [phased missions guide](../guide/phased-missions.md) for what each
    state means.

    Parameters
    ----------
    state : str, optional
        ``"working"`` (the default): running, ageing, able to fail.
        ``"standby"``: cold standby, idle (not ageing, unable to fail)
        until the phase's structure needs it, then running for the rest of
        the phase; the switch is perfect and instant. ``"guaranteed"``:
        running and ageing, but unable to fail in the phase.
        ``"not_needed"``: the phase's structure takes it as working,
        whatever its condition, while it runs, ages and can fail.
    covariates : array_like, optional
        For a block whose model is a ``RegressionNode`` with fixed
        covariates: its covariate vector in this phase (by default, the
        node's own).
    model : optional
        The block's life model in this phase, in place of its node's (by
        default, the node's): the block behaves differently in this phase
        (a pump in its start-up phase, say), without a second block. Its
        age is carried into and out of the phase, as for covariates; it
        is checked against the node when the mission is made, as a model
        the node could have been given. With ``covariates`` too, it must
        be a ``RegressionNode`` with fixed covariates, run at these.

    Raises
    ------
    ValueError
        If ``state`` is not one of the four, ``covariates`` is not a
        non-empty vector of finite numbers, ``model`` is a probability of
        failing, or ``covariates`` come with a ``model`` that is not a
        ``RegressionNode`` with fixed covariates of their length.
    TypeError
        If ``model`` is not a model with a survival function, or is a
        surpyval distribution itself.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import BlockSetting
    >>> BlockSetting(state="standby")
    BlockSetting(state='standby', covariates=None, model=None)
    >>> BlockSetting(covariates=[1.2, 40]).covariates
    (1.2, 40.0)
    >>> start_up = BlockSetting(model=surv.Weibull.from_params([40.0, 0.8]))
    >>> start_up.state
    'working'
    """

    state: str = WORKING
    covariates: Optional[Tuple[float, ...]] = None
    model: Any = None

    def __post_init__(self):
        for field, value in _checked(self).items():
            object.__setattr__(self, field, value)


def _checked(setting: BlockSetting) -> Dict[str, Any]:
    """Each field of ``setting`` checked and in its stored form (the one
    place a ``BlockSetting``'s fields are checked)."""
    one_of("state", setting.state, STATES)
    covariates = setting.covariates
    if covariates is not None:
        try:
            values = real_array(covariates, "covariates")
        except TypeError as error:
            raise ValueError(str(error)) from None
        if (
            values.ndim > 1
            or values.size == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "covariates must be a vector of finite numbers, got "
                f"{covariates!r}."
            )
        covariates = tuple(float(v) for v in np.atleast_1d(values))
    model = setting.model
    if model is not None:
        _check_phase_model(model, covariates)
    return {"state": setting.state, "covariates": covariates, "model": model}


def _check_phase_model(model, covariates) -> None:
    """Raise if ``model`` cannot be a block's model in a phase, whatever
    the block (``covariates`` given with it, or None); the node's own
    checks follow when the mission is made (:func:`phase_model`)."""
    from repyability.rbd.regression_node import RegressionNode

    what = "The model of a BlockSetting"
    no_distribution(model, what)
    if not callable(getattr(model, "sf", None)):
        raise TypeError(
            f"{what} is {model!r}, which has no survival function (sf): "
            "give a life model, such as a surpyval distribution "
            "(surpyval.Weibull.from_params([alpha, beta])) or a "
            "RegressionNode."
        )
    if is_fixed_probability(model):
        raise ValueError(
            f"{what} is a probability of failing, not a life: a block's "
            "age is carried from phase to phase, and a probability has no "
            "age to carry. Give a distribution of its life in the phase."
        )
    if covariates is None:
        return
    if not isinstance(model, RegressionNode) or model.covariates is None:
        raise ValueError(
            f"{what} comes with covariates, which set a regression "
            "model's covariates, but the model is not a RegressionNode "
            "with fixed covariates: give the model alone, or a "
            "RegressionNode(model, covariates=...) to run at these "
            "covariates."
        )
    if len(covariates) != len(model.covariates):
        raise ValueError(
            f"{what} takes {len(model.covariates)} covariate(s); the "
            f"setting gives {len(covariates)}."
        )


def setting_to_dict(setting: BlockSetting) -> Dict[str, Any]:
    """A ``BlockSetting`` as plain data, for a mission's summary: its
    state and covariates, and its model, when given, in its saved form
    (``serialisation.serialise_model``), or as its ``repr`` where it has
    none."""
    from repyability.rbd._model_utils import SAVE_ERRORS
    from repyability.rbd.serialisation import serialise_model

    out: Dict[str, Any] = {
        "state": setting.state,
        "covariates": (
            None if setting.covariates is None else list(setting.covariates)
        ),
    }
    if setting.model is not None:
        try:
            out["model"] = serialise_model(setting.model)
        except SAVE_ERRORS:
            out["model"] = repr(setting.model)
    return out


@dataclasses.dataclass(frozen=True)
class MissionPhase:
    """One phase of a mission drawn as one diagram (see
    [`PhasedMission.from_diagram`][repyability.PhasedMission.from_diagram]).

    Parameters
    ----------
    name : Hashable
        The phase's name, distinct within the mission.
    duration : float
        How long the phase lasts: finite, and not negative (a phase of no
        duration needs its structure working at that instant).
    blocks : dict, optional
        For some blocks (by node name), how each behaves in this phase: a
        state name (``"working"``, ``"standby"``, ``"guaranteed"``,
        ``"not_needed"``) or a [`BlockSetting`][repyability.BlockSetting].
        A block not named is ``"working"``, at its node's covariates. The
        order matters for standby blocks: they are started in this order
        when the structure needs them.
    k : dict, optional
        For some k-out-of-n nodes, the number of their inputs that must
        work in this phase (by default, the diagram's).

    Raises
    ------
    ValueError
        If the duration is not a finite number at least 0, a block's
        setting is neither a state name nor a ``BlockSetting``, or a vote
        count is not a whole number of at least 1.

    Examples
    --------
    >>> from repyability import MissionPhase
    >>> phase = MissionPhase("take-off", 0.2, blocks={"apu": "standby"})
    >>> phase.blocks["apu"].state
    'standby'
    """

    name: Hashable
    duration: float
    blocks: Mapping[Hashable, Any] = dataclasses.field(default_factory=dict)
    k: Mapping[Hashable, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        duration = self.duration
        if not (is_number(duration) and 0.0 <= float(duration) < np.inf):
            raise ValueError(
                f"Phase {self.name!r}: the duration must be a finite number, "
                f"not negative, got {duration!r}."
            )
        object.__setattr__(self, "duration", float(duration))
        blocks: Dict[Hashable, BlockSetting] = {}
        for node, setting in dict(self.blocks or {}).items():
            if isinstance(setting, str):
                setting = BlockSetting(state=setting)
            if not isinstance(setting, BlockSetting):
                raise ValueError(
                    f"Phase {self.name!r}: the setting of block {node!r} "
                    "must be a state name or a BlockSetting, got "
                    f"{setting!r}."
                )
            blocks[node] = setting
        object.__setattr__(self, "blocks", blocks)
        votes: Dict[Hashable, int] = {}
        for node, count in dict(self.k or {}).items():
            if not (
                is_number(count) and float(count) == int(count) and count >= 1
            ):
                raise ValueError(
                    f"Phase {self.name!r}: the vote count of node {node!r} "
                    f"must be a whole number of at least 1, got {count!r}."
                )
            votes[node] = int(count)
        object.__setattr__(self, "k", votes)

    def setting(self, node: Hashable) -> BlockSetting:
        """The setting of block ``node`` in this phase (working, on its
        node's model at its covariates, when not named)."""
        return self.blocks.get(node, _DEFAULT)


_DEFAULT = BlockSetting()


class Behaviour(NamedTuple):
    """How a block behaves in a phase, read from its setting."""

    #: Whether the phase's structure looks at it (else taken as working).
    in_structure: bool
    #: Whether it is idle until the structure needs it (cold standby).
    standby: bool
    #: Whether it spends hazard while it runs (not when guaranteed).
    spends: bool
    #: Its covariates in the phase, or None for its node's.
    covariates: Optional[Tuple[float, ...]]
    #: Its life model in the phase, or None for its node's.
    model: Any


def behaviour(setting: BlockSetting) -> Behaviour:
    """How a block with ``setting`` behaves in its phase: the one place a
    setting's fields are read."""
    return Behaviour(
        in_structure=setting.state != NOT_NEEDED,
        standby=setting.state == STANDBY,
        spends=setting.state != GUARANTEED,
        covariates=setting.covariates,
        model=setting.model,
    )


# ----------------------------------------------------------------------
# A block's life on its running clock
# ----------------------------------------------------------------------


def _hazard(model, x) -> np.ndarray:
    """The cumulative hazard of the life ``model`` at the ages ``x``."""
    from repyability.rbd.regression_node import RegressionNode

    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        if isinstance(model, RegressionNode):
            if model.schedule is not None:
                out = model.model.Hf_tvc(x, model.schedule)
            else:
                out = model.model.Hf(x, model._Z(x.size))
        elif callable(getattr(model, "Hf", None)):
            out = model.Hf(x)
        else:
            out = -np.log(np.clip(np.asarray(model.sf(x), float), 0, 1))
    return np.reshape(np.asarray(out, dtype=float), x.shape)


def _inverse(model, target: np.ndarray) -> np.ndarray:
    """The ages at which the life ``model`` has spent ``target``."""
    from repyability.rbd.regression_node import _inverse_hazard

    return _inverse_hazard(lambda t: _hazard(model, t), target)


def _time_ratio(before, after) -> Optional[float]:
    """The factor by which a block's age is rescaled, carried from the
    model ``before`` to ``after``: where both are one fitted regression
    model (the same object) at fixed covariates, of a family whose
    covariates rescale time (accelerated failure time or life), the ratio
    of their time scales; None otherwise, the age then kept, with the
    hazard left to spend.

    Raises
    ------
    ValueError
        If they are one regression model at two covariate vectors of a
        family whose covariates neither rescale time nor act on its
        hazard or odds, so that there is no age to carry.
    """
    from surpyval.univariate.regression._kinds import (
        ACCELERATED_FAILURE_TIME,
        ACCELERATED_LIFE,
        ADDITIVE_HAZARD,
        PROPORTIONAL_HAZARD,
        PROPORTIONAL_ODDS,
    )

    from repyability.rbd.regression_node import RegressionNode

    if not (
        isinstance(before, RegressionNode)
        and isinstance(after, RegressionNode)
        and before.covariates is not None
        and after.covariates is not None
        and before.model is after.model
    ):
        return None
    kind = getattr(before.model, "kind", None)
    if kind in (ACCELERATED_FAILURE_TIME, ACCELERATED_LIFE):
        # Each model's quantile is the baseline's over its rate, so their
        # ratio is the ratio of the rates, at any probability.
        half = np.array([0.5])
        return float(after._quantiles(half)[0] / before._quantiles(half)[0])
    if kind in (PROPORTIONAL_HAZARD, ADDITIVE_HAZARD, PROPORTIONAL_ODDS):
        return None
    raise ValueError(
        "its covariates change between phases, and its age is carried "
        "from one phase to the next only for a proportional-hazards, "
        "additive-hazards, proportional-odds, accelerated-failure-time or "
        f"accelerated-life model; its model is {kind!r}."
    )


def _carried(before, after, age):
    """The age ``age`` on the model ``before``, on ``after``'s clock."""
    if before is after:
        return age
    ratio = _time_ratio(before, after)
    return age if ratio is None else age * ratio


class PhasedLife:
    """A block's life on its running clock where that clock is the
    mission's (it is never on standby): in phase ``j``, which starts at
    ``s_j``, the phase's model ``M_j``, entered at the age ``e_j`` on its
    clock (carried from the phase before, see ``_carried``) with ``S_j``
    spent before, so that its cumulative hazard at ``t`` is ``S_j +
    H_j(e_j + t - s_j) - H_j(e_j)``. After the last phase it runs on the
    last phase's model."""

    def __init__(
        self, models: List[Any], starts: List[float], durations: List[float]
    ):
        self.models = list(models)
        self.starts = np.asarray(starts, dtype=float)
        entered, spent = [0.0], [0.0]
        base = [float(_hazard(self.models[0], 0.0))]
        for j in range(1, len(self.models)):
            before, after = self.models[j - 1], self.models[j]
            left = entered[-1] + durations[j - 1]
            spent.append(spent[-1] + float(_hazard(before, left)) - base[-1])
            entered.append(_carried(before, after, left))
            base.append(float(_hazard(after, entered[-1])))
        self._entered, self._spent, self._base = entered, spent, base

    def Hf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        phase = np.searchsorted(self.starts, x, side="right") - 1
        phase = np.clip(phase, 0, len(self.models) - 1)
        out = np.empty(x.shape)
        for j in np.unique(phase):
            at = phase == j
            ages = self._entered[j] + (x[at] - self.starts[j])
            out[at] = (
                self._spent[j] + _hazard(self.models[j], ages) - self._base[j]
            )
        return out

    def sf(self, x) -> np.ndarray:
        with np.errstate(over="ignore"):
            return np.exp(-self.Hf(x))

    def ff(self, x) -> np.ndarray:
        with np.errstate(over="ignore"):
            return -np.expm1(-self.Hf(x))

    def _draw(self, u: np.ndarray) -> np.ndarray:
        """The lifetimes of the uniforms ``u``: the times at which the
        cumulative hazard reaches ``-log(1 - u)``."""
        from repyability.rbd.regression_node import _inverse_hazard

        u = np.asarray(u, dtype=float)
        with np.errstate(divide="ignore"):
            target = -np.log1p(-u)
        return _inverse_hazard(self.Hf, target)

    def random(self, size: int) -> np.ndarray:
        return self._draw(np.random.uniform(size=size))

    def _row_sampler(self) -> RowSampler:
        return RowSampler(
            1, lambda u: self._draw(np.ascontiguousarray(u[:, 0]))
        )

    def __repr__(self) -> str:
        return f"PhasedLife({self.models!r})"


class Life(NamedTuple):
    """A block's life through the mission: one model on its running clock
    (``per_phase`` None: its node's, a phase's, or a :class:`PhasedLife`),
    or one per phase (a block on standby whose model changes), each on the
    block's age as that phase's model reads it."""

    model: Any
    per_phase: Optional[List[Any]] = None

    def at(self, phase: int):
        """The model of the block's life in ``phase``."""
        return self.model if self.per_phase is None else self.per_phase[phase]

    def models(self) -> List[Any]:
        """The models the block's life is made of, one per phase where they
        change."""
        if self.per_phase is not None:
            return list(self.per_phase)
        if isinstance(self.model, PhasedLife):
            return self.model.models
        return [self.model]

    def hazard(self, phase: int, x: np.ndarray) -> np.ndarray:
        return _hazard(self.at(phase), x)

    def inverse(self, phase: int, target: np.ndarray) -> np.ndarray:
        return _inverse(self.at(phase), target)

    def carry(
        self, phase: int, age: np.ndarray, end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The block's age and the age at which it fails, in phase
        ``phase - 1``'s model, carried to phase ``phase``'s: rescaled by
        the ratio of the two models' time scales where they are one
        regression model whose covariates rescale time, and otherwise the
        same age, with the hazard left to spend carried over."""
        before, after = self.at(phase - 1), self.at(phase)
        if before is after:
            return age, end
        ratio = _time_ratio(before, after)
        if ratio is not None:
            return age * ratio, end * ratio
        left = _hazard(before, end) - _hazard(before, age)
        return age, np.where(
            np.isfinite(end),
            _inverse(after, _hazard(after, age) + left),
            end,
        )


def phase_model(node: Hashable, model: Any, phase: MissionPhase) -> Any:
    """The life model of block ``node`` (its node's model ``model``) in
    ``phase``: the phase's model, checked against the node
    (``_overrides.checked_override``), or the node's; at the phase's
    covariates, if given (a ``RegressionNode`` at them).

    Raises
    ------
    ValueError
        If the phase's model is not one the node could have been given,
        or the block is given covariates and its model is not a
        ``RegressionNode`` with fixed covariates, or they are not as many
        as it was fitted with.
    """
    from repyability.rbd._overrides import checked_override
    from repyability.rbd.regression_node import RegressionNode

    how = behaviour(phase.setting(node))
    life = model
    if how.model is not None:
        life = checked_override(
            node,
            model,
            how.model,
            label=f"The model of block {node!r} in phase {phase.name!r}",
        )
    if how.covariates is None:
        return life
    if not isinstance(life, RegressionNode) or life.covariates is None:
        raise ValueError(
            f"Block {node!r} is given covariates in phase {phase.name!r}, "
            "but its model is not a RegressionNode with fixed covariates, "
            "so there are no covariates to set: give the block a "
            "RegressionNode(model, covariates=...) in the diagram."
        )
    if len(how.covariates) != len(life.covariates):
        raise ValueError(
            f"Block {node!r}: its covariates in phase {phase.name!r} are "
            f"{len(how.covariates)} numbers, but its model takes "
            f"{len(life.covariates)}."
        )
    if how.covariates == tuple(float(v) for v in life.covariates):
        return life
    return RegressionNode(life.model, covariates=list(how.covariates))


def life_of(
    node: Hashable,
    model: Any,
    phases: List[MissionPhase],
    starts: List[float],
) -> Life:
    """The life of block ``node`` (its node's model ``model``) through the
    ``phases`` that start at ``starts``: each phase's model
    (:func:`phase_model`), one object for the phases whose models are the
    same, so that nothing is carried between them.

    Raises
    ------
    ValueError
        As :func:`phase_model`; or if the block's models change between
        phases and its node's model is a probability of failing, or one
        regression model of a family that carries no age is set at other
        covariates.
    """
    from repyability.rbd.non_repairable_rbd import NonRepairableRBD

    models: List[Any] = []
    for phase in phases:
        given = phase_model(node, model, phase)
        same = [m for m in models if NonRepairableRBD._same_model(m, given)]
        models.append(same[0] if same else given)
    if all(m is models[0] for m in models):
        return Life(models[0])
    if is_fixed_probability(model):
        raise ValueError(
            f"The model of block {node!r} is a probability of failing, which "
            "has no age to carry from phase to phase: give the block a "
            "distribution of its life to set its model by phase."
        )
    for j in range(1, len(models)):
        try:
            _carried(models[j - 1], models[j], 0.0)
        except ValueError as error:
            raise ValueError(f"Block {node!r}: {error}") from None
    if not any(behaviour(p.setting(node)).standby for p in phases):
        durations = [phase.duration for phase in phases]
        return Life(PhasedLife(models, starts, durations))
    return Life(models[0], models)
