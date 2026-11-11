"""A mission drawn as one diagram, each phase setting its blocks' states and
covariates (see ``PhasedMission.from_diagram``).

A phase is a :class:`MissionPhase`: a name, a duration, and for some
blocks a :class:`BlockSetting` (a state, and the covariates of a
regression block in the phase), with the k-out-of-n votes it needs. How a
block behaves in a phase is read through one function, :func:`behaviour`,
which takes its setting: a new field of ``BlockSetting`` is one more
branch there.

Every block has its own running clock (its age) and a budget of
cumulative hazard, an ``Exp(1)`` draw. While it runs under covariates
``x`` from age ``a`` to ``a + d`` it spends ``H_x(a, a + d)``; on standby,
idle, it neither ages nor spends; guaranteed, it ages but spends nothing.
It fails when it has spent its budget. A block's life on its running clock
(``_Life``) is its node's model, or for a regression block given
covariates, the node's regression model at each phase's covariates: one
model along a step schedule over its running clock where that clock is the
mission's (it is never on standby), or one model per phase, the age carried
from one to the next (``_Life.carry``), where it is not.
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

from repyability.utils.checks import is_number, one_of, real_array

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

    Raises
    ------
    ValueError
        If ``state`` is not one of the four, or ``covariates`` is not a
        non-empty vector of finite numbers.

    Examples
    --------
    >>> from repyability import BlockSetting
    >>> BlockSetting(state="standby")
    BlockSetting(state='standby', covariates=None)
    >>> BlockSetting(covariates=[1.2, 40]).covariates
    (1.2, 40.0)
    """

    state: str = WORKING
    covariates: Optional[Tuple[float, ...]] = None

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
    return {"state": setting.state, "covariates": covariates}


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
        """The setting of block ``node`` in this phase (working, at its
        node's covariates, when not named).

        Parameters
        ----------
        node : Hashable
            The block's name in the diagram.
        """
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


def behaviour(setting: BlockSetting) -> Behaviour:
    """How a block with ``setting`` behaves in its phase: the one place a
    setting's fields are read."""
    return Behaviour(
        in_structure=setting.state != NOT_NEEDED,
        standby=setting.state == STANDBY,
        spends=setting.state != GUARANTEED,
        covariates=setting.covariates,
    )


# ----------------------------------------------------------------------
# A block's life on its running clock
# ----------------------------------------------------------------------


def _hazard(model, x: np.ndarray) -> np.ndarray:
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


class Life(NamedTuple):
    """A block's life through the mission: one model on its running clock
    (``per_phase`` None), or one per phase (a regression block on standby
    with covariates that change), each on the block's age as that phase's
    model reads it."""

    model: Any
    per_phase: Optional[List[Any]] = None
    #: Whether the covariates rescale time (accelerated failure time or
    #: life), so the age carried from one phase's model to the next is
    #: rescaled with them; else it is the running age.
    scales_time: bool = False

    def at(self, phase: int):
        """The model of the block's life in ``phase``."""
        return self.model if self.per_phase is None else self.per_phase[phase]

    def hazard(self, phase: int, x: np.ndarray) -> np.ndarray:
        return _hazard(self.at(phase), x)

    def inverse(self, phase: int, target: np.ndarray) -> np.ndarray:
        return _inverse(self.at(phase), target)

    def carry(
        self, phase: int, age: np.ndarray, end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The block's age and the age at which it fails, in phase
        ``phase - 1``'s model, carried to phase ``phase``'s: rescaled by
        the ratio of the two models' time scales where the covariates
        rescale time, and otherwise the same age, with the hazard left to
        spend carried over."""
        before, after = self.at(phase - 1), self.at(phase)
        if before is after:
            return age, end
        if self.scales_time:
            # Each model's quantile is the baseline's over its rate, so
            # their ratio is the ratio of the rates, at any probability.
            ratio = float(
                after._quantiles(np.array([0.5]))[0]
                / before._quantiles(np.array([0.5]))[0]
            )
            return age * ratio, end * ratio
        left = _hazard(before, end) - _hazard(before, age)
        return age, np.where(
            np.isfinite(end),
            _inverse(after, _hazard(after, age) + left),
            end,
        )


def _schedule(starts: List[float], rows: List[Tuple[float, ...]]):
    """A surpyval step schedule: ``rows[i]`` from ``starts[i]`` on."""
    from surpyval.univariate.regression import StepSchedule

    return StepSchedule.from_changepoints(starts, [list(r) for r in rows])


def life_of(
    node: Hashable,
    model: Any,
    phases: List[MissionPhase],
    starts: List[float],
) -> Life:
    """The life of block ``node`` (its node's model ``model``) through the
    ``phases`` that start at ``starts``.

    Raises
    ------
    ValueError
        If the block is given covariates and its model is not a
        ``RegressionNode`` with fixed covariates, or they are not as many
        as it was fitted with.
    """
    from repyability.rbd.regression_node import RegressionNode

    given = [
        (phase.name, behaviour(phase.setting(node)).covariates)
        for phase in phases
    ]
    named = [name for name, covariates in given if covariates is not None]
    if not named:
        return Life(model)
    if not isinstance(model, RegressionNode) or model.covariates is None:
        raise ValueError(
            f"Block {node!r} is given covariates in phase {named[0]!r}, but "
            "its model is not a RegressionNode with fixed covariates, so "
            "there are no covariates to set: give the block a "
            "RegressionNode(model, covariates=...) in the diagram."
        )
    default = tuple(float(v) for v in model.covariates)
    rows = [default if c is None else c for _, c in given]
    for (name, _), row in zip(given, rows):
        if len(row) != len(default):
            raise ValueError(
                f"Block {node!r}: its covariates in phase {name!r} are "
                f"{len(row)} numbers, but its model takes {len(default)}."
            )
    if all(row == default for row in rows):
        return Life(model)
    if all(row == rows[0] for row in rows):
        return Life(RegressionNode(model.model, covariates=list(rows[0])))
    standby = any(behaviour(p.setting(node)).standby for p in phases)
    if not standby:
        # Its running clock is the mission's: one model along the step
        # schedule of the phases' covariates (the phases of no duration
        # have none to apply).
        kept = [
            (start, row)
            for start, row, phase in zip(starts, rows, phases)
            if phase.duration > 0.0
        ] or [(0.0, rows[0])]
        schedule = _schedule([s for s, _ in kept], [r for _, r in kept])
        return Life(RegressionNode(model.model, schedule=schedule))
    from surpyval.univariate.regression._kinds import (
        ACCELERATED_FAILURE_TIME,
        ACCELERATED_LIFE,
        ADDITIVE_HAZARD,
        PROPORTIONAL_HAZARD,
        PROPORTIONAL_ODDS,
    )

    kind = getattr(model.model, "kind", None)
    scales = kind in (ACCELERATED_FAILURE_TIME, ACCELERATED_LIFE)
    if not scales and kind not in (
        PROPORTIONAL_HAZARD,
        ADDITIVE_HAZARD,
        PROPORTIONAL_ODDS,
    ):
        raise ValueError(
            f"Block {node!r} is on standby in some phase and its covariates "
            "change between phases, which needs a proportional-hazards, "
            "additive-hazards, proportional-odds, accelerated-failure-time "
            f"or accelerated-life model; its model is {kind!r}."
        )
    made: Dict[Tuple[float, ...], Any] = {}
    for row in rows:
        if row not in made:
            made[row] = RegressionNode(model.model, covariates=list(row))
    return Life(model, [made[row] for row in rows], scales)
