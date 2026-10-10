"""A ``RepairableRBD``'s Python event loop: the components' random
streams and their stand-ins, a run's context, the queue of events and
what follows each (failures, repairs, preventive and opportunistic
maintenance, inspections, repair crews, standby groups, common causes),
one simulation (``_replicate``) and a run of them (``_run``, holding
``SIMULATIONS``), in this process, in workers or in shards. The compiled
loop (``_kernel``) must agree with this one to the last bit.
"""

import heapq
import math
import pickle
from copy import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np

from repyability.non_repairable import NonRepairable
from repyability.rbd import (
    _ccf_groups,
    _crews,
    _curves,
)
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _runs, _streams, _timeline_runs
from repyability.rbd._common import (
    _safe_mean,
)
from repyability.rbd._curves import (
    _draws_start,
)
from repyability.rbd._degradation import (
    failure_from,
    level_after,
)
from repyability.rbd._events import (
    Event,
    _Cause,
    _CauseDraws,
    _Crews,
    _EventQueue,
    _Fixed,
    _Imperfect,
    _Inspection,
    _Preventive,
    _Standby,
    _StandbyDraws,
    _StandbyGroup,
    _Unit,
)
from repyability.rbd._model_utils import (
    always_works,
    distribution_name,
    is_mixture,
    lfp_p,
)
from repyability.rbd._runs import (  # noqa: E402,F401
    _UNSTREAMED,
    failure_criticality_index_per_component_failures,
    failure_criticality_index_per_system_failures,
    restoration_criticality_index_by_component,
    restoration_criticality_index_by_system,
)
from repyability.rbd._sampling import MixtureLife, stream_sampler
from repyability.rbd._tally import (
    _CATEGORIES,
    _REPAIR,
    _CapacityRecorder,
    _Replication,
    _shard_bytes,
    _Tally,
)
from repyability.rbd.helper_classes import PerfectReliability
from repyability.rbd.node_state import NodeState
from repyability.utils.checks import (
    simulation_window,
)
from repyability.utils.wrappers import SIMULATIONS

if TYPE_CHECKING:
    from repyability.rbd.repairable_rbd import RepairableRBD


class _StreamedRBD:
    """Stands in for a nested :class:`RepairableRBD` component during
    ``availability()``: runs the nested RBD's own simulation, with its
    components drawing from their own streams through their stand-ins (see
    ``RepairableRBD._streamed_components``)."""

    def __init__(self, rbd: "RepairableRBD", sources: dict):
        self._rbd = rbd
        self._sources = sources

    def initialize_event_queue(self, t_simulation, state=None):
        """Start the nested RBD's simulation, its components from
        ``state`` (checked; see ``RepairableRBD._simulation_states``)."""
        _start_queue(
            self._rbd,
            t_simulation,
            set(),
            set(),
            "p",
            self._sources,
            state or {},
        )

    def next_event(self):
        return self._rbd.next_event(sources=self._sources)

    @property
    def system_state(self) -> bool:
        return self._rbd.system_state

    @property
    def last_change_planned(self) -> bool:
        return self._rbd.last_change_planned


def _planned(source, status: bool) -> bool:
    """Whether a nested RBD's change to ``status``, just drawn from
    ``source``, starts a planned outage."""
    return not status and getattr(source, "last_change_planned", False)


class _StreamedComponent:
    """Stands in for a :class:`NonRepairable` component during
    ``availability()``: alternates failure and repair draws exactly as
    ``NonRepairable.next_event`` does, but takes them from the component's
    own streams (see ``_streams``). A maintained or inspected component
    draws its maintenance or test times from a stream of its own too, or,
    when they cannot be streamed, from ``model.random``."""

    __slots__ = (
        "_failure",
        "_repair",
        "_fails_next",
        "_duration",
        "_model",
        "_start",
        "_test",
        "_level",
    )

    def __init__(
        self,
        failure,
        repair,
        duration=None,
        model=None,
        start=None,
        test=None,
        level=None,
    ):
        self._failure = failure
        self._repair = repair
        self._fails_next = True
        self._duration = duration
        self._model = model
        self._start = start
        self._test = test
        self._level = level

    def reset(self):
        self._fails_next = True

    def maintenance_time(self) -> float:
        if self._duration is not None:
            return self._duration.draw()
        return self._model.random(1).item()

    def level_uniform(self) -> float:
        """A uniform a unit replaced on condition by its measured level
        (#271) draws its level at an inspection, or its failure from it,
        from: from its ``LEVEL`` stream, or numpy's global RNG."""
        if self._level is not None:
            return self._level.draw()
        return float(np.random.random())

    def start_uniform(self) -> float:
        """The uniform a start from a state draws what is left of the
        unit's life, repair or maintenance from (see
        ``RepairableRBD._started``): from its ``START`` stream, or numpy's
        global RNG."""
        if self._start is not None:
            return self._start.draw()
        return float(np.random.random())

    def test_uniform(self) -> float:
        """The uniform that decides whether a test that can miss a failure
        finds it: from its ``TEST`` stream, or numpy's global RNG."""
        if self._test is not None:
            return self._test.draw()
        return float(np.random.random())

    def life_drawn(self, x: float = 0.0) -> None:
        """The unit's life has been drawn (given its age, at a start from a
        state; ``x`` its operating time from its last repair to the failure
        drawn): its next draw is its repair."""
        self._fails_next = False

    def next_event(self):
        if self._fails_next:
            self._fails_next = False
            return self._failure.draw(), False
        self._fails_next = True
        return self._repair.draw(), True


class _ModelDraws:
    """A model's draws, one at a time, from numpy's global RNG: as a
    stream gives them, and as ``NonRepairable.next_event`` draws them (one
    ``random(1)`` call each)."""

    __slots__ = ("_model",)

    def __init__(self, model):
        self._model = model

    def draw(self) -> float:
        return self._model.random(1).item()


def _exponential_sampler(rate: float):
    """The quantile function of an exponential time at ``rate``: ``inf``
    for a rate of 0 (a common-cause group member whose every cause is
    shared never fails on its own, see ``RepairableRBD._own_rate``)."""
    if rate <= 0.0:
        return lambda u: np.full(np.shape(u), np.inf)
    return lambda u: -np.log1p(-np.asarray(u, dtype=float)) / rate


class _ExponentialDraws:
    """Exponential times at ``rate``, one at a time, from numpy's global RNG
    (``inf`` for a rate of 0)."""

    __slots__ = ("_rate",)

    def __init__(self, rate: float):
        self._rate = rate

    def draw(self) -> float:
        if self._rate <= 0.0:
            return math.inf
        return float(np.random.exponential(1.0 / self._rate))


def _level_uniform(source) -> float:
    """A uniform for a unit replaced on condition by its measured level
    (#271): from its stand-in's ``LEVEL`` stream, or numpy's global RNG
    for a component that draws its own way."""
    if isinstance(source, _StreamedComponent):
        return source.level_uniform()
    return float(np.random.random())


def _aged_life(model, age: float, u: float) -> float:
    """The life left to a unit of lifetime ``model`` at virtual age
    ``age``, from the uniform ``u``: the ``x`` with ``H(age + x) = H(age) -
    log(u)`` (``H`` the cumulative hazard), as surpyval's virtual-age
    renewal models draw it (``conditional_gaps``). A plain Exponential or
    Weibull is worked in closed form, any other model by surpyval (a
    mixture with its quantile worked out, ``MixtureLife``)."""
    if is_mixture(model):
        model = MixtureLife.of(model)
    dist = getattr(model, "dist", None)
    name = getattr(dist, "name", None)
    if (
        name in ("Exponential", "Weibull")
        and lfp_p(model) == 1
        and getattr(model, "f0", None) == 0
        and not getattr(model, "gamma", 0)
    ):
        exposure = 0.0 - math.log(u) if u > 0.0 else math.inf
        if name == "Exponential":
            return exposure / float(model.params[0])
        alpha, beta = (float(p) for p in model.params)
        hazard = (age / alpha) ** beta if age > 0.0 else 0.0
        if hazard == 0.0:
            # As new (or too young for its hazard to register).
            return max(alpha * exposure ** (1.0 / beta) - age, 0.0)
        # alpha * (H(age) + exposure) ** (1 / beta) - age, without the
        # cancellation.
        return age * math.expm1(math.log1p(exposure / hazard) / beta)
    from surpyval.recurrent.renewal.renewal_model import conditional_gaps

    return float(conditional_gaps(model, np.array([age]), np.array([u]))[0])


class _ImperfectComponent(_StreamedComponent):
    """Stands in for a component repaired imperfectly (a spec's
    ``"repair"``; see ``RepairableRBD``) during a simulation, keeping the
    unit's virtual age (Kijima): a repair after an operating time ``x``
    takes it from ``v`` to ``v + q * x`` (Kijima I) or ``q * (v + x)``
    (Kijima II), and its next life is drawn given it (``_aged_life``),
    from a stream of its own. A unit as new draws from its life stream, as
    any component does. At the ``replace_after``-th failure since it was
    renewed it is replaced instead, as new, and a scheduled replacement
    renews it too (``reset``). ``operated`` is its operating time since it
    was renewed, which age replacement counts. Without streams, it draws
    from its models and numpy's global RNG."""

    __slots__ = (
        "_unit",
        "_aged",
        "_q",
        "_second",
        "_limit",
        "_x",
        "age",
        "operated",
        "count",
    )

    def __init__(
        self,
        unit,
        imperfect: "_Imperfect",
        failure=None,
        repair=None,
        aged=None,
        duration=None,
        model=None,
        test=None,
    ):
        super().__init__(failure, repair, duration, model, test=test)
        self._unit = unit
        self._aged = aged
        self._q = imperfect.q
        self._second = imperfect.kijima == "kijima2"
        self._limit = imperfect.replace_after
        self._x = 0.0
        self.age = self.operated = 0.0
        self.count = 0

    def reset(self):
        self._fails_next = True
        self.age = self.operated = 0.0
        self.count = 0

    def life_drawn(self, x: float = 0.0) -> None:
        # Its repair after the failure drawn takes its virtual age on by
        # the operating time since its last repair (#269).
        super().life_drawn(x)
        self._x = x

    @property
    def replacing(self) -> bool:
        """Whether the unit's failure (the last drawn) is the
        ``replace_after``-th since it was renewed, which replaces it."""
        return self._limit is not None and self.count + 1 >= self._limit

    def next_event(self):
        if self._fails_next:
            self._fails_next = False
            if self.age > 0.0:
                u = (
                    self._aged.draw()
                    if self._aged is not None
                    else float(np.random.random())
                )
                x = _aged_life(self._unit.reliability, self.age, u)
            elif self._failure is not None:
                x = self._failure.draw()
            else:
                x = self._unit.reliability.random(1).item()
            self._x = x
            return x, False
        self._fails_next = True
        if self.replacing:
            self.age = self.operated = 0.0
            self.count = 0
        else:
            x = self._x
            self.count += 1
            self.operated += x
            self.age = (
                self._q * (self.age + x)
                if self._second
                else self.age + self._q * x
            )
        if self._repair is not None:
            return self._repair.draw(), True
        return self._unit.time_to_replace.random(1).item(), True


def _stand_in(
    component,
    run,
    made: dict,
    path: tuple,
    duration=None,
    imperfect: Optional["_Imperfect"] = None,
):
    """What a component draws its events from during ``availability()``
    (see ``RepairableRBD._streamed_components``): a stand-in taking its
    draws from its own streams, or the component itself when they cannot be
    streamed (a model other than a surpyval parametric one, or a subclass,
    which may draw its events its own way), which then draws from numpy's
    global RNG. ``path`` is the node's place, nested RBDs included, which
    names its streams; ``duration`` its maintenance or test time model, and
    ``imperfect`` its imperfect repair (see ``_ImperfectComponent``)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if type(component) is RepairableRBD:
        return _StreamedRBD(
            component,
            _streamed_components(component, run, made, path),
        )
    if type(component) is not NonRepairable:
        return component
    failure = run.stream(path, _streams.FAILURE)
    repair = run.stream(path, _streams.REPAIR)
    if imperfect is not None:
        if failure is None or repair is None:
            return _ImperfectComponent(component, imperfect, model=duration)
        return _ImperfectComponent(
            component,
            imperfect,
            failure,
            repair,
            run.stream(path, _streams.AGED),
            run.stream(path, _streams.DURATION),
            duration,
            run.stream(path, _streams.TEST),
        )
    if failure is None or repair is None:
        return component
    return _StreamedComponent(
        failure,
        repair,
        run.stream(path, _streams.DURATION),
        duration,
        run.stream(path, _streams.START),
        run.stream(path, _streams.TEST),
        run.stream(path, _streams.LEVEL),
    )


def _uniforms(u: np.ndarray) -> np.ndarray:
    """The draws of a stream of uniforms: the uniforms themselves."""
    return u


def _standby_draws(component, run, path: tuple, arrangement: "_Standby"):
    """A standby group's draws in a run (see ``_StandbyDraws``): from
    streams if every unit's lives and repairs have one."""
    units = range(arrangement.units)
    lives = [run.stream(path + (u,), _streams.FAILURE) for u in units]
    repairs = [run.stream(path + (u,), _streams.REPAIR) for u in units]
    if any(stream is None for stream in lives + repairs):
        return _StandbyDraws(component)
    return _StandbyDraws(
        component, lives, repairs, run.stream(path, _streams.SWITCH)
    )


#: Simulations per task of a parallel run of the Python engine (see
#: ``RepairableRBD``'s ``availability``).
PARALLEL_BLOCK = 250


#: A parallel run's worker process's run (see ``_start_worker``).
_WORKER: Dict[str, Any] = {}


def _start_worker(run: bytes) -> None:
    """Start a worker process of a parallel run (see ``_PythonRunner``):
    unpickle the run, the system and what its simulations share, once, for
    every block the worker runs."""
    rbd, args, curve_points, replacements, histories = pickle.loads(run)
    _WORKER["run"] = (
        rbd,
        _context(rbd, *args),
        curve_points,
        replacements,
        histories,
    )


def _simulate_block(span: Tuple[int, int]) -> "_Tally":
    """Simulations ``start`` to ``stop - 1`` of a parallel run, in a worker
    process (see ``_start_worker``): their totals, compact."""
    rbd, context, curve_points, replacements, histories = _WORKER["run"]
    tally = _Tally(
        list(rbd.components),
        rbd.costs,
        context.t_simulation,
        curve_points,
    )
    if replacements:
        tally.replacements = []
    if histories:
        tally.histories = _timeline_runs.Records(len(rbd.components))
    for replication in range(*span):
        tally.add(_replicate(rbd, context, replication))
    return tally.compact()


class _Context(NamedTuple):
    """What a run's simulations share (see ``RepairableRBD._context``)."""

    t_simulation: float
    working: set
    broken: set
    method: str
    capacity: Optional[_CapacityRecorder]
    run: _streams.Run
    sources: dict
    charges: tuple
    downtime_cost_rates: dict
    has_costs: bool
    position: dict
    plain: set
    works: Callable
    #: The components' states at the start (see ``_simulation_states``),
    #: and whether the system is up at the start with every component up
    #: but those held broken.
    states: dict
    initial_up: bool
    #: Whether each simulation records its components' changes (see
    #: ``simulate_timelines``).
    history: bool = False


class _PythonRunner:
    """Runs a run's simulations in Python: one after another in this
    process, or, with ``jobs`` above 1, in blocks of ``PARALLEL_BLOCK`` in
    that many processes. Each worker process is given the run once, when it
    starts (see ``_start_worker``), and sends back each block's totals,
    compact (#152), which are merged into the tally in order: the same
    totals, kept exactly, as adding the simulations one by one. ``args``
    are ``RepairableRBD._context``'s."""

    def __init__(self, rbd, tally, progress, args: tuple, jobs):
        self._rbd = rbd
        self._tally = tally
        self._progress = progress
        self._context: Optional[_Context] = None
        self._executor: Any = None
        if jobs is not None and jobs > 1:
            run = montecarlo.dumps(
                (
                    rbd,
                    args,
                    tally.curve_points,
                    tally.replacements is not None,
                    tally.histories is not None,
                )
            )
            self._executor = montecarlo.process_pool(
                jobs, initializer=_start_worker, initargs=(run,)
            )
        else:
            self._context = _context(rbd, *args)

    def __call__(self, start: int, stop: int) -> None:
        """Simulations ``start`` to ``stop - 1``."""
        if self._executor is None:
            assert self._context is not None
            for replication in range(start, stop):
                self._tally.add(
                    _replicate(self._rbd, self._context, replication)
                )
                self._progress.update()
            return
        spans = [
            (first, min(first + PARALLEL_BLOCK, stop))
            for first in range(start, stop, PARALLEL_BLOCK)
        ]
        for block in self._executor.map(_simulate_block, spans):
            self._tally.merge(block)
            self._progress.update(block.n)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()


class _ShardRunner:
    """Runs a run's simulations as shards (see ``repyability.rbd.shards``):
    each range of simulations asked for is cut into shards at the multiples
    of ``step``, which ``shard_map(run_shard, shards)`` runs wherever it
    sends them, and the partials it gives back, checked to be those
    shards', are merged into the tally in order. ``template`` is what
    every shard of the run holds (see ``RepairableRBD._shard_template``)."""

    def __init__(self, rbd, tally, progress, shard_map, template, step):
        from repyability.rbd.chunks import _run_key

        self._tally = tally
        self._progress = progress
        self._map = shard_map
        self._template = template
        self._step = step
        self._key = _run_key(
            {**template, "fingerprint": _runs._fingerprint(rbd)}
        )

    def __call__(self, start: int, stop: int) -> None:
        """Simulations ``start`` to ``stop - 1``."""
        from repyability.rbd.chunks import SimulationChunk, _run_key
        from repyability.rbd.shards import run_shard

        if stop <= start:
            return
        step = self._step
        cuts = [start, *range((start // step + 1) * step, stop, step), stop]
        ranges = list(zip(cuts, cuts[1:]))
        shards = [_shard_bytes(self._template, a, b) for a, b in ranges]
        chunks = []
        for saved in self._map(run_shard, shards):
            chunk = SimulationChunk._load(saved)
            chunks.append(chunk)
            self._progress.update(chunk.n_simulations)
        chunks.sort(key=lambda chunk: chunk.ranges[0])
        if [chunk.ranges for chunk in chunks] != [[r] for r in ranges] or any(
            _run_key(chunk.settings) != self._key for chunk in chunks
        ):
            raise ValueError(
                "shard_map gave back other partials than those of the "
                "shards it was given: it must give back run_shard's result "
                "for each shard, as map(run_shard, shards) does."
            )
        for chunk in chunks:
            self._tally.merge(chunk._tally)

    def close(self) -> None:
        pass


#: The per-action costs: charged at each preventive action or inspection.
_ACTION_COST_KEYS = ("preventive_cost", "inspection_cost")


def _validate_crews(crews) -> Optional[int]:
    """``repair_crews``, validated: a whole number of crews, at least 1, or
    None for as many as are needed."""
    if crews is None:
        return None
    if isinstance(crews, bool) or int(crews) != crews or crews < 1:
        raise ValueError(
            f"repair_crews must be a whole number, 1 or more, or None; got "
            f"{crews!r}."
        )
    return int(crews)


def _validate_priority(node, priority) -> float:
    """A component's ``"priority"`` for a crew, validated: a finite
    number."""
    if isinstance(priority, bool) or not isinstance(
        priority, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Component {node!r}: priority must be a number, got "
            f"{priority!r}."
        )
    if not np.isfinite(priority):
        raise ValueError(
            f"Component {node!r}: priority must be finite, got {priority!r}."
        )
    return float(priority)


def _failures_between(model, ages: np.ndarray) -> np.ndarray:
    """The probability that a unit of ``model`` that has survived to each
    of ``ages`` but the last fails before the next, ``1 - sf(next) /
    sf(age)``: from its cumulative hazard where the model gives one, which
    keeps a small probability precise."""
    ages = np.asarray(ages, dtype=float)
    out = np.full(len(ages) - 1, np.nan)
    Hf = getattr(model, "Hf", None)
    if Hf is not None:
        with np.errstate(all="ignore"):
            hazard = np.asarray(Hf(ages), dtype=float)
            out = -np.expm1(-np.maximum(np.diff(hazard), 0.0))
    missing = np.flatnonzero(np.isnan(out))
    if missing.size:
        with np.errstate(all="ignore"):
            survival = np.asarray(model.sf(ages), dtype=float)
            now, later = survival[missing], survival[missing + 1]
            fails = np.where(now > 0.0, 1.0 - later / now, 1.0)
        out[missing] = np.clip(np.nan_to_num(fails, nan=1.0), 0.0, 1.0)
    return out


def _on_duty(components: dict) -> dict:
    """The components, each spec with a ``"duty"`` given its life on the
    calendar (see ``_duty.on_calendar``): one object for each model and
    duty, so that components sharing a model still share it."""
    from repyability.rbd._duty import duty_fraction, on_calendar

    moved: dict = {}
    out = {}
    for name, component in components.items():
        if isinstance(component, dict) and component.get("duty") is not None:
            d = duty_fraction(name, component["duty"])
            life = component["reliability"]
            key = (id(life), d)
            if key not in moved:
                moved[key] = on_calendar(name, life, d)
            component = {**component, "reliability": moved[key]}
        out[name] = component
    return out


def _is_junction(name, component) -> bool:
    """Whether ``component`` makes node ``name`` a junction, which never
    fails: ``PerfectReliability`` itself, or a spec whose life it is (#175,
    #182) or is a probability of failing of 0. Such a spec may give a
    repair model, never used, but nothing that only a part that fails or
    is maintained has."""
    if component is PerfectReliability:
        return True
    if not (
        isinstance(component, dict)
        and always_works(component.get("reliability"))
    ):
        return False
    life = (
        "PerfectReliability"
        if component["reliability"] is PerfectReliability
        else "a probability of failing of 0"
    )
    other = sorted(
        str(key)
        for key, value in component.items()
        if key not in ("reliability", "repairability") and value is not None
    )
    if other:
        raise ValueError(
            f"Component {name!r} never fails (its reliability is {life}), "
            f"so it takes no {', '.join(other)}: give it as "
            "PerfectReliability alone, a junction that always works."
        )
    return True


def _test_finds(source, coverage: float) -> bool:
    """Whether a test that finds a failure with probability ``coverage``
    finds one: from the component's detection stream, or numpy's global
    RNG."""
    if isinstance(source, _StreamedComponent):
        return source.test_uniform() < coverage
    return float(np.random.random()) < coverage


def _at_zero(model) -> bool:
    """Whether ``model`` is a time of exactly 0 (an instant repair is)."""
    return distribution_name(model) == "ExactEventTime" and not np.any(
        np.ravel(model.params)
    )


def _never_settles(component) -> bool:
    """Whether a component fails at once and is repaired at once, so that
    the event loop would change its state without end."""
    return _at_zero(getattr(component, "reliability", None)) and _at_zero(
        getattr(component, "time_to_replace", None)
    )


def _stream_specs(
    rbd,
    t_simulation: float,
    prefix: tuple = (),
    specs: Optional[dict] = None,
    states: Optional[dict] = None,
) -> Tuple[dict, bool]:
    """The streams this RBD's simulations draw from (see ``_streams``),
    by name, nested RBDs' included; and whether every draw comes from
    one. A component whose draws cannot be streamed (a model other than
    a surpyval parametric one, or a subclass of the component classes)
    has none, nor has a maintenance or test time that cannot: they draw
    from numpy's global RNG. ``prefix`` is this RBD's place in the RBD
    simulated: only that RBD's own costs are drawn. A component started
    from a state in ``states`` (see ``_simulation_states``) that draws
    what is left of its life or repair has a stream for that too."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    specs = {} if specs is None else specs
    states = {} if states is None else states
    complete = True

    def add(path: tuple, kind: int, sampler, expected: float) -> None:
        rows = _streams.first_rows(expected)
        specs[(path, kind)] = _streams.Spec(
            path, kind, sampler, rows, _streams.block_width(rows)
        )

    for name, component in rbd.components.items():
        path = prefix + (name,)
        if type(component) is RepairableRBD:
            nested = _stream_specs(
                component, t_simulation, path, specs, states.get(name)
            )
            complete = complete and nested[1]
            continue
        failure = repair = None
        if type(component) is NonRepairable:
            # A common-cause group's member draws its own cause's life.
            own = _ccf_groups._own_rate(rbd, name)
            failure = (
                stream_sampler(component.reliability)
                if own is None
                else _exponential_sampler(own)
            )
            repair = stream_sampler(component.time_to_replace)
        if failure is None or repair is None:
            complete = False
            continue
        arrangement = rbd._standby.get(name)
        if arrangement is not None:
            # Each unit's lives and repairs, and the group's switches.
            lives, switches = _standby_draw_counts(rbd, name, t_simulation)
            for unit in range(arrangement.units):
                add(path + (unit,), _streams.FAILURE, failure, lives)
                add(path + (unit,), _streams.REPAIR, repair, lives)
            if 0.0 < arrangement.switching_probability < 1.0:
                add(path, _streams.SWITCH, _uniforms, switches)
            continue
        expected = _expected_draws(rbd, name, t_simulation)
        add(path, _streams.FAILURE, failure, expected["failure"])
        add(path, _streams.REPAIR, repair, expected["repair"])
        if _draws_start(states.get(name)):
            add(path, _streams.START, _uniforms, 1.0)
        if name in rbd._imperfect:
            # A life after each repair, given the unit's virtual age.
            add(path, _streams.AGED, _uniforms, expected["repair"])
        schedule = rbd._preventive.get(name)
        if schedule is not None and schedule.level is not None:
            # A level and a failure from it at each inspection (#271).
            add(path, _streams.LEVEL, _uniforms, 2.0 * expected["action"])
        inspection = rbd._inspection.get(name)
        if inspection is not None and inspection.partial:
            # Whether a test finds each failure: one for each failure
            # a test can miss.
            add(path, _streams.TEST, _uniforms, expected["failure"])
        duration = _duration_model(rbd, name)
        if duration is not None:
            sampler = stream_sampler(duration)
            if sampler is None:
                complete = False
            else:
                add(path, _streams.DURATION, sampler, expected["action"])
    for cause in _ccf_groups._shared_causes(rbd):
        # A shared common cause's strikes (#158), and at each the
        # uniform its failures' tests decide by.
        strikes = cause.rate * t_simulation + 1.0
        add(
            prefix + cause.struck,
            _streams.CAUSE,
            _exponential_sampler(cause.rate),
            strikes,
        )
        if cause.coins:
            add(
                prefix + cause.struck,
                _streams.CAUSE_TEST,
                _uniforms,
                strikes,
            )
    if not prefix:
        for node, node_costs in rbd.costs.items():
            expected = _expected_draws(rbd, node, t_simulation)
            for key, kind in _streams.COST_KINDS.items():
                cost = node_costs.get(key)
                if cost is None or isinstance(cost, float):
                    continue
                count = expected[
                    "action" if key in _ACTION_COST_KEYS else "repair"
                ]
                add((node,), kind, _streams.cost_sampler(cost), count)
    return specs, complete


def _duration_model(rbd, node):
    """The model of ``node``'s maintenance or test time, or None."""
    schedule = rbd._preventive.get(node) or rbd._inspection.get(node)
    return None if schedule is None else schedule.duration


def _standby_draw_counts(
    rbd, node, t_simulation: float
) -> Tuple[float, float]:
    """Roughly how many lives (and repairs) a simulation draws for each
    unit of a standby group, and how many switches: its operating units
    fail at their rate and its spares at the dormant one (see
    ``_expected_draws``)."""
    arrangement = rbd._standby[node]
    life = _safe_mean(rbd.components[node].reliability)
    if not life > 0.0:
        return float("nan"), float("nan")
    active = (
        arrangement.k
        + (arrangement.units - arrangement.k) * arrangement.dormancy_factor
    )
    failures = t_simulation * active / life
    return failures / arrangement.units + 1.0, failures + 1.0


def _expected_draws(rbd, node, t_simulation: float) -> Dict[str, float]:
    """Roughly how many times a simulation draws ``node``'s time to
    failure (``"failure"``: one per unit put into service), its repair
    time and per-failure costs (``"repair"``), and its maintenance or
    test time and their costs (``"action"``). NaN where it is not known.
    Only how the draws are computed depends on these (see
    ``_streams``), never what they are."""
    component = rbd.components[node]
    if node in rbd._standby:
        # Its units' failures, each a repair charged for.
        _, failures = _standby_draw_counts(rbd, node, t_simulation)
        return {"failure": failures, "repair": failures, "action": 0.0}
    life = repair = float("nan")
    if isinstance(component, NonRepairable):
        life = _safe_mean(component.reliability)
        repair = _safe_mean(component.time_to_replace)
    if life == np.inf:
        failures = 0.0
    elif life + repair > 0.0:
        failures = t_simulation / (life + repair)
    else:
        failures = float("nan")
    # A member renewed early at its group's stops (#108) has more lives
    # and actions than these: its streams then take more chunks, which
    # changes only how its draws are computed.
    lives, actions = failures + 1.0, 0.0
    schedule = rbd._preventive.get(node)
    inspection = rbd._inspection.get(node)
    if schedule is not None:
        actions = t_simulation / schedule.interval + 1.0
        if schedule.policy == "age":
            lives = t_simulation / np.fmin(life, schedule.interval) + 1.0
        else:
            lives = failures + actions
    elif inspection is not None:
        actions = t_simulation / inspection.interval + 1.0
    return {
        "failure": float(lives),
        "repair": failures + 1.0,
        "action": actions,
    }


def _stream_plan(
    rbd,
    t_simulation: float,
    entropy,
    antithetic: bool,
    states: Optional[dict] = None,
) -> Tuple[_streams.Plan, bool]:
    """A run's streams, and whether every draw comes from one; the
    components start from ``states``."""
    specs, complete = _stream_specs(rbd, t_simulation, states=states)
    return _streams.Plan(entropy, antithetic, specs), complete


def _streamed_components(
    rbd,
    run: _streams.Run,
    made: Optional[dict] = None,
    prefix: tuple = (),
) -> dict[Any, Any]:
    """What each component draws its events from in a run: a stand-in
    replaying its draws from its own streams, nested RBDs' components
    included, or the component itself when they cannot be streamed
    (see ``_stand_in``). The stand-ins are called at exactly the points
    the components would be.

    A component object used for several nodes has one simulation state,
    so it gets one stand-in (``made``, by object). Since each node has
    its own copy of a ``NonRepairable``, only a nested RBD object can be.
    """
    made = {} if made is None else made
    streamed: dict[Any, Any] = {}
    for name, component in rbd.components.items():
        if id(component) not in made:
            arrangement = rbd._standby.get(name)
            made[id(component)] = (
                _stand_in(
                    component,
                    run,
                    made,
                    prefix + (name,),
                    _duration_model(rbd, name),
                    rbd._imperfect.get(name),
                )
                if arrangement is None
                else _standby_draws(
                    component, run, prefix + (name,), arrangement
                )
            )
            own = _ccf_groups._own_rate(rbd, name)
            if own is not None and made[id(component)] is component:
                # Not streamed: its own cause's life all the same.
                made[id(component)] = _own_draws(rbd, name, own)
        streamed[name] = made[id(component)]
    for cause in _ccf_groups._shared_causes(rbd):
        path = prefix + cause.struck
        streamed[cause] = _CauseDraws(
            cause.rate,
            run.stream(path, _streams.CAUSE),
            run.stream(path, _streams.CAUSE_TEST),
        )
    return streamed


def _own_draws(rbd, node, own: float) -> "_StreamedComponent":
    """A common-cause group member's draws from numpy's global RNG: its
    own cause's life (at the rate ``own``), and its repair, maintenance
    and test times as it would draw them."""
    component = rbd.components[node]
    return _StreamedComponent(
        _ExponentialDraws(own),
        _ModelDraws(component.time_to_replace),
        model=_duration_model(rbd, node),
    )


def _keyed_charges(rbd, run: _streams.Run) -> Tuple[
    dict[Any, list[tuple[int, Any]]],
    dict[Any, Any],
    dict[Any, Any],
]:
    """For each costed node, its ``(category, charges)`` pairs: what is
    charged at each of its failures (the categories are positions in
    ``_CATEGORIES``); and, for each node with a preventive-maintenance
    or an inspection cost, what is charged at each preventive action or
    inspection. ``charges.draw()`` gives the next amount: a fixed cost,
    or the next draw from the cost's own stream (see ``_streams``)."""

    def charges(node, key: str):
        cost = rbd.costs[node][key]
        if isinstance(cost, float):
            return _Fixed(cost)
        return run.stream((node,), _streams.COST_KINDS[key])

    failures = {
        node: [
            (
                _CATEGORIES.index(key.removesuffix("_cost")),
                charges(node, key),
            )
            for key in rbd.PER_FAILURE_COST_KEYS
            if key in node_costs
        ]
        for node, node_costs in rbd.costs.items()
    }
    preventive = {
        node: charges(node, "preventive_cost")
        for node, node_costs in rbd.costs.items()
        if "preventive_cost" in node_costs
    }
    inspection = {
        node: charges(node, "inspection_cost")
        for node, node_costs in rbd.costs.items()
        if "inspection_cost" in node_costs
    }
    return failures, preventive, inspection


def _context(
    rbd,
    t_simulation: float,
    working_nodes,
    broken_nodes,
    method: str,
    capacity: Optional[_CapacityRecorder],
    entropy,
    antithetic: bool,
    states: Optional[dict] = None,
    history: bool = False,
) -> "_Context":
    """Everything a run's simulations share: the streams, what each
    component draws from, the charges, the structure function, and the
    components' ``states`` at the start (checked, as
    ``_simulation_states`` gives them; None for new). With
    ``history``, each simulation records its histories (see
    ``_replicate``)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    states = {} if states is None else states
    plan, complete = _stream_plan(
        rbd, t_simulation, entropy, antithetic, states
    )
    run = _streams.Run(plan, reseed=not complete)
    has_costs = rbd.has_costs
    nodes = list(rbd.components)
    return _Context(
        t_simulation=t_simulation,
        working=set(working_nodes),
        broken=set(broken_nodes),
        method=method,
        capacity=capacity,
        run=run,
        sources=_streamed_components(rbd, run),
        charges=_keyed_charges(rbd, run) if has_costs else ({}, {}, {}),
        downtime_cost_rates={
            node: c["downtime_cost"]
            for node, c in rbd.costs.items()
            if "downtime_cost" in c
        },
        has_costs=has_costs,
        position={node: i for i, node in enumerate(nodes)},
        # Components with no maintenance or inspection that are not
        # nested RBDs: their next event is simply their next draw.
        plain={
            node
            for node, component in rbd.components.items()
            if node not in rbd._preventive
            and node not in rbd._inspection
            and node not in rbd._standby
            and not isinstance(component, RepairableRBD)
        },
        works=rbd._decomposition().structure_function(method),
        states=states,
        initial_up=bool(
            rbd.is_system_working(
                {c: c not in broken_nodes for c in rbd.components}, method
            )
        ),
        history=history,
    )


def initialize_event_queue(
    rbd,
    t_simulation,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    method: str,
    sources: Optional[dict],
    state,
):
    """See ``RepairableRBD.initialize_event_queue``."""
    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    states = _curves._simulation_states(
        rbd, state, working_nodes | broken_nodes
    )
    _ccf_groups._require_free_members(rbd, working_nodes, broken_nodes)
    _ccf_groups._require_groups_simulated(rbd, states)
    _start_queue(
        rbd, t_simulation, working_nodes, broken_nodes, method, sources, states
    )


def _start_queue(
    rbd,
    t_simulation,
    working_nodes: set,
    broken_nodes: set,
    method: str,
    sources: Optional[dict],
    states: dict,
) -> None:
    """``initialize_event_queue``, with ``states`` checked (see
    ``_simulation_states``)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    # What the components draw their events from: themselves (an
    # imperfectly repaired one through a stand-in keeping its virtual
    # age, and one started from a state through one whose next draw can
    # be set; kept for next_event), or stand-ins drawing from their own
    # streams (see _streamed_components), which the caller passes to
    # next_event too.
    if sources is None:
        sources = rbd._step_sources = _own_sources(rbd, states)
    else:
        rbd.__dict__.pop("_step_sources", None)
    # The window's end, which the first events already look to (see
    # _replaced_at).
    rbd.t_simulation = t_simulation
    # Each calendar a state's phase shifts: by node, the time since its
    # last scheduled replacement or test (see _due).
    rbd._phases = {
        node: start.phase
        for node, start in states.items()
        if isinstance(start, NodeState)
        and (
            start.phase
            or (
                # On a calendar from 0, not at its offset.
                start.phase is not None
                and node in rbd._inspection
                and rbd._inspection[node].offset
            )
        )
    }

    # Keep record of component status', initially they're all working
    component_status: dict[Any, bool] = {
        component: True for component in rbd.components.keys()
    }

    for component in broken_nodes:
        component_status[component] = False

    # The queue supplies failure/repair events in chronological order
    event_queue = _EventQueue()
    # When each working node with hidden failures is due to fail (None
    # once it has failed).
    rbd._pending_failure = {}
    # When each working node replaced on condition is due to fail, and
    # to be replaced (see _replaced_at).
    rbd._in_service = {}
    # Each working node replaced on condition by its measured level
    # (#271): when its level was last known, and the level then.
    rbd._levels = {}
    # Opportunistic maintenance (#108): when each member that can be
    # renewed early was put into service as new, and its pending event;
    # the pending events early renewals have cancelled, and the early
    # renewals queued, by identity, and their members (see _stop).
    rbd._renewed_at = {}
    rbd._pending_event = {}
    rbd._cancelled = {}
    rbd._early = {}
    rbd._renewing = set()
    # Common-cause groups (#158): each member's next event, which a
    # shared cause that strikes it first supersedes, and the uniform a
    # strike has decided its tests by (see _strike).
    rbd._ccf_members = frozenset(
        m for group in rbd.ccf_groups for m in group.members
    )
    rbd._ccf_pending = {}
    rbd._ccf_coins = {}
    # The components down at the start in a repair or maintenance going
    # on, which holds a repair crew.
    in_hand: list = []

    # For each component add in the initial failure
    for component_id in rbd.components.keys():
        component = rbd.components[component_id]
        if component_id in working_nodes:
            continue
        elif component_id in broken_nodes:
            continue
        elif component_id in rbd._standby:
            continue  # started below, once the queue and crews exist
        source = sources[component_id]
        start = states.get(component_id)
        if isinstance(component, RepairableRBD):
            source.initialize_event_queue(t_simulation, state=start)
            component_status[component_id] = source.system_state
            t_event, event = source.next_event()
            first = Event(
                t_event, component_id, event, _planned(source, event)
            )
        elif start is not None and not start.new:
            up, first = _started(rbd, component_id, source, start)
            component_status[component_id] = up
            if not up and first.status:
                in_hand.append(component_id)
        elif component_id in rbd._preventive:
            # Put into service as new at 0: it fails, or is maintained.
            source.reset()
            first = _renewal(
                rbd, component_id, 0.0, source, rbd._preventive[component_id]
            )
        elif component_id in rbd._inspection:
            # Put into service as new at 0: it fails, or is inspected.
            source.reset()
            first = _inspected_renewal(
                rbd, component_id, 0.0, source, rbd._inspection[component_id]
            )
        else:
            source.reset()
            t_event, event = source.next_event()
            first = Event(t_event, component_id, event)

        # Only consider it if it occurs within the simulation window
        if first.time < t_simulation:
            event_queue.put(first)
        if component_id in rbd._ccf_members:
            rbd._ccf_pending[component_id] = first
    for cause in _ccf_groups._shared_causes(rbd):
        first = Event(sources[cause].gap(), cause, False)
        if first.time < t_simulation:
            event_queue.put(first)
    rbd._event_queue = event_queue
    # The repair crews, when there are fewer than the components that
    # may need one (with enough, no job ever waits).
    crews = rbd.repair_crews
    rbd._crews = (
        _Crews(crews, _crews._crew_served(rbd), rbd._priority)
        if crews is not None and _crews._crews_limited(rbd)
        else None
    )
    if rbd._crews is not None:
        # A repair or maintenance going on at the start holds a crew
        # (_simulation_states has checked that there are enough).
        for node in in_hand:
            if node in rbd._crews.served:
                rbd._crews.free -= 1
                rbd._crews.holding.add(node)
    # Each standby group's units, which queue the group's first event.
    rbd._groups = {
        node: _StandbyGroup(
            rbd, node, arrangement, sources[node], t_simulation
        )
        for node, arrangement in rbd._standby.items()
        if node not in working_nodes and node not in broken_nodes
    }
    rbd.last_change_planned = False
    # The initial system state must reflect any forced-broken components
    # (e.g. a broken component in series starts the system down), and
    # any component that starts down, rather than assuming everything is
    # up.
    rbd.system_state = rbd.is_system_working(component_status, method)
    rbd.component_status = component_status


def _started(rbd, node, source, start: NodeState) -> Tuple[bool, Event]:
    """Whether component ``node`` is up at 0, started from ``start``
    (not new), and its first event.

    One uniform, from the component's own stream (see
    ``_streamed_components``), draws what is left of its repair or
    maintenance given how long it has taken so far, or of its life
    given its age (``_aged_life``: ``S(a + x) / S(a)`` is the chance
    of more than ``x`` left). Up at age 0 it is new, and draws its life
    as a new unit does. Its calendar is shifted by the state's phase
    (see ``_due``); under age replacement, its replacement is due when
    it reaches the age, at once if it has. A unit with hidden failures
    is known to have been up only at its last test (or when put into
    service, if since): its life left is drawn from then, and if it
    has run out it failed unseen, and the next test finds it. After
    the first, its units are new, as from new.
    """
    component = rbd.components[node]
    schedule = rbd._preventive.get(node)
    inspection = rbd._inspection.get(node)
    source.reset()
    virtual = start.virtual_age or 0.0
    if virtual:
        # Repaired imperfectly (#269): at its virtual age at its last
        # repair (or once its repair going on is over).
        source.age = virtual
    if not start.alive:
        # Down: up again, as new, once its repair or maintenance is
        # done.
        model = (
            schedule.duration  # type: ignore[union-attr]
            if start.maintenance
            else component.time_to_replace
        )
        left = _aged_life(model, start.down_for, source.start_uniform())
        if inspection is not None:
            rbd._pending_failure[node] = None
        return False, Event(left, node, True, start.maintenance)
    if not start.age and not virtual:
        # New at 0, on a calendar at its phase.
        if schedule is not None:
            return True, _renewal(rbd, node, 0.0, source, schedule)
        if inspection is not None:
            return True, _inspected_renewal(rbd, node, 0.0, source, inspection)
        delay, status = source.next_event()
        return True, Event(delay, node, status)
    since = 0.0
    if inspection is not None:
        # Last known up at its last test, or when put into service.
        since = min(start.age, start.phase or 0.0)
    life = (
        _aged_life(
            component.reliability,
            virtual + start.age - since,
            source.start_uniform(),
        )
        - since
    )
    source.life_drawn(start.age + life)
    if inspection is not None:
        if life < 0.0:
            # Failed, unseen, since it was last known up.
            rbd._pending_failure[node] = None
            found = _finds(rbd, node, inspection, life)
            return False, Event(found, node, False, inspection=True)
        rbd._pending_failure[node] = life
        return True, _inspected_next(rbd, node, 0.0, inspection)
    if schedule is None:
        return True, Event(life, node, False)
    return True, _renewal(
        rbd, node, 0.0, source, schedule, life=life, start=-start.age
    )


def _due(rbd, node, schedule, t: float) -> float:
    """The next scheduled action of ``schedule`` (a ``_Preventive`` or
    ``_Inspection``) after ``t``, on ``node``'s calendar: shifted, from
    a state, by its phase, so that the last action fell that long
    before 0."""
    phase = rbd._phases.get(node) if rbd._phases else None
    if phase is None:
        return schedule.due(t)
    if isinstance(schedule, _Inspection):
        # The phase places the calendar: no offset.
        schedule = schedule.without_offset()
    due = schedule.due(t + phase) - phase
    return due if due > t else due + schedule.interval


def _finds(rbd, node, inspection: _Inspection, t: float) -> float:
    """The test that finds a failure at ``t``, on ``node``'s calendar
    (see ``_due``): the first at or after it, and not before 0."""
    phase = rbd._phases.get(node) if rbd._phases else None
    if phase is None:
        return inspection.finds(t)
    inspection = inspection.without_offset()
    found = inspection.finds(t + phase) - phase
    if found < t or found < 0.0:
        found += inspection.interval
    return found


def _follow_up(rbd, event: Event, source) -> Event:
    """The next event of ``event``'s component, drawn from ``source``.

    A component's ``next_event()`` gives the time *to* its next event,
    measured from ``event``. A nested RBD's gives the time *of* its next
    state change: its simulation runs on the same clock, from 0.
    """
    from repyability.rbd.repairable_rbd import RepairableRBD

    node = event.component
    schedule = rbd._preventive.get(node)
    if schedule is not None:
        return _maintained_follow_up(rbd, event, source, schedule)
    inspection = rbd._inspection.get(node)
    if inspection is not None:
        return _inspected_follow_up(rbd, event, source, inspection)
    t, status = source.next_event()
    if isinstance(rbd.components[node], RepairableRBD):
        return Event(t, node, status, _planned(source, status))
    return Event(event.time + t, node, status)


def _strike(rbd, event: Event, source) -> List[Event]:
    """A shared common cause strikes (#158, see ``_Cause``): each member
    it names that is up fails now, its own next event superseded (a
    member down stays down), and with tests that can miss a failure,
    one uniform decides for all its failures. Returns the events to
    queue: the members' failures, and the cause's next strike."""
    cause: Any = event.component
    t = event.time
    status = rbd.component_status
    out: List[Event] = []
    coin = source.coin() if cause.coins else None
    for member in cause.struck:
        if not status[member]:
            continue
        pending = rbd._ccf_pending.pop(member, None)
        if pending is not None and pending.time < rbd.t_simulation:
            rbd._cancelled[id(pending)] = pending
        if coin is not None:
            rbd._ccf_coins[member] = coin
        out.append(Event(t, member, False))
    later = Event(t + source.gap(), cause, False)
    if later.time < rbd.t_simulation:
        out.append(later)
    return out


def _crew_follow_up(rbd, event: Event, next_event: Event) -> Optional[Event]:
    """``next_event`` as the repair crews allow (see ``_Crews``): a job
    that falls due when ``event`` takes a component down waits for a
    crew (None while it does), and a crew that finishes one at
    ``event`` starts the next waiting job, whose end is queued here."""
    crews, node = rbd._crews, event.component
    if crews is None or node not in crews.served:
        return next_event
    if event.status:
        if node in crews.holding:
            started = crews.release(node, event.time)
            if started is not None:
                _crew_started(rbd, started)
        return next_event
    if next_event.status:
        return crews.request(node, event.time, next_event)
    return next_event


def _crew_started(rbd, started: Tuple[Any, float, Event]) -> None:
    """A crew has started a waiting job, ``started`` as
    ``_Crews.release`` gives it: queue the end of a component's, or
    tell its standby group of a unit's."""
    key, wait, ends = started
    if isinstance(key, _Unit):
        rbd._groups[key.node].crew_started(key.unit, ends.time)
        return
    ends = _started_late(rbd, key, wait, ends)
    if ends.time < rbd.t_simulation:
        rbd._event_queue.put(ends)


def _started_late(rbd, node, wait: float, ends: Event) -> Event:
    """A job of ``node`` that a crew starts ``wait`` after it fell due,
    and so ends with ``ends``: a component off-line for a test does not
    age while it waits, so its hidden failure is ``wait`` later."""
    pending = rbd._pending_failure.get(node)
    if pending is not None:
        rbd._pending_failure[node] = pending + wait
    return ends


def _maintained_follow_up(
    rbd, event: Event, source, schedule: _Preventive
) -> Event:
    """The next event of a component under preventive maintenance."""
    node, t = event.component, event.time
    if event.inspection:
        # Inspected, and kept (replacement on condition): its failure is
        # still ahead of it.
        return _condition_next(rbd, node, t, schedule, source)
    if event.preventive:
        # Replaced: the new unit has its own life ahead of it (the
        # failure drawn for the old one never happens).
        source.reset()
        if not event.status:
            # Down for maintenance: back, as new, once it is done.
            if isinstance(source, _StreamedComponent):
                duration = source.maintenance_time()
            else:
                duration = schedule.duration.random(1).item()
            return Event(t + duration, node, True, True)
    elif not event.status:
        # Failed: back, as new, once it is repaired.
        repair, status = source.next_event()
        return Event(t + repair, node, status)
    # Working, as new, from t.
    return _renewal(rbd, node, t, source, schedule)


def _renewal(
    rbd,
    node,
    t: float,
    source,
    schedule: _Preventive,
    life: Optional[float] = None,
    start: Optional[float] = None,
) -> Event:
    """The first event of a unit put into service as new at ``t``: its
    failure, or its preventive replacement if that is due first (a
    failure at the same time comes first); replaced on condition, its
    failure or its first inspection (see ``_condition_next``). Started
    from a state (see ``_started``), the unit was put into service at
    ``start``, before ``t``, and has ``life`` left."""
    if life is None:
        life, _ = source.next_event()
    if schedule.policy == "condition":
        failure = t + life
        if schedule.level is not None:
            # As new: at the process's starting level, its failure
            # drawn from there; each inspection decides (#271).
            level = float(rbd.components[node].reliability.y0)
            rbd._levels[node] = (t, level)
            rbd._in_service[node] = (failure, math.inf)
            return _condition_next(rbd, node, t, schedule, source)
        rbd._in_service[node] = (
            failure,
            _replaced_at(
                rbd, node, t if start is None else start, failure, schedule, t
            ),
        )
        return _condition_next(rbd, node, t, schedule, source)
    if start is None:
        # Back from an imperfect repair, the unit is as old as its
        # operating time since it was renewed, which age replacement
        # counts.
        start = t - source.operated if node in rbd._imperfect else t
    due = _due(rbd, node, schedule, start if schedule.policy == "age" else t)
    if due < t:
        due = t  # past its age at the start: replaced at once
    if t + life <= due:
        event = Event(t + life, node, False)
    else:
        # Maintenance that takes time is a planned outage; in zero time
        # the unit is renewed in place, and stays up.
        event = Event(due, node, schedule.duration is None, True)
    if node in rbd._early_members:
        # Kept, should a stop of its group renew it early (see _stop).
        rbd._renewed_at[node] = start
        rbd._pending_event[node] = event
    return event


def _own_sources(rbd, states: Optional[dict] = None) -> dict:
    """What the components draw their events from when no streams are
    given: themselves, but for an imperfectly repaired one, a stand-in
    keeping its virtual age (see ``_ImperfectComponent``), and for one
    started from a state in ``states`` that draws what is left of its
    life or repair, a stand-in whose next draw can be set (see
    ``_started``): drawing from their models and numpy's global
    RNG."""
    starting = [
        node for node, start in (states or {}).items() if _draws_start(start)
    ]
    if not rbd._imperfect and not starting and not rbd.ccf_groups:
        return rbd.components
    sources = dict(rbd.components)
    for group in rbd.ccf_groups:
        # Each member's own cause's life, and the shared causes (#158).
        for member in group.members:
            own = _ccf_groups._own_rate(rbd, member)
            if own is not None:
                sources[member] = _own_draws(rbd, member, own)
    for cause in _ccf_groups._shared_causes(rbd):
        sources[cause] = _CauseDraws(cause.rate)
    for node, imperfect in rbd._imperfect.items():
        sources[node] = _ImperfectComponent(
            rbd.components[node],
            imperfect,
            model=_duration_model(rbd, node),
        )
    for node in starting:
        component = rbd.components[node]
        sources[node] = _StreamedComponent(
            _ModelDraws(component.reliability),
            _ModelDraws(component.time_to_replace),
            model=_duration_model(rbd, node),
        )
    return sources


def _stop(rbd, group, t: float, trigger=None) -> List[Event]:
    """A stop of maintenance ``group`` at ``t``, opened by ``trigger``
    (a member taken down for a failure or its scheduled replacement) or
    by the system going down: each other member that is working, and at
    least its opportunity age, is renewed now, as its scheduled
    replacement would renew it, and the failure or replacement it had
    ahead of it is cancelled. A member whose own failure or replacement
    is due at ``t`` keeps it: that joins the stop. Returns the renewals
    to queue."""
    out: List[Event] = []
    for member in rbd._maintenance[group].members:
        if (
            member == trigger
            or member not in rbd._early_members
            or member not in rbd._renewed_at
            or member in rbd._renewing
            or not rbd.component_status[member]
        ):
            continue
        schedule = rbd._preventive[member]
        if t - rbd._renewed_at[member] < schedule.opportunity:
            continue
        pending = rbd._pending_event.get(member)
        if pending is not None and pending.time <= t:
            continue  # due now, as part of this stop
        rbd._pending_event.pop(member, None)
        if pending is not None and pending.time < rbd.t_simulation:
            rbd._cancelled[id(pending)] = pending
        renewal = Event(t, member, schedule.duration is None, True)
        rbd._early[id(renewal)] = renewal
        rbd._renewing.add(member)
        out.append(renewal)
    return out


def _replaced_at(
    rbd,
    node,
    renewed: float,
    failure: float,
    schedule: _Preventive,
    after: Optional[float] = None,
) -> float:
    """When a unit put into service as new at ``renewed``, and due to
    fail at ``failure``, is replaced on condition: at the first
    inspection before its failure (and the window's end), and after
    ``after`` (by default ``renewed``), at which it is more likely than
    the threshold to fail before the next, given its age (the time
    since ``renewed``); ``inf`` if at none."""
    end = min(failure, rbd.t_simulation)
    dues = []
    due = _due(rbd, node, schedule, renewed if after is None else after)
    while due < end:
        dues.append(due)
        due = _due(rbd, node, schedule, due)
    if not dues:
        return math.inf
    # Its ages at those inspections, and at the one after the last.
    ages = np.array(dues + [due]) - renewed
    likely = _failures_between(rbd.components[node].reliability, ages)
    above = np.flatnonzero(likely > schedule.threshold)
    return dues[above[0]] if above.size else math.inf


def _condition_next(
    rbd, node, t: float, schedule: _Preventive, source
) -> Event:
    """The next event, from ``t``, of a working unit replaced on
    condition: its failure, or the next inspection, which replaces the
    unit (see ``_replaced_at``) or only checks it. A failure at an
    inspection's time comes first.

    Replaced by its measured level (#271), the inspection measures it,
    drawn from the unit's ``LEVEL`` stream given that it has not failed
    by then (``level_after``, the Markov property: the failure drawn
    from the last level is past the inspection, and nothing else of it
    is kept), and replaces the unit at or past ``schedule.level``; kept,
    the unit's failure is drawn afresh from the level found."""
    failure, replaced = rbd._in_service[node]
    due = _due(rbd, node, schedule, t)
    if failure <= due:
        return Event(failure, node, False)
    if schedule.level is not None:
        process = rbd.components[node].reliability
        known, level = rbd._levels[node]
        level = level_after(
            process, level, due - known, _level_uniform(source)
        )
        if level >= schedule.level:
            return Event(due, node, schedule.duration is None, True)
        rbd._levels[node] = (due, level)
        left = failure_from(process, level, _level_uniform(source))
        rbd._in_service[node] = (due + left, math.inf)
        return Event(due, node, True, inspection=True)
    if due >= replaced:
        # Replaced, as under block replacement.
        return Event(due, node, schedule.duration is None, True)
    return Event(due, node, True, inspection=True)


def _inspected_follow_up(
    rbd, event: Event, source, inspection: _Inspection
) -> Event:
    """The next event of a component whose failures are hidden."""
    node, t = event.component, event.time
    if not event.inspection:
        if event.status:
            # Repaired: as new, from t.
            return _inspected_renewal(rbd, node, t, source, inspection)
        # Failed, unseen: found by the first inspection at or after t,
        # unless that one can miss it and does (a shared common cause
        # has decided for all its failures, see _strike).
        rbd._pending_failure[node] = None
        found = _finds(rbd, node, inspection, t)
        coin = rbd._ccf_coins.pop(node, None) if rbd._ccf_coins else None
        if not inspection.is_full(found) and not (
            coin < inspection.coverage
            if coin is not None
            else _test_finds(source, inspection.coverage)
        ):
            return Event(found, node, False, inspection=True, missed=True)
        return Event(found, node, False, inspection=True)
    if event.missed:
        # A test that missed the failure: the next test misses it too,
        # unless it is a full test.
        due = _due(rbd, node, inspection, t)
        return Event(
            due,
            node,
            False,
            inspection=True,
            missed=not inspection.is_full(due),
        )
    if event.status:
        # Tested in zero time, or back from a test: working, with its
        # failure still ahead of it.
        return _inspected_next(rbd, node, t, inspection)
    # A test that takes time, of a failed unit (repaired once the test is
    # done) or of a working one (off-line, and not ageing, until then).
    duration = 0.0
    if inspection.duration is not None:
        if isinstance(source, _StreamedComponent):
            duration = source.maintenance_time()
        else:
            duration = inspection.duration.random(1).item()
    failure = rbd._pending_failure[node]
    if failure is None:
        repair, status = source.next_event()
        return Event(t + duration + repair, node, status)
    rbd._pending_failure[node] = failure + duration
    return Event(t + duration, node, True, inspection=True)


def _inspected_renewal(
    rbd, node, t: float, source, inspection: _Inspection
) -> Event:
    """The first event of a unit with hidden failures put into service
    as new at ``t``."""
    life, _ = source.next_event()
    rbd._pending_failure[node] = t + life
    return _inspected_next(rbd, node, t, inspection)


def _inspected_next(rbd, node, t: float, inspection: _Inspection) -> Event:
    """The next event of a working unit with hidden failures, at ``t``:
    its failure, or the next inspection if that comes first (a failure
    at the same time comes first, and is found by it). A test that takes
    time takes the unit off-line; one in zero time leaves it up."""
    failure = rbd._pending_failure[node]
    due = _due(rbd, node, inspection, t)
    if failure <= due:  # type: ignore[operator]
        return Event(failure, node, False)  # type: ignore[arg-type]
    return Event(due, node, inspection.duration is None, inspection=True)


def next_event(rbd, method, sources: Optional[dict]):
    """See ``RepairableRBD.next_event``."""
    if not hasattr(rbd, "_event_queue"):
        raise ValueError("Need to initialize the event queue")
    # The components' draws come from the same sources the queue was
    # initialised with (see initialize_event_queue).
    if sources is None:
        sources = getattr(rbd, "_step_sources", rbd.components)
    new_system_state = copy(rbd.system_state)

    # Use a while loop to find the next time/event at which the system
    # status changes.
    while new_system_state == rbd.system_state:
        if rbd._event_queue.qsize() == 0:
            del rbd._event_queue
            rbd.last_change_planned = False
            return rbd.t_simulation, rbd.system_state

        event = rbd._event_queue.get()
        if rbd._cancelled and rbd._cancelled.get(id(event)) is event:
            del rbd._cancelled[id(event)]
            continue  # superseded by an early renewal, or a strike
        if event.component.__class__ is _Cause:
            # A shared common cause strikes (#158, see _strike).
            for follow in _strike(rbd, event, sources[event.component]):
                rbd._event_queue.put(follow)
            continue
        renewing_early = False
        if rbd._early and rbd._early.get(id(event)) is event:
            del rbd._early[id(event)]
            rbd._renewing.discard(event.component)
            renewing_early = True
        group = rbd._groups.get(event.component)
        if group is not None:
            # A standby group's own event (see _StandbyGroup).
            if not group.holds(event):
                continue  # superseded by an earlier one
            now, _ = group.advance(event.time)
            if now == rbd.component_status[event.component]:
                continue
            rbd.component_status[event.component] = now
            if now != rbd.system_state:
                new_system_state = rbd.is_system_working(
                    rbd.component_status, method
                )
            continue  # the group has queued its next event
        was = rbd.component_status[event.component]
        rbd.component_status[event.component] = event.status
        node = event.component
        if node in rbd._member_group and not renewing_early:
            # A failure, or a scheduled replacement starting (or done in
            # place), opens a stop of its maintenance group.
            if (not event.status and not event.inspection) or (
                event.preventive and event.status and was
            ):
                for renewal in _stop(
                    rbd, rbd._member_group[node], event.time, node
                ):
                    rbd._event_queue.put(renewal)
        # Only a change against the system's state can change it (the
        # structure is coherent; see _replicate).
        if event.status != rbd.system_state:
            new_system_state = rbd.is_system_working(
                rbd.component_status, method
            )
            if rbd.system_state and not new_system_state:
                for name, spec in rbd._maintenance.items():
                    if spec.system_down:
                        for renewal in _stop(rbd, name, event.time):
                            rbd._event_queue.put(renewal)

        follow = _follow_up(rbd, event, sources[event.component])
        next_event: Optional[Event] = (
            follow
            if rbd._crews is None
            else _crew_follow_up(rbd, event, follow)
        )
        # But only queue up the event if it occurs before the end
        # of the simulation
        if next_event is not None and next_event.time < rbd.t_simulation:
            rbd._event_queue.put(next_event)
        if node in rbd._ccf_members:
            rbd._ccf_pending[node] = next_event

    rbd.system_state = new_system_state
    # A system taken down by maintenance or a test is a planned outage.
    rbd.last_change_planned = (
        event.preventive or event.inspection
    ) and not new_system_state

    return event.time, rbd.system_state


def _run(rbd, *args, **kwargs) -> "_Tally":
    """A run of simulations (see ``_run_alone``), one at a time in the
    process (``SIMULATIONS``, #216): the event loop keeps its state on
    the diagram, and draws that cannot be streamed come from numpy's
    global RNG. Not a ``sharded`` run, whose shards each run (and take
    it) where they are sent: a ``shard_map`` on threads would wait for
    it."""
    if kwargs.get("sharded") is not None:
        return _run_alone(rbd, *args, **kwargs)
    with SIMULATIONS:
        return _run_alone(rbd, *args, **kwargs)


def _run_alone(
    rbd,
    t_simulation: float,
    working_nodes,
    broken_nodes,
    method: str,
    N: int,
    verbose: bool,
    seed,
    antithetic: bool = False,
    stop: Optional[Callable[["_Tally"], int]] = None,
    capacity: Optional[_CapacityRecorder] = None,
    jobs: Optional[int] = None,
    engine: str = "auto",
    entropy: Any = None,
    common: bool = False,
    replacements: bool = False,
    first: int = 0,
    states: Optional[dict] = None,
    curve_points: Optional[int] = None,
    sharded: Optional[tuple] = None,
    histories: bool = False,
) -> "_Tally":
    """Run ``N`` simulations, from simulation ``first`` (then more,
    while ``stop`` asks for them), and return their totals: each from
    the components' ``states`` (checked; see ``_simulation_states``),
    new by default.

    Every draw comes from a stream of its own (see ``_streams``), seeded
    from the run's ``entropy``: the ``seed``, or without one a number
    drawn from numpy's global RNG. Draws that cannot be streamed come
    from the global RNG, seeded afresh for each simulation. So each
    simulation is the same whichever ``engine`` runs it, in however many
    processes or threads (``jobs``), and in a run to a tolerance as in a
    run of its final size. The global RNG is left as it was, except for
    the number drawn for the entropy. ``compare`` passes both systems
    the same ``entropy``, and ``common``: that, like
    ``antithetic``, needs every draw to come from a stream. With
    ``capacity``, each simulation also follows the system's capacity;
    with ``replacements``, the tally keeps each simulation's
    replacements of each component, which only the Python engine
    counts; with ``curve_points``, it counts the changes of state on a
    grid (see ``availability``). With ``sharded``, ``(shard_map,
    template, step)``, the simulations run as shards (see
    ``_ShardRunner``), on the engine each shard's worker chooses. With
    ``histories``, the tally keeps each simulation's histories in place
    of every total (see ``simulate_timelines``), on the Python engine
    or numba's.
    """
    _ccf_groups._require_free_members(rbd, working_nodes, broken_nodes)
    _ccf_groups._require_groups_simulated(rbd, states)
    t_simulation = simulation_window(t_simulation)
    from tqdm import tqdm

    state = np.random.get_state()
    after = state
    tally = _Tally(list(rbd.components), rbd.costs, t_simulation, curve_points)
    if replacements:
        # Only the Python engine counts them.
        tally.replacements = []
        engine = "python"
    if histories:
        tally.histories = _timeline_runs.Records(len(rbd.components))
    runner: Any = None
    try:
        if entropy is None:
            entropy = _streams.entropy_of(seed)
            if seed is None:
                after = np.random.get_state()
        plan, complete = _stream_plan(
            rbd, t_simulation, entropy, antithetic, states
        )
        if (antithetic or common) and not complete:
            raise NotImplementedError(_UNSTREAMED)
        engine = _runs._simulation_engine(
            rbd,
            engine,
            plan,
            capacity,
            N,
            here=sharded is None,
            states=states,
        )
        progress = tqdm(
            total=N, disable=not verbose, desc="Running simulations"
        )
        if sharded is not None:
            runner = _ShardRunner(rbd, tally, progress, *sharded)
        elif engine == "numba":
            from repyability.rbd import _compiled

            runner = _compiled.Runner(
                rbd,
                plan,
                tally,
                progress,
                working_nodes,
                broken_nodes,
                method,
                jobs,
                capacity=capacity,
            )
        elif engine != "python":
            from repyability.rbd import engines

            runner = engines.registered()[engine].runner(
                rbd,
                plan,
                tally,
                progress,
                working_nodes,
                broken_nodes,
                method,
                jobs,
            )
        else:
            runner = _PythonRunner(
                rbd,
                tally,
                progress,
                (
                    t_simulation,
                    working_nodes,
                    broken_nodes,
                    method,
                    capacity,
                    entropy,
                    antithetic,
                    states,
                    histories,
                ),
                jobs,
            )
        start, goal = first, first + N
        while True:
            runner(start, goal)
            more = 0 if stop is None else stop(tally)
            if not more:
                break
            start, goal = goal, goal + more
            progress.total = goal
            progress.refresh()
        progress.close()
    finally:
        if runner is not None:
            runner.close()
        if sharded is None:
            # A sharded run draws nothing here, and another thread's
            # run may have the global RNG now.
            np.random.set_state(after)
        _forget_run(rbd)
    return tally


def _forget_run(rbd) -> None:
    """Clean up the interim variables of a simulation, this RBD's and
    its nested RBDs': left behind, a nested standby group's draws (whose
    samplers cannot be pickled) would keep the RBD from going to the
    processes of a later parallel run."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    for name in rbd._RUN_STATE:
        rbd.__dict__.pop(name, None)
    for component in rbd.components.values():
        if isinstance(component, RepairableRBD):
            _forget_run(component)


def _replicate(rbd, ctx: "_Context", replication: int) -> _Replication:
    """Simulation ``replication`` of ``[0, t_simulation]``.

    Besides the counts, it follows each component's up time and its
    overlaps with the system's as it goes: when a component changes
    state, the time since its last change is added to its up time (if
    it was up), and the system's up (or down) time over the same span
    to the time both were up (or down), as the difference of the
    system's up (or down) time so far at either end.
    """
    ctx.run.begin(replication)
    t_simulation = ctx.t_simulation
    sources = ctx.sources
    failure_charges, preventive_charges, inspection_charges = ctx.charges
    _start_queue(
        rbd,
        t_simulation,
        ctx.working,
        ctx.broken,
        ctx.method,
        sources,
        ctx.states,
    )
    status = rbd.component_status
    index, plain, works = ctx.position, ctx.plain, ctx.works
    inspected = rbd._inspection
    crews = rbd._crews
    groups = rbd._groups
    n = len(index)
    # Each component's failures, the system failures they caused, its
    # restorations and the system restorations they caused.
    failed, caused_down = [0] * n, [0] * n
    restored, caused_up = [0] * n, [0] * n
    # Each component's replacements: the spares it used.
    replaced = [0] * n
    # Imperfect repair (#109): a failure repaired, not replaced, uses no
    # spare and is charged only its repair.
    imperfect = rbd._imperfect
    # Opportunistic maintenance (#108): each member's early renewals.
    maintenance, member_group = rbd._maintenance, rbd._member_group
    cancelled, early = rbd._cancelled, rbd._early
    opportunistic = [0] * n
    failures = restorations = planned = 0
    changes: list = []
    deltas: list = []
    if ctx.initial_up and not rbd.system_state:
        # Down at the start, from the components' states: the curve of
        # the availability over time counts it as a change at 0.
        changes.append(0.0)
        deltas.append(-1)
    by_category = [0.0] * len(_CATEGORIES)
    by_node = dict.fromkeys(rbd.costs, 0.0)
    rep_cost = 0.0

    def pay(node, category: int, charges) -> float:
        """The next charge of ``charges`` (nothing without any)."""
        if charges is None:
            return 0.0
        charge = charges.draw()
        by_category[category] += charge
        by_node[node] += charge
        return charge

    trace = None if ctx.capacity is None else ctx.capacity.trace(status)
    # With ctx.history (see simulate_timelines): each component's state
    # at 0, the times of its changes, and which of them (their places
    # among its changes) were planned changes down, maintenance or a
    # test off line; and the system's changes: their times, the
    # component whose change made each, which of its changes that was,
    # and whether it was planned.
    history = ctx.history
    at_start = [status[node] for node in index] if history else None
    times_of: list = [[] for _ in range(n)] if history else []
    planned_of: list = [[] for _ in range(n)] if history else []
    system_record: tuple = ([], [], [], []) if history else ()
    # The system's state, and its up and down time until ``since``, its
    # last change; each component's last change, and the system's up and
    # down time until then.
    up = rbd.system_state
    started_up = up
    system_up = system_down = since = 0.0
    last, up_at, down_at = [0.0] * n, [0.0] * n, [0.0] * n
    node_up, both_up, both_down = [0.0] * n, [0.0] * n, [0.0] * n

    # The queue's heap, worked directly (see _EventQueue).
    heap = rbd._event_queue._heap
    push, pop = heapq.heappush, heapq.heappop

    # When each group last stopped: the work started at one instant is
    # one stop, with one set-up.
    stopped: Dict[Hashable, float] = {}
    # Common-cause groups' members, whose next events a shared cause can
    # supersede (see _strike).
    ccf_members, ccf_pending = rbd._ccf_members, rbd._ccf_pending

    def corrective(node, c: int) -> float:
        """Charge a failure's corrective action, and count the spare a
        replacement uses: an imperfect repair (see _ImperfectComponent)
        uses none, and is charged only as a repair."""
        charge = 0.0
        renewed = not imperfect or node not in imperfect
        renewed = renewed or sources[node].replacing
        if renewed:
            replaced[c] += 1
        for category, charges in failure_charges.get(node, ()):
            if renewed or category == _REPAIR:
                charge += pay(node, category, charges)
        return charge

    def stop(group, t: float, trigger=None) -> float:
        """Open a stop of maintenance ``group`` (see ``_stop``): queue
        its early renewals, and return its set-up cost, charged once
        per stop, if a member opened it or it renews any."""
        renewals = _stop(rbd, group, t, trigger)
        for renewal in renewals:
            push(heap, (renewal.time, renewal))
        if (trigger is None and not renewals) or stopped.get(group) == t:
            return 0.0
        stopped[group] = t
        setup = maintenance[group].setup_cost
        if setup:
            by_category[6] += setup
        return setup

    # No event at or after the end of the window is queued, so the
    # simulation runs until the queue is empty.
    while heap:
        event = pop(heap)[1]
        if cancelled and cancelled.get(id(event)) is event:
            del cancelled[id(event)]
            continue  # superseded by an early renewal, or a strike
        node = event.component
        if node.__class__ is _Cause:
            # A shared common cause strikes (#158).
            for follow in _strike(rbd, event, sources[node]):
                push(heap, (follow.time, follow))
            continue
        renewing_early = False
        if early and early.get(id(event)) is event:
            del early[id(event)]
            rbd._renewing.discard(node)
            opportunistic[index[node]] += 1
            renewing_early = True
        grouped = False
        if groups and node in groups:
            # A standby group's own event (see _StandbyGroup): a unit
            # failure, charged as a repair, or a repair's end.
            group = groups[node]
            if not group.holds(event):
                continue  # superseded by an earlier one
            now, broken = group.advance(event.time)
            if broken:
                replaced[index[node]] += 1
                for category, charges in failure_charges.get(node, ()):
                    rep_cost += pay(node, category, charges)
            if now == status[node]:
                continue
            event, grouped = Event(event.time, node, now), True
        if (event.preventive or event.inspection) and (
            event.status == status[node]
        ):
            # No change of state: maintenance or a test in zero time of a
            # working unit (renewed in place, or found working), or a
            # test that finds a hidden failure, whose repair starts now.
            if event.preventive:
                replaced[index[node]] += 1
                rep_cost += pay(node, 2, preventive_charges.get(node))
                # Replaced on condition: at an inspection, charged too.
                rep_cost += pay(node, 3, inspection_charges.get(node))
                if node in member_group and not renewing_early:
                    # Renewed in place on schedule: a stop of its group.
                    rep_cost += stop(member_group[node], event.time, node)
            else:
                rep_cost += pay(node, 3, inspection_charges.get(node))
                if not event.status and not event.missed:
                    rep_cost += corrective(node, index[node])
            follow = _follow_up(rbd, event, sources[node])
            next_event: Optional[Event] = (
                follow
                if crews is None
                else _crew_follow_up(rbd, event, follow)
            )
            if next_event is not None and next_event.time < t_simulation:
                push(heap, (next_event.time, next_event))
            if ccf_members and node in ccf_members:
                ccf_pending[node] = next_event
            continue
        t = event.time
        c = index[node]
        if up:
            system_up_t, system_down_t = (
                system_up + (t - since),
                system_down,
            )
        else:
            system_up_t, system_down_t = system_up, system_down + (t - since)
        if status[node]:
            node_up[c] += t - last[c]
            both_up[c] += system_up_t - up_at[c]
        else:
            both_down[c] += system_down_t - down_at[c]
        last[c], up_at[c], down_at[c] = t, system_up_t, system_down_t
        status[node] = event.status
        if trace is not None:
            trace.change(t, node, event.status)
        if history:
            times_of[c].append(t)
            if not event.status and (event.preventive or event.inspection):
                planned_of[c].append(len(times_of[c]) - 1)
        if event.status:
            restored[c] += 1
        elif event.preventive:
            # A planned outage: charged the preventive cost, and the
            # inspection's if replaced on condition (a nested RBD's own
            # costs are not counted).
            replaced[c] += 1
            rep_cost += pay(node, 2, preventive_charges.get(node))
            rep_cost += pay(node, 3, inspection_charges.get(node))
            if node in member_group and not renewing_early:
                rep_cost += stop(member_group[node], t, node)
        elif event.inspection:
            # A test that takes a working unit off-line.
            rep_cost += pay(node, 3, inspection_charges.get(node))
        else:
            failed[c] += 1
            # Repair and replace are charged per corrective action, at
            # the failure that triggers it (for a hidden failure, when an
            # inspection finds it; for a standby group, at each unit's).
            if node not in inspected and not grouped:
                if imperfect and node in imperfect:
                    rep_cost += corrective(node, c)
                else:
                    replaced[c] += 1
                    for category, charges in failure_charges.get(node, ()):
                        rep_cost += pay(node, category, charges)
            if node in member_group:
                rep_cost += stop(member_group[node], t, node)

        # The structure is coherent: a restoration can't take the
        # system down, nor a failure bring it up, so only a change
        # against the system's state needs the structure function.
        if event.status != up and works(status) != up:
            system_up, system_down, since = system_up_t, system_down_t, t
            up = not up
            changes.append(t)
            if history:
                system_record[0].append(t)
                system_record[1].append(c)
                system_record[2].append(len(times_of[c]) - 1)
                system_record[3].append(
                    not up and bool(event.preventive or event.inspection)
                )
            if up:
                deltas.append(1)
                restorations += 1
                caused_up[c] += 1
            else:
                deltas.append(-1)
                if event.preventive or event.inspection:
                    planned += 1
                else:
                    failures += 1
                    caused_down[c] += 1
                for name, spec in maintenance.items():
                    if spec.system_down:
                        # Any system outage is a stop of this group.
                        rep_cost += stop(name, t)

        if grouped:
            continue  # the group has queued its next event
        # The component's next event: its next failure if it has just
        # been restored, or its restoration if it has just gone down;
        # queued only if it falls inside the window.
        if node in plain:
            delay, next_status = sources[node].next_event()
            next_event = Event(t + delay, node, next_status)
        else:
            next_event = _follow_up(rbd, event, sources[node])
        if crews is not None:
            # A job waits for a repair crew (see _Crews).
            next_event = _crew_follow_up(rbd, event, next_event)
        if next_event is not None and next_event.time < t_simulation:
            push(heap, (next_event.time, next_event))
        if ccf_members and node in ccf_members:
            ccf_pending[node] = next_event
    rbd.system_state = up

    # Close the system's time, and each component's, at the end of the
    # window.
    if up:
        system_up += t_simulation - since
    else:
        system_down += t_simulation - since
    for c, node in enumerate(index):
        if status[node]:
            node_up[c] += t_simulation - last[c]
            both_up[c] += system_up - up_at[c]
        else:
            both_down[c] += system_down - down_at[c]

    rates = ctx.downtime_cost_rates
    if rates:
        for c, node in enumerate(index):
            rate = rates.get(node)
            if rate is not None:
                charge = rate * (t_simulation - node_up[c])
                rep_cost += charge
                by_category[4] += charge
                by_node[node] += charge
    if ctx.has_costs:
        charge = rbd.downtime_cost_rate * (t_simulation - system_up)
        rep_cost += charge
        by_category[5] += charge

    rec = _Replication()
    rec.uptime = system_up
    rec.node_up, rec.both_up, rec.both_down = node_up, both_up, both_down
    rec.counts = (failed, caused_down, restored, caused_up)
    rec.failures, rec.restorations = failures, restorations
    rec.planned = planned
    rec.changes, rec.deltas = changes, deltas
    rec.cost = rep_cost if ctx.has_costs else None
    rec.by_category, rec.by_node = by_category, by_node
    rec.capacity_changes = rec.capacity_time = rec.delivered = None
    rec.replacements = replaced
    rec.opportunistic = opportunistic
    rec.history = (
        (at_start, times_of, planned_of, started_up, system_record)
        if history
        else None
    )
    if trace is not None:
        trace.finish(t_simulation, rec)
    return rec
