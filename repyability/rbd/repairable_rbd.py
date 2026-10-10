"""Repairable reliability block diagrams: the ``RepairableRBD`` class.

Besides the class, the module holds the helpers of its availability
simulation: the ``Event`` records of its event queue and the failure and
restoration criticality index ratios. The simulation's random streams are
in ``_streams``, and its compiled engine in ``_compiled`` and
``_kernel``.
"""

import dataclasses
import hashlib
import heapq
import json
import math
import pickle
from collections.abc import Mapping
from contextlib import contextmanager
from copy import copy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from surpyval import ExactEventTime

from repyability._version import __version__
from repyability.non_repairable import NonRepairable
from repyability.rbd import (
    _ccf_modules,
    _conditional,
    _crew_chain,
    _importance_time,
)
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import _sensitivity, _spares, _streams, _timeline_runs
from repyability.rbd._block_replacement import (
    _check_life,
    _Duration,
)
from repyability.rbd._degradation import (
    failure_from,
    level_after,
)
from repyability.rbd._exact import (
    bin_totals,
)
from repyability.rbd._hidden_life import (
    TestedLife,
)
from repyability.rbd._hidden_tests import TestedUnit
from repyability.rbd._hidden_tests import check as check_tested
from repyability.rbd._model_utils import (
    SAVE_ERRORS,
    always_works,
    distribution_name,
    failure_time_scale,
    is_fixed_probability,
    is_mixture,
    lfp_p,
    model_mean,
    refuse_nonparametric,
)
from repyability.rbd._sampling import MixtureLife, stream_sampler
from repyability.rbd.degrading_node import DegradingNode
from repyability.rbd.helper_classes import PerfectReliability, perfect_class
from repyability.rbd.load_sharing_node import LoadSharingModel
from repyability.rbd.node_state import NodeState
from repyability.rbd.rbd import RBD, _check_on_infeasible_rbd, _close_name
from repyability.rbd.results import (
    AvailabilityAllocation,
    AvailabilityResult,
    CapacityDistribution,
    ConditionalRun,
    ConfidenceInterval,
    ControlVariate,
    CostResult,
    Criticalities,
    ExpectedCost,
    ExpectedEvents,
    FailureCriticalityIndex,
    Lever,
    MaintenancePlan,
    RateBreakdown,
    RestorationCriticalityIndex,
    SparesDemand,
    SparesStock,
    TimelineSimulation,
    TotalCostAllocation,
    UncertaintyImportance,
    UncertaintyResult,
    UpDownImportance,
)
from repyability.rbd.standby_node import StandbyModel

if TYPE_CHECKING:
    from repyability.rbd.chunks import SimulationChunk

from repyability.rbd import (
    _ccf_groups,
    _costs,
    _curves,
    _intervals,
    _long_run,
    _repairable_allocation,
    _repairable_capacity,
    _repairable_importance,
    _spec,
    _windows,
)
from repyability.rbd._common import (
    _common_period,
    _constant_rate,
    _curve_points,
    _repair_rate,
    _safe_mean,
    _safe_ratio,
    _sf_values,
    _times_first,
)
from repyability.rbd._curves import (
    _draws_start,
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
from repyability.rbd._tally import (
    _CATEGORIES,
    _REPAIR,
    _acquisition_only,
    _Breakdown,
    _CapacityRecorder,
    _check_shard_map,
    _chunk_means,
    _exact_controls,
    _Exacts,
    _ModuleRun,
    _Replication,
    _shard_bytes,
    _Tally,
)
from repyability.rbd._time_order import (
    _capacity_totals,
    _group_totals,
    _working_over_time,
)
from repyability.rbd.routes import AnalysisRoute, Refused
from repyability.utils.checks import (
    no_distribution,
    one_of,
    simulation_window,
)
from repyability.utils.wrappers import SIMULATIONS


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
        self._rbd._start_queue(
            t_simulation, set(), set(), "p", self._sources, state or {}
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
    if type(component) is RepairableRBD:
        return _StreamedRBD(
            component, component._streamed_components(run, made, path)
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


def _state_key(states: dict) -> Optional[str]:
    """The components' ``states`` at the start of a run (checked; see
    ``RepairableRBD._simulation_states``) as JSON, the same in every
    process, for a chunk's settings (see ``simulate_chunk``); None for
    every component new."""

    def plain(states: dict) -> list:
        out = []
        for node, start in states.items():
            if isinstance(start, NodeState):
                if start.new:
                    continue
                value: Any = dataclasses.asdict(start)
            else:
                value = plain(start)
                if not value:
                    continue
            out.append([repr(node), value])
        return sorted(out, key=lambda item: item[0])

    entries = plain(states)
    return json.dumps(entries, sort_keys=True) if entries else None


def _hot_units(spec: dict) -> "RepairableRBD":
    """A standby group's twin (see ``RepairableRBD._twin``): its units
    operating together, each failing and repaired on its own, ``k`` of them
    needed, as a nested RBD whose components are named ``0, 1, ...``, as
    the group's units are, so that their streams are the units'."""
    standby = spec["standby"]
    units, k = int(standby.get("units", 2)), int(standby.get("k", 1))
    keys = ("reliability", "repairability") + RepairableRBD.COST_KEYS
    unit = {key: spec[key] for key in keys if key in spec}
    return RepairableRBD(
        [("s", u) for u in range(units)] + [(u, "t") for u in range(units)],
        {u: dict(unit) for u in range(units)},
        k={"t": k} if k > 1 else None,
    )


def _states_from_key(rbd, key: Optional[str]) -> dict:
    """The components' states ``_state_key`` saved, for ``rbd`` (a copy of
    the system the key was made for, as a shard rebuilds it): each node by
    its ``repr``, a nested RBD's own states in turn."""

    def states(rbd, entries: list) -> dict:
        nodes = {repr(node): node for node in rbd.components}
        out: dict = {}
        for name, value in entries:
            node = nodes[name]
            if isinstance(value, dict):
                out[node] = NodeState(**value)
            else:
                out[node] = states(rbd.components[node], value)
        return out

    return {} if key is None else states(rbd, json.loads(key))


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


def _stopping_rule(
    N: int,
    tolerance: Optional[float],
    confidence: float,
    max_N: Optional[int],
    antithetic: bool,
    target: str,
    t_simulation: float,
) -> Optional[Callable[..., int]]:
    """``None`` for a fixed number of replications; otherwise a function
    of the tally so far giving how many more replications to run (0 once
    the confidence interval of the mean availability over the window, or of
    the mean cost, is at most ``tolerance`` either side, or ``max_N`` have
    run: then with a warning). Given the simulations' ``values`` too (a
    controlled run's, see ``_controlled_run``), it judges those; given why
    they cannot be judged yet (``unjudged``), it runs on to the limit."""
    montecarlo.check_confidence(confidence)
    limit = montecarlo.sample_limit(
        N, tolerance, max_N, antithetic, ("mc_samples", "max_samples")
    )
    if limit is None:
        return None

    def stop(
        tally: "_Tally", values=None, unjudged: Optional[str] = None
    ) -> int:
        if values is not None:
            # The controlled values of a run with an exact twin.
            values = np.asarray(values, dtype=float)
        elif target == "cost":
            values = np.asarray(tally.cost_samples, dtype=float)
        else:
            values = np.asarray(tally.uptimes, dtype=float) / t_simulation
        return montecarlo.more_samples(
            values,
            N,
            tolerance,  # type: ignore[arg-type]
            confidence,
            limit,
            antithetic,
            target,
            "max_samples",
            unjudged,
        )

    return stop


#: A parallel run's worker process's run (see ``_start_worker``).
_WORKER: Dict[str, Any] = {}


def _start_worker(run: bytes) -> None:
    """Start a worker process of a parallel run (see ``_PythonRunner``):
    unpickle the run, the system and what its simulations share, once, for
    every block the worker runs."""
    rbd, args, curve_points, replacements, histories = pickle.loads(run)
    _WORKER["run"] = (
        rbd,
        rbd._context(*args),
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
        tally.add(rbd._replicate(context, replication))
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
            self._context = rbd._context(*args)

    def __call__(self, start: int, stop: int) -> None:
        """Simulations ``start`` to ``stop - 1``."""
        if self._executor is None:
            for replication in range(start, stop):
                self._tally.add(
                    self._rbd._replicate(self._context, replication)
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
        self._key = _run_key({**template, "fingerprint": rbd._fingerprint()})

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


#: Why antithetic pairs and common random numbers (``compare``) are refused
#: when some component's draws do not come from a stream.
_UNSTREAMED = (
    "Antithetic and common random numbers need every component's draws to "
    "be replayable (surpyval parametric distributions and the composite "
    "models built from them)."
)


def _test_finds(source, coverage: float) -> bool:
    """Whether a test that finds a failure with probability ``coverage``
    finds one: from the component's detection stream, or numpy's global
    RNG."""
    if isinstance(source, _StreamedComponent):
        return source.test_uniform() < coverage
    return float(np.random.random()) < coverage


def failure_criticality_index_per_system_failures(FCI, system_failures):
    """Share of all system failures that each node caused.

    A node causes a system failure when its own failure takes the system
    from up to down. For each node this is the number of system failures
    it caused divided by ``system_failures``; every node gets 0 when there
    were no system failures.

    Parameters
    ----------
    FCI : dict
        Node -> counts, with the number of system failures the node caused
        under ``"system_failures"`` (as accumulated by
        ``RepairableRBD.availability``).
    system_failures : int
        The total number of system failures.

    Returns
    -------
    dict
        Node -> share of the system failures, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     failure_criticality_index_per_system_failures as per_system,
    ... )
    >>> counts = {"a": {"system_failures": 3}, "b": {"system_failures": 1}}
    >>> per_system(counts, 4)
    {'a': 0.75, 'b': 0.25}
    """
    fci = {}
    for node in FCI.keys():
        if system_failures == 0:
            # No system failures occurred (e.g. a redundant node was forced
            # working, or the system was highly reliable over the simulated
            # window), so no node can be credited with causing one.
            fci[node] = 0
        else:
            fci[node] = FCI[node]["system_failures"] / system_failures
    return fci


def failure_criticality_index_per_component_failures(FCI):
    """Share of each node's own failures that caused a system failure.

    For each node: the number of system failures it caused (its failures
    that took the system from up to down) divided by its number of
    failures, or 0 if it never failed.

    Parameters
    ----------
    FCI : dict
        Node -> counts, with the system failures the node caused under
        ``"system_failures"`` and its own failures under
        ``"component_failures"`` (as accumulated by
        ``RepairableRBD.availability``).

    Returns
    -------
    dict
        Node -> share of the node's failures, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     failure_criticality_index_per_component_failures as per_component,
    ... )
    >>> per_component({"a": {"system_failures": 3, "component_failures": 4}})
    {'a': 0.75}
    """
    fci = {}
    for node in FCI.keys():
        try:
            fci[node] = (
                FCI[node]["system_failures"] / FCI[node]["component_failures"]
            )
        except ZeroDivisionError:
            # If there were no component failures then there were no
            # system failures caused by that node
            fci[node] = 0
    return fci


def restoration_criticality_index_by_system(RCI, system_restorations):
    """Share of all system restorations that each node caused.

    A node causes a system restoration when its own restoration takes the
    system from down to up. For each node this is the number of system
    restorations it caused divided by ``system_restorations``; every node
    gets 0 when there were no system restorations.

    Parameters
    ----------
    RCI : dict
        Node -> counts, with the number of system restorations the node
        caused under ``"system_restorations"`` (as accumulated by
        ``RepairableRBD.availability``).
    system_restorations : int
        The total number of system restorations.

    Returns
    -------
    dict
        Node -> share of the system restorations, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     restoration_criticality_index_by_system as by_system,
    ... )
    >>> by_system({"a": {"system_restorations": 1}}, 2)
    {'a': 0.5}
    """
    rci = {}
    for node in RCI.keys():
        if system_restorations == 0:
            # No system restorations occurred, so no node can be credited with
            # causing one.
            rci[node] = 0
        else:
            rci[node] = RCI[node]["system_restorations"] / system_restorations
    return rci


def restoration_criticality_index_by_component(RCI):
    """Share of each node's own restorations that restored the system.

    For each node: the number of system restorations it caused (its
    restorations that took the system from down to up) divided by its
    number of restorations, or 0 if it was never restored.

    Parameters
    ----------
    RCI : dict
        Node -> counts, with the system restorations the node caused under
        ``"system_restorations"`` and its own restorations under
        ``"component_restorations"`` (as accumulated by
        ``RepairableRBD.availability``).

    Returns
    -------
    dict
        Node -> share of the node's restorations, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     restoration_criticality_index_by_component as by_component,
    ... )
    >>> counts = {"system_restorations": 1, "component_restorations": 4}
    >>> by_component({"a": counts})
    {'a': 0.25}
    """
    rci = {}
    for node in RCI.keys():
        try:
            rci[node] = (
                RCI[node]["system_restorations"]
                / RCI[node]["component_restorations"]
            )
        except ZeroDivisionError:
            # If there were no component restorations then there were no
            # times the system was restored by restoring this node.
            rci[node] = 0
    return rci


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


class RepairableRBD(RBD):
    """A reliability block diagram of repairable components.

    Each component alternates between working and failed: it fails after a
    time drawn from its reliability model, and is then repaired, as good
    as new, after a time drawn from its repairability model. Components fail
    and are repaired independently of one another and of the system state
    (unless ``repair_crews`` makes them wait for a crew), and a component
    keeps running (and can fail) while the system is down. The system is up
    whenever its working components connect the input node to the output
    node.

    Two kinds of analysis are offered:

    - Exact long-run (steady-state) metrics, built from each component's
      availability ``MTTF / (MTTF + MTTR)`` and failure frequency
      ``1 / (MTTF + MTTR)``: ``mean_availability``, ``node_availability``,
      ``system_failure_frequency``, ``mean_up_time``, ``mean_down_time``,
      ``mean_time_between_failures``, ``expected_cost_rate``,
      ``capacity_distribution`` (with node capacities) and the importance
      measures. A component under age or block replacement enters through
      its renewal cycle instead (see ``node_availability``). With fewer
      ``repair_crews`` than components, the long-run values come from the
      Markov chain of the components and the repair queue, for exponential
      components (see ``mean_availability``).
    - Monte-Carlo simulation of a finite window ``[0, t_simulation]`` that
      starts with every component working: ``availability`` (availability
      over time, criticality measures and, with node capacities, the
      capacity over time and the fraction of a demand delivered) and
      ``cost`` (the distribution of the window's cost).

    Times are in the time unit of the component models, and costs in the
    currency they are given in.

    Parameters
    ----------
    edges : Iterable[tuple[Hashable, Hashable]]
        The directed edges of the diagram, e.g.
        ``[("s", "a"), ("a", "b"), ("b", "t")]`` for ``"a"`` and ``"b"`` in
        series between the input node ``"s"`` and the output node ``"t"``.
    components : dict
        One entry per node other than the input and output nodes, keyed by
        node name (the input and output nodes never fail: a component given
        for one is refused, as its edge to the output node, or from the
        input node, is most likely missing). Each is one of:

        - A spec dict with ``"reliability"`` (a time-to-failure model, such
          as a fitted surpyval distribution or ``MixtureModel``, a fitted
          Wiener or gamma degradation process (its time to the threshold),
          or a model surpyval fits to a repairable unit's failures,
          ``CrowAMSAA``, ``Duane``, ``HPP`` or ``GeneralizedRenewal``,
          which is the life and ``"repair"`` it is:
          see the guide's imperfect repair) and ``"repairability"`` (a
          time-to-repair model, or ``"instant"`` for repair in zero time:
          the component still fails, and any repair or replace cost is
          charged, but it is never down), plus optional costs.
          ``"repair_cost"`` and ``"replace_cost"`` are charged at every
          failure of the component: each is a number, or a distribution of
          the cost (anything with ``qf`` and ``mean``, such as a fitted
          surpyval model) drawn afresh at each failure. ``"downtime_cost"``
          is a number charged per unit time the component is down, whether
          or not the system is. A cost left out, None or 0 prices nothing.
          ``"preventive"`` schedules preventive maintenance: a dict with an
          ``"interval"`` and optional ``"policy"``, ``"duration"`` and
          ``"cost"``. Under ``"policy": "age"`` (the default) the unit is
          replaced ``interval`` after it was last put into service as new
          (at time 0, or at the end of a repair or of the last preventive
          action), unless it fails first; under ``"block"`` at every
          multiple of ``interval``, whatever its age, unless it is down
          then. Under ``"condition"`` it is inspected at every multiple of
          ``interval`` while it is up, in no time, and replaced if it is
          then more likely than ``"threshold"`` (a probability) to fail
          before the next inspection, given its age ``a``:
          ``1 - R(a + interval) / R(a)``; or, for a life that is a
          degradation process, with ``"level"`` in place of
          ``"threshold"``, if its measured level is then at or past
          ``"level"`` (simulated only; see the guide's replacement on
          condition). Each inspection is charged
          ``"inspection_cost"``, a number or a distribution drawn afresh
          each time. The replacement takes a time drawn from ``"duration"``, a
          time-to-maintain model, during which the unit is down (a planned
          outage); ``"instant"`` (the default) takes no time, renewing the
          unit in place. Each replacement is charged ``"cost"``, a number
          or a distribution drawn afresh each time. The unit comes back as
          new: a failure it had yet to reach never happens. An
          ``interval`` of ``inf`` never maintains. Under ``"age"``, an
          ``"opportunity"`` (an age from 0 to ``interval``) renews the
          unit early, from that age, at a stop of its maintenance group
          (see ``"group"``), as its scheduled replacement would.
          ``"inspection"`` makes the component's failures *hidden*: a
          failure takes it down, but nobody knows until an inspection (a
          proof test) finds it, and only then does its repair start. It
          is a dict with an ``"interval"`` and optional ``"duration"``,
          ``"cost"``, ``"offset"``, ``"coverage"`` and ``"full_test"``: the
          component is inspected at every multiple of the (positive,
          finite) interval, the first at the interval, or from its
          ``"offset"`` (the time of its first test, at least 0 and less than
          the interval: tests of redundant components staggered). An offset
          of 0, the default, is no offset, with no test at 0: a positive one
          puts a test there, near the start, so an offset just above 0 adds
          one test (one within a billionth of the interval is taken as
          0). A test finds a failure with
          probability ``"coverage"`` (by default 1); a failure it misses
          stays hidden until a full test, every ``"full_test"`` (required
          with a coverage below 1, a whole multiple of the interval, from
          the offset), which finds every failure. The test takes a time
          drawn from ``"duration"``, a time-to-test model, during which the
          component is off-line (a planned outage) and does not age;
          ``"instant"`` (the default) takes no time. A failure found by a
          test is repaired once the test is done, and the repair and
          replace costs are charged when it is found. An inspection due
          while the component is being repaired is skipped. Each
          inspection is charged ``"cost"``, a number or a distribution
          drawn afresh each time. A component cannot have both a
          ``"preventive"`` schedule and an ``"inspection"``.
          ``"acquisition_cost"`` is the one-off cost of buying the unit, a
          number: it is not a running cost, so it is left out of
          ``expected_cost_rate`` and the simulated costs, and counted by
          ``total_cost`` and ``allocate_redundancy``. ``"priority"`` is the
          component's place in the queue for a repair crew (see
          ``repair_crews``): a number, higher first, by default 0.
          ``"group"`` names the component's maintenance group (see
          ``maintenance_groups``), any hashable value.
          ``"repair"`` makes its repairs imperfect, by Kijima's
          virtual-age models: a dict of a ``"model"``, ``"kijima1"`` or
          ``"kijima2"``, and a restoration factor ``"q"`` in [0, 1]. A
          repair after the unit has operated ``x`` since the last takes
          its virtual age from ``v`` to ``v + q * x`` (Kijima I) or
          ``q * (v + x)`` (Kijima II), and each life is drawn given it:
          ``H(v + X) = H(v) + E``, ``E`` exponential, ``H`` the cumulative
          hazard. ``q = 0`` renews the unit at every repair (the default)
          and ``q = 1`` is minimal repair. ``"replace_after"``, a whole
          number ``N`` (with a ``"repair"`` of ``q`` above 0), replaces the
          unit, as new, at the ``N``-th failure since it was renewed. A
          repair is then charged only its ``"repair_cost"``; a replacement
          its ``"repair_cost"`` and ``"replace_cost"``, and it uses a spare.
          A preventive replacement renews the unit, and age replacement
          counts its operating time since it was renewed. The exact
          methods refuse such a component, and the simulations follow it,
          in Python; but minimally repaired (``q = 1``) in no time, with no
          ``"replace_after"``, maintenance or tests, it is up throughout
          and fails ``H(t)`` times by ``t`` on average (``H`` its life's
          cumulative hazard), which the values over a window from new
          (``expected_failures`` and the others) take in exactly.
          ``"duty"``, a fraction ``d`` in (0, 1], is the share of the time
          the component operates, for a ``"reliability"`` fitted in
          operating time: it ages only while it operates, so its life on
          the diagram's clock is its operating life over ``d``
          (``R(d t)``), which every method then takes; its repairs,
          maintenance and tests stay on the clock. The life must be a
          surpyval parametric distribution of a time.
          ``"standby"`` makes the node a standby group of identical units,
          each failing and repaired as ``"reliability"`` and
          ``"repairability"`` say: a dict of ``"units"`` (by default 2),
          ``"k"`` (how many must operate, by default 1, fewer than
          ``"units"``), ``"dormancy_factor"`` (how fast a spare ages, as a
          fraction of an operating unit: 0, the default, for cold standby,
          1 for hot) and ``"switching_probability"`` (that switching a spare
          in works, by default 1). The group is up while ``k`` units
          operate. When one fails, the spare that has waited longest is
          switched in; a failed switch leaves the position empty until a
          repaired unit fills it. A spare that fails in standby is found at
          once. Each failed unit is repaired on its own, a job for the
          repair crews at the group's ``"priority"``, and then fills an
          empty position or waits as a spare. ``"repair_cost"`` and
          ``"replace_cost"`` are charged at each unit's failure, and
          ``"downtime_cost"`` while the group is down. A group takes no
          ``"preventive"`` or ``"inspection"`` schedule, and no
          ``"instant"`` repair. Its long-run values are exact from its
          Markov chain when its units' lives and repair times are
          exponential; it is simulated in Python.
        - A [`NonRepairable`][repyability.NonRepairable], pairing a
          reliability model with a time-to-replace model. Each node gets
          its own copy, so one object can be given for several identical
          nodes.
        - A nested ``RepairableRBD``, which acts as one node that is up
          while its own system is up. In a simulation it runs its own
          simulation, on the same clock. It is not copied, so give each
          node its own object: nodes sharing one would share its
          simulation state, and ``availability`` then typically raises a
          ValueError. Its own costs are not counted in this RBD's costs.
        - [`PerfectReliability`][repyability.PerfectReliability] itself
          (or a spec whose ``"reliability"`` it is, with no costs or
          maintenance), for a junction: a node that never fails, such as a
          k-out-of-n vote point. It is no component: the analyses
          leave it out, the simulations draw nothing for it, and it passes
          whatever reaches it, up to a ``capacity`` if it is given one.
    k : dict[Hashable, int], optional
        k-out-of-n nodes, as ``{node: k}``: the node passes only if at
        least ``k`` of the branches entering it are working (and the node
        itself is working). By default None, meaning every k is 1.
    input_node : Hashable, optional
        The input (source) node. By default None: the only node with no
        incoming edge. If given, it must be that node, or a ValueError is
        raised.
    output_node : Hashable, optional
        The output (sink) node. By default None: the only node with no
        outgoing edge. If given, it must be that node, or a ValueError is
        raised.
    on_infeasible_rbd : str, optional
        ``"raise"`` (the default), ``"warn"`` or ``"ignore"``: what to do if
        the diagram is invalid (it has a cycle, a node other than the input
        or output node with no incoming or no outgoing edge, an unusable
        ``k``, or a node with no entry in ``components``): raise a
        ValueError, warn and build the RBD anyway, or build it silently.
        The problems found are recorded in ``structure_check``.
    downtime_cost_rate : float, optional
        Cost per unit time the whole system is down (e.g. lost
        production), by default 0.0 (not priced).
    capacity : dict[Hashable, float or dict], optional
        Each node's capacity, keyed by node name, by default None: the
        throughput it passes while it is up, a positive number in any unit
        (the same for every node), or a dict ``{level: probability}`` of the
        levels it works at and the probability of each while it is up; for
        [`capacity_distribution`][repyability.RepairableRBD.capacity_distribution].
        A node that is down passes nothing. A node with no capacity given
        limits nothing (``inf``), unless its model has capacities of its
        own: a reliability that is a
        [`DegradingNode`][repyability.DegradingNode], whose stages give its
        levels, or a nested ``RepairableRBD`` with capacities, whose
        distribution it then has.
    repair_crews : int, optional
        How many repair crews work on the components, by default None: as
        many as are needed, so no work waits. At most this many jobs
        proceed at once: a job is what brings a component back up (a repair
        or replacement, preventive maintenance that takes time, or a test
        that takes time), and the component is down from when the job falls
        due. A job that finds every crew busy waits, the component down,
        until a crew is free, which then takes the waiting job of the
        highest ``"priority"``, and of those the one that fell due first;
        a crew stays with a job until it is done. Maintenance or a test in
        no time needs no crew. A nested ``RepairableRBD``'s components are
        worked on by its own crews. With fewer crews than components,
        components wait for each other: the simulations (``availability``,
        ``cost``, ``compare``) follow the queue, in Python. The exact
        long-run values then come from a Markov chain of the components'
        states and the queue when their lives and repairs are exponential
        (see ``mean_availability``); the other exact methods, which assume
        independent components, raise ``NotImplementedError``. With at
        least as many, no job waits, and every result is as without crews.
    maintenance_groups : dict, optional
        Options for the maintenance groups the components' ``"group"``
        keys form, by group name, by default None: for each, a dict of
        ``"setup_cost"`` (a number, by default 0) and ``"system_down"``
        (by default False). A member's failure, or its scheduled
        replacement, opens a *stop* of its group, at which every other
        member that is working, has an ``"opportunity"`` and is at least
        that old, is renewed too (opportunistic maintenance), taking its
        own maintenance time; a member whose own failure or replacement is
        due at that instant keeps it. With ``"system_down"``, any outage
        of the system is a stop of the group as well. The set-up cost is
        charged once per stop: once for all that starts at one instant.
        The simulations count each component's early renewals
        (``AvailabilityResult.opportunistic_renewals``) and charge the
        set-ups under ``"setup"``. Group members cannot have hidden
        failures or be standby groups. The exact methods refuse a
        component that can be renewed early; with none, a group's set-up
        is charged at each failure and each preventive replacement of a
        member in ``expected_cost_rate``. Simulated in Python.
    ccf_groups : list of CCFGroup, optional
        Common-cause groups (see [`CCFGroup`][repyability.CCFGroup]), by
        default none: identical components (the same life and repair
        models) that fail together. A group's model (a ``BetaFactor`` or
        ``MGL``) splits the members' failure rate between causes, each
        member's own and shared ones, and each cause fails the members it
        names that are up, at once (whatever the model's ``basis``: a
        repairable component's failures are a rate). Each member alone
        still fails at its rate, so its own values are as without the
        group; the system's long-run values (``mean_availability``,
        ``mean_unavailability``, ``system_failure_frequency``, MTBF, MUT,
        MDT, the cost rate and the interval choices built on them) take
        the group in exactly, from a Markov chain of which members are
        down together. The members need exponential lives, and either
        hidden failures (an ``"inspection"``, with instant tests and
        repairs and one coverage for the group; a test finds a cause's
        failures alike, and each member is found by its own tests) or
        revealed ones with exponential repairs. The importance measures
        take the groups in too, a member's conditioned on its state at
        each time, and so do the allocations (a ``BetaFactor`` member's
        copies join its group; the availability allocations keep the
        members' availability) and the values over time from new, each
        group's chain followed from every member up. The simulations draw
        each cause as a Poisson process, failing the members it names that
        are up; they need exponential lives alone.


    Attributes
    ----------
    components : dict
        Node name -> the model the node is simulated with: a
        ``NonRepairable`` (built from the spec dict, or a copy of the one
        given) or a nested ``RepairableRBD``.
    repairability : dict
        Node name -> its time-to-repair model (an ``ExactEventTime`` at 0
        for ``"instant"``), or None for a nested ``RepairableRBD``.
    costs : dict
        Node name -> ``{cost key: number or distribution}``, for the nodes
        that declare at least one non-zero cost; a preventive-maintenance
        cost is under ``"preventive_cost"`` and an inspection cost under
        ``"inspection_cost"``.
    downtime_cost_rate : float
        The system downtime cost rate.
    capacity : dict
        The capacities given, keyed by node name, as floats.
    repair_crews : int or None
        The number of repair crews (None: as many as are needed).
    acquisition_costs : dict
        Node name -> the one-off cost of buying the unit, for the nodes that
        declare a non-zero ``"acquisition_cost"``.
    input_node : Hashable
        The input node.
    output_node : Hashable
        The output node.
    nodes : list
        The names of the nodes other than the input and output nodes.
    structure_check : dict
        The results of the structural checks made at construction.
    COST_KEYS : tuple[str, ...]
        The optional cost keys of a spec dict: ``"repair_cost"``,
        ``"replace_cost"`` and ``"downtime_cost"``.
    PER_FAILURE_COST_KEYS : tuple[str, ...]
        The cost keys charged per failure, which may be distributions:
        ``"repair_cost"`` and ``"replace_cost"``.
    COMPONENT_SPEC_KEYS : tuple[str, ...]
        Every key a spec dict may carry.
    PREVENTIVE_KEYS : tuple[str, ...]
        The keys of a ``"preventive"`` spec: ``"interval"``, ``"policy"``,
        ``"duration"`` and ``"cost"``, a ``"condition"`` policy's
        ``"threshold"`` and ``"inspection_cost"``, and an ``"age"``
        policy's ``"opportunity"``.
    GROUP_KEYS : tuple[str, ...]
        The keys of a maintenance group's options: ``"setup_cost"`` and
        ``"system_down"``.
    REPAIR_MODELS : tuple[str, ...]
        The imperfect repair models a ``"repair"`` can name:
        ``"kijima1"`` and ``"kijima2"``.
    INSPECTION_KEYS : tuple[str, ...]
        The keys of an ``"inspection"`` spec: ``"interval"``, ``"duration"``,
        ``"cost"``, ``"offset"``, ``"coverage"`` and ``"full_test"``.
    STANDBY_KEYS : tuple[str, ...]
        The keys of a ``"standby"`` spec: ``"units"``, ``"k"``,
        ``"dormancy_factor"`` and ``"switching_probability"``.

    Raises
    ------
    ValueError
        If a spec dict has an unknown key, or a ``"repairability"`` string
        other than ``"instant"``; if a cost is not a finite, non-negative
        number, a ``"downtime_cost"`` or ``downtime_cost_rate`` is a
        distribution, or a cost distribution has no finite mean or puts
        appreciable probability on a negative cost (its 1e-12 quantile is
        below 0); if a ``"preventive"`` spec is not a dict of its keys with
        a positive ``interval``, a ``policy`` of ``"age"``, ``"block"`` or
        ``"condition"`` and a ``duration`` that is a model or
        ``"instant"``; if an ``"inspection"`` spec is not a dict of its
        keys with a positive, finite ``interval``, a ``duration`` that
        is a model or ``"instant"``, an ``offset`` from 0 to less than the
        interval, a ``coverage`` from 0 to 1 and, with a coverage below 1,
        a ``full_test`` that is a whole multiple of the interval, or a
        component has both; if a
        ``"standby"`` spec is not a dict of its keys with whole numbers
        ``units`` above ``k`` of at least 1, and a ``dormancy_factor`` and
        ``switching_probability`` in [0, 1], or its component has a
        schedule or instant repair; if a reliability model is not a
        surpyval parametric model or a ``StandbyModel``, or is a surpyval
        non-parametric one (fit a parametric distribution in surpyval);
        if ``input_node`` or ``output_node`` is not in the diagram, or is
        not its source or sink;
        if ``on_infeasible_rbd`` is not ``"raise"``, ``"warn"`` or
        ``"ignore"``; if the diagram is invalid and ``on_infeasible_rbd``
        is ``"raise"``; if a capacity is not a positive number or is for
        the input or output node or a node not in the diagram; if
        ``repair_crews`` is not a whole number of at least 1, or a
        ``"priority"`` is not a finite number; or if an ``"opportunity"``
        is not an age from 0 to an age policy's interval, is given to a
        component in no group, or a group's member has hidden failures or
        is a standby group, or ``maintenance_groups`` names a group no
        component is in, or has options other than a non-negative
        ``"setup_cost"`` and a boolean ``"system_down"``; or if a
        ``"repair"`` is not a dict of a known ``"model"`` and a ``"q"`` in
        [0, 1], a ``"replace_after"`` is not a whole number of at least 1
        given with a ``q`` above 0, or an imperfectly repaired component
        is a standby group, is replaced on condition, or has a lifetime
        model with no ``Hf`` and ``qf``; or if ``ccf_groups`` holds
        anything but ``CCFGroup`` instances, a member that is not a
        component (or is a nested RBD or a standby group), a component in
        two groups, or a group of components with different life or
        repair models.
    TypeError
        If a component is not a spec dict, a ``NonRepairable`` or a
        ``RepairableRBD`` (a ``Repairable``, which models imperfect repair,
        cannot be a node).
    KeyError
        If a spec dict has no ``"reliability"`` or no ``"repairability"``.

    Examples
    --------
    Two pumps in parallel, each failing on average every 10 hours and
    taking 1 hour on average to repair:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> pump = {
    ...     "reliability": surv.Exponential.from_params([0.1]),
    ...     "repairability": surv.Exponential.from_params([1.0]),
    ... }
    >>> pumps = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": pump, "b": pump},
    ... )
    >>> round(pumps.mean_availability(), 4)  # 1 - (1 / 11) ** 2
    0.9917
    >>> result = pumps.availability(t_simulation=50, mc_samples=200, seed=0)
    >>> window = result.n_simulations * result.time_simulated_to
    >>> round(float(result.system_uptime) / window, 4)  # simulated
    0.9914

    The pair as one node of a larger system, in series with a valve that
    is replaced instantly at a cost of 250 per failure, while lost
    production costs 1000 per hour of system downtime:

    >>> plant = RepairableRBD(
    ...     [("s", "pumps"), ("pumps", "valve"), ("valve", "t")],
    ...     {
    ...         "pumps": pumps,
    ...         "valve": {
    ...             "reliability": surv.Weibull.from_params([500.0, 2.0]),
    ...             "repairability": "instant",
    ...             "replace_cost": 250.0,
    ...         },
    ...     },
    ...     downtime_cost_rate=1000.0,
    ... )
    >>> round(plant.mean_availability(), 4)  # the valve is never down
    0.9917
    >>> round(plant.expected_cost_rate(), 2)  # cost per hour, long run
    8.83

    A ``NonRepairable`` can stand in for a spec dict, and one object can
    serve several nodes, since each node gets its own copy:

    >>> from repyability import NonRepairable
    >>> unit = NonRepairable(
    ...     surv.Exponential.from_params([0.1]),
    ...     surv.Exponential.from_params([1.0]),
    ... )
    >>> same = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": unit, "b": unit},
    ... )
    >>> round(same.mean_availability(), 4)
    0.9917
    >>> same.components["a"] is same.components["b"]
    False
    """

    COST_KEYS = _spec.COST_KEYS
    PER_FAILURE_COST_KEYS = _spec.PER_FAILURE_COST_KEYS
    COMPONENT_SPEC_KEYS = _spec.COMPONENT_SPEC_KEYS
    REPAIR_MODELS = _spec.REPAIR_MODELS
    #: The curves kept while an analysis shares them (see
    #: ``_sharing_curves``); None otherwise.
    _curve_memo: Optional[dict] = None
    PREVENTIVE_KEYS = _spec.PREVENTIVE_KEYS
    PREVENTIVE_POLICIES = _spec.PREVENTIVE_POLICIES
    GROUP_KEYS = _spec.GROUP_KEYS
    INSPECTION_KEYS = _spec.INSPECTION_KEYS
    STANDBY_KEYS = _spec.STANDBY_KEYS
    _PER_ACTION_COST_KEYS = _spec._PER_ACTION_COST_KEYS

    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        components: dict[Any, Any],
        k: Optional[dict[Any, int]] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
        downtime_cost_rate: float = 0.0,
        capacity: Optional[dict[Any, float]] = None,
        repair_crews: Optional[int] = None,
        maintenance_groups: Optional[dict[Any, dict]] = None,
        ccf_groups: Optional[Sequence[Any]] = None,
    ):
        _check_on_infeasible_rbd(on_infeasible_rbd)
        # Capture the constructor inputs verbatim (before any mutation) so the
        # RBD can be faithfully serialised via to_dict()/to_json().
        edges = list(edges)
        from repyability.rbd._processes import as_spec
        from repyability.rbd.ccf import as_groups

        # PerfectReliability() stands for the class (#232), given alone or
        # as a spec's life; a fitted process as a spec's life is the life
        # and repair it is (#269).
        components = {
            name: (
                as_spec(
                    name,
                    {
                        **component,
                        "reliability": perfect_class(component["reliability"]),
                    },
                )
                if isinstance(component, dict) and "reliability" in component
                else perfect_class(component)
            )
            for name, component in components.items()
        }
        ccf_groups = as_groups(ccf_groups) or None
        self._init_args = {
            "edges": [tuple(e) for e in edges],
            "components": dict(components),
            "k": dict(k) if k else None,
            "input_node": input_node,
            "output_node": output_node,
            "on_infeasible_rbd": on_infeasible_rbd,
            "downtime_cost_rate": downtime_cost_rate,
            "capacity": dict(capacity) if capacity else None,
            "repair_crews": repair_crews,
            "maintenance_groups": (
                dict(maintenance_groups) if maintenance_groups else None
            ),
            "ccf_groups": list(ccf_groups) if ccf_groups else None,
        }
        # A component that operates part of the time takes its life on
        # the calendar from here on; the spec as given keeps its life in
        # operating time, for its levers, draws and saving.
        components = _on_duty(components)
        self.repair_crews = _validate_crews(repair_crews)
        # Each component's place in the queue for a crew (higher first).
        self._priority: dict[Any, float] = {}
        self.downtime_cost_rate = _spec._validate_cost(
            "<system>", "downtime_cost_rate", downtime_cost_rate
        )
        # Per-node cost fields, pulled out of the component specs: a number,
        # or (per-failure costs only) a distribution. Only nodes that declare
        # at least one non-zero cost appear here.
        self.costs: dict[Any, dict[str, Any]] = {}
        # Scheduled preventive maintenance, by node (only schedules that
        # maintain: an infinite interval never does).
        self._preventive: dict[Any, _Preventive] = {}
        # Periodic inspection, by node: the nodes whose failures are hidden.
        self._inspection: dict[Any, _Inspection] = {}
        # Standby groups, by node: the nodes that are groups of units.
        self._standby: dict[Any, _Standby] = {}
        # One-off purchase costs, by node (only non-zero ones).
        self.acquisition_costs: dict[Any, float] = {}
        # Each component's maintenance group, by node (see _maintenance).
        self._member_group: dict[Any, Hashable] = {}
        # Components repaired imperfectly, by node (see _ImperfectComponent).
        self._imperfect: dict[Any, _Imperfect] = {}
        # The base class checks the names given components against the
        # edges; a component for a name in no edge is not part of the
        # diagram (the structure check reports it).
        self._models_given = list(components)
        in_edges = {node for edge in edges for node in edge}
        components = {
            name: component
            for name, component in components.items()
            if name in in_edges
        }
        # Junctions: nodes that never fail, such as a k-out-of-n vote point
        # (#182). They are no components: the structure is folded with them
        # working (see RBD._decomposition).
        self._junction_nodes = frozenset(
            name
            for name, component in components.items()
            if _is_junction(name, component)
        )
        self._perfect_given = self._junction_nodes
        components = {
            name: component
            for name, component in components.items()
            if name not in self._junction_nodes
        }
        reliability = {}
        repairability = {}
        for name, component in components.items():
            if isinstance(component, dict):
                _spec._validate_component_spec(name, component)
                node_costs = {}
                for key in self.COST_KEYS:
                    if component.get(key) is None:
                        continue
                    cost = _spec._validate_component_cost(
                        name, key, component[key]
                    )
                    # A cost of 0 prices nothing, so it is left out.
                    if not (isinstance(cost, float) and cost == 0.0):
                        node_costs[key] = cost
                if component.get("preventive") is not None:
                    schedule, cost, inspecting = _spec._validate_preventive(
                        name, component["preventive"], component["reliability"]
                    )
                    if cost is not None:
                        node_costs["preventive_cost"] = cost
                    if np.isfinite(schedule.interval):
                        self._preventive[name] = schedule
                        if inspecting is not None:
                            node_costs["inspection_cost"] = inspecting
                if component.get("inspection") is not None:
                    if name in self._preventive:
                        raise ValueError(
                            f"Component {name!r} has both a preventive "
                            "schedule and an inspection; give it one or the "
                            "other."
                        )
                    inspection, cost = _spec._validate_inspection(
                        name, component["inspection"]
                    )
                    if cost is not None:
                        node_costs["inspection_cost"] = cost
                    self._inspection[name] = inspection
                if component.get("priority") is not None:
                    self._priority[name] = _validate_priority(
                        name, component["priority"]
                    )
                if component.get("standby") is not None:
                    self._standby[name] = _spec._validate_standby(
                        name, component
                    )
                if component.get("group") is not None:
                    self._member_group[name] = component["group"]
                imperfect = _spec._validate_imperfect(self, name, component)
                if imperfect is not None:
                    self._imperfect[name] = imperfect
                if component.get("acquisition_cost") is not None:
                    acquisition = _spec._validate_cost(
                        name, "acquisition_cost", component["acquisition_cost"]
                    )
                    if acquisition:
                        self.acquisition_costs[name] = acquisition
                if node_costs:
                    self.costs[name] = node_costs
                repair_model = component["repairability"]
                if isinstance(repair_model, str):
                    # "instant": repaired in zero time. The component still
                    # *fails* (failure events fire, and any repair/replace
                    # cost is charged), but each outage has zero length, so
                    # it contributes no downtime and its availability is 1.
                    # The convenient assumption when repairs are much faster
                    # than the timescale being studied, or when no
                    # repair-time data exists.
                    if repair_model != "instant":
                        raise ValueError(
                            f"Component {name!r}: unknown repairability "
                            f"{repair_model!r}. Pass a fitted time-to-repair "
                            "model, or the string 'instant' for repair in "
                            "zero time."
                        )
                    repair_model = ExactEventTime.from_params(0)
                # By the spec's own keys, not the NonRepairable's (#233).
                no_distribution(
                    component["reliability"],
                    f"The reliability of component {name!r}",
                )
                no_distribution(
                    repair_model, f"The repairability of component {name!r}"
                )
                try:
                    components[name] = NonRepairable(
                        component["reliability"], repair_model
                    )
                except (TypeError, ValueError) as error:
                    raise type(error)(f"Component {name!r}: {error}") from None
                reliability[name] = component["reliability"]
                repairability[name] = repair_model
            elif isinstance(component, RepairableRBD):
                reliability[name] = component
                repairability[name] = None
            elif isinstance(component, NonRepairable):
                # A NonRepairable carries its simulation state (whether it
                # fails or is repaired next), so each node gets its own copy:
                # one object can then be given for several identical parts,
                # here or in a nested RBD.
                components[name] = copy(component)
                reliability[name] = component.reliability
                repairability[name] = component.time_to_replace
            else:
                raise TypeError(_spec._unknown_component(name, component))
        refuse_nonparametric(
            {
                name: spec
                for name, spec in self._init_args["components"].items()
                if not isinstance(spec, RepairableRBD)
            }
        )

        super().__init__(
            edges,
            None,
            k,
            input_node,
            output_node,
            on_infeasible_rbd,
            capacity=capacity,
        )

        # Every intermediate graph node needs a component definition (the
        # input/output nodes do not): the base class has checked, and
        # reported the missing ones, rather than a KeyError mid-simulation.
        missing = list(self.structure_check["nodes_with_no_model"])
        self.structure_check["is_missing_components"] = bool(missing)
        self.structure_check["nodes_with_no_component"] = missing

        for node, component in components.items():
            if _never_settles(component):
                raise ValueError(
                    f"Component {node!r} fails at once and is repaired at "
                    "once, so a simulation would change its state without "
                    "end: give its life or its repair some length."
                )
        self.components = components
        self.repairability = copy(repairability)
        self._maintenance = _spec._validate_groups(self, maintenance_groups)
        self.ccf_groups = _spec._validate_ccf_groups(self, ccf_groups)

    #: The junctions are folded out of the structure (see ``RBD``).
    _FOLDS_JUNCTIONS = True
    _SIMULATE_INSTEAD = (
        "availability, cost and simulate_timelines simulate it (in Python)."
    )

    def _junctions(self) -> frozenset:
        """The junctions: nodes given ``PerfectReliability``, which never
        fail (see ``RBD._junctions``)."""
        return getattr(self, "_junction_nodes", frozenset())

    def _repr_details(self) -> List[str]:
        """What shapes the repairable diagram, for ``repr``: its
        maintenance, tests, standby groups, nested RBDs and repair crews."""
        counts = [
            (len(getattr(self, "_preventive", {})), "maintained"),
            (len(getattr(self, "_inspection", {})), "tested"),
            (len(getattr(self, "_standby", {})), "standby group(s)"),
            (len(getattr(self, "_imperfect", {})), "repaired imperfectly"),
            (
                sum(
                    isinstance(c, RepairableRBD)
                    for c in getattr(self, "components", {}).values()
                ),
                "nested RBD(s)",
            ),
        ]
        counts.append(
            (len(getattr(self, "ccf_groups", ())), "common-cause group(s)")
        )
        out = [f"{count} {what}" for count, what in counts if count]
        if getattr(self, "repair_crews", None) is not None:
            out.append(f"{self.repair_crews} repair crew(s)")
        junctions = self._junctions()
        if junctions:
            out.append(
                "junction(s) "
                + ", ".join(repr(n) for n in sorted(junctions, key=str))
            )
        return out

    def _ccf_plan(self, structure) -> "_ccf_modules.Plan":
        """Each common-cause group's owner in ``structure`` (the
        decomposition, or its dual), the smallest module holding its
        members (see ``_ccf_modules.Plan``): once for each structure."""
        cache = self.__dict__.setdefault("_ccf_plans", {})
        entry = cache.get(id(structure))
        if entry is None or entry[0] is not structure:
            entry = cache[id(structure)] = (
                structure,
                _ccf_modules.Plan(structure, self.ccf_groups),
            )
        return entry[1]

    def _ccf_measure(
        self,
        measure: str,
        inputs: tuple,
        weights: Optional[np.ndarray],
        kind: str = "failure",
        fv_type: str = "c",
        method: str = "exact",
    ) -> dict:
        """An importance measure of every node with the common-cause groups
        (``measure``, one of ``_importance_time``'s names), at the points
        ``inputs`` gives (see ``_ccf_tabled``): averaged with ``weights``,
        each ratio a ratio of the averages, or at each point without.

        A node outside the groups is independent of them, so its measures
        come from the system's probabilities with it held working and
        failed, as without groups. A member's state says something of its
        group's, so it is conditioned on: ``A(1_i)`` and ``A(0_i)`` at each
        point are the system's availability given the member up and given
        it down then, from the joint probabilities summed over its group's
        combinations (see ``_ccf_modules.Tabled.joints``). Birnbaum's
        measure is the difference between the two, taken from whichever
        end keeps its precision, the system's unavailability or its
        availability. Fussell-Vesely sums each combination's numerators,
        the nodes being independent given it."""
        one_of("kind", kind, ("failure", "success"))
        if fv_type not in ("c", "p"):
            raise ValueError(
                "fv_type must be either 'c' (cut-set) or 'p' (path-set), "
                f"fv_type={fv_type!r} was given."
            )
        one_of("method", method, ("exact", "rare_event"))
        evaluation = _ccf_groups._ccf_tabled(self, *inputs)
        R_t, Q_t = evaluation.system()

        def mean(value):
            if weights is None:
                return np.broadcast_to(
                    np.asarray(value, dtype=float), R_t.shape
                )
            return np.atleast_1d(weights @ np.asarray(value, dtype=float))

        R, Q = mean(R_t), mean(Q_t)
        out: dict = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            if measure == _importance_time.FUSSELL_VESELY:
                shares = _ccf_groups._ccf_fv_shares(
                    self, evaluation, inputs, fv_type, method
                )
                return {node: mean(shares[node]) / Q for node in self.nodes}
            for node in self.nodes:
                values, joint = evaluation.joints(node)
                works, fails = values["works"], values["fails"]
                if joint:
                    r1, q1 = values["up_ok"] / works, values["down_ok"] / works
                    r0, q0 = (
                        values["up_bad"] / fails,
                        values["down_bad"] / fails,
                    )
                else:
                    r1, q1 = values["up_ok"], values["down_ok"]
                    r0, q0 = values["up_bad"], values["down_bad"]
                change = np.where(Q_t <= R_t, q0 - q1, r1 - r0)
                if measure == _importance_time.BIRNBAUM:
                    value = mean(change)
                elif measure == _importance_time.IMPROVEMENT:
                    value = mean(fails * change)
                elif measure == _importance_time.RAW:
                    value = mean(q0) / Q
                elif measure == _importance_time.RRW:
                    value = Q / mean(q1)
                elif kind == "failure":
                    value = mean(fails * change) / Q
                else:
                    value = mean(works * change) / R
                out[node] = value
        return out

    @property
    def has_costs(self) -> bool:
        """Whether any cost has been declared (per-component or system-wide).

        True if some component declares a non-zero ``"repair_cost"``,
        ``"replace_cost"``, ``"downtime_cost"``, or preventive-maintenance
        or inspection ``"cost"`` (a cost distribution always counts), a
        maintenance group has a set-up cost, or ``downtime_cost_rate`` is
        non-zero: whether running the system costs anything. Costs of 0
        price nothing, costs declared inside a nested ``RepairableRBD`` do
        not count, and neither does an
        ``"acquisition_cost"`` (a one-off cost, see ``total_cost``). When
        nothing is priced there is no cost model to evaluate, so the cost
        methods short-circuit rather than doing the work:
        ``expected_cost_rate`` returns 0.0, and ``cost`` and the ``cost`` of
        ``availability``'s result have no running cost, exactly, with the
        acquisition cost beside it, or are None if that is not
        given either.

        Returns
        -------
        bool
            True if anything is priced.
        """
        return (
            bool(self.costs)
            or bool(self.downtime_cost_rate)
            or any(
                group.setup_cost
                for group in getattr(self, "_maintenance", {}).values()
            )
        )

    def _stream_specs(
        self,
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
        specs = {} if specs is None else specs
        states = {} if states is None else states
        complete = True

        def add(path: tuple, kind: int, sampler, expected: float) -> None:
            rows = _streams.first_rows(expected)
            specs[(path, kind)] = _streams.Spec(
                path, kind, sampler, rows, _streams.block_width(rows)
            )

        for name, component in self.components.items():
            path = prefix + (name,)
            if type(component) is RepairableRBD:
                nested = component._stream_specs(
                    t_simulation, path, specs, states.get(name)
                )
                complete = complete and nested[1]
                continue
            failure = repair = None
            if type(component) is NonRepairable:
                # A common-cause group's member draws its own cause's life.
                own = _ccf_groups._own_rate(self, name)
                failure = (
                    stream_sampler(component.reliability)
                    if own is None
                    else _exponential_sampler(own)
                )
                repair = stream_sampler(component.time_to_replace)
            if failure is None or repair is None:
                complete = False
                continue
            arrangement = self._standby.get(name)
            if arrangement is not None:
                # Each unit's lives and repairs, and the group's switches.
                lives, switches = self._standby_draw_counts(name, t_simulation)
                for unit in range(arrangement.units):
                    add(path + (unit,), _streams.FAILURE, failure, lives)
                    add(path + (unit,), _streams.REPAIR, repair, lives)
                if 0.0 < arrangement.switching_probability < 1.0:
                    add(path, _streams.SWITCH, _uniforms, switches)
                continue
            expected = self._expected_draws(name, t_simulation)
            add(path, _streams.FAILURE, failure, expected["failure"])
            add(path, _streams.REPAIR, repair, expected["repair"])
            if _draws_start(states.get(name)):
                add(path, _streams.START, _uniforms, 1.0)
            if name in self._imperfect:
                # A life after each repair, given the unit's virtual age.
                add(path, _streams.AGED, _uniforms, expected["repair"])
            schedule = self._preventive.get(name)
            if schedule is not None and schedule.level is not None:
                # A level and a failure from it at each inspection (#271).
                add(path, _streams.LEVEL, _uniforms, 2.0 * expected["action"])
            inspection = self._inspection.get(name)
            if inspection is not None and inspection.partial:
                # Whether a test finds each failure: one for each failure
                # a test can miss.
                add(path, _streams.TEST, _uniforms, expected["failure"])
            duration = self._duration_model(name)
            if duration is not None:
                sampler = stream_sampler(duration)
                if sampler is None:
                    complete = False
                else:
                    add(path, _streams.DURATION, sampler, expected["action"])
        for cause in _ccf_groups._shared_causes(self):
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
            for node, node_costs in self.costs.items():
                expected = self._expected_draws(node, t_simulation)
                for key, kind in _streams.COST_KINDS.items():
                    cost = node_costs.get(key)
                    if cost is None or isinstance(cost, float):
                        continue
                    count = expected[
                        "action" if key in _ACTION_COST_KEYS else "repair"
                    ]
                    add((node,), kind, _streams.cost_sampler(cost), count)
        return specs, complete

    def _duration_model(self, node):
        """The model of ``node``'s maintenance or test time, or None."""
        schedule = self._preventive.get(node) or self._inspection.get(node)
        return None if schedule is None else schedule.duration

    def _standby_draw_counts(
        self, node, t_simulation: float
    ) -> Tuple[float, float]:
        """Roughly how many lives (and repairs) a simulation draws for each
        unit of a standby group, and how many switches: its operating units
        fail at their rate and its spares at the dormant one (see
        ``_expected_draws``)."""
        arrangement = self._standby[node]
        life = _safe_mean(self.components[node].reliability)
        if not life > 0.0:
            return float("nan"), float("nan")
        active = (
            arrangement.k
            + (arrangement.units - arrangement.k) * arrangement.dormancy_factor
        )
        failures = t_simulation * active / life
        return failures / arrangement.units + 1.0, failures + 1.0

    def _expected_draws(self, node, t_simulation: float) -> Dict[str, float]:
        """Roughly how many times a simulation draws ``node``'s time to
        failure (``"failure"``: one per unit put into service), its repair
        time and per-failure costs (``"repair"``), and its maintenance or
        test time and their costs (``"action"``). NaN where it is not known.
        Only how the draws are computed depends on these (see
        ``_streams``), never what they are."""
        component = self.components[node]
        if node in self._standby:
            # Its units' failures, each a repair charged for.
            _, failures = self._standby_draw_counts(node, t_simulation)
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
        schedule = self._preventive.get(node)
        inspection = self._inspection.get(node)
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
        self,
        t_simulation: float,
        entropy,
        antithetic: bool,
        states: Optional[dict] = None,
    ) -> Tuple[_streams.Plan, bool]:
        """A run's streams, and whether every draw comes from one; the
        components start from ``states``."""
        specs, complete = self._stream_specs(t_simulation, states=states)
        return _streams.Plan(entropy, antithetic, specs), complete

    def _streamed_components(
        self,
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
        for name, component in self.components.items():
            if id(component) not in made:
                arrangement = self._standby.get(name)
                made[id(component)] = (
                    _stand_in(
                        component,
                        run,
                        made,
                        prefix + (name,),
                        self._duration_model(name),
                        self._imperfect.get(name),
                    )
                    if arrangement is None
                    else _standby_draws(
                        component, run, prefix + (name,), arrangement
                    )
                )
                own = _ccf_groups._own_rate(self, name)
                if own is not None and made[id(component)] is component:
                    # Not streamed: its own cause's life all the same.
                    made[id(component)] = self._own_draws(name, own)
            streamed[name] = made[id(component)]
        for cause in _ccf_groups._shared_causes(self):
            path = prefix + cause.struck
            streamed[cause] = _CauseDraws(
                cause.rate,
                run.stream(path, _streams.CAUSE),
                run.stream(path, _streams.CAUSE_TEST),
            )
        return streamed

    def _own_draws(self, node, own: float) -> "_StreamedComponent":
        """A common-cause group member's draws from numpy's global RNG: its
        own cause's life (at the rate ``own``), and its repair, maintenance
        and test times as it would draw them."""
        component = self.components[node]
        return _StreamedComponent(
            _ExponentialDraws(own),
            _ModelDraws(component.time_to_replace),
            model=self._duration_model(node),
        )

    def _keyed_charges(self, run: _streams.Run) -> Tuple[
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
            cost = self.costs[node][key]
            if isinstance(cost, float):
                return _Fixed(cost)
            return run.stream((node,), _streams.COST_KINDS[key])

        failures = {
            node: [
                (
                    _CATEGORIES.index(key.removesuffix("_cost")),
                    charges(node, key),
                )
                for key in self.PER_FAILURE_COST_KEYS
                if key in node_costs
            ]
            for node, node_costs in self.costs.items()
        }
        preventive = {
            node: charges(node, "preventive_cost")
            for node, node_costs in self.costs.items()
            if "preventive_cost" in node_costs
        }
        inspection = {
            node: charges(node, "inspection_cost")
            for node, node_costs in self.costs.items()
            if "inspection_cost" in node_costs
        }
        return failures, preventive, inspection

    def _context(
        self,
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
        states = {} if states is None else states
        plan, complete = self._stream_plan(
            t_simulation, entropy, antithetic, states
        )
        run = _streams.Run(plan, reseed=not complete)
        has_costs = self.has_costs
        nodes = list(self.components)
        return _Context(
            t_simulation=t_simulation,
            working=set(working_nodes),
            broken=set(broken_nodes),
            method=method,
            capacity=capacity,
            run=run,
            sources=self._streamed_components(run),
            charges=self._keyed_charges(run) if has_costs else ({}, {}, {}),
            downtime_cost_rates={
                node: c["downtime_cost"]
                for node, c in self.costs.items()
                if "downtime_cost" in c
            },
            has_costs=has_costs,
            position={node: i for i, node in enumerate(nodes)},
            # Components with no maintenance or inspection that are not
            # nested RBDs: their next event is simply their next draw.
            plain={
                node
                for node, component in self.components.items()
                if node not in self._preventive
                and node not in self._inspection
                and node not in self._standby
                and not isinstance(component, RepairableRBD)
            },
            works=self._decomposition().structure_function(method),
            states=states,
            initial_up=bool(
                self.is_system_working(
                    {c: c not in broken_nodes for c in self.components}, method
                )
            ),
            history=history,
        )

    def expected_cost_rate(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the long-run expected cost per unit time.

        Exact (no simulation), from the steady-state quantities:

        ```text
        rate = downtime_cost_rate * (1 - A_sys)
               + sum_i omega_i * (repair_cost_i + replace_cost_i)
               + sum_i nu_i * preventive_cost_i
               + sum_i inspection_cost_i / tau_i
               + sum_i (1 - A_i) * downtime_cost_i
        ```

        where ``A_sys`` is the system's long-run availability
        (``mean_availability``), ``A_i`` is node i's (see
        ``node_availability``), ``omega_i`` is node i's long-run failure
        frequency, ``1 / (MTTF_i + MTTR_i)``, and ``nu_i`` its frequency of
        preventive replacements. So ``repair_cost`` and ``replace_cost``
        are charged per corrective action (failure), the preventive cost
        per preventive action, and the downtime rates per unit time down.
        A cost given as a distribution enters through its mean. The result
        is in cost per unit of the component models' time.

        Under age replacement at ``T`` a component renews at a failure or
        at a preventive replacement, whichever comes first, in cycles of
        mean length ``C = integral_0^T R + F(T) * MTTR + R(T) * MTTP``
        (``MTTP`` the mean maintenance time), so ``omega_i = F(T) / C`` and
        ``nu_i = R(T) / C``: the renewal-reward rate. With instant repair
        and maintenance the component's own cost rate is
        ``(c_p * R(T) + c_u * F(T)) / integral_0^T R``, as
        ``NonRepairable.cost_rate`` computes.

        Under block replacement at ``T`` the renewals are the block times at
        which the component is up (see ``node_availability``): with ``L``
        the mean time between them and ``N`` the mean number of failures in
        between, ``omega_i = N / L`` and ``nu_i = 1 / L``. With instant
        repair and replacement that is ``(c_p + c_u * M(T)) / T``, ``M`` the
        renewal function of the lives.

        A component with hidden failures is inspected every ``tau_i`` and,
        with a constant failure rate ``lambda``, fails ``(1 - exp(-lambda *
        tau_i)) / tau_i`` times per unit time (at most once per interval).
        With inspection cost ``c_i`` and downtime cost rate ``c_d`` it
        costs ``c_i / tau + c_d * U(tau)``, ``U`` its unavailability (see
        ``node_availability``): frequent tests cost more, and rare ones
        leave failures hidden for longer. The rate is least near ``tau =
        sqrt(2 * c_i / (lambda * c_d))``.

        With limited ``repair_crews``, ``A_sys``, ``A_i`` and ``omega_i``
        come from the crews' Markov chain (see ``mean_availability``): a
        component waiting for a crew is down, and pays its downtime cost,
        and it fails at its constant rate while it is up.

        Every cost is optional and defaults to 0, so any subset can be
        priced; with nothing priced (see ``has_costs``) this is 0.0. Costs
        are undiscounted, and only this RBD's own costs count: costs
        declared inside a nested ``RepairableRBD`` are left out. The
        simulated counterpart is ``cost``, whose ``cost_rate`` converges to
        this value as the window grows.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working: they never fail, so
            they incur no corrective or downtime cost, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed: they incur no corrective
            cost (they never change state) but are down for all time, by
            default None.

        Returns
        -------
        float
            Expected cost per unit time.

        Raises
        ------
        ValueError
            If something is priced and a working/broken node is unknown, is
            the input or output node, or is in both sets. (With nothing
            priced this returns 0.0 without any checks.)
        NotImplementedError
            If something is priced and a component is under block
            replacement with models its exact values do not cover (see
            ``node_availability``), or has hidden failures its numerical
            values do not cover (see ``mean_availability``); or, while a
            component can wait for a repair crew, as for
            ``mean_availability``: simulate it with ``cost``.

        Examples
        --------
        One component with MTTF 10 and MTTR 1 is up 10/11 of the time and
        fails 1/11 times per unit time, so at 100 per repair and 50 per unit
        time of outage the cost rate is ``(100 + 50) / 11``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...             "repair_cost": 100.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=50.0,
        ... )
        >>> round(rbd.expected_cost_rate(), 4)
        13.6364
        """
        return _costs.expected_cost_rate(
            self, working_nodes=working_nodes, broken_nodes=broken_nodes
        )

    @property
    def acquisition_cost(self) -> float:
        """The one-off cost of buying the components: the sum of their
        ``"acquisition_cost"`` (0.0 if none is given). Only this RBD's own
        components count, not those inside a nested ``RepairableRBD``."""
        return _costs.acquisition_cost(self)

    def total_cost(
        self,
        horizon,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        discount_rate: float = 0.0,
    ) -> Union[float, np.ndarray]:
        """Returns the total cost of owning the system for ``horizon``.

        The life-cycle cost: buying the components, then running the system
        for ``horizon`` at the long-run cost rate,

        ```text
        total = acquisition_cost + expected_cost_rate() * horizon
        ```

        with ``acquisition_cost`` the sum of the components'
        ``"acquisition_cost"``. With a ``discount_rate`` ``r`` it is the
        present value: the components are bought at the start, and
        the running costs, spent at a steady rate, are discounted
        continuously, so the horizon counts as ``(1 - exp(-r * horizon)) /
        r``. The running cost is the long-run rate,
        exact over a horizon long compared with the components' cycles;
        ``expected_cost(horizon).total`` is the exact expected cost of
        owning the system from new, and ``cost`` simulates a window from new
        (whose ``CostResult`` gives the same ``acquisition_cost``
        separately).

        Parameters
        ----------
        horizon : float or array-like
            How long the system is owned, in the time unit of the component
            models: non-negative, and finite unless the costs are discounted
            (``inf``, owned for ever). An array of horizons gives a
            total for each.
        working_nodes : Collection[Hashable], optional
            As for ``expected_cost_rate``, by default None.
        broken_nodes : Collection[Hashable], optional
            As for ``expected_cost_rate``, by default None.
        discount_rate : float, optional
            The continuous discount rate per unit time of the component
            models, by default 0 (undiscounted): for an annual rate of 7%
            with models in hours, ``math.log(1.07) / 8760``. A rate that
            discounts the costs away (one per year with models in hours,
            say) is warned of.

        Returns
        -------
        float or numpy.ndarray
            The total cost over the horizon, or over each.

        Raises
        ------
        ValueError
            If ``horizon`` is negative, or infinite without a
            ``discount_rate``; if ``discount_rate`` is not a finite,
            non-negative number; or as for ``expected_cost_rate``.
        NotImplementedError
            As for ``expected_cost_rate``.

        Examples
        --------
        A pump bought for 20,000, failing on average every 1000 hours,
        repaired in 10 at 500 per repair, with lost production at 100 per
        hour, owned for ten years:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "p"), ("p", "t")],
        ...     {
        ...         "p": {
        ...             "reliability": surv.Exponential.from_params([1e-3]),
        ...             "repairability": surv.Exponential.from_params([0.1]),
        ...             "repair_cost": 500.0,
        ...             "acquisition_cost": 20000.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=100.0,
        ... )
        >>> round(rbd.expected_cost_rate(), 4)  # (500 + 100 * 10) / 1010
        1.4851
        >>> round(rbd.total_cost(87600.0))
        150099

        Its present value at 7% a year, the ten years counting as 63,656
        hours:

        >>> import math
        >>> r = math.log(1.07) / 8760  # per hour
        >>> round(rbd.total_cost(87600.0, discount_rate=r))
        114538
        """
        return _costs.total_cost(
            self,
            horizon=horizon,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            discount_rate=discount_rate,
        )

    def _mean_lives(self) -> List[float]:
        """The components' mean lives (their failure-time scales), against
        which an endless horizon's discount rate is judged (#231)."""
        return [
            failure_time_scale(component.reliability)
            for component in self.components.values()
            if isinstance(component, NonRepairable)
        ]

    def spares_demand(
        self,
        horizon: float,
        *,
        nodes: Optional[Collection[Hashable]] = None,
        fleet: int = 1,
        method: str = "exact",
        mc_samples: Optional[int] = None,
        seed=None,
        parts: Optional[Mapping[Hashable, Collection[Hashable]]] = None,
    ) -> Dict[Hashable, SparesDemand]:
        """How many spares each component uses over ``[0, horizon)``, from
        new: the distribution of its replacements, for one system or a
        fleet of ``fleet``; and how many each part that several components
        use (``parts``) takes from its one shelf.

        A component uses a spare at each failure and at each preventive
        replacement; a standby group at each of its units' failures. Its
        replacements are a renewal process: each up time (its life, or the
        replacement age if that comes first) ends in one, then a repair or
        maintenance time, and it starts again as new. So the ``n``-th
        replacement comes after ``n - 1`` whole cycles and an up time, and
        the probability of ``n`` or more by the horizon is that of their
        sum falling before it (a replacement at the horizon itself falls
        after it, as in the simulation). ``method="exact"`` works that out
        on a grid, to about 1e-6, for components with corrective repair
        alone or under age replacement. Under block replacement, a unit's
        next replacement is at its failure or at the next block time,
        whichever comes first (a unit down then skips it), counted block
        interval by block interval on the grid. With hidden failures, the
        replacements fall on the tests that find them, and are counted
        there: exactly, tested and repaired in no time by tests that find
        every failure, and otherwise from the cycle between those tests,
        followed test by test (with tests that can miss a failure, over
        where each cycle starts between the full tests). A fleet's count
        is the sum of its systems', independent and each from new.
        ``method="simulate"`` counts them in ``mc_samples`` simulations of
        the whole system instead, which also covers standby groups, repair
        crews and tests that can last as long as their interval. With
        constant failure rates and instant repair, the counts are Poisson.

        Identical parts often share one shelf: the seals of a station's
        three pumps come from one bin. ``parts={part: [nodes]}`` pools the
        spares of each part's components: their counts are
        independent, so the part's is their sum (by simulation, the sum in
        each simulation). Its components may differ, one under age
        replacement and the others not.

        Parameters
        ----------
        horizon : float
            The time counted over, from new.
        nodes : Collection[Hashable], optional
            The components to count for, by default every component but a
            nested RBD (whose own ``spares_demand`` counts its components'),
            or none when ``parts`` are given.
        fleet : int, optional
            How many systems, by default 1.
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``.
        mc_samples : int, optional
            The number of simulations for ``method="simulate"``, by default
            10_000.
        seed : int, optional
            Seeds the simulations, by default None.
        parts : dict, optional
            ``{part: [nodes]}``: the components whose spares come from one
            shelf, counted together under the part's name. A node is in one
            part at most, and a part is named apart from the components.

        Returns
        -------
        dict[Hashable, SparesDemand]
            Per component, and per part, the distribution of the spares it
            uses, with its ``mean``, ``std``, and ``stock(probability)``:
            the fewest spares that cover the horizon with that probability.

        Raises
        ------
        ValueError
            If ``horizon`` is negative or not finite, ``fleet`` is not a
            whole number of at least 1, ``method`` is unknown, ``nodes`` or
            a part names a node that is not a component, or a nested RBD,
            or ``parts`` is not as above.
        NotImplementedError
            For a part with two members in one common-cause group; and
            with ``method="exact"``, for a standby group, a component with
            hidden failures whose tests can last as long as their interval,
            or while a component can wait for a repair crew (see
            ``repair_crews``), or if more than 2,000 replacements are
            likely: count them by simulation.

        Examples
        --------
        A pump with a constant failure rate of 0.01 an hour, replaced in no
        time, for a fleet of 5 over 1,000 hours: Poisson with mean 50.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([0.01]),
        ...             "repairability": "instant",
        ...         }
        ...     },
        ... )
        >>> demand = rbd.spares_demand(1000.0, fleet=5)["pump"]
        >>> round(demand.mean, 4)
        50.0
        >>> demand.stock(0.95)
        62
        """
        return _spares.spares_demand(
            self,
            horizon=horizon,
            nodes=nodes,
            fleet=fleet,
            method=method,
            mc_samples=mc_samples,
            seed=seed,
            parts=parts,
        )

    def spares_stock(
        self,
        lead_time: float,
        *,
        fill_rate: Optional[float] = None,
        stockout_probability: Optional[float] = None,
        nodes: Optional[Collection[Hashable]] = None,
        fleet: int = 1,
        parts: Optional[Mapping[Hashable, Collection[Hashable]]] = None,
    ) -> Dict[Hashable, SparesStock]:
        """The fewest spares of each component (or each part several
        share, ``parts``) to hold for a fill rate, or to run out no more
        than a fraction of the time, when each spare used is reordered at
        once and arrives ``lead_time`` later (one-for-one, or ``(S - 1,
        S)``, replenishment), in the long run.

        With a stock of ``S`` a spare is on the shelf while fewer than
        ``S`` are on order: those used in the last lead time. The demand is
        the component's replacements (see ``spares_demand``), in the long
        run: from a random time the first comes at the end of what is left
        of an up time, or after what is left of a repair or maintenance and
        an up time, and the next after whole cycles. So the stock-out
        probability is that of ``S`` or more replacements in a lead time
        from a random time, and the fill rate that of fewer than ``S`` in
        the lead time before a replacement. For a fleet, the systems'
        demands add up. Worked out on a grid, to about 1e-6, for components
        with corrective repair alone or under age replacement; and on its
        tests, for a component with hidden failures (from a random time,
        the next replacement is ``j`` tests on with probability ``P(C >=
        j) / S``, ``C`` a cycle's length in tests and ``S`` its mean):
        exactly, tested and repaired in no time by tests that find every
        failure, and otherwise from the cycle followed test by test (with
        tests that can miss a failure, over where each cycle starts between
        the full tests, see ``_spares``).

        Under block replacement every ``T``, with repairs and block
        replacements in no time, each block interval starts with a new
        unit, so the demand repeats every interval, and a lead time's
        depends on where in the interval it starts: it is averaged over
        that, uniform on ``[0, T)`` from a random time, and as the
        replacements fall from a replacement (a failure, at the renewal
        density, or a block replacement), on a grid of the interval, to
        about 1e-6. With repairs or block replacements that take time, a
        unit down at a block time is not replaced there, and an interval
        need not start new: the demand is counted from a typical
        replacement in the long run, a failure at each phase of the
        interval or a block replacement, each followed block interval by
        block interval, the replacements before one as those after it,
        and from a random time by Campbell's formula; on grids of the
        interval, extrapolated, to about 1e-6. A unit dead on arrival is
        refused while its repairs or block replacements may take no time,
        as replacements could then come several at one instant. A fleet's
        systems are taken as on block schedules of their own, out of step
        with each other.

        A part's components draw on one shelf, which needs fewer
        spares than a shelf each: its spares on order are the sum of its
        components' independent ones, and a demand comes from component
        ``i`` with the share ``rate_i / sum(rates)`` of their long-run
        replacement rates, finding ``i``'s as its own demands do and the
        others' as at a random time (see the spares
        [guide](../guide/spares.md#one-shelf-for-interchangeable-parts)).

        Parameters
        ----------
        lead_time : float
            The time a spare takes to arrive once ordered.
        fill_rate : float, optional
            The fraction of demands to meet from the shelf, in ``(0, 1)``.
        stockout_probability : float, optional
            The most fraction of time to be out of stock, in ``(0, 1)``.
            At least one of the two targets must be given; the stock meets
            both.
        nodes : Collection[Hashable], optional
            The components, by default every component but a nested RBD
            (none when ``parts`` are given).
        fleet : int, optional
            How many systems share the stock, by default 1.
        parts : dict, optional
            ``{part: [nodes]}``: the components that draw on one shelf, as
            for ``spares_demand``.

        Returns
        -------
        dict[Hashable, SparesStock]
            Per component, and per part, the stock and what it achieves,
            with the distributions behind them.

        Raises
        ------
        ValueError
            If ``lead_time`` is negative or not finite, neither target is
            given or one is not in ``(0, 1)``, ``fleet`` is not a whole
            number of at least 1, ``nodes`` or a part names a node that is
            not a component, or a nested RBD, or ``parts`` is not as for
            ``spares_demand``.
        NotImplementedError
            For a part with two members in one common-cause group, or two
            under block replacement, a component under block replacement
            whose repairs or block replacements take time or whose life
            may end at 0, with hidden failures whose tests can last
            as long as their interval, a standby group or a life that may
            never end, or while a component can wait for a repair crew, or
            if more than 2,000 replacements are likely in a lead time.

        Examples
        --------
        A pump with a constant failure rate of 0.01 an hour, replaced in no
        time, and 300 hours to get a spare: 3 are on order on average, and
        7 on the shelf meet 96.6% of demands.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([0.01]),
        ...             "repairability": "instant",
        ...         }
        ...     },
        ... )
        >>> stock = rbd.spares_stock(300.0, fill_rate=0.95)["pump"]
        >>> stock.stock
        7
        >>> round(stock.fill_rate, 4)
        0.9665
        """
        return _spares.spares_stock(
            self,
            lead_time=lead_time,
            fill_rate=fill_rate,
            stockout_probability=stockout_probability,
            nodes=nodes,
            fleet=fleet,
            parts=parts,
        )

    def allocate_redundancy(
        self,
        horizon: float,
        *,
        nodes: Optional[Collection[Hashable]] = None,
        trains: Optional[Mapping] = None,
        min_availability: Optional[float] = None,
        max_units: Union[int, Dict[Hashable, int], None] = None,
        method: str = "exact",
        discount_rate: float = 0.0,
    ) -> TotalCostAllocation:
        """Choose the redundancy with the lowest total cost of ownership.

        How many identical copies of each node to fit in active parallel
        (or of each train of nodes to add alongside it, see ``trains``) so
        that owning the system for ``horizon`` costs least: each copy costs
        its ``"acquisition_cost"`` to buy and its running costs (repairs,
        replacements, preventive maintenance, inspections, its own downtime
        cost) to keep, and together the copies save the cost of the system
        being down (``downtime_cost_rate``). That is ``total_cost`` for the
        system with the copies drawn out:

        ```text
        total = sum_i n_i * (a_i + horizon * r_i)
                + horizon * downtime_cost_rate * (1 - A_sys)
                + the cost of the nodes not considered
        ```

        with ``n_i`` copies of node ``i``, each bought for ``a_i`` and
        running at ``r_i`` per unit time, and ``A_sys`` the system's
        long-run availability. With a ``discount_rate`` ``r`` it is the
        present value, as ``total_cost`` gives it: the copies are bought at
        the start and the running costs discounted, ``horizon`` counting as
        ``(1 - exp(-r * horizon)) / r``. The copies fail and are repaired
        independently, so ``n`` copies of a node of long-run availability
        ``A`` are all down a fraction ``(1 - A) ** n`` of the time, and each
        design is scored exactly, with no simulation. (Copies of a
        component with hidden failures are inspected together, and are
        scored over the inspection period, as ``mean_availability`` does.)

        More copies cost more and save less and less downtime, so the total
        is not monotone in them. ``method="exact"`` finds a proven optimum
        from the greedy solution: a design no worse than that spends no more
        on copies than its total, which caps every node's copies, and
        (without ``min_availability``) no node takes a copy that could not
        pay for itself, since the ``k + 1``-th copy of a node of
        unavailability ``U`` can save at most ``horizon *
        downtime_cost_rate * U ** k * (1 - U)``. When every node considered
        is in series with the rest of the system (it lies on every path)
        and has no hidden failures, the system's availability is theirs
        times the rest's, and a dynamic program over the nodes finds the
        optimum however many there are; otherwise a branch and bound over
        the designs does, which suits a handful of nodes.
        ``method="greedy"`` adds or removes one copy at a time while that
        lowers the total: fast, but not guaranteed optimal.

        Parameters
        ----------
        horizon : float
            How long the system is owned, in the time unit of the component
            models: finite and non-negative.
        nodes : Collection[Hashable], optional
            The components that may be given copies, by default every one
            with an ``"acquisition_cost"``. The others stay as they are, and
            their costs are counted once. Nested ``RepairableRBD`` nodes
            cannot be given copies. Given ``trains``, by default none.
        trains : dict, optional
            Trains of components that may be given copies, ``{name:
            [nodes]}``: each a chain in series, its first node fed by any
            nodes, each of the others by the one before it alone, and its
            last feeding one node alone (a pump train of seal, bearing and
            motor, into a vote). A copy of a train is another path
            alongside it, fed as its first node is and feeding the node its
            last feeds, which keeps its ``k``: copies of a train of a
            2-out-of-3 vote make it 2-out-of-4, where copies of its nodes,
            each in parallel with its own, would not. Name one of identical
            trains, as its copies stand in for any of them. A copy costs its
            nodes' acquisition and running costs, and ``units`` counts a
            train and its copies under its name, which is not a node's. A
            node is in one train at most, and not in ``nodes`` as well.
        min_availability : float, optional
            Only consider designs whose long-run availability is at least
            this, in (0, 1), by default no limit.
        max_units : int or dict, optional
            The most copies (at least 1) of every node or train considered
            (an int) or of particular ones (a dict; those it leaves out are
            unlimited), by default unlimited. A node or train whose copies
            cost nothing over the horizon needs one.
        method : str, optional
            ``"exact"`` (the default) or ``"greedy"``. The exact search
            gives up with an explanatory error after examining 500,000
            designs (the dynamic program, after holding 2,000,000 partial
            ones).
        discount_rate : float, optional
            The continuous discount rate per unit time of the component
            models, by default 0 (undiscounted), as for ``total_cost``: the
            design with the lowest present value is chosen.

        Returns
        -------
        TotalCostAllocation
            The chosen ``units`` per node and train, with the design's
            ``total_cost``, ``acquisition_cost``, ``cost_rate`` and
            ``availability`` (see
            [`TotalCostAllocation`][repyability.TotalCostAllocation]).

        Raises
        ------
        ValueError
            If ``horizon``, ``nodes``, ``trains``, ``min_availability``,
            ``max_units``, ``method`` or ``discount_rate`` is invalid; if
            no component has an acquisition cost and neither ``nodes`` nor
            ``trains`` is given;
            if a node's or train's copies cost nothing and are not capped;
            if ``min_availability`` cannot be reached; or if the exact search
            examines more than 500,000 designs.
        NotImplementedError
            If a component is under block replacement with models its exact
            values do not cover (see ``node_availability``), or has hidden
            failures its numerical values do not cover (see
            ``mean_availability``): its long-run values are not known. Or if
            a component can wait for a repair crew (see ``repair_crews``):
            the search assumes that none does. Or if a common-cause group's
            chain does not cover it (see ``ccf_groups``), a member given
            copies is in an ``MGL`` group (its letters are for its group's
            size), or a train holds a member.

        Examples
        --------
        A pump that fails on average every 1000 hours and takes 10 to
        repair, bought for 20,000 and repaired for 500, when an hour without
        pumping costs 100. Over ten years (87,600 hours) a second pump pays
        for itself; a third would not:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([1e-3]),
        ...             "repairability": surv.Exponential.from_params([0.1]),
        ...             "repair_cost": 500.0,
        ...             "acquisition_cost": 20000.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=100.0,
        ... )
        >>> round(rbd.total_cost(87600.0))  # one pump
        150099
        >>> best = rbd.allocate_redundancy(87600.0)
        >>> best.units, round(best.total_cost)
        ({'pump': 2}, 127591)

        Over one year the second pump does not pay for itself:

        >>> rbd.allocate_redundancy(8760.0).units
        {'pump': 1}

        A station needing two of three pump trains, each a pump and its
        motor, when an hour without pumping costs 2,000: is a fourth train
        worth it? Name one train, and its copies join the vote:

        >>> pump = {
        ...     "reliability": surv.Exponential.from_params([1e-3]),
        ...     "repairability": surv.Exponential.from_params([0.1]),
        ...     "repair_cost": 500.0,
        ...     "acquisition_cost": 20000.0,
        ... }
        >>> motor = {
        ...     "reliability": surv.Exponential.from_params([2e-4]),
        ...     "repairability": surv.Exponential.from_params([0.05]),
        ...     "acquisition_cost": 10000.0,
        ... }
        >>> trains = (1, 2, 3)
        >>> station = RepairableRBD(
        ...     [("s", f"pump {i}") for i in trains]
        ...     + [(f"pump {i}", f"motor {i}") for i in trains]
        ...     + [(f"motor {i}", "t") for i in trains],
        ...     {
        ...         **{f"pump {i}": pump for i in trains},
        ...         **{f"motor {i}": motor for i in trains},
        ...     },
        ...     k={"t": 2},
        ...     downtime_cost_rate=2000.0,
        ... )
        >>> round(station.total_cost(87600.0))  # three trains
        319927
        >>> best = station.allocate_redundancy(
        ...     87600.0, trains={"train 1": ["pump 1", "motor 1"]}
        ... )
        >>> best.units, round(best.total_cost)
        ({'train 1': 2}, 295306)
        """
        return _repairable_allocation.allocate_redundancy(
            self,
            horizon=horizon,
            nodes=nodes,
            trains=trains,
            min_availability=min_availability,
            max_units=max_units,
            method=method,
            discount_rate=discount_rate,
        )

    def availability_allocation(
        self,
        target: float,
        method: str = "cost_based",
        *,
        fixed: Optional[Collection[Hashable]] = None,
        weights: Optional[Dict] = None,
        max_availabilities: Optional[Dict] = None,
        feasibility: Optional[Dict] = None,
    ) -> AvailabilityAllocation:
        """What availability each component needs for the system to meet a
        target, and the MTTF or MTTR that gives it.

        Availability allocation: one of the reliability allocation methods
        (see ``cost_based_allocation``), run on the components' long-run
        availabilities (``node_availability``), with the system scored as
        ``mean_availability`` scores it. A component's availability is
        ``MTTF / (MTTF + MTTR)``, so the availability ``A`` allocated to it
        is met by an MTTF of ``MTTR * A / (1 - A)`` at its current MTTR, or
        by an MTTR of ``MTTF * (1 - A) / A`` at its current MTTF (or by any
        pair in the same ratio): both are reported. To choose the cheapest
        pair instead, use ``mttf_mttr_allocation``.

        Only components with corrective repair alone are allocated an
        availability: those that fail and take time to repair, with no
        ``"preventive"`` or ``"inspection"`` schedule. The others keep
        theirs: components with a schedule (whose intervals
        ``optimal_replacement_intervals`` and
        ``optimal_inspection_intervals`` choose), components repaired
        instantly (never down), nested ``RepairableRBD`` nodes, components
        with units that never fail or repairs that may never end, the
        members of common-cause groups (whose MTTF and MTTR are their
        group's; the system is scored over the groups' joint states), and
        the components in ``fixed``. A held component that is inspected or
        block-replaced is up with a probability that varies over its
        schedule, together with the others on the same calendar; it enters
        with that variation, as it does in ``mean_availability``, so that
        the allocation meets the target exactly.

        Parameters
        ----------
        target : float
            The system's long-run availability to reach, in [0, 1].
        method : str, optional
            How to allocate it, from the current availabilities:
            ``"cost_based"`` (the default) for Mettas's cheapest allocation
            (see ``cost_based_allocation``); ``"improvement"`` to scale
            every unavailability by one factor (``improvement_allocation``);
            ``"minimum_effort"``, for a series system, to raise the least
            available components to one level
            (``minimum_effort_allocation``); or ``"equal"`` to give every
            component allocated an availability the same one.
        fixed : Collection[Hashable], optional
            Components that keep their availability.
        weights : dict, optional
            With ``"improvement"`` only: a weight for each component
            allocated an availability (see ``improvement_allocation``).
        max_availabilities : dict, optional
            With ``"cost_based"`` only: the most a component's availability
            can reach. A component without one can approach 1.
        feasibility : dict, optional
            With ``"cost_based"`` only: a component's feasibility, in
            [0, 1), 0.5 without one: the higher, the easier to improve.

        Returns
        -------
        AvailabilityAllocation
            The ``availability`` of every component; for each one allocated
            an availability, the ``mttf`` that gives it at the current MTTR
            and the ``mttr`` that gives it at the current MTTF; and the
            ``system_availability`` with them. The solver's result is
            stored on the RBD as ``res``, as the method run stores it.

        Raises
        ------
        ValueError
            If ``target`` is not in [0, 1] or cannot be reached (the message
            gives how far the system can go); if ``method`` is unknown, or
            an option is given to a method that does not take it; if
            ``fixed`` or an option names a node that is not a component, or
            an option names a component that keeps its availability; if no
            component can be allocated an availability; if
            or if ``"minimum_effort"`` is asked of a system that is not a
            series.
        KeyError
            If ``weights`` has no entry for a component allocated an
            availability.
        NotImplementedError
            As for ``mean_availability``, or if a component can wait for a
            repair crew (see ``repair_crews``): the allocation assumes
            independent components.

        Examples
        --------
        Two pumps in parallel, each with an MTTF of 10 h and an MTTR of
        1 h, in series with a valve with an MTTF of 50 h and an MTTR of
        2 h, are up 95.4% of the time. For 98%, the cheapest allocation
        asks most of the valve, which every path goes through:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def unit(mttf, mttr):
        ...     return {
        ...         "reliability": surv.Exponential.from_params([1 / mttf]),
        ...         "repairability": surv.Exponential.from_params([1 / mttr]),
        ...     }
        >>> plant = RepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {"p1": unit(10, 1), "p2": unit(10, 1), "v": unit(50, 2)},
        ... )
        >>> round(plant.mean_availability(), 4)
        0.9536
        >>> need = plant.availability_allocation(0.98)
        >>> {node: round(a, 4) for node, a in need.availability.items()}
        {'p1': 0.9318, 'p2': 0.9318, 'v': 0.9846}

        The valve needs an MTTF of 128 h at its 2 h repairs, or repairs of
        0.78 h at its 50 h MTTF:

        >>> round(need.mttf["v"]), round(need.mttr["v"], 2)
        (128, 0.78)
        """
        return _repairable_allocation.availability_allocation(
            self,
            target=target,
            method=method,
            fixed=fixed,
            weights=weights,
            max_availabilities=max_availabilities,
            feasibility=feasibility,
        )

    def mttf_mttr_allocation(
        self,
        target: float,
        *,
        levers: str = "both",
        fixed: Optional[Collection[Hashable]] = None,
        max_mttf: Optional[Dict] = None,
        min_mttr: Optional[Dict] = None,
        mttf_feasibility: Optional[Dict] = None,
        mttr_feasibility: Optional[Dict] = None,
    ) -> AvailabilityAllocation:
        """The cheapest MTTFs and MTTRs that meet a system availability
        target.

        A component's availability, ``MTTF / (MTTF + MTTR)``, can be raised
        with a longer MTTF (reliability) or a shorter MTTR
        (maintainability), which cost different amounts. This is Mettas's
        cost-based allocation (see ``cost_based_allocation``) with both
        levers: it finds each component's MTTF and MTTR that meet the target
        at the least total cost, where lowering its failure rate
        ``lambda = 1 / MTTF`` from ``lambda_0`` towards its least,
        ``lambda_min = 1 / max_mttf``, costs

            exp((1 - f) * (lambda_0 - lambda) / (lambda - lambda_min)),

        and cutting its MTTR from ``MTTR_0`` towards its least,
        ``min_mttr``, costs

            exp((1 - f) * (MTTR_0 - MTTR) / (MTTR - min_mttr)),

        each with its own feasibility ``f`` in [0, 1): the lower it is, the
        faster the cost rises. Each cost is 1 while its lever is unused and
        grows without bound towards its limit. Without limits, raising the
        MTTF by a factor ``k`` costs ``exp((1 - f) * (k - 1))``, and so
        does cutting the MTTR by one: the availability depends on the ratio
        of the two alone, so at equal feasibilities both are used alike.
        The cheaper lever is used first, and the dearer one only once the
        cheaper one costs as much at the margin.

        With ``levers="mttr"`` the failure behaviour is held and only the
        repairs are allocated (maintainability allocation): a shorter
        repair is worth most on a component that is often down, so, other
        things equal, the components that fail most often get the shortest
        repairs. With ``levers="mttf"`` only the MTTFs change.

        The components allocated, and those that keep their availability,
        are as for ``availability_allocation``, and the system is scored as
        ``mean_availability`` scores it. It is solved with
        ``scipy.optimize.minimize`` (SLSQP) with exact gradients, from the
        point where every lever has closed the same fraction of its gap to
        its limit, and the design returned meets the target. The solver's
        result is stored on the RBD as ``res``, replacing any earlier one;
        its ``fun`` is the log of the total cost of the levers that may
        change.

        Parameters
        ----------
        target : float
            The system's long-run availability to reach, in [0, 1].
        levers : str, optional
            ``"both"`` (the default), ``"mttr"`` to change only the MTTRs, or
            ``"mttf"`` to change only the MTTFs.
        fixed : Collection[Hashable], optional
            Components that keep their MTTF and MTTR.
        max_mttf : dict, optional
            The most a component's MTTF can reach, at least its current
            MTTF (which holds it). A component without one has no limit.
        min_mttr : dict, optional
            The least a component's MTTR can reach, from 0 (the default for
            a component without one) to its current MTTR (which holds it).
        mttf_feasibility : dict, optional
            The feasibility of raising a component's MTTF, in [0, 1), 0.5
            without one.
        mttr_feasibility : dict, optional
            The feasibility of cutting a component's MTTR, in [0, 1), 0.5
            without one.

        Returns
        -------
        AvailabilityAllocation
            Every component's ``availability``; for each one allocated, its
            ``mttf`` and ``mttr`` in the cheapest design (together); and the
            ``system_availability``. A target the system already meets
            returns the current values.

        Raises
        ------
        ValueError
            If ``target`` is not in [0, 1] or cannot be reached (the message
            gives the most the limits allow: they are only approached, at an
            ever-growing cost); if ``levers`` is unknown; if ``fixed`` or an
            option names a node that is not a component, or an option names
            one that keeps its availability; if a limit is on the wrong side
            of the current value, or a feasibility is outside [0, 1); or if
            no lever can change.
        NotImplementedError
            As for ``mean_availability``, or if a component can wait for a
            repair crew (see ``repair_crews``): the allocation assumes
            independent components.

        Warns
        -----
        UserWarning
            If the solver stops before converging; the design returned
            still meets the target, but may not be the cheapest.

        Examples
        --------
        The pumps and valve of ``availability_allocation``, to 98%: at
        equal feasibilities each component's MTTF rises by the factor its
        MTTR falls by, and the valve's the most:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def unit(mttf, mttr):
        ...     return {
        ...         "reliability": surv.Exponential.from_params([1 / mttf]),
        ...         "repairability": surv.Exponential.from_params([1 / mttr]),
        ...     }
        >>> plant = RepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {"p1": unit(10, 1), "p2": unit(10, 1), "v": unit(50, 2)},
        ... )
        >>> design = plant.mttf_mttr_allocation(0.98)
        >>> {node: round(t, 1) for node, t in design.mttf.items()}
        {'p1': 10.6, 'p2': 10.6, 'v': 85.5}
        >>> {node: round(t, 2) for node, t in design.mttr.items()}
        {'p1': 0.94, 'p2': 0.94, 'v': 1.17}
        >>> round(design.system_availability, 4)
        0.98

        Holding the failure behaviour, only the repairs change:

        >>> repairs = plant.mttf_mttr_allocation(0.98, levers="mttr")
        >>> {node: round(t, 2) for node, t in repairs.mttr.items()}
        {'p1': 0.74, 'p2': 0.74, 'v': 0.78}
        """
        return _repairable_allocation.mttf_mttr_allocation(
            self,
            target=target,
            levers=levers,
            fixed=fixed,
            max_mttf=max_mttf,
            min_mttr=min_mttr,
            mttf_feasibility=mttf_feasibility,
            mttr_feasibility=mttr_feasibility,
        )

    def _allocation_probability(
        self, probabilities: Dict[Any, float]
    ) -> float:
        if "_allocation_calendar" not in self.__dict__:
            return super()._allocation_probability(probabilities)
        return _repairable_allocation._calendar_means(
            self,
            *_repairable_allocation._calendar_arrays(
                self, probabilities, None
            ),
        )[0]

    def _log_odds(
        self, p: Dict[Any, float], q: Dict[Any, float]
    ) -> Tuple[float, Dict[Any, float]]:
        if "_allocation_calendar" not in self.__dict__:
            return super()._log_odds(p, q)
        weights, _, free = self.__dict__["_allocation_calendar"]
        works, fails = _repairable_allocation._calendar_arrays(self, p, q)
        up, down = _repairable_allocation._calendar_means(self, works, fails)
        with np.errstate(divide="ignore"):
            log_odds = float(np.log(up) - np.log(down))
        scale = up * down
        gradient = dict.fromkeys(self.nodes, 0.0)
        if scale:
            ones, zeros = np.ones(len(weights)), np.zeros(len(weights))
            for node in free:
                # The system's availability is linear in the node's, with
                # slope P(down | node down) - P(down | node up).
                if_up = _repairable_allocation._calendar_means(
                    self, {**works, node: ones}, {**fails, node: zeros}
                )[1]
                if_down = _repairable_allocation._calendar_means(
                    self, {**works, node: zeros}, {**fails, node: ones}
                )[1]
                gradient[node] = (if_down - if_up) / scale
        return log_odds, gradient

    def optimal_replacement_intervals(
        self,
        nodes: Optional[Collection[Hashable]] = None,
        *,
        allowed=None,
        min_availability: Optional[float] = None,
        max_cost_rate: Optional[float] = None,
        assume_unlimited_crews: bool = False,
    ) -> MaintenancePlan:
        """Choose the age-replacement intervals for the system as a whole.

        ``NonRepairable.find_optimal_replacement`` chooses one unit's
        replacement age on its own. In a system the components should be
        chosen together: a unit whose failure stops the system is worth
        replacing sooner than one with a standby, and replacing a unit takes
        the system down if its replacement takes time and nothing covers
        for it. This chooses the age-replacement interval of each component
        in ``nodes`` for:

        - the lowest long-run cost rate (``expected_cost_rate``), by
          default;
        - the lowest cost rate that keeps the system's long-run availability
          (``mean_availability``) at least ``min_availability``;
        - or the highest availability within a cost rate of
          ``max_cost_rate``.

        The long-run values are exact (see ``expected_cost_rate``), so the
        choice is too, to the precision of the search: a gradient search
        (SLSQP) over the logarithms of the intervals, from several starting
        points, keeping the best. Each interval ranges from a thousandth to a
        thousand times the component's mean life; one found at the top of
        that range is compared with never replacing the component (an
        interval of ``inf``), which is taken if no worse. The cost rate is
        usually flat near its minimum, so intervals some way from the ones
        found cost almost the same.

        Replacements are often made on a calendar (quarterly, yearly): the
        intervals are then chosen from ``allowed``, as
        ``optimal_inspection_intervals`` chooses tests', every combination
        tried when there are at most 2000 of them, which gives the
        optimum, and a local search (one interval changed at a time, from
        several starting points) made otherwise.

        The plan is the best for the long run. A plant starts with every
        unit new, so the units' first replacements fall due together, and
        where they take the system down together (redundant units whose
        replacements take time) its first years cost more than the plan's
        cost rate says: ``expected_cost`` gives the cost from new, and from
        units of other ages (``state``) a staggered start.

        With limited ``repair_crews`` a component can wait for a crew, and
        the exact long-run values do not hold: the choice is refused unless
        ``assume_unlimited_crews``, which chooses the intervals as if
        every repair started at once. Simulate that plan with the crews,
        ``with_intervals(plan).cost()``, to see what the waiting costs.

        Parameters
        ----------
        nodes : Collection[Hashable], optional
            The components whose intervals to choose, each under age
            replacement (a ``"preventive"`` schedule with ``"policy":
            "age"``; its interval is one of the starting points). By default
            every component under age replacement. The others keep their
            schedules.
        allowed : sequence of float or dict, optional
            The intervals to choose from: one sequence for every node, or a
            dict of a sequence per node; ``inf`` among them is never
            replacing the component. By default None: each interval is
            searched continuously.
        min_availability : float, optional
            The least long-run system availability allowed, in (0, 1).
        max_cost_rate : float, optional
            The highest long-run cost rate allowed: the intervals then give
            the highest availability within it.
        assume_unlimited_crews : bool, optional
            With limited ``repair_crews``, choose the intervals as if every
            repair started at once (the plan's cost rate and availability
            are then those without waiting), by default False: refused.

        Returns
        -------
        MaintenancePlan
            The interval of each component in ``nodes`` (``inf`` for never),
            and the system's cost rate and availability with them.

        Raises
        ------
        ValueError
            If a node is not a component under age replacement; if nothing
            is priced; if both targets are given, or one is out of range;
            if ``allowed`` gives a node no intervals, or one that is not a
            positive number; or if no intervals meet the target (the
            message gives the best they can do).
        NotImplementedError
            If the long-run values are not known exactly (see
            ``expected_cost_rate``), or a component can wait for a repair
            crew and ``assume_unlimited_crews`` is not given.

        Examples
        --------
        A pump that wears out, in series with a pair of them in parallel;
        repairs take about 23 hours and replacements about 7, and the plant
        loses 500 an hour while it is down:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def pump():
        ...     return {
        ...         "reliability": surv.Weibull.from_params([1000, 2.5]),
        ...         "repairability": surv.LogNormal.from_params([3.0, 0.5]),
        ...         "replace_cost": 5000.0,
        ...         "preventive": {
        ...             "interval": 1000.0,
        ...             "duration": surv.Weibull.from_params([8, 3]),
        ...             "cost": 1000.0,
        ...         },
        ...     }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("a", "b1"), ("a", "b2"), ("b1", "t"),
        ...      ("b2", "t")],
        ...     {"a": pump(), "b1": pump(), "b2": pump()},
        ...     downtime_cost_rate=500.0,
        ... )
        >>> plan = plant.optimal_replacement_intervals()
        >>> {node: round(t) for node, t in plan.intervals.items()}
        {'a': 590, 'b1': 497, 'b2': 497}
        >>> round(plan.cost_rate, 2), round(plan.availability, 4)
        (20.12, 0.9803)

        The pump alone in the line is replaced later than those with a
        standby: its replacements stop the plant too.
        """
        return _intervals.optimal_replacement_intervals(
            self,
            nodes=nodes,
            allowed=allowed,
            min_availability=min_availability,
            max_cost_rate=max_cost_rate,
            assume_unlimited_crews=assume_unlimited_crews,
        )

    def optimal_inspection_intervals(
        self,
        nodes: Optional[Collection[Hashable]] = None,
        *,
        allowed=None,
        min_availability: Optional[float] = None,
        max_cost_rate: Optional[float] = None,
        offsets=None,
        assume_unlimited_crews: bool = False,
        offset_shares=None,
    ) -> MaintenancePlan:
        """Choose the proof-test intervals of components with hidden
        failures, for the system as a whole.

        Testing a component with hidden failures more often costs more
        tests, and testing it less often leaves its failures hidden for
        longer (see ``node_availability``). This chooses the inspection
        interval of each component in ``nodes`` for:

        - the lowest long-run cost rate (``expected_cost_rate``: tests,
          repairs and downtime), by default;
        - the lowest cost rate that keeps the system's long-run availability
          at least ``min_availability``: for a safety function, whose
          unavailability is its average probability of failure on demand,
          a PFDavg of at most ``1 - min_availability``;
        - or the highest availability within a cost rate of
          ``max_cost_rate``.

        Tests are usually made on a calendar (monthly, quarterly, yearly),
        and components tested at the same times are down together, so the
        long-run values depend on how the schedules line up. So the
        intervals are chosen from ``allowed``: every combination is tried
        when there are at most 2000 of them, which gives the optimum, and a
        local search (changing one interval at a time, from several
        starting points) is made otherwise. With ``allowed`` left out, the
        RBD must have one component with hidden failures, whose interval
        is then searched continuously, as by
        ``optimal_replacement_intervals``.

        When redundant components are tested matters as well: two tested
        at once are down together for as long as a failure of both stays
        hidden, where tests half an interval apart find a common-cause
        failure twice as soon. ``offset_shares`` chooses the times of the
        first tests with the intervals.

        A component whose tests can miss a failure (a ``coverage`` below
        1) keeps its full tests' interval (``full_test``), a whole number
        of its test intervals: its interval is chosen among those that
        divide it, from ``allowed`` (which must) or, left out, from every
        one in range.

        Parameters
        ----------
        nodes : Collection[Hashable], optional
            The components whose intervals to choose, each with hidden
            failures (an ``"inspection"`` schedule). By default every
            component with one. The others keep their intervals.
        allowed : sequence of float or dict, optional
            The intervals to choose from: one sequence for every node, or a
            dict of a sequence per node. Left out, the one component with
            hidden failures has its interval chosen from all.
        min_availability : float, optional
            The least long-run system availability allowed, in (0, 1).
        max_cost_rate : float, optional
            The highest long-run cost rate allowed: the intervals then give
            the highest availability within it.
        offsets : sequence of float, dict or str, optional
            Deprecated: ``offset_shares``, which it is renamed, as its
            values are shares of the interval where ``with_intervals``'
            and the plan's ``offsets`` are times. Refused in 0.14.
        assume_unlimited_crews : bool, optional
            With limited ``repair_crews``, choose as if every repair started
            at once, as ``optimal_replacement_intervals`` does, by default
            False: refused.
        offset_shares : sequence of float, dict or str, optional
            Choose each node's offset, the time of its first test, as well:
            as a share of its interval, in [0, 1), from these, one sequence
            for every node or a dict of one per node. ``"stagger"`` with
            ``n`` nodes is the shares ``0, 1/n, ..., (n - 1)/n``, among
            which are tests of one interval spread evenly over it. Shifting
            every test by one time changes no long-run value, so when the
            nodes are all the components with hidden failures, the first
            one's tests stay from 0. Needs ``allowed``. By default each
            offset keeps its share of the interval. The plan's ``offsets``
            are the times they give (shares times the intervals).

        Returns
        -------
        MaintenancePlan
            The interval of each component in ``nodes``, its ``offsets``
            (the time of its first test), and the system's cost rate and
            availability (1 - PFDavg) with them.

        Raises
        ------
        ValueError
            If a node is not a component, or has no hidden failures; if
            ``allowed`` is left out with more than one component with
            hidden failures (or with ``offset_shares``), names a node not
            chosen, or holds something other than positive, finite
            intervals, or intervals that do not divide a component's full
            tests' interval; if ``offset_shares`` names a node not chosen,
            or holds something other than shares in [0, 1); if nothing is
            priced; if both targets are given, or one is out of range; or
            if no intervals meet the target (the message gives the best
            they can do).
        NotImplementedError
            If a component's hidden failures have no exact long-run values
            (see ``node_availability``), the intervals repeat together only
            after too many tests to average over, or a component can wait
            for a repair crew and ``assume_unlimited_crews`` is not given.

        Examples
        --------
        A shutdown valve whose dangerous failures are hidden, at ``2e-6``
        per hour, each proof test costing 500: the cheapest monthly,
        quarterly, half-yearly, yearly or two-yearly tests (in hours) that
        keep the PFDavg at most ``1e-3``, for one valve and for two in
        parallel (1oo2):

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def valve():
        ...     return {
        ...         "reliability": surv.Exponential.from_params([2e-6]),
        ...         "repairability": "instant",
        ...         "inspection": {"interval": 8760.0, "cost": 500.0},
        ...     }
        >>> calendar = [730.0, 2190.0, 4380.0, 8760.0, 17520.0]
        >>> one = RepairableRBD([("s", "v"), ("v", "t")], {"v": valve()})
        >>> one.optimal_inspection_intervals(
        ...     allowed=calendar, min_availability=1 - 1e-3
        ... ).intervals
        {'v': 730.0}
        >>> pair = RepairableRBD(
        ...     [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        ...     {"v1": valve(), "v2": valve()},
        ... )
        >>> plan = pair.optimal_inspection_intervals(
        ...     allowed=calendar, min_availability=1 - 1e-3
        ... )
        >>> plan.intervals
        {'v1': 17520.0, 'v2': 17520.0}
        >>> round(1 - plan.availability, 6)
        0.000399

        The redundant pair meets the target with tests every two years; the
        single valve needs them monthly. With a 10% common cause between
        the two, and a PFDavg of at most ``5e-4``, testing them at
        different times as well finds a common-cause failure sooner, and
        meets the target with yearly tests six months apart, for a third
        less:

        >>> from repyability import BetaFactor, CCFGroup
        >>> common = RepairableRBD(
        ...     [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        ...     {"v1": valve(), "v2": valve()},
        ...     ccf_groups=[CCFGroup(["v1", "v2"], BetaFactor(0.1))],
        ... )
        >>> target = 1 - 5e-4
        >>> plan = common.optimal_inspection_intervals(
        ...     allowed=calendar, min_availability=target
        ... )
        >>> plan.intervals, round(plan.cost_rate, 4)
        ({'v1': 4380.0, 'v2': 8760.0}, 0.1712)
        >>> staggered = common.optimal_inspection_intervals(
        ...     allowed=calendar,
        ...     min_availability=target,
        ...     offset_shares="stagger",
        ... )
        >>> staggered.intervals, staggered.offsets
        ({'v1': 8760.0, 'v2': 8760.0}, {'v1': 0.0, 'v2': 4380.0})
        >>> round(staggered.cost_rate, 4), round(1 - staggered.availability, 7)
        (0.1142, 0.0004925)
        """
        return _intervals.optimal_inspection_intervals(
            self,
            nodes=nodes,
            allowed=allowed,
            min_availability=min_availability,
            max_cost_rate=max_cost_rate,
            offsets=offsets,
            assume_unlimited_crews=assume_unlimited_crews,
            offset_shares=offset_shares,
        )

    def with_intervals(self, intervals, offsets=None) -> "RepairableRBD":
        """A copy of this RBD with some maintenance or test intervals
        changed.

        To simulate a plan that ``optimal_replacement_intervals`` or
        ``optimal_inspection_intervals`` chose, with the repair crews it was
        chosen without (``assume_unlimited_crews``), or to ``compare`` it
        with the schedules this RBD has. The copy is built as this RBD was,
        with each named component's schedule given the new interval.

        Parameters
        ----------
        intervals : MaintenancePlan or dict
            A plan (its ``intervals``, and its ``offsets`` if it has them),
            or ``{node: interval}``: a positive interval, or for a
            component under age replacement ``inf``, to replace it only when
            it fails.
        offsets : dict, optional
            ``{node: time of the first test}`` for components with hidden
            failures, each from 0 to less than its interval. By default a
            plan's own; otherwise each keeps its share of the interval.

        Returns
        -------
        RepairableRBD
            The new RBD.

        Raises
        ------
        ValueError
            If a node has no maintenance or test schedule (or an offset no
            test schedule), a block-replaced component is given ``inf``, or
            an interval or offset is invalid (as the constructor checks
            them).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> pump = {
        ...     "reliability": surv.Weibull.from_params([1000, 2.5]),
        ...     "repairability": surv.Exponential.from_params([0.05]),
        ...     "replace_cost": 5000.0,
        ...     "preventive": {"interval": 1000.0, "cost": 1000.0},
        ... }
        >>> rbd = RepairableRBD([("s", "p"), ("p", "t")], {"p": pump})
        >>> plan = rbd.optimal_replacement_intervals()
        >>> rbd.with_intervals(plan).expected_cost_rate() == plan.cost_rate
        True
        """
        return _intervals.with_intervals(
            self, intervals=intervals, offsets=offsets
        )

    def _require_component(self, node, given: str) -> None:
        """Raise if ``node``, given in ``given``, is not a component (see
        ``_not_a_component``)."""
        if node not in self.components:
            raise self._not_a_component(node, given)

    def _not_a_component(self, node, given: str) -> ValueError:
        """The error for ``node``, given in ``given``, that is not a
        component: a junction, which never fails, or an unknown name, with
        the closest component's name if one is close (#232), and those
        that are (#222)."""
        if node in self._junction_nodes:
            return ValueError(
                f"Node {node!r} given in {given} is a junction "
                "(PerfectReliability): it never fails, so it is no component "
                "to fail, repair, maintain or stock spares for."
            )
        close = _close_name(node, self.components)
        hint = "" if close is None else f" Did you mean {close!r}?"
        return ValueError(
            f"Unknown node {node!r} given in {given}; it is not a component "
            f"of the RBD.{hint} Its components are: {list(self.components)}."
        )

    def initialize_event_queue(
        self,
        t_simulation,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        sources: Optional[dict] = None,
        state=None,
    ):
        """Start one simulation of the system over ``[0, t_simulation)``.

        Advanced, event-stepping API: ``availability`` runs its simulations
        through it, and a parent RBD uses it to drive a nested
        ``RepairableRBD``, which therefore has the same event API as a
        ``NonRepairable`` component. Every component starts working and new
        (or from its ``state``) except the ``broken_nodes``, which stay down
        for the whole window; the ``working_nodes`` never fail. The first
        failure of each other component is queued (a nested RBD starts its
        own simulation and queues its first state change), and events at or
        after ``t_simulation`` are dropped. Then call ``next_event``
        repeatedly to step through the system's state changes; see it for
        an example.

        Unlike ``availability``, this takes no seed and does not check the
        working/broken nodes: the draws come from numpy's global RNG, so
        seed that (``np.random.seed``) for a reproducible run. The state is
        kept on the RBD itself (``system_state``, ``component_status``,
        ``t_simulation``, ``last_change_planned``), and ``availability``
        uses and then deletes the same state, so do not interleave the two.

        Parameters
        ----------
        t_simulation : float
            The end of the simulated window.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window, by default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``
            or ``"paths"``, the default) or the minimal cut sets (``"c"``
            or ``"cuts"``); both give the same state.
        sources : dict, optional
            Internal: node name -> the object each component's events are
            drawn from, which ``availability`` passes so that the
            components draw from random streams of their own (see
            ``_streams``). By default None: the components themselves.
        state : dict, optional
            Start the components it names from their states rather than new,
            as for ``availability``: ``{node: NodeState}``, and a nested
            RBD's own such dict for a nested RBD. By default None: every
            component new.

        Raises
        ------
        ValueError
            If ``method`` is not ``"p"`` or ``"c"``, or ``state`` is not
            one the components can be in.
        NotImplementedError
            If ``state`` is ``"stationary"``, or gives a state to a
            component whose state is not taken (see ``availability``).
        """
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        states = _curves._simulation_states(
            self, state, working_nodes | broken_nodes
        )
        _ccf_groups._require_free_members(self, working_nodes, broken_nodes)
        _ccf_groups._require_groups_simulated(self, states)
        self._start_queue(
            t_simulation, working_nodes, broken_nodes, method, sources, states
        )

    def _start_queue(
        self,
        t_simulation,
        working_nodes: set,
        broken_nodes: set,
        method: str,
        sources: Optional[dict],
        states: dict,
    ) -> None:
        """``initialize_event_queue``, with ``states`` checked (see
        ``_simulation_states``)."""
        # What the components draw their events from: themselves (an
        # imperfectly repaired one through a stand-in keeping its virtual
        # age, and one started from a state through one whose next draw can
        # be set; kept for next_event), or stand-ins drawing from their own
        # streams (see _streamed_components), which the caller passes to
        # next_event too.
        if sources is None:
            sources = self._step_sources = self._own_sources(states)
        else:
            self.__dict__.pop("_step_sources", None)
        # The window's end, which the first events already look to (see
        # _replaced_at).
        self.t_simulation = t_simulation
        # Each calendar a state's phase shifts: by node, the time since its
        # last scheduled replacement or test (see _due).
        self._phases: dict = {
            node: start.phase
            for node, start in states.items()
            if isinstance(start, NodeState)
            and (
                start.phase
                or (
                    # On a calendar from 0, not at its offset.
                    start.phase is not None
                    and node in self._inspection
                    and self._inspection[node].offset
                )
            )
        }

        # Keep record of component status', initially they're all working
        component_status: dict[Any, bool] = {
            component: True for component in self.components.keys()
        }

        for component in broken_nodes:
            component_status[component] = False

        # The queue supplies failure/repair events in chronological order
        event_queue = _EventQueue()
        # When each working node with hidden failures is due to fail (None
        # once it has failed).
        self._pending_failure: dict[Any, Optional[float]] = {}
        # When each working node replaced on condition is due to fail, and
        # to be replaced (see _replaced_at).
        self._in_service: dict[Any, Tuple[float, float]] = {}
        # Each working node replaced on condition by its measured level
        # (#271): when its level was last known, and the level then.
        self._levels: dict[Any, Tuple[float, float]] = {}
        # Opportunistic maintenance (#108): when each member that can be
        # renewed early was put into service as new, and its pending event;
        # the pending events early renewals have cancelled, and the early
        # renewals queued, by identity, and their members (see _stop).
        self._renewed_at: dict[Any, float] = {}
        self._pending_event: dict[Any, Event] = {}
        self._cancelled: dict[int, Event] = {}
        self._early: dict[int, Event] = {}
        self._renewing: set = set()
        # Common-cause groups (#158): each member's next event, which a
        # shared cause that strikes it first supersedes, and the uniform a
        # strike has decided its tests by (see _strike).
        self._ccf_members = frozenset(
            m for group in self.ccf_groups for m in group.members
        )
        self._ccf_pending: dict[Any, Optional[Event]] = {}
        self._ccf_coins: dict[Any, float] = {}
        # The components down at the start in a repair or maintenance going
        # on, which holds a repair crew.
        in_hand: list = []

        # For each component add in the initial failure
        for component_id in self.components.keys():
            component = self.components[component_id]
            if component_id in working_nodes:
                continue
            elif component_id in broken_nodes:
                continue
            elif component_id in self._standby:
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
                up, first = self._started(component_id, source, start)
                component_status[component_id] = up
                if not up and first.status:
                    in_hand.append(component_id)
            elif component_id in self._preventive:
                # Put into service as new at 0: it fails, or is maintained.
                source.reset()
                first = self._renewal(
                    component_id, 0.0, source, self._preventive[component_id]
                )
            elif component_id in self._inspection:
                # Put into service as new at 0: it fails, or is inspected.
                source.reset()
                first = self._inspected_renewal(
                    component_id, 0.0, source, self._inspection[component_id]
                )
            else:
                source.reset()
                t_event, event = source.next_event()
                first = Event(t_event, component_id, event)

            # Only consider it if it occurs within the simulation window
            if first.time < t_simulation:
                event_queue.put(first)
            if component_id in self._ccf_members:
                self._ccf_pending[component_id] = first
        for cause in _ccf_groups._shared_causes(self):
            first = Event(sources[cause].gap(), cause, False)
            if first.time < t_simulation:
                event_queue.put(first)
        self._event_queue = event_queue
        # The repair crews, when there are fewer than the components that
        # may need one (with enough, no job ever waits).
        crews = self.repair_crews
        self._crews = (
            _Crews(crews, self._crew_served(), self._priority)
            if crews is not None and self._crews_limited()
            else None
        )
        if self._crews is not None:
            # A repair or maintenance going on at the start holds a crew
            # (_simulation_states has checked that there are enough).
            for node in in_hand:
                if node in self._crews.served:
                    self._crews.free -= 1
                    self._crews.holding.add(node)
        # Each standby group's units, which queue the group's first event.
        self._groups: dict[Any, _StandbyGroup] = {
            node: _StandbyGroup(
                self, node, arrangement, sources[node], t_simulation
            )
            for node, arrangement in self._standby.items()
            if node not in working_nodes and node not in broken_nodes
        }
        self.last_change_planned = False
        # The initial system state must reflect any forced-broken components
        # (e.g. a broken component in series starts the system down), and
        # any component that starts down, rather than assuming everything is
        # up.
        self.system_state = self.is_system_working(component_status, method)
        self.component_status = component_status

    def _started(self, node, source, start: NodeState) -> Tuple[bool, Event]:
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
        component = self.components[node]
        schedule = self._preventive.get(node)
        inspection = self._inspection.get(node)
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
                self._pending_failure[node] = None
            return False, Event(left, node, True, start.maintenance)
        if not start.age and not virtual:
            # New at 0, on a calendar at its phase.
            if schedule is not None:
                return True, self._renewal(node, 0.0, source, schedule)
            if inspection is not None:
                return True, self._inspected_renewal(
                    node, 0.0, source, inspection
                )
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
                self._pending_failure[node] = None
                found = self._finds(node, inspection, life)
                return False, Event(found, node, False, inspection=True)
            self._pending_failure[node] = life
            return True, self._inspected_next(node, 0.0, inspection)
        if schedule is None:
            return True, Event(life, node, False)
        return True, self._renewal(
            node, 0.0, source, schedule, life=life, start=-start.age
        )

    def _due(self, node, schedule, t: float) -> float:
        """The next scheduled action of ``schedule`` (a ``_Preventive`` or
        ``_Inspection``) after ``t``, on ``node``'s calendar: shifted, from
        a state, by its phase, so that the last action fell that long
        before 0."""
        phase = self._phases.get(node) if self._phases else None
        if phase is None:
            return schedule.due(t)
        if isinstance(schedule, _Inspection):
            # The phase places the calendar: no offset.
            schedule = schedule.without_offset()
        due = schedule.due(t + phase) - phase
        return due if due > t else due + schedule.interval

    def _finds(self, node, inspection: _Inspection, t: float) -> float:
        """The test that finds a failure at ``t``, on ``node``'s calendar
        (see ``_due``): the first at or after it, and not before 0."""
        phase = self._phases.get(node) if self._phases else None
        if phase is None:
            return inspection.finds(t)
        inspection = inspection.without_offset()
        found = inspection.finds(t + phase) - phase
        if found < t or found < 0.0:
            found += inspection.interval
        return found

    def mean_unavailability(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> float:
        """Returns the system's long-run (steady-state) unavailability.

        ``1 - mean_availability()``: the long-run fraction of time the
        system is down, exact as
        [`mean_availability`][repyability.RepairableRBD.mean_availability]
        is, and over the same cases. It is worked out in its own right,
        from each component's unavailability (``MTTR / (MTTF + MTTR)``,
        say, or ``1 - exp(-lambda * u)`` at a time ``u`` since a test), as
        a sum of products over the structure, so that a small one keeps its
        full relative precision: a PFDavg of ``1e-12`` gets ``1e-12``,
        where one less the availability would keep only about four digits
        of it.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (unavailability 0), by
            default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (unavailability 1), by
            default None.
        method : str, optional
            ``"p"`` or ``"c"``, as for ``mean_availability``; both give the
            same result, computed the same way.

        Returns
        -------
        float
            Long-run unavailability of the system, in ``[0, 1]``.

        Raises
        ------
        ValueError
            As for ``mean_availability``.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        Two components in parallel, each down a millionth of the time, are
        down together a millionth of a millionth of it:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> unit = {
        ...     "reliability": surv.Exponential.from_params([1e-6]),
        ...     "repairability": surv.ExactEventTime.from_params([1.0]),
        ... }
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> f"{rbd.mean_unavailability():.6e}"
        '9.999980e-13'
        """
        return _long_run.mean_unavailability(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
        )

    def analysis_routes(self) -> Dict[str, "AnalysisRoute"]:
        """How each analysis of this RBD is computed, found without running
        it: exactly, numerically, by simulation, or not at all.

        The route follows from the method, then from the components: the
        long-run values are exact from each component's long-run
        availability, which is a closed form, or numerical for a component
        under preventive maintenance, or refused for a case they do not
        cover (hidden failures at a rate that is not constant, say); the
        availability over time solves each component's renewal equation on
        a grid; ``availability``, ``cost`` and ``compare`` always simulate.
        A component whose life has no exact or numerical reliability (a
        ``StandbyModel`` or ``LoadSharingModel`` that only simulations take)
        makes the values it enters refuse, as it does. A refusal is found by
        the check the method itself runs, and its reason is the message it
        would raise; limits that only computing shows (a grid grown too
        large) are not foreseen.

        A method that takes nodes, targets or intervals is described for
        its defaults: ``optimal_replacement_intervals`` choosing for every
        component under age replacement, say.

        For a simulation, ``engine`` is the engine ``engine="auto"`` runs a
        long simulation on: when the compiled engine simulates the system,
        ``"numba"`` if numba is installed (or the name of an engine another
        package adds, see ``repyability.rbd.engines``), else ``"python"``,
        with the reason in ``engine_reason``.

        Returns
        -------
        dict[str, AnalysisRoute]
            For each public analysis, by method name: its ``route``
            (``"exact"``, ``"numerical"``, ``"simulated"`` or
            ``"refused"``), the ``reason``, and the ``nodes`` that decide it
            (see [`AnalysisRoute`][repyability.AnalysisRoute]).

        Examples
        --------
        A pump found failed only by monthly tests that take it off line
        for about an hour: its values are numerical, its cycle from one
        test that finds a failure to the next followed test by test. Tests
        that could last as long as the month (of 1000 hours on average,
        say) are refused, and the simulation is the way:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def pump(test_hours):
        ...     test = surv.Exponential.from_params([1.0 / test_hours])
        ...     return RepairableRBD(
        ...         [("s", "pump"), ("pump", "t")],
        ...         {
        ...             "pump": {
        ...                 "reliability": surv.Weibull.from_params([500, 2]),
        ...                 "repairability": "instant",
        ...                 "inspection": {"interval": 720, "duration": test},
        ...             }
        ...         },
        ...     )
        >>> pump(1.0).analysis_routes()["mean_availability"].route
        'numerical'
        >>> routes = pump(1000.0).analysis_routes()
        >>> routes["mean_availability"].route
        'refused'
        >>> routes["mean_availability"].nodes
        ('pump',)
        >>> routes["availability"].route
        'simulated'
        """
        from repyability.rbd import routes as r

        out: Dict[str, r.AnalysisRoute] = {}

        def give(names, route) -> None:
            for name in names:
                out[name] = route

        long_run = _long_run._long_run_route(self)

        long_run_nodes = _long_run._long_run_nodes(self)

        def from_long_run(route, reason):
            """A method of ``route``, explained by ``reason``, built on the
            long-run values: refused as they are, and numerical or simulated
            as the components' are."""
            if long_run.route == r.REFUSED:
                return long_run
            return r.with_nodes(
                route, reason, long_run_nodes, "long-run values"
            )

        give(("mean_availability", "mean_unavailability"), long_run)
        out["node_availability"] = (
            _long_run._long_run_route(self, groups=False)
            if self.ccf_groups
            else long_run
        )
        frequencies = (
            None
            if long_run.route == r.REFUSED
            else r.refusal(partial(_ccf_groups._require_ccf_frequencies, self))
        )
        give(
            (
                "system_failure_frequency",
                "mean_time_between_failures",
                "mean_up_time",
                "mean_down_time",
            ),
            r.refused(frequencies) if frequencies else long_run,
        )
        give(
            ("barlow_proschan_importance",),
            (
                r.refused(frequencies)
                if frequencies
                else (
                    long_run
                    if long_run.route == r.REFUSED
                    else dataclasses.replace(
                        long_run,
                        reason="Each component's share of the system's "
                        "long-run failure frequency (the Birnbaum/Vesely "
                        "formula's terms). " + long_run.reason,
                    )
                )
            ),
        )
        # The allocations assume independent components, which limited
        # repair crews make them not; the common-cause groups' chains take
        # theirs in.
        allocation = r.refusal(
            partial(
                self._require_unlimited_crews,
                *_repairable_allocation._ALLOCATION_CREWS,
            )
        ) or r.refusal(partial(_ccf_groups._require_ccf_long_run, self))
        conditioned = long_run
        if self.ccf_groups and long_run.route != r.REFUSED:
            conditioned = dataclasses.replace(
                long_run,
                reason=long_run.reason
                + " A common-cause group's member is conditioned on its "
                "state at each time, as the shared causes tie the other "
                "members' to it.",
            )
        if self._crews_couple() and long_run.route != r.REFUSED:
            conditioned = dataclasses.replace(
                long_run,
                reason=long_run.reason
                + " With the crews, Birnbaum's measure, the improvement "
                "potential, RAW and RRW hold each node working and failed "
                "in the chain, solved without it; the criticality and "
                "Fussell-Vesely measures are probabilities over its "
                "states.",
            )
        give(
            (
                "birnbaum_importance",
                "improvement_potential",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "criticality_importance",
            ),
            conditioned,
        )
        pairs = r.refusal(self._require_no_ccf_pairs)
        give(
            ("joint_importance",),
            (
                r.refused(pairs)
                if pairs
                else (
                    conditioned
                    if conditioned.route == r.REFUSED
                    else dataclasses.replace(
                        conditioned,
                        reason="Each component's Birnbaum importance with "
                        "the other held up, less with it held down. "
                        + conditioned.reason,
                    )
                )
            ),
        )
        give(
            ("differential_importance",),
            (
                conditioned
                if conditioned.route == r.REFUSED
                else dataclasses.replace(
                    conditioned,
                    reason="Birnbaum's measure (or, for a proportional "
                    "change, the criticality importance), shared out; "
                    "over='parameters', parameter_sensitivity's "
                    "derivatives. " + conditioned.reason,
                )
            ),
        )
        give(
            ("fussell_vesely",),
            from_long_run(
                r.EXACT,
                "The exact probability that a minimal cut set containing "
                "each node is down, over the system's unavailability, from "
                "the exact long-run values (method='rare_event' sums the cut "
                "sets' probabilities instead).",
            ),
        )
        give(
            ("parameter_sensitivity",),
            from_long_run(
                r.NUMERICAL,
                "Central differences of the long-run availability (or cost "
                "rate) in each lever, the diagram rebuilt with the lever "
                "moved; at times or over a window, of the values over time.",
            ),
        )
        if self.has_costs:
            setups = r.refusal(self._require_separate_setups)
            give(
                ("expected_cost_rate", "total_cost"),
                (
                    r.refused(setups)
                    if setups
                    else from_long_run(
                        r.EXACT,
                        "The long-run cost rate, from the exact long-run "
                        "values (total_cost multiplies it by the time).",
                    )
                ),
            )
        else:
            give(
                ("expected_cost_rate", "total_cost"),
                r.AnalysisRoute(
                    r.EXACT,
                    "No running cost is priced, so the cost rate is 0 "
                    "(total_cost adds any acquisition costs).",
                ),
            )
        capacity = (
            None
            if long_run.route == r.REFUSED
            else _repairable_capacity._capacity_refusal(self)
        )
        out["capacity_distribution"] = (
            r.refused(*capacity)
            if capacity
            else from_long_run(
                r.EXACT,
                "The exact long-run distribution of the system's capacity, "
                + (
                    "averaged over the states of the repair crews' Markov "
                    "chain."
                    if self._crews_couple()
                    else "from the components' long-run availabilities."
                ),
            )
        )
        no_capacity = r.refusal(self._require_numbered_capacities) or (
            r.refusal(self._require_capacity)
        )
        out["system_capacity"] = (
            r.refused(no_capacity)
            if no_capacity
            else r.AnalysisRoute(
                r.EXACT,
                "The exact distribution of the system's capacity from the "
                "node probabilities given.",
            )
        )
        give(
            (
                "point_availability",
                "mission_availability",
                "point_unavailability",
                "mission_unavailability",
            ),
            _curves._over_time(self),
        )
        give(
            ("availability_rate",),
            (
                _curves._crew_over_time(self, "rate")
                if self._crews_couple()
                else _curves._over_time(
                    self,
                    "Each component's renewal equation solved on a grid (to "
                    "about 1e-7) and differentiated on it, times its Birnbaum "
                    "importance at the components' availabilities; the jumps "
                    "at scheduled events split along the path between the "
                    "values either side."
                    + (
                        " A common-cause group's part is its chain's moves, "
                        "split by cause and by member repair, at the "
                        "system's availability with the group in each of "
                        "its states (exact); the jumps at a hidden group's "
                        "tests are split by the Shapley value of what jumps."
                        if self.ccf_groups
                        else ""
                    ),
                )
            ),
        )
        over_time_capacity = _repairable_capacity._capacity_refusal(self)
        give(
            ("point_capacity", "mission_capacity"),
            (
                r.refused(*over_time_capacity)
                if over_time_capacity
                else _curves._over_time(
                    self,
                    "Each component's renewal equation solved on a grid (to "
                    "about 1e-7), a degrading component's stages through its "
                    "renewals too, and the system's capacity distribution "
                    "exactly at its components' at each time.",
                    kind="capacity",
                )
            ),
        )
        window = _curves._over_time(
            self,
            "Each component's expected events from its renewal equation, "
            "solved on a grid (to about 1e-7), and the system's failures by "
            "the time-dependent Birnbaum/Vesely formula.",
            kind="window",
        )
        give(("expected_failures", "expected_events"), window)
        out["expected_cost"] = (
            window
            if self.has_costs
            else r.AnalysisRoute(
                r.EXACT,
                "No running cost is priced, so the expected cost is 0 (with "
                "any acquisition costs beside it).",
            )
        )
        # A group's members' MTTF and MTTR are their group's: the
        # availability allocations hold them, and a member's copies join its
        # group.
        held_members = (
            " A common-cause group's members keep their availability: "
            "their MTTF and MTTR are their group's."
            if self.ccf_groups
            else ""
        )
        out["allocate_redundancy"] = (
            r.refused(allocation)
            if allocation
            else from_long_run(
                r.EXACT,
                "A search over the exact long-run values of the candidates."
                + (
                    " A common-cause group member's copies join its group, "
                    "whose chain takes them in: a BetaFactor group's (an MGL "
                    "group's member refuses)."
                    if self.ccf_groups
                    else ""
                ),
            )
        )
        out["availability_allocation"] = (
            r.refused(allocation)
            if allocation
            else from_long_run(
                r.EXACT,
                "A search over the exact long-run values of the candidates."
                + held_members,
            )
        )
        out["mttf_mttr_allocation"] = (
            r.refused(allocation)
            if allocation
            else from_long_run(
                r.NUMERICAL,
                "A solver for the components' MTTF and MTTR targets, over the "
                "exact long-run values." + held_members,
            )
        )
        if out["availability_allocation"].route != r.REFUSED:
            # Which components are allocated one follows from the diagram
            # (and fixed, which only holds more).
            none = r.refusal(
                partial(_repairable_allocation._allocated_levers, self, set())
            )
            if none:
                give(
                    ("availability_allocation", "mttf_mttr_allocation"),
                    r.refused(none),
                )
        # The optimisers, called for every node they can choose for.
        targets = r.refusal(
            partial(
                _intervals._interval_targets,
                self,
                None,
                None,
            )
        )
        interval_crews = r.refusal(
            partial(_intervals._require_interval_crews, self)
        )
        refusal = (
            r.refusal(partial(_intervals._maintained, self, None))
            or targets
            or interval_crews
        )
        out["optimal_replacement_intervals"] = (
            r.refused(refusal)
            if refusal
            else from_long_run(
                r.NUMERICAL,
                "An optimiser over the age-replacement intervals, with the "
                "exact long-run values at each.",
            )
        )
        refusal = (
            r.refusal(partial(_intervals._inspected, self, None))
            or targets
            or interval_crews
            or next(
                (
                    message
                    for message in (
                        r.refusal(partial(self._require_tested_exact, node))
                        for node in self._inspection
                    )
                    if message
                ),
                None,
            )
            or r.refusal(self._require_one_inspected)
        )
        out["optimal_inspection_intervals"] = (
            r.refused(refusal)
            if refusal
            else from_long_run(
                r.NUMERICAL,
                "An optimiser over the inspection interval, with the exact "
                "long-run values at each.",
            )
        )
        # Spares: each component's replacements, counted as a renewal
        # process, in the order the methods check them.
        for name, long_run_count in (
            ("spares_demand", False),
            ("spares_stock", True),
        ):
            refusal = r.refusal(
                partial(
                    self._require_unlimited_crews,
                    *(
                        _spares._STOCK_CREWS
                        if long_run_count
                        else _spares._SPARES_CREWS
                    ),
                )
            )
            which: Tuple[Hashable, ...] = ()
            for node in _spares._spares_nodes(self, None):
                if refusal:
                    break
                refusal = r.refusal(
                    partial(_spares._replacements, self, node, long_run_count)
                )
                which = (node,)
            out[name] = (
                r.refused(refusal, which)
                if refusal
                else r.AnalysisRoute(
                    r.NUMERICAL,
                    "Each component's replacements, a renewal process, "
                    "counted on a grid (to about 1e-6): under block "
                    + (
                        "replacement, averaged over where the lead time "
                        "falls in the block interval (with repairs or "
                        "block replacements that take time, from a typical "
                        "replacement, a failure at each phase of the "
                        "interval or a block replacement, in the long run)"
                        if long_run_count
                        else "replacement, from one block interval to the "
                        "next"
                    )
                    + "; with hidden failures, on its tests: exactly, tested "
                    "and repaired in no time by tests that find every "
                    "failure, and otherwise from the cycle between the tests "
                    "that find failures, followed test by test (#159).",
                )
            )
        plan, streamed = self._stream_plan(1.0, 0, False)
        paired = (
            ""
            if streamed
            else " Antithetic pairs are refused: some components' draws "
            "do not come from a stream."
        )
        groups = r.refusal(
            partial(_ccf_groups._require_groups_simulated, self)
        )
        given = r.refusal(self._require_capacities_given)
        engine, why = self._engine_choice(capacity=self._has_capacity())
        causes = (
            " Each common-cause group's shared causes strike as Poisson "
            "processes, each failing the members it names that are up."
            if _ccf_groups._has_ccf(self)
            else ""
        )
        out["availability"] = (
            r.refused(groups)
            if groups
            else (
                r.refused(given, tuple(self._capacity_models()))
                if given
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "A discrete-event simulation of the components' failures, "
                    "repairs and maintenance." + causes + paired,
                    engine=engine,
                    engine_reason=why,
                )
            )
        )
        out["simulate_chunk"] = out["availability"]
        unsaved = r.refusal(self._shard_system)
        out["shards"] = (
            out["availability"]
            if out["availability"].route == r.REFUSED
            else (
                r.refused(unsaved)
                if unsaved
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "The run's simulations as shards, to simulate anywhere "
                    "with run_shard (see shards); availability_from_chunks "
                    "merges their partials into its result.",
                )
            )
        )
        out["availability_from_chunks"] = (
            out["availability"]
            if given or groups
            else r.AnalysisRoute(
                r.SIMULATED,
                "Simulated chunks of a run (see simulate_chunk), merged into "
                "its result.",
            )
        )
        engine, why = _timeline_runs.engine_choice(self, plan)
        out["simulate_timelines"] = (
            r.refused(groups)
            if groups
            else r.AnalysisRoute(
                r.SIMULATED,
                "The simulations availability runs, their histories kept "
                "as timelines: recorded by the event loop as it runs."
                + paired,
                engine=engine,
                engine_reason=why,
            )
        )
        engine, why = self._engine_choice(capacity=False)
        out["cost"] = (
            r.refused(groups)
            if groups and self.has_costs
            else r.AnalysisRoute(
                r.SIMULATED,
                "A discrete-event simulation of the components' failures, "
                "repairs and maintenance, and what they cost." + paired,
                engine=engine,
                engine_reason=why,
            )
        )
        twin = self._twin_report()
        # The expected values over the window, where the methods over time
        # work them out, need no simulation, and a run takes them (#187).
        exact_means = ""
        if all(
            out[name].route in (r.EXACT, r.NUMERICAL)
            for name in ("mission_availability", "expected_events")
            + (("expected_cost",) if self.has_costs else ())
        ):
            exact_means = (
                " Its expected values over the window need none: "
                "mission_availability, expected_events and expected_cost "
                "work them out, and by default a run's means are theirs, "
                "with no error, so a run to a tolerance stops at "
                "once. Simulate for their spread: each simulation's values, "
                "percentiles, the chance of no failure."
            )
        # Or a run's means are taken given its modules' histories (#189).
        if not exact_means:
            modules = self._conditional_applies()
            if modules:
                names = ", ".join(map(repr, modules))
                exact_means = (
                    " By default a run's means take each "
                    "simulation's expected values given the histories of "
                    f"{names}, the rest exact given their states, which "
                    "vary less than its own; with conditional=True only "
                    f"{names} are simulated, for those means at a fraction "
                    "of the work, and no spread."
                )
        for name in ("availability", "cost"):
            if out[name].route == r.SIMULATED:
                out[name] = dataclasses.replace(
                    out[name], reason=out[name].reason + exact_means, twin=twin
                )
        # The difference of exact expected values needs no simulation
        # (#236): where _exact_means works them out.
        needed = [out["mission_availability"]] + (
            [out["expected_cost"]] if self.has_costs else []
        )
        # A diagram too meshed to work out has no exact expected values,
        # so compare simulates it.
        if self._too_meshed() is None and all(
            route.route in (r.EXACT, r.NUMERICAL) for route in needed
        ):
            out["compare"] = r.AnalysisRoute(
                (
                    r.NUMERICAL
                    if any(route.route == r.NUMERICAL for route in needed)
                    else r.EXACT
                ),
                "The difference of the two systems' expected values over "
                "the window (mission_availability, and expected_cost their "
                "costs), where the other's are worked out too (#236); "
                "otherwise, or with control_variate=False, the two systems "
                "simulated with common random numbers.",
            )
        elif groups:
            out["compare"] = r.refused(groups)
        elif streamed:
            out["compare"] = r.AnalysisRoute(
                r.SIMULATED,
                "The two systems simulated with common random numbers.",
                engine=engine,
                engine_reason=why,
            )
        else:
            out["compare"] = r.refused(_UNSTREAMED)
        give(
            ("initialize_event_queue", "next_event"),
            (
                r.refused(groups)
                if groups
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "One simulation, stepped through event by event, drawing "
                    "from numpy's global RNG.",
                )
            ),
        )
        out["structural_importance"] = r.AnalysisRoute(
            r.EXACT, "From the structure alone."
        )
        out["system_probability"] = r.AnalysisRoute(
            r.EXACT,
            "The structure function over the node probabilities given.",
        )
        out["path_set_probabilities"] = r.AnalysisRoute(
            r.EXACT, "From the node probabilities given."
        )
        allocation_by_probability = r.AnalysisRoute(
            r.NUMERICAL,
            "A solver over the exact system probability, from the node "
            "probabilities given (not the component models).",
        )
        give(
            (
                "improvement_allocation",
                "equal_allocation",
                "simple_allocation",
                "cost_based_allocation",
            ),
            allocation_by_probability,
        )
        series = r.refusal(self._require_series)
        out["minimum_effort_allocation"] = (
            r.refused(series) if series else allocation_by_probability
        )
        # Each component's own values, and the long-run costs but the
        # system's downtime, need no structure; nor the expected cost when
        # nothing is priced.
        free = {"node_availability", "spares_demand", "spares_stock"}
        if not self.downtime_cost_rate:
            free |= {"expected_cost_rate", "total_cost"}
        if not self.has_costs:
            free.add("expected_cost")
        meshed = self._too_meshed()
        if meshed is not None and self._has_capacity():
            # The simulations that follow the system's capacity do so
            # through its reduced diagram, which a structure too meshed has
            # not.
            for name in (
                "availability",
                "simulate_chunk",
                "availability_from_chunks",
                "shards",
            ):
                out[name] = r.refused(meshed)
        out = self._meshed_routes(out, free)
        # Parameter uncertainty (#200): each draw's diagram rebuilt and its
        # value worked out as the diagram's own, so refused where that is
        # (a structure too meshed included).
        for name, method in (
            ("mean_availability_uncertainty", "mean_availability"),
            ("point_availability_uncertainty", "point_availability"),
            ("mission_availability_uncertainty", "mission_availability"),
            ("expected_cost_rate_uncertainty", "expected_cost_rate"),
        ):
            own = out[method]
            out[name] = (
                own
                if own.route == r.REFUSED
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "The components' models drawn from their uncertainty, "
                    f"and each draw's diagram's {method} worked out as its "
                    f"own ({own.route}).",
                )
            )
        sensitivity = out["parameter_sensitivity"]
        out["uncertainty_importance"] = (
            sensitivity
            if sensitivity.route == r.REFUSED
            else r.AnalysisRoute(
                r.NUMERICAL,
                "The delta method: parameter_sensitivity's derivatives in "
                "each uncertain parameter, with their covariance "
                "(method='sobol' estimates Sobol indices from draws "
                "instead).",
            )
        )
        return dict(sorted(out.items()))

    def _engine_choice(self, capacity: bool) -> Tuple[str, str]:
        """The engine ``engine="auto"`` runs a long simulation on, and why
        (see ``_simulation_engine``)."""
        from repyability.rbd import _compiled

        plan = self._stream_plan(1.0, 0, False)[0]
        engine, reason = _compiled.choice(
            self, plan, object() if capacity else None
        )
        if reason is not None:
            return "python", f"the compiled engine does not simulate {reason}"
        if engine is None:
            return (
                "python",
                "numba is not installed; pip install 'repyability[fast]' "
                "for the compiled engine",
            )
        return (
            engine,
            "compiled for a run long enough to repay loading it; a short "
            "one runs in Python",
        )

    def mean_availability(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> float:
        """Returns the system's long-run (steady-state) availability.

        Exact, with no simulation. Each component's long-run availability
        is ``MTTF / (MTTF + MTTR)`` (1 if it is repaired instantly; a nested
        ``RepairableRBD``'s is its own ``mean_availability``; under age
        replacement, see ``node_availability``). Since the components fail
        and are repaired independently, the system's is
        the RBD's structure function evaluated exactly at those
        availabilities. It is the long-run fraction of time the system is
        up; for availability over time, from a start with everything
        working, use ``availability``.

        A component with hidden failures is up with probability
        ``exp(-lambda * u)`` at a time ``u`` since its last inspection, so
        components inspected at the same times are down together more
        often than independent ones would be. The system's availability is
        then averaged over time, over one period of the inspection
        schedules (the least common multiple of their intervals): for two
        such components in parallel, inspected together every ``tau``, the
        unavailability is ``(1 / tau) * integral_0^tau (1 -
        exp(-lambda * t)) ** 2 dt``, about ``(lambda * tau) ** 2 / 3``: the
        average probability of failure on demand (PFDavg) of a 1oo2 safety
        function. This is exact for a constant failure rate, and numerical
        (summed over the test intervals) for any other life, with instant
        tests and instant repair (see ``node_availability``).

        With fewer ``repair_crews`` than components, a failed component can
        wait for a crew, and the components no longer fail and recover
        independently. When the components the crews work on have
        exponential lives and exponential (or instant) repairs, with no
        scheduled maintenance or inspection, the system is a Markov chain:
        its state is which components are under repair and which are
        waiting, in the order the crews will take them. Its long-run
        distribution is solved exactly (for up to 15,000 states), and the
        availability is the system's, state by state, averaged over it. A
        node held working never fails and one held broken is never
        repaired, so neither needs a crew: the chain is of the others. A
        nested ``RepairableRBD`` has crews of its own, and enters through
        its own long-run availability.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (availability 1), by
            default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (availability 0), by
            default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the minimal cut sets
            (``"c"`` or ``"cuts"``); both
            give the same exact result.

        Returns
        -------
        float
            Long-run availability of the system, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            If a component's life has no exact or numerical mean (a
            ``StandbyModel`` or ``LoadSharingModel`` that only simulations
            take), is under block replacement with models its exact values
            do not cover (see ``node_availability``), or has hidden
            failures whose tests or repairs take time, or whose tests can
            miss a failure, with a life that is not a surpyval parametric
            model with a density, repairs or tests that may never end, or a
            test that can last as long as its interval; or, while a
            component can wait for a repair crew, if the Markov chain does
            not cover the components (a life
            or repair that is not exponential, scheduled maintenance, an
            inspection or a common-cause group) or would have more than
            15,000 states: simulate it with ``availability``.

        Examples
        --------
        A single component with mean time to failure 10 and mean time to
        repair 1 has long-run availability ``10 / (10 + 1)``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> round(rbd.mean_availability(), 4)
        0.9091
        """
        return _long_run.mean_availability(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
        )

    def point_availability(
        self,
        x,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The probability that the system is up at each time ``x``, with
        every component new at 0: exact, with no simulation.

        Each component's point availability ``A(t)`` follows from the
        distributions of its up and down times by the renewal equation,
        solved numerically (see ``repyability/rbd/_point_availability.py``):
        its error is about 4e-7 (up to 4e-6 soon after the start). Since
        the components fail and are repaired independently, the system's is
        the structure function evaluated exactly at theirs, at each time.
        It starts at 1 (less any components dead on arrival) and settles at
        [`mean_availability`][repyability.RepairableRBD.mean_availability]
        (or, with components replaced or inspected on a calendar, repeats
        with the calendar about it);
        [`availability`][repyability.RepairableRBD.availability] estimates
        the same curve by simulation. At a time something happens -- a
        block replacement, say -- it is the availability just after.

        Components under age or block replacement are covered, as in
        ``mean_availability``, and nested RBDs through their own point
        availability. A component with hidden failures is up with the
        probability that it has not failed since its last test, ``u``
        before (``exp(-lambda * u)`` for a constant failure rate), or
        after it with its tests and repairs taking time or its tests
        missing some failures: as in ``mean_availability``.

        With fewer ``repair_crews`` than components, a component can wait
        for a crew and the components are not independent: the system's
        point availability is then that of the crews' Markov chain (see
        ``mean_availability``), followed over time from its state at 0 by
        uniformization, to about 1e-13; a nested RBD, which has crews of
        its own, enters through its own point availability. A standby
        group's comes from its units' chain in the same way, and a
        common-cause group's members' joint states from theirs (see
        ``ccf_groups``), followed from every member up through their tests
        or by uniformization, with the system summed over them.

        Each component's curve is computed on a grid of 1,000 steps over its
        typical up time. Near a time at which its units start or stop on a
        schedule -- at 0, at its scheduled replacements, and at the failures
        of a lifetime known exactly -- what happens faster than a step, such
        as a short repair, is smoothed over the step: a point value within
        a step of such a time can be off by up to about the probability
        that the component is under repair then. ``mission_availability``
        is not affected.

        Parameters
        ----------
        x : float or array-like
            Times, from 0 (every component new).
        working_nodes : Collection[Hashable], optional
            Nodes that always work (availability 1), by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed (availability 0), by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the cut sets (``"c"``
            or ``"cuts"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The system's point availability at each time, in ``x``'s shape.

        Raises
        ------
        ValueError
            If a time is negative or not finite, or a working/broken node is
            invalid (see ``mean_availability``).
        NotImplementedError
            If a component has hidden failures its numerical values do not
            cover, a model that block replacement's exact values do not
            cover (see ``mean_availability``), a common-cause group's member
            starts from a state, or time scales too far apart for the
            grid (years of running, seconds of repair, over centuries); or,
            while a component can wait for a repair crew (see
            ``repair_crews``), if the crews' Markov chain does not cover the
            components (see ``mean_availability``) or its rates are too far
            apart to follow it over time: simulate it with
            ``availability``.

        Examples
        --------
        One component with failure rate 0.1 and repair rate 1 is up at time
        ``t`` with probability ``1 / 1.1 + (0.1 / 1.1) * exp(-1.1 t)``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> rbd.point_availability([0.0, 1.0, 100.0]).round(4).tolist()
        [1.0, 0.9394, 0.9091]
        """
        return _windows.point_availability(
            self,
            x=x,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            state=state,
        )

    def point_unavailability(
        self,
        x,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The probability that the system is down at each time ``x``, with
        every component new at 0: one less ``point_availability``, worked
        out in its own right so that a small one keeps its precision,
        as ``mean_unavailability`` is in the long run.

        Each component's probability of being down enters the structure
        function's sum of products for the system's failing, so that a
        redundant system's ``1e-16`` is not lost below one less a number
        next to 1, as ``1 - point_availability(t)`` loses it; a
        common-cause group's members' joint states enter theirs in the same
        way. A component new at 0 with an exponential life and an
        exponential (or instant) repair, and no maintenance or tests, is
        down with ``lambda / (lambda + mu) (1 - exp(-(lambda + mu) t))``,
        in closed form; any other, with one less its point availability
        (see
        [`point_availability`][repyability.RepairableRBD.point_availability]),
        whose grid smooths over a step what changes faster than it. With
        components waiting for repair crews, the crews' chain gives it as
        one less the availability (to about 1e-13).

        Parameters
        ----------
        x : float or array-like
            Times, from 0 (every component new).
        working_nodes : Collection[Hashable], optional
            Nodes that always work (never down), by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed (always down), by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the cut sets (``"c"``
            or ``"cuts"``), as for ``point_availability``.
        state : dict or str, optional
            Start from the components' current states rather than new,
            as for ``point_availability``: ``{node: NodeState}`` or
            ``"stationary"``. By default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The system's point unavailability at each time, in ``x``'s
            shape.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_availability``.

        Examples
        --------
        Two valves in parallel, each failing at a rate of ``1e-9`` an hour
        and repaired in a mean of eight hours: each is down ``8e-9`` of
        the time, and the pair, both at once, ``6.4e-17``, which one less
        the availability rounds to 0 or ``1.1e-16``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> valve = {
        ...     "reliability": surv.Exponential.from_params([1e-9]),
        ...     "repairability": surv.Exponential.from_params([1 / 8]),
        ... }
        >>> pair = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": valve, "b": valve},
        ... )
        >>> f"{pair.point_unavailability(1000.0):.3e}"
        '6.400e-17'
        """
        return _windows.point_unavailability(
            self,
            x=x,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            state=state,
        )

    def mission_availability(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The expected fraction of ``[0, t]`` the system is up, with every
        component new at 0: exact, with no simulation.

        It is the mean of
        [`point_availability`][repyability.RepairableRBD.point_availability]
        over the window, integrated by 4-point Gauss-Legendre quadrature on
        pieces that end where the components' curves bend (a scheduled
        replacement, a test, a down time from a known instant) and are at
        most a few steps of the finest grid the curves are worked out on,
        each halved until the quadrature on it agrees with that on its
        halves, to about 1e-8: it takes little longer than the point
        availability. That runs up to the time when the curves have all
        settled at their long-run values (or into repeating with their
        inspections and block replacements); past it, the integral is
        extended exactly, so a mission of decades costs no more than one of
        a few years. It is what
        [`availability`][repyability.RepairableRBD.availability] estimates
        by simulation as the mean of each simulation's uptime divided by
        ``t_simulation``. As ``t`` grows it approaches ``mean_availability``,
        from which it differs by about ``b / t``, ``b`` a constant of the
        components' up and down times: positive unless their lives vary
        more than exponential ones do, and largest for components that wear
        out (Weibull shape above 1, say), which from new fail less early
        on.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths (one mission average for each).
        working_nodes : Collection[Hashable], optional
            Nodes that always work (availability 1), by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed (availability 0), by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the cut sets (``"c"``
            or ``"cuts"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The mission availability for each window, in ``t``'s shape (at
            ``t = 0``, the point availability at 0).

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_availability``.

        Examples
        --------
        One component with failure rate 0.1 and repair rate 1, over 10 time
        units: ``1 / 1.1 + 0.1 / 1.1 ** 2 * (1 - exp(-11)) / 10``.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> round(rbd.mission_availability(10.0), 4)
        0.9174
        """
        return _windows.mission_availability(
            self,
            t=t,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            state=state,
        )

    def mission_unavailability(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The expected fraction of ``[0, t]`` the system is down, with
        every component new at 0: one less ``mission_availability``, worked
        out in its own right so that a small one keeps its precision:
        the mean of
        [`point_unavailability`][repyability.RepairableRBD.point_unavailability]
        over the window, integrated as ``mission_availability`` integrates
        the availability.

        Parameters
        ----------
        t : float or array-like
            The missions' lengths, from 0 (every component new).
        working_nodes : Collection[Hashable], optional
            Nodes that always work (never down), by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed (always down), by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the cut sets (``"c"``
            or ``"cuts"``), as for ``mission_availability``.
        state : dict or str, optional
            Start from the components' current states rather than new,
            as for ``mission_availability``: ``{node: NodeState}`` or
            ``"stationary"``. By default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The expected fraction of each mission the system is down, in
            ``t``'s shape (its point unavailability at 0 for a mission of
            length 0).

        Raises
        ------
        ValueError, NotImplementedError
            As for ``mission_availability``.

        Examples
        --------
        The pair of valves of ``point_unavailability``, down together
        ``6.4e-17`` of the time once they settle, from new over a year:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> valve = {
        ...     "reliability": surv.Exponential.from_params([1e-9]),
        ...     "repairability": surv.Exponential.from_params([1 / 8]),
        ... }
        >>> pair = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": valve, "b": valve},
        ... )
        >>> f"{pair.mission_unavailability(8760.0):.3e}"
        '6.391e-17'
        """
        return _windows.mission_unavailability(
            self,
            t=t,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            state=state,
        )

    def expected_failures(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The expected number of system failures in ``[0, t)``, with every
        component new at 0: exact, with no simulation.

        A component's failure takes the system down if the component is
        critical then: if the system is up with it and down without it.
        The components fail and are repaired independently, so at a time
        ``s`` that is so with probability ``I_B^i(s)``, the component's
        Birnbaum importance at the components' point availabilities then,
        and the system's expected failures are

        ```text
        integral from 0 to t of  sum_i I_B^i(s) dM_i(s)
        ```

        ``M_i(s)`` the component's expected failures by ``s``: the
        time-dependent form of the Birnbaum/Vesely formula, whose long-run
        rate is ``system_failure_frequency``. Each ``M_i`` follows from the
        component's renewal equation, solved on the same grid as its point
        availability (see ``point_availability``), to about 1e-7; the
        integral is summed over the pieces ``mission_availability``
        integrates over, on each with the importance taken as the cubic
        through its values at the quadrature points, and extended exactly
        past the time the curves have settled. Failures at exact
        times (units dead on arrival, an exact lifetime) are taken together
        when several fall at once. It is what ``availability`` estimates as
        ``system_failures / n_simulations``: every system failure counts,
        including the zero-length outages an instantly repaired component
        causes; planned outages do not (see ``expected_events``).

        With fewer ``repair_crews`` than components, the components are not
        independent: the rate of system failures is then that of the crews'
        Markov chain in each of its states (a component up failing at its
        rate where it is critical), integrated over the chain's states over
        time by uniformization, to about 1e-13 (see ``point_availability``).
        The crews' chain takes no nested RBD's events in, as yet.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths (one count for each).
        working_nodes : Collection[Hashable], optional
            Nodes that always work, and never fail, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed, by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the cut sets (``"c"``
            or ``"cuts"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The expected number of system failures in each window, in
            ``t``'s shape.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_availability``.

        Examples
        --------
        One component with failure rate 0.1 and repair rate 1 fails at the
        rate 0.1 while it is up, ``0.1 * (10 / 1.1 + 0.1 / 1.1 ** 2 * (1 -
        exp(-11)))`` times in 10 time units:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> round(rbd.expected_failures(10.0), 4)
        0.9174
        """
        return _windows.expected_failures(
            self,
            t=t,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            state=state,
        )

    def expected_events(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ) -> ExpectedEvents:
        """What the system and each component are expected to do in
        ``[0, t)``, with every component new at 0: exact, with no
        simulation.

        The system's failures are those of ``expected_failures``, and its
        planned outages are counted the same way, from its components'
        preventive maintenance that takes time; its down time is ``t * (1
        - mission_availability(t))``. Each component's failures, corrective
        and preventive actions and tests follow from its renewal equation,
        solved on the grid of its point availability (see
        ``point_availability``), to about 1e-7, and its down time is the
        integral of its unavailability. The maintenance due at exact times
        (at a replacement age, at block times) is counted exactly, and
        maintenance of several components due at the same time takes the
        system down at most once. It is the mean of what ``availability``
        counts in each simulation of the window, and events at ``t`` itself
        fall outside the window, as there.

        A component's corrective actions, at each of which its
        ``repair_cost`` and ``replace_cost`` are charged, are its failures,
        except for hidden failures: those found by a test in the window. A
        nested RBD's failures are its system failures, and its own
        maintenance is not this RBD's to count. A forced node does nothing:
        held working it is never down, held broken it is down throughout.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths.
        working_nodes : Collection[Hashable], optional
            Nodes that always work, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed, by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the cut sets (``"c"``
            or ``"cuts"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        ExpectedEvents
            The system's expected failures, planned outages and down time,
            and each component's expected failures, corrective and
            preventive actions, tests and down time: floats for one window,
            arrays in ``t``'s shape for several.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_availability``.

        Examples
        --------
        A pump that wears out, replaced at 600 hours in about 7 (a planned
        outage), over a year:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "p"), ("p", "t")],
        ...     {
        ...         "p": {
        ...             "reliability": surv.Weibull.from_params([1000, 2.5]),
        ...             "repairability": surv.LogNormal.from_params([3, 0.5]),
        ...             "preventive": {
        ...                 "interval": 600.0,
        ...                 "duration": surv.Weibull.from_params([8, 3]),
        ...             },
        ...         }
        ...     },
        ... )
        >>> year = rbd.expected_events(8760.0)
        >>> round(year.system_failures, 3), round(year.node_preventive["p"], 3)
        (3.706, 11.275)
        >>> round(year.system_planned_outages, 3)  # each replacement
        11.275
        """
        return _windows.expected_events(
            self,
            t=t,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            state=state,
        )

    def expected_cost(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
        *,
        discount_rate: float = 0.0,
    ) -> ExpectedCost:
        """The expected cost of running the system for ``t``, from new:
        exact, with no simulation.

        Each category is its events' expected number over ``[0, t)`` (see
        ``expected_events``) times its mean cost, as ``cost`` charges them:
        ``repair_cost`` and ``replace_cost`` at each corrective action, the
        preventive cost at each preventive replacement, the inspection cost
        at each test, ``downtime_cost`` over each component's expected down
        time, ``downtime_cost_rate`` over the system's, and a maintenance
        group's set-up cost once per stop (each failure or replacement of a
        member, those at one instant one stop). It is the mean that
        ``cost`` estimates by simulation, with its distribution;
        ``total_cost`` is the long-run approximation, exact only over a
        window long compared with the components' cycles. As ``t`` grows
        this approaches ``expected_cost_rate() * t`` plus a constant, of the
        early years from new.

        Every cost is optional, and a cost given as a distribution enters
        through its mean. Only this RBD's own costs count, not those inside
        a nested ``RepairableRBD``. With nothing priced (see
        ``has_costs``), every category is 0, without any checks.

        With a ``discount_rate`` ``r`` each value is its present value,
        the costs discounted continuously from when they fall:
        ``integral from 0 to t of exp(-r s) dC(s)``, ``C`` the expected
        cost from new, worked out by parts, ``exp(-r t) C(t) + r *
        integral from 0 to t of exp(-r s) C(s) ds``, the integral by
        Gauss-Kronrod quadrature on pieces halved until the total is
        within 1e-8 of itself. It is the exact form of ``total_cost``'s
        discounted value, which spends the long-run rate from the start:
        the early years, which count most, are those that differ from it.
        The acquisition cost, at the start, is not discounted.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths.
        working_nodes : Collection[Hashable], optional
            Nodes that always work: they incur no corrective, preventive,
            inspection or downtime cost, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed: they incur no corrective cost, but
            their own downtime cost throughout, by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"`` or ``"paths"``, the default) or the cut sets (``"c"``
            or ``"cuts"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.
        discount_rate : float, optional
            The continuous discount rate per unit time of the component
            models, as for ``total_cost``, by default 0 (undiscounted).

        Returns
        -------
        ExpectedCost
            The expected cost of each window, by category and component,
            with the acquisition cost beside it: present values with a
            ``discount_rate``.

        Raises
        ------
        ValueError, NotImplementedError
            When something is priced, as for ``point_availability``.

        Examples
        --------
        A pump bought for 20,000, failing on average every 1000 hours,
        repaired in 10 at 500 per repair, with lost production at 100 per
        hour, owned for a year:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "p"), ("p", "t")],
        ...     {
        ...         "p": {
        ...             "reliability": surv.Exponential.from_params([1e-3]),
        ...             "repairability": surv.Exponential.from_params([0.1]),
        ...             "repair_cost": 500.0,
        ...             "acquisition_cost": 20000.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=100.0,
        ... )
        >>> year = rbd.expected_cost(8760.0)
        >>> round(year.mean), round(year.total)
        (13000, 33000)
        >>> round(rbd.total_cost(8760.0))  # at the long-run rate
        33010

        Owned for ten years at 7% a year, from new and at the long-run rate:

        >>> import math
        >>> r = math.log(1.07) / 8760  # per hour
        >>> round(rbd.expected_cost(87600.0, discount_rate=r).total)
        114528
        >>> round(rbd.total_cost(87600.0, discount_rate=r))
        114538
        """
        return _windows.expected_cost(
            self,
            t=t,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            state=state,
            discount_rate=discount_rate,
        )

    @contextmanager
    def _sharing_curves(self):
        """Within it, the curves ``_availability_curves`` builds are kept
        and shared, so that an analysis asking for several exact values
        builds each node's curve once (#185): keyed by the node, the
        horizon, the stages followed and its state, a curve that counts its
        events serving one that need not. Out of it, every call builds its
        own, so that a change to the diagram or its models is taken."""
        if self._curve_memo is not None:
            yield
            return
        self._curve_memo = {}
        try:
            yield
        finally:
            self._curve_memo = None

    def capacity_distribution(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> CapacityDistribution:
        """The exact long-run distribution of the system's capacity.

        Each node carries its ``capacity`` (given when the RBD was built)
        while it is up, at one level or several, and nothing while it is
        down; the system's capacity is the most that can flow through the
        nodes that are up from the input to the output. A node given no
        capacity limits nothing, unless its model has capacities of its own:
        a degrading component spends its up time in its stages in
        proportion to their mean times (by the renewal-reward theorem), and
        a nested RBD with capacities brings its own long-run
        distribution. The probability of each level is the long-run fraction of
        time the system spends at it; the capacity is positive exactly when
        the system is up, so the fraction of time it is positive is
        [`mean_availability`][repyability.RepairableRBD.mean_availability].
        With three pumps of half the demand each, one down costs nothing
        and two cost half the output.

        Exact, with no simulation: worked out from each node's long-run
        availability as ``mean_availability`` is (see
        [`RBD.system_capacity`][repyability.RBD.system_capacity]), and
        averaged over the schedules of components inspected or replaced on
        a calendar, which are down together more often than independent
        ones would be. With limited ``repair_crews``, it is averaged over
        the states of the crews' Markov chain instead (see
        ``mean_availability``).

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always being up, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being down, by default None.

        Returns
        -------
        CapacityDistribution
            The capacities the system can have, and the long-run fraction of
            time at each. Its ``meets(demand)`` is the fraction of time the
            capacity meets a demand, ``mean()`` the average capacity, and
            ``delivered_fraction(demand)`` the fraction of the demand
            delivered: the production availability.

        Raises
        ------
        ValueError
            If no node has a capacity, or as for ``mean_availability``.
        NotImplementedError
            As for ``mean_availability``, or if a degrading component is
            maintained or inspected on a schedule.

        Examples
        --------
        Three pumps of 50 each, each up 10 / 11 of the time, against a
        demand of 100:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> pump = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("s", "c"),
        ...      ("a", "t"), ("b", "t"), ("c", "t")],
        ...     {"a": pump, "b": pump, "c": pump},
        ...     capacity={"a": 50, "b": 50, "c": 50},
        ... )
        >>> capacity = plant.capacity_distribution()
        >>> capacity.levels.tolist()
        [0.0, 50.0, 100.0, 150.0]
        >>> round(capacity.meets(100), 4)  # at least two up
        0.9767
        >>> round(capacity.delivered_fraction(100), 4)
        0.988
        >>> round(capacity.mean, 2)  # 150 * 10 / 11
        136.36
        """
        return _repairable_capacity.capacity_distribution(
            self, working_nodes=working_nodes, broken_nodes=broken_nodes
        )

    def point_capacity(
        self,
        x,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        state=None,
    ) -> CapacityDistribution:
        """The distribution of the system's capacity at each time ``x``,
        with every component new at 0: exact, with no simulation.

        It is ``capacity_distribution`` at a time rather than in the long
        run: the components fail and are repaired independently, so each
        is up at ``t`` with its point availability (see
        ``point_availability``), and the system's capacity distribution is
        the same exact computation at those (see
        [`RBD.system_capacity`][repyability.RBD.system_capacity]). A node
        with levels while it works is at each with the level's share of its
        availability. A degrading component (a ``DegradingNode`` as its
        reliability) is in each stage with the probability that its first
        unit is, or a unit put into service later is, by its renewals,
        solved on the grid of its availability; a nested RBD with
        capacities brings its own distribution over time. The probability
        of a capacity above 0 is ``point_availability(x)``, and the
        distribution settles at ``capacity_distribution()``;
        ``availability(demand=...)`` estimates its mean by simulation.

        With fewer ``repair_crews`` than components the components are not
        independent: the probability of each capacity is then that of the
        states of the crews' Markov chain at it, followed over time by
        uniformization (see ``point_availability``). The chain takes no
        nested RBD's capacities in, as yet.

        Parameters
        ----------
        x : float or array-like
            Times, from 0 (every component new).
        working_nodes : Collection[Hashable], optional
            Nodes that are always up, by default None. One that takes its
            capacity from its model (a degrading component, a nested RBD)
            is at the levels it is up at, in proportion.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always down, by default None.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        CapacityDistribution
            The capacities the system can have and their probabilities:
            one per level for a scalar ``x``, else one row per level and one
            column per time (``x`` flattened). Its ``meets(demand)``,
            ``mean()`` and ``delivered_fraction(demand)`` are per time.

        Raises
        ------
        ValueError
            If no node has a capacity, or as for ``point_availability``.
        NotImplementedError
            If a degrading component is maintained or inspected on a
            schedule, or as for ``point_availability``.

        Examples
        --------
        Three pumps of 50 each, new at 0, with failure rate 0.1 and repair
        rate 1, each up at ``t`` with probability ``1 / 1.1 + 0.1 / 1.1 *
        exp(-1.1 t)``, against a demand of 100:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> pump = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("s", "c"),
        ...      ("a", "t"), ("b", "t"), ("c", "t")],
        ...     {"a": pump, "b": pump, "c": pump},
        ...     capacity={"a": 50, "b": 50, "c": 50},
        ... )
        >>> capacity = plant.point_capacity([0.0, 1.0, 100.0])
        >>> capacity.levels.tolist()
        [0.0, 50.0, 100.0, 150.0]
        >>> capacity.meets(100).round(4).tolist()  # at least two up
        [1.0, 0.9894, 0.9767]
        >>> round(plant.capacity_distribution().meets(100), 4)  # long run
        0.9767
        """
        return _repairable_capacity.point_capacity(
            self,
            x=x,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            state=state,
        )

    def mission_capacity(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        state=None,
    ) -> CapacityDistribution:
        """The distribution of the system's capacity over ``[0, t]``, with
        every component new at 0: the expected fraction of the window it
        spends at each level. Exact, with no simulation.

        It is the mean of ``point_capacity`` over the window, integrated as
        ``mission_availability`` integrates the availability: by
        Gauss-Legendre quadrature on pieces cut where the components' curves
        bend, and extended exactly past the time they have settled.
        Its ``delivered_fraction(demand)`` is the expected fraction of the
        demand delivered over the window, the production availability of a
        contract period from new, which ``availability(demand=...)``
        estimates by simulation as ``delivered_fraction``; ``meets(demand)``
        is the expected fraction of the window the capacity meets the
        demand, and ``mean()`` the average capacity. As ``t`` grows it
        approaches ``capacity_distribution()``.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths (one distribution for each).
        working_nodes : Collection[Hashable], optional
            Nodes that are always up, by default None (see
            ``point_capacity``).
        broken_nodes : Collection[Hashable], optional
            Nodes that are always down, by default None.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        CapacityDistribution
            The capacities and the expected fraction of each window spent
            at each: one per level for a scalar ``t``, else one row per
            level and one column per window (at a window of 0, the
            distribution at 0).

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_capacity``.

        Examples
        --------
        The three pumps above over 10 time units, against a demand of 100:
        more than in the long run, as every pump starts up.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> pump = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("s", "c"),
        ...      ("a", "t"), ("b", "t"), ("c", "t")],
        ...     {"a": pump, "b": pump, "c": pump},
        ...     capacity={"a": 50, "b": 50, "c": 50},
        ... )
        >>> round(plant.mission_capacity(10.0).delivered_fraction(100), 4)
        0.9896
        >>> round(plant.capacity_distribution().delivered_fraction(100), 4)
        0.988
        """
        return _repairable_capacity.mission_capacity(
            self,
            t=t,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            state=state,
        )

    def _capacity_models(self) -> dict:
        """``{node: model}`` for the nodes with no capacity entry whose model
        has a capacity distribution of its own: a nested ``RepairableRBD``
        with capacities, or a component whose reliability is a
        ``DegradingNode``."""
        out: Dict[Any, Any] = {}
        for node, component in self.components.items():
            if node in self.capacity:
                continue
            if isinstance(component, RepairableRBD):
                if component._has_capacity():
                    out[node] = component
            elif isinstance(component.reliability, DegradingNode):
                out[node] = component.reliability
        return out

    def _follow_up(self, event: Event, source) -> Event:
        """The next event of ``event``'s component, drawn from ``source``.

        A component's ``next_event()`` gives the time *to* its next event,
        measured from ``event``. A nested RBD's gives the time *of* its next
        state change: its simulation runs on the same clock, from 0.
        """
        node = event.component
        schedule = self._preventive.get(node)
        if schedule is not None:
            return self._maintained_follow_up(event, source, schedule)
        inspection = self._inspection.get(node)
        if inspection is not None:
            return self._inspected_follow_up(event, source, inspection)
        t, status = source.next_event()
        if isinstance(self.components[node], RepairableRBD):
            return Event(t, node, status, _planned(source, status))
        return Event(event.time + t, node, status)

    def _strike(self, event: Event, source) -> List[Event]:
        """A shared common cause strikes (#158, see ``_Cause``): each member
        it names that is up fails now, its own next event superseded (a
        member down stays down), and with tests that can miss a failure,
        one uniform decides for all its failures. Returns the events to
        queue: the members' failures, and the cause's next strike."""
        cause: Any = event.component
        t = event.time
        status = self.component_status
        out: List[Event] = []
        coin = source.coin() if cause.coins else None
        for member in cause.struck:
            if not status[member]:
                continue
            pending = self._ccf_pending.pop(member, None)
            if pending is not None and pending.time < self.t_simulation:
                self._cancelled[id(pending)] = pending
            if coin is not None:
                self._ccf_coins[member] = coin
            out.append(Event(t, member, False))
        later = Event(t + source.gap(), cause, False)
        if later.time < self.t_simulation:
            out.append(later)
        return out

    def _crew_follow_up(
        self, event: Event, next_event: Event
    ) -> Optional[Event]:
        """``next_event`` as the repair crews allow (see ``_Crews``): a job
        that falls due when ``event`` takes a component down waits for a
        crew (None while it does), and a crew that finishes one at
        ``event`` starts the next waiting job, whose end is queued here."""
        crews, node = self._crews, event.component
        if crews is None or node not in crews.served:
            return next_event
        if event.status:
            if node in crews.holding:
                started = crews.release(node, event.time)
                if started is not None:
                    self._crew_started(started)
            return next_event
        if next_event.status:
            return crews.request(node, event.time, next_event)
        return next_event

    def _crew_started(self, started: Tuple[Any, float, Event]) -> None:
        """A crew has started a waiting job, ``started`` as
        ``_Crews.release`` gives it: queue the end of a component's, or
        tell its standby group of a unit's."""
        key, wait, ends = started
        if isinstance(key, _Unit):
            self._groups[key.node].crew_started(key.unit, ends.time)
            return
        ends = self._started_late(key, wait, ends)
        if ends.time < self.t_simulation:
            self._event_queue.put(ends)

    def _started_late(self, node, wait: float, ends: Event) -> Event:
        """A job of ``node`` that a crew starts ``wait`` after it fell due,
        and so ends with ``ends``: a component off-line for a test does not
        age while it waits, so its hidden failure is ``wait`` later."""
        pending = self._pending_failure.get(node)
        if pending is not None:
            self._pending_failure[node] = pending + wait
        return ends

    def _crew_served(self) -> list:
        """The jobs the repair crews work on, by key: this RBD's own
        components, not a nested RBD's (which has crews of its own), and
        each unit of a standby group (``_Unit(node, unit)``)."""
        served: list = []
        for node, component in self.components.items():
            if isinstance(component, RepairableRBD):
                continue
            arrangement = self._standby.get(node)
            if arrangement is None:
                served.append(node)
            else:
                served.extend(
                    _Unit(node, unit) for unit in range(arrangement.units)
                )
        return served

    def _crews_limited(self) -> bool:
        """Whether there are fewer repair crews than components that may
        need one, so that a job may wait."""
        return self.repair_crews is not None and self.repair_crews < len(
            self._crew_served()
        )

    def _crews_couple(self) -> bool:
        """Whether a job waiting for a repair crew can tie different nodes
        together: the crews are limited, and work on more than one node's
        jobs. With a single standby group's units the only jobs, the waiting
        stays inside the group, whose own Markov chain counts the crews (see
        ``_standby_long_run``), and the nodes stay independent."""
        if not self._crews_limited():
            return False
        owners = {
            key.node if isinstance(key, _Unit) else key
            for key in self._crew_served()
        }
        return len(owners) > 1

    def _require_unlimited_crews(
        self,
        assumes: str = "these values assume",
        advice: str = "Simulate the system with availability() or cost().",
    ) -> None:
        """Raise if a job may wait for a repair crew: components then no
        longer fail and recover independently, which ``assumes`` (what
        assumes it, and ``assume``): do ``advice`` instead."""
        if self._crews_couple():
            raise NotImplementedError(
                f"With {self.repair_crews} repair crew(s) for "
                f"{len(self._crew_served())} components, a component can "
                "wait for a crew, so the components no longer fail and "
                f"recover independently, which {assumes}. {advice}"
            )

    def _no_crew_chain(self, why: str) -> NoReturn:
        """Raise that the repair crews' Markov chain does not cover this RBD,
        because ``why``."""
        raise NotImplementedError(
            f"With {self.repair_crews} repair crew(s) for "
            f"{len(self._crew_served())} components, a component can wait "
            "for a crew, and the exact long-run values come from a Markov "
            "chain of the components' states and the repair queue: "
            f"{why}. Simulate the system with availability() or cost()."
        )

    def _crew_chain_rates(self) -> Dict[Any, Tuple[float, float]]:
        """The failure and repair rates (``inf`` for an instant repair) of
        the components the repair crews work on, for their Markov chain
        (see ``_crew_chain.py``). Raise if the chain does not cover them:
        scheduled maintenance or inspection, a life or repair time that is
        not exponential, or more states than it is solved for."""
        assert self.repair_crews is not None  # limited crews only
        for node in self._standby:
            self._no_crew_chain(
                "it has no place for a standby group, which component "
                f"{node!r} is"
            )
        rates: Dict[Any, Tuple[float, float]] = {}
        for node in self._crew_served():
            component = self.components[node]
            if node in self._preventive:
                self._no_crew_chain(
                    "it has no place for scheduled maintenance, which "
                    f"component {node!r} has"
                )
            if node in self._inspection:
                self._no_crew_chain(
                    "it has no place for inspections, which component "
                    f"{node!r} has"
                )
            if node in self._imperfect:
                self._no_crew_chain(
                    "it has no place for imperfect repair, which component "
                    f"{node!r} has"
                )
            life = _constant_rate(component.reliability)
            if life is None:
                self._no_crew_chain(
                    "it needs exponential lives (a constant failure rate), "
                    f"and the life of component {node!r} is not one"
                )
            repair = _repair_rate(component.time_to_replace)
            if repair is None:
                self._no_crew_chain(
                    "it needs exponential repair times (or instant repair), "
                    f"and the repair times of component {node!r} are not"
                )
            rates[node] = (life, repair)
        count = _crew_chain.state_count(
            [self._priority.get(node, 0.0) for node in rates],
            [math.isinf(repair) for _, repair in rates.values()],
            self.repair_crews,
        )
        if count > _crew_chain.MAX_STATES:
            self._no_crew_chain(
                f"here it has {count:,} states, more than the "
                f"{_crew_chain.MAX_STATES:,} it is solved for"
            )
        return rates

    def _require_crew_chain(self) -> None:
        """Raise if the exact long-run values with limited repair crews
        cannot be computed: the Markov chain does not take common-cause
        groups in (#251, see ``_ccf_rates``), does not cover the components
        (see ``_crew_chain_rates``), or nested RBDs' calendars fall together
        (see ``_require_calendars``)."""
        _ccf_groups._require_ccf_long_run(self)
        self._crew_chain_rates()
        self._require_calendars()

    def _crew_chain(
        self, forced: frozenset = frozenset()
    ) -> "_crew_chain.CrewChain":
        """The Markov chain of the components the repair crews work on, and
        its long-run distribution (see ``_crew_chain.py``), leaving out the
        nodes in ``forced``: held working a node never fails, and held
        broken it is never repaired, so neither needs a crew. Solved once
        for each set of rates and kept."""
        self._require_crew_chain()
        rates = self._crew_chain_rates()
        assert self.repair_crews is not None  # limited crews only
        nodes = [node for node in rates if node not in forced]
        priorities = [self._priority.get(node, 0.0) for node in nodes]
        key = (
            self.repair_crews,
            tuple(zip(nodes, (rates[node] for node in nodes), priorities)),
        )
        cache = self.__dict__.setdefault("_crew_chains", {})
        if key not in cache:
            cache[key] = _crew_chain.solve(
                nodes,
                [rates[node][0] for node in nodes],
                [rates[node][1] for node in nodes],
                priorities,
                self.repair_crews,
            )
        return cache[key]

    def _chain_probabilities(
        self, working_nodes, broken_nodes
    ) -> Tuple[dict, np.ndarray]:
        """With limited repair crews, every node's availability in each
        state of their Markov chain (1 or 0 for the components the crews
        work on and those held working or broken, and a nested RBD's own
        long-run availability, as it has crews of its own), and the states'
        long-run probabilities: the long-run values are then averages over
        the states, as over the times of ``_long_run_grid``."""
        working_nodes = set(working_nodes or ())
        broken_nodes = set(broken_nodes or ())
        self._validate_node_overrides(working_nodes, broken_nodes)
        chain = self._crew_chain(frozenset(working_nodes | broken_nodes))
        size = len(chain.probabilities)
        out: dict = {
            node: chain.up[:, k].astype(float)
            for k, node in enumerate(chain.nodes)
        }
        for node, component in self.components.items():
            if node in working_nodes:
                out[node] = np.ones(size)
            elif node in broken_nodes:
                out[node] = np.zeros(size)
            elif node not in out:
                out[node] = np.full(size, float(component.mean_availability()))
        for node in self.in_or_out:
            out[node] = np.ones(size)
        return out, chain.probabilities

    def _chain_outage_terms(
        self, working_nodes, broken_nodes
    ) -> Tuple[List[Tuple[Any, float]], float]:
        """``_outage_terms`` with limited repair crews, over the
        states of their Markov chain: in each, a component that is up fails
        at its constant rate, and takes the system down if it is critical
        there. A nested RBD enters through its own frequencies, as it has
        crews of its own."""
        availability, unavailability, weights = (
            _long_run._long_run_unavailabilities(
                self, working_nodes, broken_nodes
            )
        )
        forced = set(working_nodes or ()) | set(broken_nodes or ())
        rates = self._crew_chain_rates()
        birnbaum = self._birnbaum_importance(
            availability, node_failures=unavailability
        )
        terms: List[Tuple[Any, float]] = []
        planned = 0.0
        for node in self.components:
            if node in forced:
                continue
            importance = np.asarray(birnbaum[node])
            if node in rates:
                life = rates[node][0]
                node_failures, node_planned = life * availability[node], 0.0
            else:
                node_failures, _, node_planned = _long_run._node_frequencies(
                    self, node
                )
            terms.append((node, float(weights @ (importance * node_failures))))
            planned += float(weights @ (importance * node_planned))
        return terms, planned

    def _maintained_follow_up(
        self, event: Event, source, schedule: _Preventive
    ) -> Event:
        """The next event of a component under preventive maintenance."""
        node, t = event.component, event.time
        if event.inspection:
            # Inspected, and kept (replacement on condition): its failure is
            # still ahead of it.
            return self._condition_next(node, t, schedule, source)
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
        return self._renewal(node, t, source, schedule)

    def _renewal(
        self,
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
                level = float(self.components[node].reliability.y0)
                self._levels[node] = (t, level)
                self._in_service[node] = (failure, math.inf)
                return self._condition_next(node, t, schedule, source)
            self._in_service[node] = (
                failure,
                self._replaced_at(
                    node, t if start is None else start, failure, schedule, t
                ),
            )
            return self._condition_next(node, t, schedule, source)
        if start is None:
            # Back from an imperfect repair, the unit is as old as its
            # operating time since it was renewed, which age replacement
            # counts.
            start = t - source.operated if node in self._imperfect else t
        due = self._due(
            node, schedule, start if schedule.policy == "age" else t
        )
        if due < t:
            due = t  # past its age at the start: replaced at once
        if t + life <= due:
            event = Event(t + life, node, False)
        else:
            # Maintenance that takes time is a planned outage; in zero time
            # the unit is renewed in place, and stays up.
            event = Event(due, node, schedule.duration is None, True)
        if node in self._early_members:
            # Kept, should a stop of its group renew it early (see _stop).
            self._renewed_at[node] = start
            self._pending_event[node] = event
        return event

    def _own_sources(self, states: Optional[dict] = None) -> dict:
        """What the components draw their events from when no streams are
        given: themselves, but for an imperfectly repaired one, a stand-in
        keeping its virtual age (see ``_ImperfectComponent``), and for one
        started from a state in ``states`` that draws what is left of its
        life or repair, a stand-in whose next draw can be set (see
        ``_started``): drawing from their models and numpy's global
        RNG."""
        starting = [
            node
            for node, start in (states or {}).items()
            if _draws_start(start)
        ]
        if not self._imperfect and not starting and not self.ccf_groups:
            return self.components
        sources = dict(self.components)
        for group in self.ccf_groups:
            # Each member's own cause's life, and the shared causes (#158).
            for member in group.members:
                own = _ccf_groups._own_rate(self, member)
                if own is not None:
                    sources[member] = self._own_draws(member, own)
        for cause in _ccf_groups._shared_causes(self):
            sources[cause] = _CauseDraws(cause.rate)
        for node, imperfect in self._imperfect.items():
            sources[node] = _ImperfectComponent(
                self.components[node],
                imperfect,
                model=self._duration_model(node),
            )
        for node in starting:
            component = self.components[node]
            sources[node] = _StreamedComponent(
                _ModelDraws(component.reliability),
                _ModelDraws(component.time_to_replace),
                model=self._duration_model(node),
            )
        return sources

    def _stop(self, group, t: float, trigger=None) -> List[Event]:
        """A stop of maintenance ``group`` at ``t``, opened by ``trigger``
        (a member taken down for a failure or its scheduled replacement) or
        by the system going down: each other member that is working, and at
        least its opportunity age, is renewed now, as its scheduled
        replacement would renew it, and the failure or replacement it had
        ahead of it is cancelled. A member whose own failure or replacement
        is due at ``t`` keeps it: that joins the stop. Returns the renewals
        to queue."""
        out: List[Event] = []
        for member in self._maintenance[group].members:
            if (
                member == trigger
                or member not in self._early_members
                or member not in self._renewed_at
                or member in self._renewing
                or not self.component_status[member]
            ):
                continue
            schedule = self._preventive[member]
            if t - self._renewed_at[member] < schedule.opportunity:
                continue
            pending = self._pending_event.get(member)
            if pending is not None and pending.time <= t:
                continue  # due now, as part of this stop
            self._pending_event.pop(member, None)
            if pending is not None and pending.time < self.t_simulation:
                self._cancelled[id(pending)] = pending
            renewal = Event(t, member, schedule.duration is None, True)
            self._early[id(renewal)] = renewal
            self._renewing.add(member)
            out.append(renewal)
        return out

    def _replaced_at(
        self,
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
        end = min(failure, self.t_simulation)
        dues = []
        due = self._due(node, schedule, renewed if after is None else after)
        while due < end:
            dues.append(due)
            due = self._due(node, schedule, due)
        if not dues:
            return math.inf
        # Its ages at those inspections, and at the one after the last.
        ages = np.array(dues + [due]) - renewed
        likely = _failures_between(self.components[node].reliability, ages)
        above = np.flatnonzero(likely > schedule.threshold)
        return dues[above[0]] if above.size else math.inf

    def _condition_next(
        self, node, t: float, schedule: _Preventive, source
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
        failure, replaced = self._in_service[node]
        due = self._due(node, schedule, t)
        if failure <= due:
            return Event(failure, node, False)
        if schedule.level is not None:
            process = self.components[node].reliability
            known, level = self._levels[node]
            level = level_after(
                process, level, due - known, _level_uniform(source)
            )
            if level >= schedule.level:
                return Event(due, node, schedule.duration is None, True)
            self._levels[node] = (due, level)
            left = failure_from(process, level, _level_uniform(source))
            self._in_service[node] = (due + left, math.inf)
            return Event(due, node, True, inspection=True)
        if due >= replaced:
            # Replaced, as under block replacement.
            return Event(due, node, schedule.duration is None, True)
        return Event(due, node, True, inspection=True)

    def _inspected_follow_up(
        self, event: Event, source, inspection: _Inspection
    ) -> Event:
        """The next event of a component whose failures are hidden."""
        node, t = event.component, event.time
        if not event.inspection:
            if event.status:
                # Repaired: as new, from t.
                return self._inspected_renewal(node, t, source, inspection)
            # Failed, unseen: found by the first inspection at or after t,
            # unless that one can miss it and does (a shared common cause
            # has decided for all its failures, see _strike).
            self._pending_failure[node] = None
            found = self._finds(node, inspection, t)
            coin = self._ccf_coins.pop(node, None) if self._ccf_coins else None
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
            due = self._due(node, inspection, t)
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
            return self._inspected_next(node, t, inspection)
        # A test that takes time, of a failed unit (repaired once the test is
        # done) or of a working one (off-line, and not ageing, until then).
        duration = 0.0
        if inspection.duration is not None:
            if isinstance(source, _StreamedComponent):
                duration = source.maintenance_time()
            else:
                duration = inspection.duration.random(1).item()
        failure = self._pending_failure[node]
        if failure is None:
            repair, status = source.next_event()
            return Event(t + duration + repair, node, status)
        self._pending_failure[node] = failure + duration
        return Event(t + duration, node, True, inspection=True)

    def _inspected_renewal(
        self, node, t: float, source, inspection: _Inspection
    ) -> Event:
        """The first event of a unit with hidden failures put into service
        as new at ``t``."""
        life, _ = source.next_event()
        self._pending_failure[node] = t + life
        return self._inspected_next(node, t, inspection)

    def _inspected_next(
        self, node, t: float, inspection: _Inspection
    ) -> Event:
        """The next event of a working unit with hidden failures, at ``t``:
        its failure, or the next inspection if that comes first (a failure
        at the same time comes first, and is found by it). A test that takes
        time takes the unit off-line; one in zero time leaves it up."""
        failure = self._pending_failure[node]
        due = self._due(node, inspection, t)
        if failure <= due:  # type: ignore[operator]
            return Event(failure, node, False)  # type: ignore[arg-type]
        return Event(due, node, inspection.duration is None, inspection=True)

    def next_event(self, method="p", sources: Optional[dict] = None):
        """Advance the current simulation to the system's next state change.

        Advanced, event-stepping API, after ``initialize_event_queue``. It
        takes the queued component events in time order, updating each
        component's state and queueing its next event (if before
        ``t_simulation``), until the system changes state, and returns
        that change. It gives a ``RepairableRBD`` the same event API as a
        ``NonRepairable``, so it can be a node of another RBD. Unlike
        ``NonRepairable.next_event``, which returns the time *to* the
        component's next event, this returns the time *of* the change, on
        the simulation's clock.

        When no further change happens before ``t_simulation``, it returns
        ``(t_simulation, current state)`` and discards the queue, so the
        next call raises a ValueError until ``initialize_event_queue`` is
        called again. After each call ``last_change_planned`` says whether
        the change was the start of a planned outage (preventive
        maintenance that takes the system down).

        Parameters
        ----------
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``
            or ``"paths"``, the default) or the minimal cut sets (``"c"``
            or ``"cuts"``); both give the same state.
        sources : dict, optional
            Internal: the same ``sources`` given to
            ``initialize_event_queue``. By default None: the components
            themselves.

        Returns
        -------
        tuple[float, bool]
            The time of the change, and the system's new state (True for
            working, False for failed).

        Raises
        ------
        ValueError
            If the event queue has not been initialised, or was used up by
            an earlier call; or if ``method`` is not ``"p"`` or ``"c"``.

        Examples
        --------
        A component that fails exactly 10 hours after each repair, and
        takes exactly 2 hours to repair:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.ExactEventTime.from_params(10),
        ...             "repairability": surv.ExactEventTime.from_params(2),
        ...         }
        ...     },
        ... )
        >>> rbd.initialize_event_queue(30.0)
        >>> changes = [rbd.next_event()]
        >>> while changes[-1][0] < 30.0:
        ...     changes.append(rbd.next_event())
        >>> changes[:4]
        [(10.0, False), (12.0, True), (22.0, False), (24.0, True)]

        The next failure, at 34, is outside the window, so the last call
        returns the end of the window and the state the system is left in:

        >>> changes[4]
        (30.0, True)
        """
        if not hasattr(self, "_event_queue"):
            raise ValueError("Need to initialize the event queue")
        # The components' draws come from the same sources the queue was
        # initialised with (see initialize_event_queue).
        if sources is None:
            sources = getattr(self, "_step_sources", self.components)
        new_system_state = copy(self.system_state)

        # Use a while loop to find the next time/event at which the system
        # status changes.
        while new_system_state == self.system_state:
            if self._event_queue.qsize() == 0:
                del self._event_queue
                self.last_change_planned = False
                return self.t_simulation, self.system_state

            event = self._event_queue.get()
            if self._cancelled and self._cancelled.get(id(event)) is event:
                del self._cancelled[id(event)]
                continue  # superseded by an early renewal, or a strike
            if event.component.__class__ is _Cause:
                # A shared common cause strikes (#158, see _strike).
                for follow in self._strike(event, sources[event.component]):
                    self._event_queue.put(follow)
                continue
            renewing_early = False
            if self._early and self._early.get(id(event)) is event:
                del self._early[id(event)]
                self._renewing.discard(event.component)
                renewing_early = True
            group = self._groups.get(event.component)
            if group is not None:
                # A standby group's own event (see _StandbyGroup).
                if not group.holds(event):
                    continue  # superseded by an earlier one
                now, _ = group.advance(event.time)
                if now == self.component_status[event.component]:
                    continue
                self.component_status[event.component] = now
                if now != self.system_state:
                    new_system_state = self.is_system_working(
                        self.component_status, method
                    )
                continue  # the group has queued its next event
            was = self.component_status[event.component]
            self.component_status[event.component] = event.status
            node = event.component
            if node in self._member_group and not renewing_early:
                # A failure, or a scheduled replacement starting (or done in
                # place), opens a stop of its maintenance group.
                if (not event.status and not event.inspection) or (
                    event.preventive and event.status and was
                ):
                    for renewal in self._stop(
                        self._member_group[node], event.time, node
                    ):
                        self._event_queue.put(renewal)
            # Only a change against the system's state can change it (the
            # structure is coherent; see _replicate).
            if event.status != self.system_state:
                new_system_state = self.is_system_working(
                    self.component_status, method
                )
                if self.system_state and not new_system_state:
                    for name, spec in self._maintenance.items():
                        if spec.system_down:
                            for renewal in self._stop(name, event.time):
                                self._event_queue.put(renewal)

            follow = self._follow_up(event, sources[event.component])
            next_event: Optional[Event] = (
                follow
                if self._crews is None
                else self._crew_follow_up(event, follow)
            )
            # But only queue up the event if it occurs before the end
            # of the simulation
            if next_event is not None and next_event.time < self.t_simulation:
                self._event_queue.put(next_event)
            if node in self._ccf_members:
                self._ccf_pending[node] = next_event

        self.system_state = new_system_state
        # A system taken down by maintenance or a test is a planned outage.
        self.last_change_planned = (
            event.preventive or event.inspection
        ) and not new_system_state

        return event.time, self.system_state

    def availability(
        self,
        t_simulation: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        mc_samples: Optional[int] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        *,
        tolerance: Optional[float] = None,
        confidence: float = 0.95,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
        demand: Optional[float] = None,
        engine: str = "auto",
        state=None,
        curve_points: Optional[int] = None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        control_variate: Optional[bool] = None,
        conditional: Optional[bool] = None,
    ) -> AvailabilityResult:
        """Simulate the system's availability over ``[0, t_simulation]``.

        Runs ``mc_samples`` independent Monte-Carlo (discrete-event)
        simulations of the system from time 0, each starting with every
        component working and new (except ``broken_nodes``), or from the
        components' current ``state``. Each component alternates failure and
        repair independently of the others and of the system state: it
        fails after a time drawn from its reliability model and is
        restored, as good as new, after a time drawn from its repairability
        model. A component under preventive maintenance is also replaced on
        its schedule (see the class docstring), down for the maintenance
        time if it takes any: a planned outage, which counts in every up
        and down time but is not a failure. A nested ``RepairableRBD`` runs
        its own simulation on the same clock, and is down while its system
        is down. The system state is re-evaluated at every component event.

        The result holds the estimated availability over time, the system's
        and every node's total up and down time, the number of system
        failures and restorations, and criticality measures that attribute
        the system's up and down time, failures and restorations to the
        nodes (see [`Criticalities`][repyability.Criticalities]). When
        something is priced (see ``has_costs``), the same simulations also
        accumulate costs, and the result's ``cost`` is a
        [`CostResult`][repyability.CostResult]; otherwise it is None. For
        exact long-run values, use ``mean_availability`` and the other
        steady-state methods.

        When nodes have capacities (see the ``capacity`` argument of the
        class), the same simulations also follow what the system can
        deliver: its capacity after every component event (a failure that
        leaves the system up can still take some of its capacity), the time
        it spends at each capacity, and the fraction of ``demand`` it
        delivers (see
        [`AvailabilityResult`][repyability.AvailabilityResult]). The
        simulation draws only whether each component is up, so a node
        working at several levels counts at each in proportion to its
        probability. Nodes taking their capacity from their models (a
        ``DegradingNode``'s stages, or a nested RBD's capacities) are not
        followed: give them a capacity, or use ``capacity_distribution`` for
        the long run.

        Parameters
        ----------
        t_simulation : float
            Length of the window each simulation covers, in the time unit
            of the component models. Events at or after it are not
            simulated.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window: they never fail. By
            default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window: they are down from time
            0 and never repaired. By default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``
            or ``"paths"``, the default) or the minimal cut sets (``"c"``
            or ``"cuts"``); the results are identical.
        mc_samples : int, optional
            Number of simulations, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None. Each random
            quantity -- a component's times to failure, its repair times,
            its maintenance or test times, a cost given as a distribution
            -- is drawn from a stream of its own, named by the component's
            place and the quantity, and seeded from ``seed``: simulation
            ``r`` draws the same numbers however many simulations the run
            has, in however many processes (``n_jobs``), and whichever
            ``engine`` runs it, and one component's draws never depend on
            another's. With None, the seed is a number drawn from numpy's
            global RNG, so seeding that with ``np.random.seed(s)`` makes
            the run the one ``seed=s`` gives; the global RNG is otherwise
            left as it was. Models whose draws cannot be streamed (other
            than surpyval's parametric ones, and the component classes'
            subclasses) draw from the global RNG, seeded afresh for each
            simulation.
        tolerance : float, optional
            Simulate until the mean availability over the window (the
            fraction of it the system is up) is known to within
            ``tolerance`` either side, at ``confidence``: after the first
            ``mc_samples`` simulations, and each further ``mc_samples``, the
            run stops once the half-width of the confidence interval of
            ``result.mean_availability_interval()`` is at most
            ``tolerance``, or ``max_samples`` simulations have run (then
            with a RuntimeWarning). By default None: exactly
            ``mc_samples``. A run that stops after ``n`` simulations gives
            the result of a run of ``mc_samples=n``.
        confidence : float, optional
            The confidence level ``tolerance`` is judged at, by default
            0.95.
        max_samples : int, optional
            The most simulations a run to ``tolerance`` makes, by default
            100 times ``mc_samples``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs, by default False: in
            the second simulation of a pair, each stream (see ``seed``)
            draws ``1 - u`` for every uniform ``u`` it drew in the first, in
            the same order, so a failure or repair that was early in one is
            late in the other. Each simulation is still a correct one, but
            the pair's results are negatively correlated, so their mean
            varies less than two independent simulations': a narrower
            interval for the same ``mc_samples``. The pairs, not the
            simulations, are independent, and the result's intervals are
            worked out from the pairs' means. ``mc_samples`` (and
            ``max_samples``) must be even, and every draw must come from a
            stream (surpyval parametric models), else
            ``NotImplementedError``.
        n_jobs : int, optional
            Run the simulations in parallel on ``n_jobs`` CPUs (-1: all of
            them), by default None: one. The compiled engine runs on that
            many threads; the Python one sends blocks of ``PARALLEL_BLOCK``
            (250) simulations to that many processes, each with a copy of
            the RBD, which pays off for long simulations. The result is the
            same to the last bit for any ``n_jobs``, and without it.
        engine : str, optional
            What runs the simulations: ``"python"``, ``"numba"`` (compiled,
            with numba, an optional dependency: ``pip install
            "repyability[fast]"``) or ``"auto"`` (the default), which
            compiles when numba is installed, the system is one the
            compiled engine simulates, and the run is long enough to pay
            for loading it. The engines give the same results to the last
            bit; what the compiled engine does not simulate runs in Python,
            and another package can add an engine of its own (see
            ``repyability.rbd.engines``). See
            [The compiled engine](../guide/simulation.md#the-compiled-engine).
        demand : float, optional
            The demand the delivered fraction is measured against, in the
            capacities' units, when nodes have capacities. By default the
            system's capacity with every component up (at its highest
            level): its design capacity. If that is unlimited, no delivered
            fraction is worked out unless a demand is given.
        state : dict, optional
            Start each simulation from the components' current states
            rather than new: ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), and a nested RBD's own
            such dict for a nested RBD. A component up at an age draws what
            is left of its life given the age, one down draws what is left
            of its repair (or maintenance) given how long it has taken so
            far, from a stream of its own, and its calendar (block
            replacement, inspections) is shifted by its phase; a component
            left out starts new. Such a run is simulated in Python. By
            default None: every component new at 0.
        curve_points : int, optional
            Keep the availability over time on a grid of ``curve_points``
            steps, ``t_simulation * k / curve_points`` for ``k`` from 0,
            rather than at every time a simulated system changed state:
            the simulations count their changes in the grid's steps, so
            the curve costs ``curve_points`` counts however many run, and
            its values at the grid's times are those the full curve takes
            there, exactly (and only there). With capacities, so does the
            capacity curve (``capacity_timeline``), each step's changes
            summed exactly. For a large run, whose full curve has a point
            at every change of every simulation. By default None: the full
            curve. Everything else in the result is the same either way.
        shard_map : callable, optional
            Run the simulations as shards (see ``shards``) through this
            map, wherever it sends them: ``shard_map(run_shard, shards)``
            must give back ``run_shard``'s result for each shard, in any
            order, as ``map`` does. The built-in ``map`` runs them here;
            ``concurrent.futures.ProcessPoolExecutor(...).map`` in other
            processes; Ray's, Dask's or a batch system's on other machines
            (see [Shards](../guide/simulation.md#shards)). The result is the
            same to the last bit; a run to a ``tolerance`` maps a round of
            shards at a time. Each shard is simulated by ``engine`` where it
            runs, and carries the system as JSON, so the system must save
            (``to_dict``). Not with ``n_jobs``: give the map's workers the
            CPUs. By default None.
        shard_size : int, optional
            The simulations to a shard, with ``shard_map``: as ``shards``'
            ``size``, by default as many as make 1024 or more.
        control_variate : bool, optional
            Control the estimate of the mean availability by the system's
            exact twin: the same diagram, its components failing and
            repaired independently, whose mean availability over the
            window is exact, simulated alongside with common random
            numbers. By default None: where the exact methods work out the
            system's expected values over the window, the result's means
            (``mean_availability``, the cost's ``mean`` and breakdowns) and
            their intervals are those values, with no error, the twin
            being the system itself; otherwise they may be
            taken given the modules (see ``conditional``). False forces a
            plain simulation, whose means are the simulations' own. True
            controls the run by the twin, whose values the result's
            ``control_variate`` holds (see
            [`ControlVariate`][repyability.ControlVariate]); every draw must
            then come from a stream (surpyval parametric models). See
            [An exact twin](../guide/simulation.md#an-exact-twin).
        conditional : bool, optional
            Take the expected values given the histories of the dependent
            modules: the nodes whose values over time the exact
            methods do not work out (``analysis_routes`` names them), every
            other node independent of them. Each simulation's expected
            values given its modules' histories are exact, and their mean
            has less variance than the simulations' own. By default None:
            where the system has modules and the exact methods do not take
            it whole, the result's means and their intervals are those
            expected values (see
            [`ConditionalRun`][repyability.ConditionalRun]), the whole
            system simulated as a plain run simulates it. False keeps the
            simulations' own means. True simulates only the modules, for a
            fraction of the work, but each simulation's values are then
            expected values: the cost's ``percentile`` and ``std`` refuse,
            and ``criticalities`` is None. See
            [Conditional runs](../guide/simulation.md#conditional-runs).
        Returns
        -------
        AvailabilityResult
            The availability over time (``timeline``, ``availability``; on
            the grid, with ``curve_points``), the up and down totals summed
            over the simulations, the system
            failure, planned outage and restoration counts, the
            ``criticalities``, the ``cost`` and each simulation's up time
            (``uptimes``); with capacities, also the capacity over time,
            the time at each capacity and the delivered fraction.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets, if ``method`` is not ``"p"`` or
            ``"c"``, or if ``mc_samples``, ``tolerance``, ``confidence``,
            ``max_samples``, ``n_jobs`` or ``engine`` is invalid
            (``mc_samples`` odd with ``antithetic``, ``max_samples`` without
            a tolerance or below ``mc_samples``, ...); or if a ``demand`` is
            not a positive, finite number, or is
            given for an RBD with no capacities; or if ``shard_map`` is not
            callable or comes with ``n_jobs``, or with ``control_variate``
            for a twin other than the system itself, or ``shard_size`` is
            not a whole number at least 1 or comes without ``shard_map``.
        NotImplementedError
            With ``antithetic``, if a component's draws cannot be replayed;
            with capacities, if a node takes its capacity from its model;
            with ``engine="numba"``, if the compiled engine does not
            simulate the system; with ``shard_map``, if the system cannot
            be saved as JSON; with ``control_variate``, if the system has
            no exact twin (a component's model is a probability, or its
            life is simulated) or a draw cannot come from a stream; with
            ``conditional=True``, if the system's other nodes cannot be
            taken exactly given the modules (see ``conditional``).
        ImportError
            With ``engine="numba"``, if numba is not installed.

        Examples
        --------
        Two components in series, each with MTTF 10 and MTTR 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> unit = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")], {"a": unit, "b": unit}
        ... )
        >>> result = rbd.availability(t_simulation=50, mc_samples=200, seed=0)
        >>> float(result.timeline[0]), float(result.availability[0])
        (0.0, 1.0)
        >>> float(result.timeline[-1])
        50.0

        The fraction of the window the system was up is close to the
        long-run availability, ``(10 / 11) ** 2`` (over a short window it is
        a little above it on average, as every simulation starts up):

        >>> window = result.n_simulations * result.time_simulated_to
        >>> round(float(result.system_uptime) / window, 4)
        0.832
        >>> round(rbd.mean_availability(), 4)
        0.8264

        In series the system is up only while every node is up:

        >>> oci = result.criticalities.operational_criticality_index
        >>> {node: round(float(v), 4) for node, v in oci.up.items()}
        {'a': 1.0, 'b': 1.0}
        """
        N = 10_000 if mc_samples is None else mc_samples

        return self._simulated(
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            verbose,
            seed,
            tolerance=tolerance,
            confidence=confidence,
            max_N=max_samples,
            antithetic=antithetic,
            n_jobs=n_jobs,
            target="availability",
            demand=demand,
            engine=engine,
            state=state,
            curve_points=_curve_points(curve_points),
            shard_map=shard_map,
            shard_size=shard_size,
            control_variate=control_variate,
            conditional=conditional,
        )

    def simulate_timelines(
        self,
        t_simulation: float,
        mc_samples: Optional[int] = None,
        seed: Optional[int] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        antithetic: bool = False,
        engine: str = "auto",
        n_jobs: Optional[int] = None,
        start: int = 0,
        state=None,
    ) -> "TimelineSimulation":
        """Simulate the system over ``[0, t_simulation]`` and keep each
        simulation's up/down histories as timelines: every component's and
        the system's, with the component that caused each of the system's
        changes.

        These are the simulations
        [`availability`][repyability.RepairableRBD.availability] runs with
        the same ``seed`` (and options), kept whole rather than added up:
        so any measure of a history can be read off them (the time to the
        first system failure, the longest outage, the outages per year),
        and they merge with other timelines (see
        ``repyability.timelines``). Their histories are the event loop's,
        whichever engine makes them: each simulation's system uptime is
        ``availability``'s to the last bit.

        The event loop records them as it runs, compiled with numba where
        it can (``pip install "repyability[fast]"``), as ``availability``
        runs. On the Python engine, independent components (plain units
        with streamed models, and nested RBDs of them) have their histories
        drawn straight from their streams instead, a batch of simulations
        at once, and the system's merged from theirs (see
        [`RBD.system_timeline`][repyability.RBD.system_timeline]): a
        simulation in which components change at the same instant is run
        in the event loop, which orders them as it does.

        Parameters
        ----------
        t_simulation : float
            The window's end.
        mc_samples : int, optional
            The number of simulations. By default 1,000.
        seed : int, optional
            Seeds the simulations, as for ``availability``. By default
            unseeded.
        working_nodes : collection, optional
            Nodes held working throughout.
        broken_nodes : collection, optional
            Nodes held broken throughout.
        antithetic : bool, optional
            Simulate in antithetic pairs (``mc_samples`` even), as for
            ``availability``. By default False.
        engine : str, optional
            ``"auto"`` (the default), ``"python"`` or ``"numba"``, as for
            ``availability``; ``"auto"`` compiles a run long enough to
            repay loading numba, when it is installed and simulates the
            system. Another package's engine records no histories.
        n_jobs : int, optional
            Run on this many processes (the Python event loop) or threads
            (numba, and the streams), ``-1`` for one per CPU. The histories
            are the same however many. By default one.
        start : int, optional
            The run's first simulation to make: simulations ``start`` to
            ``start + mc_samples - 1`` of the run ``seed`` seeds (so parts
            of one run, made apart, join with ``TimelineSimulation.join``).
            Even with ``antithetic``. By default 0.
        state : dict, optional
            Start from the components' current states rather than new, as
            for ``availability``: ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD. A component down at 0 starts its history
            down, and the system its own in its state then, with no change
            at 0. These are the simulations ``availability(state=...)``
            runs; a component started from a state draws its first life,
            repair or maintenance from a stream of its own, which the event
            loop follows, in Python. By default None: every component new.

        Returns
        -------
        TimelineSimulation
            ``system`` and ``components`` as
            [`Timelines`][repyability.Timelines], one history per
            simulation, and how they were made (``engine``, ``method``).

        Raises
        ------
        ValueError
            If ``t_simulation`` is not a positive, finite time,
            ``mc_samples`` is not a positive integer (even with
            ``antithetic``), a working or broken node is unknown, the input
            or output node, or in both, ``engine`` is not one of those,
            ``start`` is negative (or odd with ``antithetic``), ``start``
            is given without a ``seed``, or a state is not one a
            simulation can start from (see ``availability``).
        NotImplementedError
            If a common-cause group's members are not exponential, are
            held, or start from a state (see ``availability``); with
            ``antithetic``, if a component's draws cannot be replayed; or,
            with ``engine="numba"``, if the compiled engine does not
            simulate the system.
        ImportError
            With ``engine="numba"``, if numba is not installed.

        Examples
        --------
        Two pumps in parallel: how long until the plant first fails, and
        which pump's failure did it:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> unit = {
        ...     "reliability": surv.Weibull.from_params([100, 1.5]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> runs = rbd.simulate_timelines(1000.0, mc_samples=2000, seed=1)
        >>> bool(runs.system.first_failure.min() > 0)
        True
        >>> sorted(runs.system.failures_by_cause())
        ['a', 'b']

        The same simulations ``availability`` runs:

        >>> result = rbd.availability(1000.0, mc_samples=2000, seed=1)
        >>> bool(np.array_equal(runs.system.uptime, result.uptimes))
        True
        """
        return _timeline_runs.simulate(
            self,
            t_simulation,
            1000 if mc_samples is None else mc_samples,
            seed,
            working_nodes,
            broken_nodes,
            antithetic,
            engine,
            n_jobs,
            start,
            state,
        )

    def simulate_chunk(
        self,
        t_simulation: float,
        start: int,
        stop: int,
        *,
        seed: int,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        antithetic: bool = False,
        demand: Optional[float] = None,
        engine: str = "auto",
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        state=None,
        curve_points: Optional[int] = None,
        control_variate: Optional[bool] = None,
        conditional: Optional[bool] = None,
    ) -> "SimulationChunk":
        """Run simulations ``start`` to ``stop - 1`` of the run
        ``availability(t_simulation, mc_samples=N, seed=seed, ...)`` makes,
        for any ``N`` of at least ``stop``, and return them as a chunk to
        merge with the run's others.

        Each simulation draws from streams of its own, seeded from ``seed``
        and its position in the run (see
        [Random streams](../guide/simulation.md#random-streams)), so it
        comes out the same wherever and whenever it runs, by either engine,
        in any company. A run can so be split across processes, machines or
        preemptible workers: each runs its chunk, saves it
        (``SimulationChunk.to_json``) and sends it back, and
        ``availability_from_chunks`` merges the chunks into the run's
        [`AvailabilityResult`][repyability.AvailabilityResult]. Chunks of
        simulations ``0`` to ``N - 1`` give the result of
        ``availability(..., mc_samples=N)``, to the last bit: the same
        simulations, so the same per-simulation values and timeline, and
        the same totals, which are kept exactly however the run is cut.

        Parameters
        ----------
        t_simulation : float
            The window each simulation covers, from 0.
        start : int
            The first simulation to run, by its position in the run (from
            0; even with ``antithetic``, so that a chunk holds whole pairs).
        stop : int
            The simulation after the last to run (above ``start``; even with
            ``antithetic``).
        seed : int
            The run's seed, required: chunks of a run share it.
        working_nodes : Collection[Hashable], optional
            Nodes held working, as for ``availability``.
        broken_nodes : Collection[Hashable], optional
            Nodes held broken, as for ``availability``.
        method : str, optional
            ``"p"`` or ``"c"``, as for ``availability``, by default ``"p"``.
        antithetic : bool, optional
            Simulate in antithetic pairs, as for ``availability``, by
            default False.
        demand : float, optional
            A demand on the system's capacity, as for ``availability``.
        engine : str, optional
            ``"auto"``, ``"python"`` or ``"numba"``, as for
            ``availability``: the chunk is the same either way.
        n_jobs : int, optional
            Run the chunk on several CPUs, as for ``availability``.
        verbose : bool, optional
            Show a progress bar, by default False.
        state : dict, optional
            The components' states at the start, as for ``availability``
            (the chunks of a run share it); by default None: new.
        curve_points : int, optional
            Count the curve on a grid, as for ``availability`` (the chunks
            of a run share it): the chunk then holds the grid's counts
            rather than every change. By default None: every change.
        control_variate : bool, optional
            None (the default) or False, as for ``availability``:
            the means the merged result takes, kept with the chunk (the
            chunks of a run share it). The simulations are the same either
            way; False gives the merged result the simulations' own means.
            True, which simulates a twin alongside the system, is not taken
            by a chunk.
        conditional : bool, optional
            None (the default) or False, as for ``availability``, kept with
            the chunk likewise: False leaves the merged result's means the
            simulations' own where no exact ones are taken. True, which
            simulates the modules alone, is not taken by a chunk.

        Returns
        -------
        SimulationChunk
            The simulations' totals, with the run's settings.

        Raises
        ------
        ValueError
            If ``seed`` is None, ``start`` and ``stop`` are not whole
            numbers with ``0 <= start < stop`` (even, with ``antithetic``),
            or the other arguments are invalid, as for ``availability``.
        NotImplementedError
            As for ``availability``.

        See Also
        --------
        shards : The run's simulations as plain data, to run anywhere.

        Examples
        --------
        Two chunks of a run, merged, are the run:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "t")],
        ...     {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
        ... )
        >>> first = rbd.simulate_chunk(100.0, 0, 300, seed=1)
        >>> rest = rbd.simulate_chunk(100.0, 300, 500, seed=1)
        >>> merged = rbd.availability_from_chunks([first, rest])
        >>> whole = rbd.availability(100.0, mc_samples=500, seed=1)
        >>> bool((merged.uptimes == whole.uptimes).all())
        True
        """
        if seed is None:
            raise ValueError(
                "A chunk needs the run's seed: the chunks of a run share it."
            )
        for name, value in (("start", start), ("stop", stop)):
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"{name} must be a whole number, got {value!r}."
                )
        if not 0 <= start < stop:
            raise ValueError(
                f"The chunk must be simulations start to stop - 1, with 0 <= "
                f"start < stop; got start={start!r}, stop={stop!r}."
            )
        if antithetic and (start % 2 or stop % 2):
            raise ValueError(
                "With antithetic pairs, start and stop must be even, so that "
                "the chunk holds whole pairs."
            )
        means = _chunk_means(control_variate, conditional)
        working = set() if working_nodes is None else set(working_nodes)
        broken = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working, broken)
        states = _curves._simulation_states(self, state, working | broken)
        chunk = self._chunk(
            t_simulation,
            int(start),
            int(stop),
            _streams.entropy_of(seed),
            working,
            broken,
            method,
            antithetic,
            demand,
            engine,
            None if n_jobs is None else montecarlo.jobs(n_jobs),
            verbose,
            states,
            _curve_points(curve_points),
        )
        # Kept only where they differ from availability's defaults, so that
        # a chunk is saved as it was before them.
        chunk.settings.update(
            {name: value for name, value in means.items() if value is False}
        )
        return chunk

    def _chunk(
        self,
        t_simulation: float,
        start: int,
        stop: int,
        entropy: int,
        working: set,
        broken: set,
        method: str,
        antithetic: bool,
        demand: Optional[float],
        engine: str,
        jobs: Optional[int],
        verbose: bool,
        states: dict,
        curve_points: Optional[int],
    ) -> "SimulationChunk":
        """Simulations ``start`` to ``stop - 1`` of the run of ``entropy``
        (see ``simulate_chunk``, whose arguments, checked, these are), as
        a chunk with the run's settings."""
        from repyability.rbd.chunks import SimulationChunk

        capacity = self._chunk_capacity(broken, method, demand)
        tally = self._run(
            t_simulation,
            working,
            broken,
            method,
            stop - start,
            verbose,
            None,
            antithetic,
            capacity=capacity,
            jobs=jobs,
            engine=engine,
            entropy=entropy,
            first=start,
            states=states,
            curve_points=curve_points,
        )
        settings = {
            "t_simulation": float(t_simulation),
            "entropy": entropy,
            "method": method,
            "working_nodes": sorted(working, key=repr),
            "broken_nodes": sorted(broken, key=repr),
            "antithetic": bool(antithetic),
            "demand": None if demand is None else float(demand),
            "fingerprint": self._fingerprint(),
            "state": _state_key(states),
            "curve_points": tally.curve_points,
        }
        return SimulationChunk([(start, stop)], settings, tally)

    def _chunk_capacity(
        self, broken: set, method: str, demand: Optional[float]
    ) -> Optional[_CapacityRecorder]:
        """Check a chunk's (or shard's) ``method`` and ``demand``, and give
        the recorder of the capacities it follows, if the system has
        them."""
        self.is_system_working(
            {c: c not in broken for c in self.components}, method
        )
        if self._has_capacity():
            return _CapacityRecorder(self, demand)
        if demand is not None:
            raise ValueError(
                "A demand is measured against capacities, and no node has "
                "one: give them with capacity={node: capacity}."
            )
        return None

    def shards(
        self,
        t_simulation: float,
        mc_samples: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        size: Optional[int] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        antithetic: bool = False,
        demand: Optional[float] = None,
        engine: str = "auto",
        state=None,
        curve_points: Optional[int] = None,
    ) -> List[bytes]:
        """The run ``availability(t_simulation, mc_samples, seed=seed,
        ...)`` makes, as shards: ranges of its simulations as plain data,
        to run anywhere (see ``repyability.rbd.shards``).

        Each shard is JSON: this system as ``to_dict`` gives it, the run's
        entropy (drawn once, here, from the seed) and settings, and its
        range. ``repyability.run_shard`` runs one in any process, on any
        machine and by any executor (``concurrent.futures``, Ray, Dask, a
        batch system with ``python -m repyability.rbd.shards < shard.json
        > partial.npz``), and gives back its partial, the simulations'
        totals as ``.npz`` bytes; ``availability_from_chunks(partials,
        mc_samples)`` puts them together, in any order, into the run's
        result, to the last bit. ``availability(..., shard_map=...)`` does
        it all through a ``map`` of your choosing.

        Parameters
        ----------
        t_simulation : float
            The window each simulation covers, from 0.
        mc_samples : int, optional
            The run's simulations, by default 10_000.
        seed : int, optional
            The run's seed, by default None: drawn from numpy's global RNG,
            as ``availability`` does.
        size : int, optional
            Simulations to a shard, rounded up to a whole number of the
            run's widest block of draws (see
            [Random streams](../guide/simulation.md#random-streams)), so that
            no two shards draw the same block; by default as many as make
            1024 or more. A shard should run for some seconds, to repay a
            worker's start.
        working_nodes : Collection[Hashable], optional
            Nodes held working, as for ``availability``.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed, as for ``availability``.
        method : str, optional
            ``"p"`` or ``"c"``, as for ``availability``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs, as for
            ``availability``: ``mc_samples`` must then be even.
        demand : float, optional
            The demand the delivered fraction is measured against, as for
            ``availability``.
        state : dict, optional
            The components' states at the start, as for ``availability``.
        engine : str, optional
            What runs each shard: as for ``availability``, in the worker,
            by default ``"auto"``.
        curve_points : int, optional
            Count the curve on a grid, as for ``availability``, so that a
            partial carries the grid's counts rather than every change. By
            default None.

        Returns
        -------
        list of bytes
            The shards, in order of their simulations, as JSON.

        Raises
        ------
        ValueError
            If an argument is invalid, as for ``availability`` (with
            ``antithetic``, ``mc_samples`` must be even).
        NotImplementedError
            If this system cannot be saved as JSON (a model that is not a
            surpyval parametric one), as a shard carries it: run it with
            ``availability(..., n_jobs=...)`` instead.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "t")],
        ...     {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
        ... )
        >>> len(rbd.shards(100.0, 10_000, seed=1, size=4096))
        3
        """
        N = 10_000 if mc_samples is None else mc_samples
        montecarlo.check_count(N, antithetic, "mc_samples")
        working = set() if working_nodes is None else set(working_nodes)
        broken = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working, broken)
        states = _curves._simulation_states(self, state, working | broken)
        capacity = self._chunk_capacity(broken, method, demand)
        _ccf_groups._require_free_members(self, working, broken)
        _ccf_groups._require_groups_simulated(self, states)
        t_simulation = simulation_window(t_simulation)
        entropy = _streams.entropy_of(seed)
        template = self._shard_template(
            t_simulation,
            entropy,
            working,
            broken,
            method,
            antithetic,
            demand,
            engine,
            states,
            _curve_points(curve_points),
        )
        # What a run checks before it simulates (see _run).
        plan, complete = self._stream_plan(
            t_simulation, entropy, antithetic, states
        )
        if antithetic and not complete:
            raise NotImplementedError(_UNSTREAMED)
        self._simulation_engine(
            engine, plan, capacity, N, here=False, states=states
        )
        step = self._shard_size(plan, size)
        return [
            _shard_bytes(template, first, min(first + step, N))
            for first in range(0, N, step)
        ]

    def _shard_template(
        self,
        t_simulation: float,
        entropy: int,
        working: set,
        broken: set,
        method: str,
        antithetic: bool,
        demand: Optional[float],
        engine: str,
        states: dict,
        curve_points: Optional[int],
    ) -> dict:
        """What every shard of a run holds (see ``shards``): all but its
        range."""
        from repyability.rbd.shards import KIND

        return {
            "kind": KIND,
            "repyability_version": __version__,
            "system": self._shard_system(),
            "t_simulation": float(t_simulation),
            "entropy": int(entropy),
            "method": method,
            "working_nodes": sorted(working, key=repr),
            "broken_nodes": sorted(broken, key=repr),
            "antithetic": bool(antithetic),
            "demand": None if demand is None else float(demand),
            "engine": engine,
            "state": _state_key(states),
            "curve_points": curve_points,
        }

    def _shard_system(self) -> dict:
        """This system as a shard carries it: ``to_dict``'s JSON data, with
        every model of a class that loads back as itself (see
        ``serialisation.exactly``), so that a worker simulates this
        system."""
        from repyability.rbd.serialisation import exactly

        try:
            with exactly():
                system = self.to_dict()
            json.dumps(system)
            if type(self) is not RepairableRBD:
                raise NotImplementedError(
                    f"it is a {type(self).__name__}, which loads as a "
                    "RepairableRBD"
                )
        except SAVE_ERRORS as error:
            raise NotImplementedError(
                "A shard carries the system as JSON, and this one cannot be "
                f"saved: {str(error).rstrip('.')}. Run it with "
                "availability(n_jobs=...) instead."
            ) from None
        return system

    @staticmethod
    def _shard_size(plan: _streams.Plan, size: Optional[int]) -> int:
        """Simulations to a shard of the run of ``plan``: ``size`` (by
        default 1024), rounded up to a whole number of its widest block of
        draws, so that no two shards draw the same block."""
        wanted = 1024 if size is None else size
        if (
            isinstance(wanted, bool)
            or not isinstance(wanted, (int, np.integer))
            or wanted < 1
        ):
            raise ValueError(
                "The size of a shard must be a whole number of simulations, "
                f"at least 1; got {size!r}."
            )
        widest = max(
            [plan.columns(spec) for spec in plan.specs.values()]
            + [2 if plan.antithetic else 1]
        )
        return int(-(-int(wanted) // widest) * widest)

    def availability_from_chunks(
        self,
        chunks,
        mc_samples: Optional[int] = None,
        *,
        allow_gaps: bool = False,
        control_variate: Optional[bool] = None,
        conditional: Optional[bool] = None,
    ) -> AvailabilityResult:
        """The result of the simulations of ``chunks`` (see
        ``simulate_chunk``, and the partials ``run_shard`` gives back from
        ``shards``), as ``availability`` gives it: merged, in order of
        their positions in the run, whatever order they come in.

        Chunks of simulations ``0`` to ``N - 1`` give the result of
        ``availability(t_simulation, mc_samples=N, seed=seed, ...)``: the
        same per-simulation values (``uptimes``, the costs' ``samples``) and
        timeline, and the same totals, to the last bit: every total is kept
        exactly and rounded once, so it does not depend on how the run was
        cut.
        The chunks must hold simulations ``0`` to ``N - 1`` with none
        missing, so a lost chunk (a shard that never came back) is not
        taken for a smaller run; with ``allow_gaps=True`` they give
        the result of whichever simulations they hold. With costs, the
        result's ``cost`` is the simulated cost distribution, as ``cost``
        gives it.

        Parameters
        ----------
        chunks : SimulationChunk, or an iterable of them
            The chunks, their ``to_dict`` data, or their ``to_npz`` bytes
            (a shard's partial): of one run of this system, holding
            different simulations.
        mc_samples : int, optional
            The run's number of simulations: the chunks must then hold
            simulations ``0`` to ``mc_samples - 1``, all of them, which
            also refuses a missing last chunk. By default None: simulations
            ``0`` to however many they hold.
        allow_gaps : bool, optional
            Take chunks with simulations missing between or before them,
            for the result of those they hold. By default False.
        control_variate : bool, optional
            None (the default): the means ``availability`` takes by
            default, exact where the exact methods work them out.
            False: the simulations' own, as ``availability(...,
            control_variate=False)`` gives them. Chunks made with
            False keep it.
        conditional : bool, optional
            None (the default) or False, as for ``availability``: False
            leaves the means the simulations' own where no exact ones are
            taken, rather than those given the modules' histories. Chunks
            made with False keep it.

        Returns
        -------
        AvailabilityResult
            The result of their simulations.

        Raises
        ------
        ValueError
            If the chunks are of different runs, of another system (or of
            it saved by another RePyability version), or their simulations
            overlap or interleave; or, unless ``allow_gaps``, some
            simulations are missing between or before them; or, with
            ``mc_samples``, they do not hold simulations ``0`` to
            ``mc_samples - 1``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD, SimulationChunk
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "t")],
        ...     {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
        ... )
        >>> saved = rbd.simulate_chunk(100.0, 0, 200, seed=3).to_json()
        >>> chunk = SimulationChunk.from_json(saved)
        >>> rbd.availability_from_chunks(chunk).n_simulations
        200
        """
        from repyability.rbd.chunks import SimulationChunk

        if isinstance(chunks, (SimulationChunk, dict, bytes, bytearray)):
            chunks = [chunks]
        chunk = SimulationChunk.merge(chunks)
        if mc_samples is not None and chunk.ranges != [(0, int(mc_samples))]:
            raise ValueError(
                f"The chunks hold simulations {chunk.ranges} (as "
                "(start, stop) ranges), not all of simulations 0 to "
                f"{int(mc_samples) - 1}: some are missing."
            )
        if not allow_gaps and chunk.ranges != [(0, chunk.n_simulations)]:
            held = ", ".join(f"{a} to {b - 1}" for a, b in chunk.ranges)
            raise ValueError(
                f"The chunks hold simulations {held}: those between or "
                "before them are missing (a chunk that never came back?). "
                "Give them all, or pass allow_gaps=True for the result of "
                "the simulations held."
            )
        settings = chunk.settings
        means = _chunk_means(control_variate, conditional)
        for name in ("control_variate", "conditional"):
            if settings.get(name) is False:
                means[name] = False
        fingerprint = self._fingerprint()
        if (
            settings["fingerprint"] is not None
            and fingerprint is not None
            and settings["fingerprint"] != fingerprint
        ):
            raise ValueError(
                "The chunks were simulated with another system (or with "
                "this one saved by another RePyability version)."
            )
        working = set(settings["working_nodes"])
        broken = set(settings["broken_nodes"])
        capacity = None
        if self._has_capacity():
            capacity = _CapacityRecorder(self, settings["demand"])
        initial_up = bool(
            self.is_system_working(
                {c: c not in broken for c in self.components},
                settings["method"],
            )
        )
        # The means the run's result takes by default (#187, #189).
        controls, record, breakdown = self._default_means(
            chunk._tally,
            float(settings["t_simulation"]),
            working,
            broken,
            settings["method"],
            int(settings["entropy"]),
            bool(settings["antithetic"]),
            _states_from_key(self, settings.get("state")),
            chunk.ranges,
            **means,
        )
        return self._availability_result(
            chunk._tally,
            settings["t_simulation"],
            initial_up,
            settings["antithetic"],
            capacity,
            controls,
            conditional=record,
            breakdown=breakdown,
        )

    def _default_means(
        self,
        tally: "_Tally",
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        entropy: int,
        antithetic: bool,
        state,
        ranges: List[Tuple[int, int]],
        control_variate: Optional[bool] = None,
        conditional: Optional[bool] = None,
    ) -> Tuple[
        Tuple[Optional[ControlVariate], ...],
        Optional[ConditionalRun],
        Optional[_Breakdown],
    ]:
        """The means the result of a plain run takes by default, for its
        simulations ``ranges`` in ``tally`` (chunks merged, see
        ``availability_from_chunks``), as ``availability`` takes them: the
        exact methods', where they work them out (#187, see
        ``_exact_controls``), or given the histories of its modules,
        simulated again from the run's ``entropy`` (#189, see
        ``_conditioned_run``), with the expected cost's split found so;
        else none. ``control_variate=False`` takes neither, and
        ``conditional=False`` not the second (#236), as ``availability``
        does."""
        T = t_simulation
        if control_variate is False:
            return (None, None), None, None
        exact_means = self._exact_means(T, working, broken, method, state)
        if exact_means is not None:
            return (
                _exact_controls(tally, T, exact_means, antithetic),
                None,
                exact_means.breakdown,
            )
        if conditional is False:
            return (None, None), None, None
        modules = self._conditional_applies(working, broken, state)
        if not modules:
            return (None, None), None, None
        run = _ModuleRun(
            self,
            T,
            working,
            broken,
            method,
            entropy,
            antithetic=antithetic,
            n_jobs=None,
            engine="auto",
            curve_points=1,
            state=state,
            modules=modules,
        )
        try:
            run.prepare()
        except NotImplementedError:
            return (None, None), None, None
        for start, stop in ranges:
            run.simulate(start, stop - start)
        record = run.means()
        return (None, None), record, run.split(record)

    def _fingerprint(self) -> Optional[str]:
        """A hash of this system saved as JSON (with the RePyability
        version), which chunks of its runs carry; None if it cannot be
        saved."""
        try:
            text = json.dumps(self.to_dict(), sort_keys=True)
        except SAVE_ERRORS:
            return None
        return hashlib.sha256(text.encode()).hexdigest()

    def compare(
        self,
        other: "RepairableRBD",
        t_simulation: float,
        mc_samples: Optional[int] = None,
        seed: Optional[int] = None,
        *,
        quantity: str = "availability",
        confidence: float = 0.95,
        n_jobs: Optional[int] = None,
        engine: str = "auto",
        state=None,
        control_variate: Optional[bool] = None,
    ) -> ConfidenceInterval:
        """How much better (or worse) this system is than ``other``: exactly
        where the exact methods work out both systems' expected values, and
        otherwise by simulation with common random numbers.

        By default, as ``availability`` and ``cost`` take their means,
        where the exact methods work out both systems' expected
        values over the window (``mission_availability``, and
        ``expected_cost`` their costs), the difference is theirs, exact,
        with no error (``method="exact"``), and nothing is simulated.
        ``control_variate=False`` simulates it whatever.

        Both systems are simulated ``mc_samples`` times over
        ``[0, t_simulation]`` (every component working and new at the start,
        or from ``state``), and in each simulation a
        component in the same place in both (the same node name, and the
        same names down through nested RBDs) draws the same random numbers
        in both: the same failures and repairs where it is modelled the
        same way, and matching ones (the same quantiles of its own models)
        where it is not. The differences between the two systems' results
        then come from how the systems differ rather than from chance, so
        their mean is a more precise estimate of the difference than the
        difference of two independent simulations of the same size: much
        more, when the systems differ in a component's models and share
        the rest.

        Parameters
        ----------
        other : RepairableRBD
            The system to compare with.
        t_simulation : float
            The window each simulation covers.
        mc_samples : int, optional
            The number of simulations of each system, by default 10_000.
        seed : int, optional
            Seed for a reproducible comparison, by default None: a number
            drawn from numpy's global RNG (see ``availability``).
        quantity : str, optional
            ``"availability"`` (the default): the fraction of the window the
            system is up. ``"cost"``: what owning it for the window from new
            costs, its running cost and its components' ``acquisition_cost``,
            as ``total_cost`` counts them (both systems must be
            priced; a cost given as a distribution draws the same numbers in
            both too, from a stream of its own).
        confidence : float, optional
            The confidence level of the interval, by default 0.95.
        n_jobs : int, optional
            Run the simulations in parallel on ``n_jobs`` CPUs (see
            ``availability``), by default None: one.
        engine : str, optional
            What runs the simulations: ``"python"``, ``"numba"`` or
            ``"auto"`` (the default), as in ``availability``.
        state : dict, optional
            The components' states at the start, as for ``availability``,
            in both systems: each must have the components it names. By
            default None: new.
        control_variate : bool, optional
            By default None: the exact difference where the exact methods
            work out both systems' expected values. False: the simulated
            difference, whatever. (True, which controls ``availability``'s
            run by a twin, is not taken: the common random numbers are the
            comparison's own.)

        Returns
        -------
        ConfidenceInterval
            The expected difference (this system's quantity minus
            ``other``'s): exact (``method="exact"``), or the mean over the
            simulations (``"simulated"``), with its standard error and a
            normal confidence interval (not clipped: the difference may be
            negative).

        Raises
        ------
        ValueError
            If ``quantity``, ``mc_samples``, ``confidence``, ``n_jobs`` or
            ``engine`` is invalid, or a system to compare by cost has no
            costs.
        NotImplementedError
            If a component's draws cannot be replayed from a stream of its
            own (a model sampled its own way), or, with
            ``engine="numba"``, the compiled engine does not simulate a
            system.
        ImportError
            With ``engine="numba"``, if numba is not installed.

        Examples
        --------
        Two pumps in parallel, each failing about every 10 hours: how much
        more of a 100-hour window is the system up if a repair takes 1 hour
        on average instead of 2?

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def pumps(mttr):
        ...     pump = {
        ...         "reliability": surv.Exponential.from_params([0.1]),
        ...         "repairability": surv.Exponential.from_params([1 / mttr]),
        ...     }
        ...     return RepairableRBD(
        ...         [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        ...         {"p1": pump, "p2": pump},
        ...     )
        >>> gain = pumps(1.0).compare(pumps(2.0), 100.0)
        >>> round(gain.estimate, 4), gain.method
        (0.0189, 'exact')

        Each pump is up ``mu / (lambda + mu) + lambda / (lambda + mu) *
        exp(-(lambda + mu) t)`` of the time at ``t``, which the exact
        methods average over the window. Simulated with common random
        numbers:

        >>> simulated = pumps(1.0).compare(
        ...     pumps(2.0), 100.0, mc_samples=2000, seed=0,
        ...     control_variate=False,
        ... )
        >>> round(simulated.estimate, 4), round(simulated.standard_error, 5)
        (0.0187, 0.00041)

        Two independent runs of 2000 simulations would estimate it with a
        standard error of about 0.00058.
        """
        N = 10_000 if mc_samples is None else mc_samples
        t_simulation = simulation_window(t_simulation)

        one_of("quantity", quantity, ("availability", "cost"))
        montecarlo.check_confidence(confidence)
        montecarlo.check_count(N, False, "mc_samples")
        if control_variate not in (None, False):
            raise ValueError(
                "compare takes control_variate=None (the exact difference "
                "where the exact methods work out both systems' expected "
                "values) or False (the simulated difference), got "
                f"{control_variate!r}: its common random numbers are its "
                "own variance reduction."
            )
        if quantity == "cost":
            for rbd in (self, other):
                if not (rbd.has_costs or rbd.acquisition_cost):
                    raise ValueError(
                        "Both systems must be priced to compare their costs."
                    )
        if control_variate is None:
            exact = self._exact_difference(
                other, t_simulation, quantity, state
            )
            if exact is not None:
                return ConfidenceInterval(
                    estimate=exact,
                    lower=exact,
                    upper=exact,
                    confidence=confidence,
                    standard_error=0.0,
                    n_samples=0,
                    method="exact",
                )
        jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
        states = [
            _curves._simulation_states(rbd, state) for rbd in (self, other)
        ]
        entropy = _streams.entropy_of(seed)
        values = []
        for rbd, start in zip((self, other), states):
            tally = rbd._run(
                t_simulation,
                set(),
                set(),
                "p",
                N,
                False,
                None,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                common=True,
                states=start,
            )
            if quantity == "cost":
                running = (
                    np.asarray(tally.cost_samples, dtype=float)
                    if rbd.has_costs
                    else np.zeros(N)
                )
                values.append(running + rbd.acquisition_cost)
            else:
                values.append(np.asarray(tally.uptimes) / t_simulation)
        differences = values[0] - values[1]
        estimate = float(np.mean(differences))
        standard_error = montecarlo.standard_error(differences, False)
        z = montecarlo.z_value(confidence)
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate - z * standard_error,
            upper=estimate + z * standard_error,
            confidence=confidence,
            standard_error=standard_error,
            n_samples=N,
            method="simulated",
        )

    def _exact_difference(
        self,
        other: "RepairableRBD",
        t_simulation: float,
        quantity: str,
        state,
    ) -> Optional[float]:
        """``compare``'s difference worked out exactly (#236): this
        system's expected value of ``quantity`` over the window less
        ``other``'s, where the exact methods work out both (see
        ``_exact_means``); else None."""
        values = []
        for rbd in (self, other):
            exacts = rbd._exact_means(t_simulation, set(), set(), "p", state)
            if exacts is None:
                return None
            if quantity == "availability":
                values.append(exacts.availability)
                continue
            if exacts.cost is not None:
                running = exacts.cost
            elif rbd.has_costs:
                return None
            else:
                running = 0.0
            values.append(running + rbd.acquisition_cost)
        return float(values[0] - values[1])

    def _twin(self) -> Tuple["RepairableRBD", List[str]]:
        """This system's exact twin (#154), and what it changes, in words.

        The twin has this system's diagram and components, with their
        models, failing and repaired independently: without the limit on
        repair crews or the maintenance groups, which tie components
        together, and, component by component, without what the exact
        methods over time do not take (see ``_twin_simpler``). Its expected
        values over a window are then exact (``mission_availability``,
        ``expected_cost``), and each of its streams is named as this
        system's is (a standby group's units as the twin's nested units),
        so that, simulated with common random numbers, it follows this
        system closely: see ``availability``'s ``control_variate``.

        Raises
        ------
        NotImplementedError
            If no such twin has exact values over time: a component's model
            is a probability, or a life is simulated.
        """
        from repyability.rbd import routes as r

        args = self._init_args
        changes = []
        if self._crews_limited():
            changes.append("the limit on repair crews")
        if self._maintenance:
            changes.append("the maintenance groups")
        specs: Dict[Any, Any] = {}
        for node, value in args["components"].items():
            if isinstance(value, RepairableRBD):
                nested, inner = value._twin()
                specs[node] = nested
                changes.extend(f"{what} in {node!r}" for what in inner)
            elif isinstance(value, dict):
                spec = {
                    key: item
                    for key, item in value.items()
                    if key not in ("priority", "group")
                }
                preventive = value.get("preventive")
                if preventive and "opportunity" in preventive:
                    # Opportunities come from a group's stops.
                    spec["preventive"] = {
                        key: item
                        for key, item in preventive.items()
                        if key != "opportunity"
                    }
                specs[node] = spec
            else:
                specs[node] = value
        twin = self._twin_from(specs)
        simpler = {}
        for node, spec in specs.items():
            if isinstance(spec, dict):
                simpler[node], change = twin._twin_simpler(node, spec)
                if change:
                    changes.append(change)
        if any(simpler[node] is not specs[node] for node in simpler):
            twin = self._twin_from({**specs, **simpler})
        if twin.ccf_groups and r.refusal(
            partial(_ccf_groups._require_groups_over_time, twin, {})
        ):
            # Its groups' chains cannot follow them over time (#158): its
            # members fail independently.
            twin = self._twin_from({**specs, **simpler}, groups=False)
            changes.append("the common-cause groups")
        route = _curves._over_time(twin)
        if route.route == r.REFUSED:
            raise NotImplementedError(
                "This system has no exact twin to control its simulation "
                f"by: {route.reason}"
            )
        if route.route == r.SIMULATED:
            raise NotImplementedError(
                "This system has no exact twin to control its simulation "
                "by: its components' values over time are simulated "
                f"({route.reason})"
            )
        return twin, changes

    def _twin_report(self) -> str:
        """What ``analysis_routes`` says of the exact twin a run with
        ``control_variate`` is controlled by (see ``AnalysisRoute.twin``):
        what it leaves out of this system, or why there is none."""
        try:
            _, changes = self._twin()
        except NotImplementedError as error:
            return f"none. {error}"
        _, complete = self._stream_specs(1.0)
        if not complete:
            return f"none. {_UNSTREAMED}"
        if not changes:
            return (
                "the system itself, whose values over the window are exact "
                "(mission_availability, expected_cost)."
            )
        return f"the system without {', '.join(changes)}."

    def _twin_from(
        self, components: dict, groups: bool = True
    ) -> "RepairableRBD":
        """A system of this one's diagram with ``components`` (see
        ``_twin``), and its common-cause groups unless not ``groups``."""
        args = self._init_args
        return RepairableRBD(
            args["edges"],
            components,
            k=args["k"],
            input_node=args["input_node"],
            output_node=args["output_node"],
            on_infeasible_rbd=args["on_infeasible_rbd"],
            downtime_cost_rate=args["downtime_cost_rate"],
            ccf_groups=args.get("ccf_groups") if groups else None,
        )

    def _twin_simpler(self, node, spec: dict) -> Tuple[Any, str]:
        """A component's spec in an exact twin (see ``_twin``), this system
        being the twin so far, and what it changes (or ""): a standby group
        whose own chain cannot follow it over time becomes its units
        operating together; imperfect repair becomes perfect (but for
        minimal repair in no time, whose values over time are exact);
        replacement on condition, inspections and block replacement that
        the exact methods do not take go."""
        from repyability.rbd import routes as r

        if node in self._standby:
            if r.refusal(partial(_long_run._standby_rates, self, node)):
                return (
                    _hot_units(spec),
                    f"the switching of standby group {node!r} (its units "
                    "operate together)",
                )
            return spec, ""
        drop: Dict[str, str] = {}
        if node in self._imperfect and r.refusal(
            partial(self._require_minimal_repair, node)
        ):
            drop["repair"] = drop["replace_after"] = "imperfect repair"
        if node in self._inspection and r.refusal(
            partial(self._require_tested_exact, node)
        ):
            drop["inspection"] = "inspections"
        schedule = self._preventive.get(node)
        if (
            schedule is not None
            and schedule.policy in ("block", "condition")
            and r.refusal(partial(self._require_block_models, node))
        ):
            drop["preventive"] = (
                "block replacement"
                if schedule.policy == "block"
                else "replacement on condition"
            )
        if not drop:
            return spec, ""
        what = list(dict.fromkeys(drop[key] for key in drop if key in spec))
        note = " (its failures are revealed)" if "inspection" in drop else ""
        return (
            {key: value for key, value in spec.items() if key not in drop},
            f"the {' and '.join(what)} of {node!r}{note}",
        )

    def _simulated(
        self,
        t_simulation: float,
        working_nodes,
        broken_nodes,
        method: str,
        N: int,
        verbose: bool,
        seed,
        *,
        tolerance: Optional[float],
        confidence: float,
        max_N: Optional[int],
        antithetic: bool,
        n_jobs: Optional[int],
        target: str,
        demand: Optional[float] = None,
        engine: str = "auto",
        state=None,
        curve_points: Optional[int] = None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        control_variate: Optional[bool] = None,
        conditional: Optional[bool] = None,
    ) -> AvailabilityResult:
        """``availability`` (and ``cost``): validate, run the replications
        (serially, in parallel, or as shards through ``shard_map``, until
        converged if asked; with ``control_variate``, alongside the system's
        exact twin; with ``conditional``, of the dependent modules alone,
        see ``_conditional_run``) and build the result, its curve on a grid
        of ``curve_points`` steps if given. By default its expected values
        are exact where the system is its own exact twin (#187), and taken
        given the modules' histories where a conditional run applies
        (#189, see ``_conditioned_run``)."""
        t_simulation = simulation_window(t_simulation)
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        states = _curves._simulation_states(
            self, state, working_nodes | broken_nodes
        )
        # The initial system state with every component up but those forced
        # broken, which can make the system start down; a simulation that
        # starts down from the components' states records the change at 0
        # (see _replicate).
        initial_status = {c: c not in broken_nodes for c in self.components}
        initial_up = bool(self.is_system_working(initial_status, method))
        montecarlo.check_count(N, antithetic, "mc_samples")
        stop = _stopping_rule(
            N, tolerance, confidence, max_N, antithetic, target, t_simulation
        )
        _check_shard_map(shard_map, shard_size, n_jobs)
        if control_variate is not None and not isinstance(
            control_variate, (bool, np.bool_)
        ):
            raise ValueError(
                f"control_variate must be True or False, got "
                f"{control_variate!r}."
            )
        if conditional is not None and not isinstance(
            conditional, (bool, np.bool_)
        ):
            raise ValueError(
                f"conditional must be True, False or None, got "
                f"{conditional!r}."
            )
        if conditional:
            if demand is not None and not self._has_capacity():
                raise ValueError(
                    "A demand is measured against capacities, and no node "
                    "has one: give them with capacity={node: capacity}."
                )
            return self._conditional_run(
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                N,
                seed,
                stop=stop,
                antithetic=antithetic,
                n_jobs=n_jobs,
                engine=engine,
                curve_points=curve_points if target == "availability" else 1,
                target=target,
                state=state,
                shard_map=shard_map,
                shard_size=shard_size,
                demand=demand,
                capacities=target == "availability" and self._has_capacity(),
                control_variate=bool(control_variate),
            )
        # By default the expected values are the exact methods', where they
        # work them out (#187): a run to a tolerance stops at once.
        auto = control_variate is None
        exact_means = (
            self._exact_means(
                t_simulation, working_nodes, broken_nodes, method, state
            )
            if auto
            else None
        )
        twin = None
        changes: List[str] = []
        exacts: Optional[_Exacts] = None
        if control_variate:
            twin, changes = self._twin()
            if shard_map is not None and changes:
                raise ValueError(
                    "A run with control_variate simulates the system's twin "
                    "alongside it, here: leave out shard_map."
                )
            exacts = self._twin_exact(
                twin, t_simulation, working_nodes, broken_nodes, method, state
            )
        # Otherwise, by default, its means are taken given the histories of
        # its modules, where a conditional run applies (#189).
        modules = (
            self._conditional_applies(working_nodes, broken_nodes, state)
            if auto and exact_means is None and conditional is None
            else []
        )
        capacity = None
        # Shards follow the capacities whatever the target, as chunks do.
        if (
            target == "availability" or shard_map is not None
        ) and self._has_capacity():
            capacity = _CapacityRecorder(self, demand)
        elif demand is not None:
            raise ValueError(
                "A demand is measured against capacities, and no node has "
                "one: give them with capacity={node: capacity}."
            )
        entropy = sharded = None
        if shard_map is not None:
            _ccf_groups._require_groups_simulated(self)
            entropy = _streams.entropy_of(seed)
            template = self._shard_template(
                t_simulation,
                entropy,
                working_nodes,
                broken_nodes,
                method,
                antithetic,
                demand,
                engine,
                states,
                curve_points,
            )
            plan, _ = self._stream_plan(
                t_simulation, entropy, antithetic, states
            )
            step = self._shard_size(plan, shard_size)
            sharded = (shard_map, template, step)
        jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
        controls: Tuple[Optional[ControlVariate], ...] = (None, None)
        # The expected cost's split, where the means are exact (#223).
        breakdown: Optional[_Breakdown] = None
        if modules:
            done = self._conditioned_run(
                modules,
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                N,
                verbose,
                seed,
                antithetic,
                stop,
                capacity=capacity,
                jobs=jobs,
                n_jobs=n_jobs,
                engine=engine,
                state=state,
                states=states,
                curve_points=curve_points,
                target=target,
                sharded=sharded,
                shard_map=shard_map,
                shard_size=shard_size,
            )
            if done is not None:
                tally, record, breakdown = done
                return self._availability_result(
                    tally,
                    t_simulation,
                    initial_up,
                    antithetic,
                    capacity,
                    conditional=record,
                    breakdown=breakdown,
                )
        if twin is not None:
            assert exacts is not None
            tally, controls = self._controlled_run(
                twin,
                exacts,
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                N,
                verbose,
                seed,
                antithetic,
                stop,
                capacity=capacity,
                jobs=jobs,
                engine=engine,
                state=state,
                states=states,
                curve_points=curve_points,
                target=target,
                itself=not changes,
                sharded=sharded,
            )
            if not changes:
                # Its means are the system's exact ones (#223).
                breakdown = exacts.breakdown
        else:
            tally = self._run(
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                N,
                verbose,
                seed,
                antithetic,
                # Exact means need no more simulations.
                None if exact_means is not None else stop,
                capacity=capacity,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                states=states,
                curve_points=curve_points,
                sharded=sharded,
            )
            if exact_means is not None:
                controls = _exact_controls(
                    tally, t_simulation, exact_means, antithetic
                )
                breakdown = exact_means.breakdown
        return self._availability_result(
            tally,
            t_simulation,
            initial_up,
            antithetic,
            capacity,
            controls,
            breakdown=breakdown,
        )

    def _exact_means(
        self,
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        state,
    ) -> Optional[_Exacts]:
        """The system's own expected values over the window, where the
        exact methods work them out (#187): its mean availability and, if
        it is priced, its expected cost (see ``_twin_exact``); else None."""
        try:
            return self._twin_exact(
                self, t_simulation, working, broken, method, state
            )
        except NotImplementedError:
            return None

    def _conditional_modules(self, working: set, broken: set) -> list:
        """The nodes a conditional run simulates (#189, see
        ``_conditional``), in the order of the components: those whose
        values over time the exact methods do not work out (see
        ``_node_over_time``), with every member of a maintenance group one
        of them is in. Raise if the others cannot be taken exactly given
        them."""
        from repyability.rbd import routes as r

        held = working | broken
        chosen = {
            node
            for node in self.components
            if node not in held
            and _curves._node_over_time(self, node, "window")[0] == r.REFUSED
        }
        if self._crews_couple():
            # A job can wait for a crew behind any other: every node the
            # crews serve is a module, simulated with them, and the rest
            # (nested RBDs, with crews of their own) taken exactly.
            chosen |= {
                key.node if isinstance(key, _Unit) else key
                for key in self._crew_served()
            } - held
        for name, spec in self._maintenance.items():
            members = set(spec.members)
            if not members & chosen:
                continue
            caught = members & held
            if caught:
                raise NotImplementedError(
                    f"Maintenance group {name!r} is simulated in a "
                    f"conditional run, and its member(s) "
                    f"{sorted(caught, key=str)} held: hold none of its "
                    "members, or run plainly (conditional=False)."
                )
            if spec.system_down:
                raise NotImplementedError(
                    f"Maintenance group {name!r} stops at every outage of "
                    "the system (system_down), which ties its members to "
                    "every component: a conditional run would simulate them "
                    "all. Run it plainly (conditional=False)."
                )
            chosen |= members
        for group in self.ccf_groups:
            caught = set(group.members) & chosen
            if caught:
                raise NotImplementedError(
                    f"Common-cause group {list(group.members)}: member(s) "
                    f"{sorted(caught, key=str)} would be simulated in a "
                    "conditional run apart from the group's causes, which it "
                    "takes exactly: run it plainly (conditional=False)."
                )
        if len(chosen) > 62:
            raise NotImplementedError(
                f"A conditional run would simulate {len(chosen)} nodes, more "
                "than the 62 it follows the joint states of: run it plainly "
                "(conditional=False)."
            )
        if chosen and not set(self.components) - held - chosen:
            raise NotImplementedError(
                "A conditional run would simulate every component (the "
                "nodes the exact methods over time do not take, and those "
                "they are tied to by repair crews or maintenance groups), "
                "leaving none to take exactly given them: run it plainly "
                "(conditional=False)."
            )
        return [node for node in self.components if node in chosen]

    def _conditional_applies(
        self, working: Collection = (), broken: Collection = (), state=None
    ) -> list:
        """The modules a conditional run would simulate (see
        ``_conditional_modules``) with the nodes ``working`` and ``broken``
        held and from the components' ``state``, or none if it would
        refuse: the others must be taken exactly given them, which a
        structure too meshed for its decision diagram, or common-cause
        groups the exact methods over time refuse, prevent."""
        from repyability.rbd import routes as r

        if self._too_meshed() is not None:
            return []
        if self.ccf_groups and r.refusal(
            partial(
                _ccf_groups._require_groups_over_time,
                self,
                state or {},
            )
        ):
            return []
        try:
            return self._conditional_modules(set(working), set(broken))
        except NotImplementedError:
            return []

    def _modules_rbd(self, modules: list) -> "RepairableRBD":
        """The ``modules`` of a conditional run (see
        ``_conditional_modules``) as a diagram of their own, each between
        the input and output nodes with its own spec: their streams are
        named as in this system, so its simulations are this system's
        modules'. It has this system's repair crews (whose jobs are the
        modules' alone: crews that tie the components together make every
        node they serve a module), the maintenance groups of its members,
        and no system downtime cost."""
        args = self._init_args
        options = args["maintenance_groups"] or {}
        groups = {
            name: spec
            for name, spec in options.items()
            if set(self._maintenance[name].members) <= set(modules)
        }
        return RepairableRBD(
            [(self.input_node, node) for node in modules]
            + [(node, self.output_node) for node in modules],
            {node: args["components"][node] for node in modules},
            input_node=self.input_node,
            output_node=self.output_node,
            repair_crews=args["repair_crews"],
            maintenance_groups=groups or None,
        )

    def _capacity_given(
        self,
        modules: list,
        state: int,
        working: set,
        broken: set,
        x: np.ndarray,
        others,
        demand: Optional[float],
    ) -> "_conditional.CapacityGiven":
        """The system's capacity given the modules in one joint ``state``
        (see ``_conditional_given``), on the grid ``x``: from its exact
        distribution over time with the modules held so (a module up at its
        levels by their probabilities, as the simulation weighs them), and
        the other nodes from their states ``others``, against ``demand``
        (see ``_conditional.capacity_given``)."""
        up = {node for j, node in enumerate(modules) if state >> j & 1}
        down = set(modules) - up
        distribution = self.point_capacity(
            x, working | up, broken | down, state=others
        )
        return _conditional.capacity_given(
            x, distribution.levels, distribution.probabilities, demand
        )

    def _up_at_start(self, node, start) -> bool:
        """Whether ``node`` is up just before 0 from its checked ``start``
        (see ``_states``; None, new): a component unless it starts down, a
        nested RBD as its system, from its components'."""
        if start is None:
            return True
        component = self.components[node]
        if isinstance(component, RepairableRBD):
            status = {
                inner: component._up_at_start(inner, start.get(inner))
                for inner in component.components
            }
            return bool(component.is_system_working(status, "p"))
        return bool(start.alive)

    def _crew_free(self) -> "RepairableRBD":
        """This system without its limit on repair crews: where every node
        the crews serve is held (a conditional run's modules), they tie
        nothing together, and the rest is worked out as if they were
        unlimited (a nested RBD keeps its own)."""
        if not self._crews_limited():
            return self
        return RepairableRBD(**{**self._init_args, "repair_crews": None})

    def _conditional_given(
        self,
        modules: list,
        state: int,
        working: set,
        broken: set,
        method: str,
        x: np.ndarray,
        others=None,
    ) -> "_conditional.Given":
        """The system given the modules in one joint ``state`` (bit ``j``
        set while module ``j`` is up), on the grid ``x``: with the modules
        held so, and the other nodes from their states ``others`` (new by
        default), its expected up time, failures and planned outages before
        each time, and its point availability (see
        ``_conditional.Given``)."""
        up = {node for j, node in enumerate(modules) if state >> j & 1}
        down = set(modules) - up
        _, _, _, counts, _, _ = _windows._window(
            self, x, working | up, broken | down, method, state=others
        )
        available = self.point_availability(
            x, working | up, broken | down, method, state=others
        )
        # Just before 0, every other node is up, but one started down.
        started = _curves._states(self, others, working | broken | up | down)
        status = {
            node: node not in broken | down
            and self._up_at_start(node, started.get(node))
            for node in self.components
        }
        return _conditional.Given(
            x,
            np.asarray(counts["uptime"], dtype=float),
            np.asarray(counts["failures"], dtype=float),
            np.asarray(counts["planned"], dtype=float),
            np.asarray(available, dtype=float),
            float(bool(self.is_system_working(status, method))),
        )

    def _conditional_run(
        self,
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        N: int,
        seed,
        *,
        stop: Optional[Callable[..., int]],
        antithetic: bool,
        n_jobs: Optional[int],
        engine: str,
        curve_points: Optional[int],
        target: str,
        state=None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        demand: Optional[float] = None,
        capacities: bool = False,
        control_variate: bool = False,
    ) -> AvailabilityResult:
        """``availability`` (and ``cost``) with ``conditional=True`` (#189,
        see ``_conditional`` and ``_ModuleRun``): simulate the modules
        alone, in rounds while ``stop`` asks for more, and take every other
        node exactly given their joint states. From the components'
        ``state``, the modules start from theirs and the others from
        theirs. With ``shard_map``, the modules' simulations run as shards
        (see ``shards``) wherever it sends them. With ``capacities``, the
        system's capacity given the modules' states too, against
        ``demand``. With ``control_variate``, the exact twin's stand-ins for
        the modules (see ``_twin``) are simulated alongside them, with
        common random numbers, and each simulation's expected values given
        theirs, whose mean is the twin's exact one, control the run's (see
        ``ControlVariate``)."""
        run = _ModuleRun(
            self,
            t_simulation,
            working,
            broken,
            method,
            _streams.entropy_of(seed),
            antithetic=antithetic,
            n_jobs=n_jobs,
            engine=engine,
            curve_points=curve_points,
            state=state,
            shard_map=shard_map,
            shard_size=shard_size,
            demand=demand,
            capacities=capacities,
            control_variate=control_variate,
        )
        run.prepare()
        first, count = 0, N
        while True:
            run.simulate(first, count)
            first += count
            if stop is None:
                break
            # A run to a tolerance judges the mean asked for (see
            # _stopping_rule), controlled if it is: not while the modules
            # have not changed state (#215).
            count = stop(None, run.judged(target), run.unjudged())
            if not count:
                break
        return run.result()

    def _conditioned_run(
        self,
        modules: list,
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        N: int,
        verbose: bool,
        seed,
        antithetic: bool,
        stop: Optional[Callable[..., int]],
        *,
        capacity: Optional[_CapacityRecorder],
        jobs: Optional[int],
        n_jobs: Optional[int],
        engine: str,
        state,
        states: dict,
        curve_points: Optional[int],
        target: str,
        sharded: Optional[tuple],
        shard_map: Optional[Callable],
        shard_size: Optional[int],
    ) -> Optional[Tuple["_Tally", ConditionalRun, Optional[_Breakdown]]]:
        """A plain run whose means are taken given its ``modules``'
        histories (#189), by default where a conditional run applies: the
        simulations, in rounds while ``stop`` asks for more (judged by
        those means), and of the same simulations each one's expected up
        time and cost given the histories of its modules, worked out as a
        conditional run works them out (see ``_ModuleRun``): the modules'
        simulations are the plain run's, drawn from the same streams. The
        simulations' totals, a record holding those values, and the
        expected cost's split given the modules (None unpriced, or while
        they have not changed state); or None, before simulating anything,
        if the rest cannot be worked out exactly given the modules."""
        entropy = _streams.entropy_of(seed)
        run = _ModuleRun(
            self,
            t_simulation,
            working,
            broken,
            method,
            entropy,
            antithetic=antithetic,
            n_jobs=n_jobs,
            # The modules' histories are the event loop's, on every engine.
            engine=engine if engine in ("python", "numba") else "auto",
            curve_points=1,
            state=state,
            shard_map=shard_map,
            shard_size=shard_size,
            modules=modules,
        )
        try:
            run.prepare()
        except NotImplementedError:
            return None
        tally: Optional[_Tally] = None
        first, count = 0, N
        while True:
            part = self._run(
                t_simulation,
                working,
                broken,
                method,
                count,
                verbose,
                None,
                antithetic,
                capacity=capacity,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                first=first,
                states=states,
                curve_points=curve_points,
                sharded=sharded,
            )
            if tally is None:
                tally = part
            else:
                tally.merge(part)
            run.simulate(first, count)
            first += count
            if stop is None:
                break
            # Its means given the modules, or the simulations' own while
            # the modules have not changed state (#215).
            more = stop(tally, None if run.unjudged() else run.judged(target))
            if not more:
                break
            count = more
        assert tally is not None
        record = run.means()
        return tally, record, run.split(record)

    def _twin_exact(
        self,
        twin: "RepairableRBD",
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        state,
    ) -> _Exacts:
        """The exact ``twin``'s (see ``_twin``) mean availability over the
        window and, if both systems are priced, its expected cost, by
        category and by component (else None), building each component's
        curve once for both (#185)."""
        priced = self.has_costs and twin.has_costs
        # The cost's curves, which count events, serve the availability too.
        with twin._sharing_curves():
            expected = (
                twin.expected_cost(
                    t_simulation, working, broken, method, state=state
                )
                if priced
                else None
            )
            exact = float(
                np.ravel(
                    twin.mission_availability(
                        t_simulation, working, broken, method, state=state
                    )
                )[0]
            )
        if expected is None:
            return _Exacts(exact)

        def first(value) -> float:
            return float(np.ravel(value)[0])

        return _Exacts(
            exact,
            first(expected.mean),
            (
                {key: first(v) for key, v in expected.by_category.items()},
                {node: first(v) for node, v in expected.by_component.items()},
            ),
        )

    def _controlled_run(
        self,
        twin: "RepairableRBD",
        exacts: _Exacts,
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        N: int,
        verbose: bool,
        seed,
        antithetic: bool,
        stop: Optional[Callable[..., int]],
        *,
        capacity: Optional[_CapacityRecorder],
        jobs: Optional[int],
        engine: str,
        state,
        states: dict,
        curve_points: Optional[int],
        target: str,
        itself: bool = False,
        sharded: Optional[tuple] = None,
    ) -> Tuple["_Tally", Tuple[Optional[ControlVariate], ...]]:
        """Run this system and its exact ``twin`` (see ``_twin``) with
        common random numbers, as ``compare`` does, in rounds while
        ``stop`` asks for more, judged by the controlled values; return
        this system's totals and the controls of its fractions up and of
        its costs (None without costs) by the twin's (see
        ``ControlVariate``), ``itself`` if the twin is this system: by its
        exact values ``exacts``, its mean availability and its expected
        cost (None unpriced; see ``_twin_exact``). A twin that is this
        system is not simulated again, so its run can be ``sharded`` (see
        ``_run``)."""
        assert sharded is None or itself
        twin_states = _curves._simulation_states(twin, state, working | broken)
        exact, exact_cost = exacts.availability, exacts.cost
        entropy = _streams.entropy_of(seed)
        tally: Optional[_Tally] = None
        twin_tally: Optional[_Tally] = None
        first, count = 0, N
        while True:
            part = self._run(
                t_simulation,
                working,
                broken,
                method,
                count,
                verbose,
                None,
                antithetic,
                capacity=capacity,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                common=True,
                first=first,
                states=states,
                curve_points=curve_points,
                sharded=sharded,
            )
            # A twin that is this system, drawn from the same streams, runs
            # as this system does, to the last bit: its run is this one.
            twin_part = (
                None
                if itself
                else twin._run(
                    t_simulation,
                    working,
                    broken,
                    method,
                    count,
                    False,
                    None,
                    antithetic,
                    jobs=jobs,
                    engine=engine,
                    entropy=entropy,
                    common=True,
                    first=first,
                    states=twin_states,
                    curve_points=1,
                )
            )
            if tally is None:
                tally, twin_tally = part, twin_part
            else:
                tally.merge(part)
                if twin_tally is not None and twin_part is not None:
                    twin_tally.merge(twin_part)
            twins = tally if twin_tally is None else twin_tally
            fractions = np.asarray(tally.uptimes, dtype=float) / t_simulation
            control = ControlVariate.of(
                fractions,
                np.asarray(twins.uptimes, dtype=float) / t_simulation,
                exact,
                antithetic,
                itself=itself,
            )
            cost_control = None
            if exact_cost is not None:
                cost_control = ControlVariate.of(
                    tally.cost_samples,
                    twins.cost_samples,
                    exact_cost,
                    antithetic,
                    itself=itself,
                )
            if stop is None:
                break
            if target == "cost":
                values = (
                    tally.cost_samples
                    if cost_control is None
                    else cost_control.controlled(tally.cost_samples)
                )
            else:
                values = control.controlled(fractions)
            more = stop(tally, values)
            if not more:
                break
            first, count = first + count, more
        return tally, (control, cost_control)

    def _run(self, *args, **kwargs) -> "_Tally":
        """A run of simulations (see ``_run_alone``), one at a time in the
        process (``SIMULATIONS``, #216): the event loop keeps its state on
        the diagram, and draws that cannot be streamed come from numpy's
        global RNG. Not a ``sharded`` run, whose shards each run (and take
        it) where they are sent: a ``shard_map`` on threads would wait for
        it."""
        if kwargs.get("sharded") is not None:
            return self._run_alone(*args, **kwargs)
        with SIMULATIONS:
            return self._run_alone(*args, **kwargs)

    def _run_alone(
        self,
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
        _ccf_groups._require_free_members(self, working_nodes, broken_nodes)
        _ccf_groups._require_groups_simulated(self, states)
        t_simulation = simulation_window(t_simulation)
        from tqdm import tqdm

        state = np.random.get_state()
        after = state
        tally = _Tally(
            list(self.components), self.costs, t_simulation, curve_points
        )
        if replacements:
            # Only the Python engine counts them.
            tally.replacements = []
            engine = "python"
        if histories:
            tally.histories = _timeline_runs.Records(len(self.components))
        runner: Any = None
        try:
            if entropy is None:
                entropy = _streams.entropy_of(seed)
                if seed is None:
                    after = np.random.get_state()
            plan, complete = self._stream_plan(
                t_simulation, entropy, antithetic, states
            )
            if (antithetic or common) and not complete:
                raise NotImplementedError(_UNSTREAMED)
            engine = self._simulation_engine(
                engine, plan, capacity, N, here=sharded is None, states=states
            )
            progress = tqdm(
                total=N, disable=not verbose, desc="Running simulations"
            )
            if sharded is not None:
                runner = _ShardRunner(self, tally, progress, *sharded)
            elif engine == "numba":
                from repyability.rbd import _compiled

                runner = _compiled.Runner(
                    self,
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
                    self,
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
                    self,
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
            self._forget_run()
        return tally

    #: The interim variables of a simulation (see _forget_run).
    _RUN_STATE = (
        "_event_queue",
        "system_state",
        "t_simulation",
        "component_status",
        "last_change_planned",
        "_pending_failure",
        "_in_service",
        "_levels",
        "_renewed_at",
        "_pending_event",
        "_cancelled",
        "_early",
        "_renewing",
        "_crews",
        "_groups",
        "_step_sources",
        "_phases",
    )

    def _forget_run(self) -> None:
        """Clean up the interim variables of a simulation, this RBD's and
        its nested RBDs': left behind, a nested standby group's draws (whose
        samplers cannot be pickled) would keep the RBD from going to the
        processes of a later parallel run."""
        for name in self._RUN_STATE:
            self.__dict__.pop(name, None)
        for component in self.components.values():
            if isinstance(component, RepairableRBD):
                component._forget_run()

    def _simulation_engine(
        self,
        engine: str,
        plan: _streams.Plan,
        capacity: Optional[_CapacityRecorder],
        N: int,
        here: bool = True,
        states: Optional[dict] = None,
    ) -> str:
        """The engine that runs a simulation: ``"python"``, ``"numba"`` or
        one another package adds (see ``availability``'s ``engine``), for a
        run from the components' ``states``. Not ``here`` (the simulations
        run as shards elsewhere), only whether it can simulate the system
        is checked, and ``engine`` is kept, for each shard's worker to
        choose by."""
        from repyability.rbd import _compiled, engines

        added = engines.registered()
        if engine not in ("auto", "python", "numba") and engine not in added:
            names = ", ".join(
                repr(name) for name in ["auto", "python", "numba", *added]
            )
            raise ValueError(f"engine must be one of {names}, got {engine!r}.")
        if engine == "python":
            return engine
        if engine != "auto":
            # What the engine does not simulate is refused first: installing
            # it would not help.
            reason = _compiled.unsupported(
                self, plan, capacity, numba=engine == "numba", states=states
            )
            if reason is not None:
                raise NotImplementedError(
                    f"The compiled engine does not simulate {reason}: use "
                    "engine='python', or 'auto', which chooses the engine "
                    "that can."
                )
            if here and engine == "numba":
                _compiled.require()
            elif here and not added[engine].available():
                raise ImportError(
                    f"The {engine!r} simulation engine cannot run here."
                )
            return _compiled.ready(engine, auto=False) if here else engine
        if not here:
            return engine
        name, _ = _compiled.choice(self, plan, capacity, states)
        if name is not None and _compiled.worthwhile(plan, N, name):
            return _compiled.ready(name, auto=True)
        return "python"

    def _replicate(self, ctx: "_Context", replication: int) -> _Replication:
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
        self._start_queue(
            t_simulation,
            ctx.working,
            ctx.broken,
            ctx.method,
            sources,
            ctx.states,
        )
        status = self.component_status
        index, plain, works = ctx.position, ctx.plain, ctx.works
        inspected = self._inspection
        crews = self._crews
        groups = self._groups
        n = len(index)
        # Each component's failures, the system failures they caused, its
        # restorations and the system restorations they caused.
        failed, caused_down = [0] * n, [0] * n
        restored, caused_up = [0] * n, [0] * n
        # Each component's replacements: the spares it used.
        replaced = [0] * n
        # Imperfect repair (#109): a failure repaired, not replaced, uses no
        # spare and is charged only its repair.
        imperfect = self._imperfect
        # Opportunistic maintenance (#108): each member's early renewals.
        maintenance, member_group = self._maintenance, self._member_group
        cancelled, early = self._cancelled, self._early
        opportunistic = [0] * n
        failures = restorations = planned = 0
        changes: list = []
        deltas: list = []
        if ctx.initial_up and not self.system_state:
            # Down at the start, from the components' states: the curve of
            # the availability over time counts it as a change at 0.
            changes.append(0.0)
            deltas.append(-1)
        by_category = [0.0] * len(_CATEGORIES)
        by_node = dict.fromkeys(self.costs, 0.0)
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
        up = self.system_state
        started_up = up
        system_up = system_down = since = 0.0
        last, up_at, down_at = [0.0] * n, [0.0] * n, [0.0] * n
        node_up, both_up, both_down = [0.0] * n, [0.0] * n, [0.0] * n

        # The queue's heap, worked directly (see _EventQueue).
        heap = self._event_queue._heap
        push, pop = heapq.heappush, heapq.heappop

        # When each group last stopped: the work started at one instant is
        # one stop, with one set-up.
        stopped: Dict[Hashable, float] = {}
        # Common-cause groups' members, whose next events a shared cause can
        # supersede (see _strike).
        ccf_members, ccf_pending = self._ccf_members, self._ccf_pending

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
            renewals = self._stop(group, t, trigger)
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
                for follow in self._strike(event, sources[node]):
                    push(heap, (follow.time, follow))
                continue
            renewing_early = False
            if early and early.get(id(event)) is event:
                del early[id(event)]
                self._renewing.discard(node)
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
                follow = self._follow_up(event, sources[node])
                next_event: Optional[Event] = (
                    follow
                    if crews is None
                    else self._crew_follow_up(event, follow)
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
                system_up_t, system_down_t = system_up, system_down + (
                    t - since
                )
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
                next_event = self._follow_up(event, sources[node])
            if crews is not None:
                # A job waits for a repair crew (see _Crews).
                next_event = self._crew_follow_up(event, next_event)
            if next_event is not None and next_event.time < t_simulation:
                push(heap, (next_event.time, next_event))
            if ccf_members and node in ccf_members:
                ccf_pending[node] = next_event
        self.system_state = up

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
            charge = self.downtime_cost_rate * (t_simulation - system_up)
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

    def _availability_result(
        self,
        tally: "_Tally",
        t_simulation: float,
        initial_up: bool,
        antithetic: bool,
        capacity: Optional[_CapacityRecorder] = None,
        controls: Tuple[Optional[ControlVariate], ...] = (None, None),
        conditional: Optional[ConditionalRun] = None,
        breakdown: Optional[_Breakdown] = None,
    ) -> AvailabilityResult:
        """The ``AvailabilityResult`` of the replications in ``tally``: its
        exact totals rounded, once; with ``controls``, the controls of its
        fractions up and its costs by an exact twin (see
        ``_controlled_run``); with ``conditional``, the record of its means
        taken given its modules' histories (see ``_conditioned_run``); with
        ``breakdown``, the split of the expected cost its mean is found
        with, exact or given the modules (#223), in place of the
        simulations' own."""
        N = tally.n
        nodes = tally.nodes
        tally._fold()

        def rounded(totals) -> list:
            return [float(total) for total in totals]

        system_uptime = float(tally.system_uptime)
        system_downtime = float(tally.system_downtime)
        both_up = rounded(tally.intersection_uptime)
        both_down = rounded(tally.intersection_downtime)
        # Collect Importance/Criticality measures from the simulation
        # reference: https://www.weibull.com/pubs/2004rm_05B_02.pdf
        # Operational Criticality Index
        oci_down = {
            k: _safe_ratio(v, system_downtime)
            for k, v in zip(nodes, both_down)
        }
        oci_up = {
            k: _safe_ratio(v, system_uptime) for k, v in zip(nodes, both_up)
        }
        # Intersection Over Union Importance
        iou_up = {
            k: _safe_ratio(both, either)
            for k, both, either in zip(
                nodes, both_up, rounded(tally.union_uptime)
            )
        }
        iou_down = {
            k: _safe_ratio(both, either)
            for k, both, either in zip(
                nodes, both_down, rounded(tally.union_downtime)
            )
        }
        FCI, RCI = tally.criticality_counts()
        # Failure Criticality Index Importance
        fci_sys = failure_criticality_index_per_system_failures(
            FCI, tally.system_failures
        )
        fci_comp = failure_criticality_index_per_component_failures(FCI)
        # Restoration Criticality Index Importance
        rci_sys = restoration_criticality_index_by_system(
            RCI, tally.system_restorations
        )
        rci_comp = restoration_criticality_index_by_component(RCI)
        criticalities = Criticalities(
            operational_criticality_index=UpDownImportance(
                up=oci_up, down=oci_down
            ),
            iou=UpDownImportance(up=iou_up, down=iou_down),
            failure_criticality_index=FailureCriticalityIndex(
                per_system_failure=fci_sys, per_component_failure=fci_comp
            ),
            restoration_criticality_index=RestorationCriticalityIndex(
                by_system=rci_sys, by_component=rci_comp
            ),
        )

        # The availability from t=0..t_simulation: how many of the N
        # simulated systems work after each time at which one changed state
        # (and at 0 and t_simulation whether or not any did), over N.
        if tally.binned is not None:
            # On the grid (#153): how many work at each of its times.
            assert tally.edges is not None
            time = tally.edges.copy()
            working = (N if initial_up else 0) + np.cumsum(tally.binned)
        else:
            changed_at, deltas = tally.state_changes()
            time, working = _working_over_time(
                changed_at,
                deltas,
                t_simulation,
                N if initial_up else 0,
            )
        system_availability = working / N

        cost_result = _acquisition_only(self, N, t_simulation, antithetic)
        if self.has_costs:
            # Per-component means cover each node's repair, replace,
            # preventive and own downtime cost. (System downtime is a
            # system-level quantity and is not attributed to components.)
            by_category, by_component = (
                breakdown
                if breakdown is not None
                else (
                    {
                        k: float(v) / N
                        for k, v in tally.cost_by_category.items()
                    },
                    {
                        k: float(v) / N
                        for k, v in tally.cost_by_component.items()
                    },
                )
            )
            cost_result = CostResult(
                samples=np.asarray(tally.cost_samples, dtype=float),
                t_simulation=t_simulation,
                n_simulations=N,
                acquisition_cost=self.acquisition_cost,
                by_category=by_category,
                by_component=by_component,
                antithetic=antithetic,
                control_variate=controls[1],
                conditional=conditional,
            )

        capacity_fields: dict = {}
        if capacity is not None:
            # The mean capacity from t=0..t_simulation: the simulated
            # systems' total expected capacity after each time at which one
            # changed (unlimited while one can carry an unlimited amount).
            if tally.capacity_binned is not None:
                # On the grid (#190): the total after each of its times.
                assert tally.edges is not None
                assert tally.unlimited_binned is not None
                times = tally.edges.copy()
                totals = np.cumsum(
                    bin_totals(tally.capacity_binned, len(times))
                )
                unlimited = np.cumsum(tally.unlimited_binned) > 0
            else:
                at, steps, starts, free_at, counts = tally.capacity_changes()
                change = _group_totals(steps, starts)
                del steps, starts  # (their memory serves what follows)
                times, totals, unlimited = _capacity_totals(
                    at, change, free_at, counts, t_simulation
                )
                del at, change
            # The mean over the systems, never below 0, and unlimited while
            # one can carry an unlimited amount: np.where(unlimited, inf,
            # np.maximum(totals / N, 0.0)), worked out in place.
            curve = np.divide(totals, N, out=totals)
            np.maximum(curve, 0.0, out=curve)
            curve[unlimited] = np.inf
            capacity_fields = dict(
                capacity_timeline=times,
                capacity=curve,
                capacity_time={
                    level: float(tally.capacity_time[level])
                    for level in sorted(tally.capacity_time)
                },
                demand=capacity.demand,
                delivered=(
                    None
                    if capacity.demand is None
                    else np.asarray(tally.delivered, dtype=float)
                ),
            )

        return AvailabilityResult(
            timeline=time,
            availability=system_availability,
            system_uptime=system_uptime,
            time_simulated_to=t_simulation,
            criticalities=criticalities,
            node_uptime=dict(zip(nodes, rounded(tally.node_uptime))),
            node_downtime=dict(zip(nodes, rounded(tally.node_downtime))),
            system_downtime=system_downtime,
            system_failures=tally.system_failures,
            system_restorations=tally.system_restorations,
            n_simulations=N,
            cost=cost_result,
            system_planned_outages=tally.system_planned_outages,
            uptimes=np.asarray(tally.uptimes, dtype=float),
            antithetic=antithetic,
            opportunistic_renewals=(
                dict(zip(nodes, tally.opportunistic))
                if self._maintenance
                else None
            ),
            control_variate=controls[0],
            conditional=conditional,
            **capacity_fields,
        )

    def cost(
        self,
        t_simulation: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        mc_samples: Optional[int] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        *,
        tolerance: Optional[float] = None,
        confidence: float = 0.95,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
        engine: str = "auto",
        state=None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        control_variate: Optional[bool] = None,
        conditional: Optional[bool] = None,
    ) -> Optional[CostResult]:
        """Simulate the cost of running the system for ``t_simulation``.

        Runs the ``availability`` simulation with cost accumulation and
        returns its [`CostResult`][repyability.CostResult]: the distribution
        of the window's total cost (``samples``, ``mean``,
        ``percentile(q)``), a confidence interval for the mean
        (``mean_interval()``), and per-category and per-component
        breakdowns. In each simulation ``repair_cost`` and ``replace_cost``
        are charged at every failure of their component (a fresh draw for a
        cost distribution), a preventive-maintenance cost at every
        preventive replacement, ``downtime_cost`` per unit time its
        component is down (planned outages included), and
        ``downtime_cost_rate`` per unit time the system is down.
        Only this RBD's own costs count: costs declared inside a nested
        ``RepairableRBD`` are left out. ``result.cost_rate`` converges to
        the exact ``expected_cost_rate`` as the window grows, which is the
        cross-check to reach for.

        When nothing is priced but the components' ``acquisition_cost``
        (see ``has_costs``), the result has no running cost, exactly, and
        the acquisition beside it, without a simulation; when
        nothing at all is priced, it is None: there is no cost model to
        evaluate. (The same result is available as ``availability(...).cost``
        if you also want the availability outputs from the same
        simulations.)

        Parameters
        ----------
        t_simulation : float
            Length of the window each simulation covers, in the time unit
            of the component models.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window, by default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``
            or ``"paths"``, the default) or the minimal cut sets (``"c"``
            or ``"cuts"``); the results are identical.
        mc_samples : int, optional
            Number of simulations, each giving one sample of the window's
            total cost, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None (see
            ``availability``). A cost given as a distribution draws from a
            stream of its own, so pricing never changes the simulated
            failures and repairs.
        tolerance : float, optional
            Simulate until the mean cost of a window is known to within
            ``tolerance`` (in the costs' currency) either side, at
            ``confidence`` (the half-width of ``mean_interval()``): as in
            ``availability``, checked after the first ``mc_samples``
            simulations and each further ``mc_samples``, up to
            ``max_samples``. By default None: exactly ``mc_samples``.
        confidence : float, optional
            The confidence level ``tolerance`` is judged at, by default
            0.95.
        max_samples : int, optional
            The most simulations a run to ``tolerance`` makes, by default
            100 times ``mc_samples``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs (see ``availability``),
            by default False. The draws of costs given as distributions are
            paired too.
        n_jobs : int, optional
            Run the simulations in parallel on ``n_jobs`` CPUs (see
            ``availability``), by default None.
        engine : str, optional
            What runs the simulations: ``"python"``, ``"numba"`` or
            ``"auto"`` (the default), as in ``availability``.
        state : dict, optional
            Start each simulation from the components' current states
            rather than new, as for ``availability``: ``{node:
            NodeState}``. Costs incurred before 0 (a repair or maintenance
            going on at 0 was charged when it started) are not counted. By
            default None: every component new at 0.
        shard_map : callable, optional
            Run the simulations as shards through this map, wherever it
            sends them, as for ``availability``. By default None.
        shard_size : int, optional
            The simulations to a shard, with ``shard_map``, as for
            ``availability``.
        control_variate : bool, optional
            Control the estimate of the mean cost by the system's exact
            twin, as for ``availability``: the result's ``mean`` and
            ``mean_interval`` are then the controlled estimate's, from the
            twin's exact ``expected_cost``, and a ``tolerance`` is judged
            on it. By default None: where the exact methods work out the
            system's expected cost over the window, ``mean`` is that cost,
            with no error, and the breakdowns its split, as
            for ``availability``. False forces a plain simulation, whose
            ``mean`` is ``sample_mean``.
        conditional : bool, optional
            Take the expected cost given the histories of the dependent
            modules, as for ``availability``: each simulation's
            expected cost given them is the modules' own costs as
            simulated, the other nodes' exact expected costs and the
            system downtime's expected cost. By default None: where a
            system has modules, ``mean``, its interval and the breakdowns
            are those of these expected costs, while ``samples``,
            ``sample_mean``, ``std`` and ``percentile`` are the
            simulations' own. False keeps the simulations' own means. True
            simulates only the modules: ``samples`` are then those
            expected costs, so ``sample_mean`` is ``mean``, and
            ``percentile`` and ``std`` refuse.

        Returns
        -------
        CostResult or None
            The simulated costs, or None if nothing is priced (not even an
            ``acquisition_cost``).

        Raises
        ------
        ValueError
            As for ``availability``, when something is priced. (With nothing
            priced this returns None without any checks; with only an
            ``acquisition_cost``, only ``mc_samples`` is checked.)
        NotImplementedError
            As for ``availability``.
        ImportError
            As for ``availability``.

        Examples
        --------
        One component with MTTF 10 and MTTR 1, at 100 per repair, and 50 per
        hour of system downtime:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...             "repair_cost": 100.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=50.0,
        ... )
        >>> result = rbd.cost(t_simulation=100.0, mc_samples=200, seed=0)
        >>> round(result.mean, 2)  # expected cost of a 100-hour window: exact
        1360.33
        >>> round(result.sample_mean, 2)  # the 200 windows' own
        1387.42
        >>> round(result.percentile(90), 2)  # 9 windows in 10 cost less
        1897.0
        >>> round(result.cost_rate, 2), round(rbd.expected_cost_rate(), 2)
        (13.6, 13.64)
        """
        N = 10_000 if mc_samples is None else mc_samples
        t_simulation = simulation_window(t_simulation)

        if not self.has_costs:
            if self.acquisition_cost:
                montecarlo.check_count(N, antithetic, "mc_samples")
            return _acquisition_only(self, N, t_simulation, antithetic)
        return self._simulated(
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            verbose,
            seed,
            tolerance=tolerance,
            confidence=confidence,
            max_N=max_samples,
            antithetic=antithetic,
            n_jobs=n_jobs,
            target="cost",
            engine=engine,
            state=state,
            # The cost result has no curve: count the changes, not keep them.
            curve_points=1,
            shard_map=shard_map,
            shard_size=shard_size,
            control_variate=control_variate,
            conditional=conditional,
        ).cost

    def node_availability(self) -> dict[Hashable, float]:
        """Returns each node's long-run (steady-state) availability.

        Exact: ``MTTF / (MTTF + MTTR)`` for each component (1.0 if it is
        repaired instantly), a nested ``RepairableRBD``'s own
        ``mean_availability``, and 1.0 for the input and output nodes. These
        are the availabilities the importance measures are evaluated at.

        A component under age replacement at ``T`` renews at a failure or
        at a preventive replacement, whichever comes first. By the
        renewal-reward theorem its availability is its mean up time per
        cycle over the mean cycle length,
        ``integral_0^T R / (integral_0^T R + F(T) * MTTR + R(T) * MTTP)``,
        with ``MTTP`` the mean maintenance time (0 for ``"instant"``).

        Under block replacement every ``T`` a unit is replaced at each
        multiple of ``T`` at which it is up, and in between it fails and is
        repaired as usual; a replacement due while it is down is skipped.
        The block times at which it is up are its renewals, so its
        availability is its mean up time between two of them over their
        mean distance apart. That needs the expected up time and number of
        failures of an alternating renewal process of lives and repairs
        within an interval, and what a repair still going on at a block time
        carries into the next: they are computed numerically, with an error
        falling as the square of the grid step (about 1e-6 or less). With
        instant repair and replacement the unit is always up, and it fails
        ``M(T)`` times per interval, ``M`` the renewal function of its
        lives.

        A component with hidden failures, a constant failure rate
        ``lambda``, inspected every ``tau`` with instant tests and instant
        repair, is up a fraction ``(1 - exp(-lambda * tau)) / (lambda *
        tau)`` of the time, about ``1 - lambda * tau / 2``: it is down, from
        a failure until the next inspection, half an interval on average.
        Other hidden failures have no exact long-run values here: simulate
        them.

        With limited ``repair_crews``, a component's is its long-run
        probability of being up in the crews' Markov chain (see
        ``mean_availability``): waiting for a crew, it is down.

        Returns
        -------
        dict[Hashable, float]
            Node name -> long-run availability, in ``[0, 1]``.

        Raises
        ------
        NotImplementedError
            If a component is under block replacement with models its exact
            values do not cover (a lifetime that is not a surpyval
            parametric model with a density, dead-on-arrival units, repairs
            that may never end, or repairs or maintenance far longer than
            the interval), or has hidden failures its numerical values do
            not cover (see ``mean_availability``); or, while a component can
            wait for a repair crew, as for ``mean_availability``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": "instant"},
        ...     },
        ... )
        >>> availability = rbd.node_availability()
        >>> {node: round(a, 4) for node, a in availability.items()}
        {'a': 0.8333, 'b': 1.0, 's': 1.0, 't': 1.0}
        """
        return _long_run.node_availability(self)

    def _require_one_inspected(self) -> None:
        """Raise if more than one component has hidden failures, when no
        intervals are given to choose from (``optimal_inspection_intervals``
        with ``allowed=None``)."""
        if len(self._inspection) > 1:
            raise Refused(
                "More than one component has hidden failures: give the "
                "intervals to choose from in allowed (tests are made on "
                "a calendar, and the long-run values depend on how the "
                "schedules line up)."
            )

    def _require_capacities_given(self) -> None:
        """Raise if a node takes its capacity from its model (a
        ``DegradingNode``'s stages, or a nested RBD's capacities), which
        the availability simulation does not follow."""
        own = self._capacity_models()
        if own:
            raise NotImplementedError(
                f"Node(s) {sorted(own, key=str)} take their capacity from "
                "their models (a DegradingNode's stages, or a nested RBD's "
                "capacities), which the simulation does not follow. Give "
                "them a capacity, or use capacity_distribution() for the "
                "long run."
            )

    def _require_reliabilities(self, node) -> None:
        """Raise, as the model itself does, if a component's life or repair
        model has no exact or numerical reliability: a ``StandbyModel`` or
        ``LoadSharingModel`` that only simulations take (#149)."""
        component = self.components[node]
        for model in (
            getattr(component, "reliability", None),
            getattr(component, "time_to_replace", None),
        ):
            if (
                isinstance(model, (StandbyModel, LoadSharingModel))
                and model.is_simulated
            ):
                raise model._no_reliability()

    def _require_time_models(self, node) -> None:
        """Raise if a component's life or repair model is a probability,
        not a distribution of times: its long-run and time-dependent values
        then have no exact value (its mean is no mean time). (First, if
        one has no reliability: see ``_require_reliabilities``.)"""
        self._require_reliabilities(node)
        self._require_times(node)

    def _require_times(self, node) -> None:
        """Raise if a component's life or repair model is a probability,
        not a distribution of times: it has no mean time, nor a value over
        time (see ``_require_time_models``)."""
        component = self.components[node]
        for what, model in [
            ("reliability", getattr(component, "reliability", None)),
            ("repairability", getattr(component, "time_to_replace", None)),
        ]:
            if is_fixed_probability(model):
                raise NotImplementedError(
                    f"Component {node!r}: its {what} model is a probability, "
                    "not a distribution of times, so it has no mean time or "
                    "availability over time: a unit fails at once with that "
                    "probability, or never. Estimate the system by "
                    "simulation, with availability()."
                )

    def _require_unscheduled_stages(self, node) -> None:
        """Raise if a degrading component is maintained or inspected on a
        schedule: its long-run time in each stage has no exact value."""
        if node in self._preventive or node in self._inspection:
            raise NotImplementedError(
                f"Component {node!r} degrades through stages and is "
                "maintained or inspected on a schedule: its long-run time in "
                "each stage has no exact value here. Give it a capacity "
                "instead."
            )

    def _imperfect_phrase(self, node) -> str:
        """How a component is repaired imperfectly, in words."""
        imperfect = self._imperfect[node]
        kind = "I" if imperfect.kijima == "kijima1" else "II"
        phrase = f"Kijima {kind}, q = {imperfect.q:g}"
        if imperfect.replace_after is not None:
            phrase += (
                f", replaced at failure {imperfect.replace_after} since it "
                "was renewed"
            )
        return phrase

    def _require_perfect_repair(self, node) -> None:
        """Raise if a component is repaired imperfectly (a spec's
        ``"repair"``): a repair does not renew it, so its long-run values
        are known only by simulation (its values over time, only by
        simulation too, but for minimal repair in no time: see
        ``_require_minimal_repair``)."""
        if node in self._imperfect:
            over_time = (
                "its values over a window from new are exact, as it fails "
                "as often as its life's cumulative hazard"
                if self._minimal_repair_blocker(node) is None
                else "nor its values over time"
            )
            raise NotImplementedError(
                f"Component {node!r} is repaired imperfectly "
                f"({self._imperfect_phrase(node)}), so a repair does not "
                f"renew it: its long-run values have no exact value here "
                f"({over_time}). Estimate them by simulation, with "
                "availability() or cost()."
            )

    def _minimal_repair_blocker(self, node) -> Optional[str]:
        """Why an imperfectly repaired component's values over time have no
        exact value, or None if they have: when it is minimally repaired
        (Kijima, ``q = 1``) in no time, with no ``replace_after``,
        preventive maintenance or tests, and a life that works at 0. Its
        failures are then a non-homogeneous Poisson process whose intensity
        is its life's hazard at its age (its age is its operating time,
        which is all the time), so that it is up throughout and fails
        ``H(t)`` times by ``t`` on average, ``H`` the life's cumulative
        hazard (see ``MinimalRepairCurve``)."""
        imperfect = self._imperfect[node]
        component = self.components[node]
        if imperfect.q < 1.0:
            return "its repairs take away some of its age (q below 1)"
        if imperfect.replace_after is not None:
            return (
                f"it is replaced at its failure {imperfect.replace_after} "
                "since it was renewed"
            )
        repair = component.time_to_replace
        if float(np.ravel(_sf_values(repair.sf, np.zeros(1)))[0]) > 0.0:
            return "its repairs take time"
        if node in self._preventive:
            return "it has preventive maintenance"
        if node in self._inspection:
            return "its failures are found only by its tests"
        life = component.reliability
        if float(np.ravel(_sf_values(life.sf, np.zeros(1)))[0]) < 1.0:
            return "its life may end at 0 (dead on arrival)"
        return None

    def _require_minimal_repair(self, node) -> None:
        """Raise unless an imperfectly repaired component's values over time
        are exact: minimal repair in no time (see
        ``_minimal_repair_blocker``)."""
        blocker = self._minimal_repair_blocker(node)
        if blocker is not None:
            raise NotImplementedError(
                f"Component {node!r} is repaired imperfectly "
                f"({self._imperfect_phrase(node)}), and {blocker}: its "
                "availability over time has no exact value here (it has "
                "for minimal repair, q = 1, in no time, with no "
                "replace_after, maintenance or tests). Estimate it by "
                "simulation, with availability() or cost()."
            )

    def _require_no_opportunities(self, node) -> None:
        """Raise if a component can be renewed early at the stops of its
        maintenance group (an ``"opportunity"`` below its interval): when
        depends on the other members, so its long-run values and its
        availability over time are known only by simulation."""
        if node in self._early_members:
            raise NotImplementedError(
                f"Component {node!r} is renewed early at the stops of its "
                f"maintenance group {self._member_group[node]!r}, which "
                "depend on the other members, so its long-run values and "
                "its availability over time have no exact value here. "
                "Estimate them by simulation, with availability() or cost()."
            )

    def _require_separate_setups(self) -> None:
        """Raise if two members of a maintenance group with a set-up cost
        can be replaced at the same instants, again and again: on block
        schedules, or never failing before an age replacement in zero time.
        Such replacements share one stop, and its set-up, which the exact
        cost rate, charging a set-up for each member's failures and
        replacements, does not count."""
        for group, spec in self._maintenance.items():
            if not spec.setup_cost:
                continue
            clocked = [node for node in spec.members if self._on_a_clock(node)]
            if len(clocked) > 1:
                raise NotImplementedError(
                    f"Components {clocked} of maintenance group {group!r} "
                    "are replaced on a clock (on block schedules, or never "
                    "failing before an age replacement in zero time), so "
                    "their replacements can fall at the same instants and "
                    "share a set-up, which the exact cost rate does not "
                    "count: estimate it by simulation, with cost()."
                )

    def _on_a_clock(self, node) -> bool:
        """Whether a component's replacements fall on a fixed lattice of
        times: under block replacement, or under age replacement in zero
        time without a failure before it."""
        schedule = self._preventive.get(node)
        if schedule is None or not math.isfinite(schedule.interval):
            return False
        if schedule.policy == "block":
            return True
        if schedule.policy != "age" or schedule.duration is not None:
            return False
        survives = self.components[node].reliability_function(
            schedule.interval
        )
        return bool(np.ravel(survives)[0] >= 1.0)

    def _require_block_models(self, node) -> None:
        """Raise unless the exact block-replacement values cover the
        component's models (the checks ``block_cycle`` makes first)."""

        self._refuse_level(node)
        component = self.components[node]
        _check_life(component.reliability, node)
        _Duration(component.time_to_replace, "repair", node)
        _Duration(self._preventive[node].duration, "replacement", node)

    def _refuse_level(self, node) -> None:
        """Raise for a component replaced on condition by its measured
        degradation level (#271): its inspections follow the level, which
        only the simulation draws."""
        schedule = self._preventive.get(node)
        if schedule is not None and schedule.level is not None:
            raise NotImplementedError(
                f"Component {node!r} is replaced on condition by its "
                "measured degradation level, which only the simulation "
                "follows: estimate it by simulation, with availability() or "
                "cost()."
            )

    def _require_calendars(self) -> None:
        """Raise if the exact long-run values cannot average over the
        components' calendars (block replacement and inspection): a nested
        RBD's calendar with another here, intervals with no common period,
        or one repeating too often in it (before any computation)."""
        nested = [
            node
            for node, c in self.components.items()
            if isinstance(c, RepairableRBD) and _long_run._has_calendar(c)
        ]
        blocks = _long_run._block_nodes(self)
        if nested and len(nested) + len(self._inspection) + len(blocks) > 1:
            raise NotImplementedError(
                f"Node(s) {sorted(nested, key=str)} are RBDs with hidden "
                "failures or block replacement, and other nodes' inspections "
                "or block replacements here fall at the same times: "
                "estimate the long-run values by simulation, with "
                "availability() or cost()."
            )
        intervals = {self._preventive[node].interval for node in blocks}
        intervals |= {
            self._inspection[node].interval for node in self._inspection
        }
        if not intervals:
            return
        period = _common_period(intervals)
        if any(round(period / interval) > 100_000 for interval in intervals):
            if blocks:
                raise NotImplementedError(
                    f"The block-replacement and inspection intervals "
                    f"{sorted(intervals)} repeat together only after too many "
                    "intervals to average over: estimate the long-run values "
                    "by simulation, with availability() or cost()."
                )
            raise NotImplementedError(
                f"The inspection intervals {sorted(intervals)} repeat "
                "together only after too many inspections to average "
                "over: estimate the long-run values by simulation, with "
                "availability() or cost()."
            )

    def _instant_tests(self, node) -> bool:
        """Whether a component with hidden failures is tested, and
        repaired, in no time."""
        return (
            self._inspection[node].duration is None
            and model_mean(self.components[node].time_to_replace) == 0.0
        )

    def _tested_kind(self, node) -> str:
        """How a component with hidden failures' values are worked out:
        ``"closed"``, closed forms, for a constant failure rate and tests
        and repair in no time; ``"life"``, summed over the test intervals
        (``TestedLife``, #144), for any other life with those and tests
        that find every failure; and ``"unit"``, its renewal cycle followed
        test by test on a grid (``TestedUnit``, #159), for tests or repairs
        that take time, or tests that can miss a failure of a life that is
        not exponential."""
        instant = self._instant_tests(node)
        if instant and _constant_rate(self.components[node].reliability):
            return "closed"
        if instant and not self._inspection[node].partial:
            return "life"
        return "unit"

    def _require_tested_exact(self, node) -> None:
        """Raise unless a component with hidden failures has exact or
        numerical values (long-run, or over time): with tests and repair in
        no time, any life (#144); with tests or repairs that take time, or
        tests that can miss a failure, a surpyval parametric life with a
        density and repair and test times that end, the tests within the
        interval (#159)."""
        if self._tested_kind(node) != "unit":
            return
        component = self.components[node]
        inspection = self._inspection[node]
        check_tested(
            component.reliability,
            component.time_to_replace,
            inspection.duration,
            inspection.interval,
            node,
        )

    def _inspected_rate(self, node) -> Tuple[float, float]:
        """The constant failure rate and the inspection interval of a
        component with hidden failures, for the closed forms of an
        exponential life tested and repaired in no time (see
        ``_tested_life`` and ``_tested_unit`` for the others)."""
        self._require_tested_exact(node)
        if self._tested_kind(node) != "closed":
            raise NotImplementedError(
                f"Component {node!r} has hidden failures, and a life that is "
                "not exponential or tests or repairs that take time, which "
                "this does not take: estimate it by simulation, with "
                "availability() or cost()."
            )
        rate = _constant_rate(self.components[node].reliability)
        assert rate is not None
        return rate, self._inspection[node].interval

    def _tested_life(self, node) -> Optional[TestedLife]:
        """For a component with hidden failures, a life other than
        exponential, and tests and repair in no time that find every
        failure, its long run under its tests (see ``_hidden_life``),
        worked out once; None for any other (see ``_tested_kind``)."""
        self._require_tested_exact(node)
        if self._tested_kind(node) != "life":
            return None
        component = self.components[node]
        interval = float(self._inspection[node].interval)
        cache = self.__dict__.setdefault("_tested_lives", {})
        key = (node, id(component.reliability), interval)
        if key not in cache:
            cache[key] = TestedLife(component.reliability, interval)
        return cache[key]

    def _tested_unit(
        self, node, any_kind: bool = False
    ) -> Optional[TestedUnit]:
        """For a component with hidden failures whose tests or repairs
        take time, or whose tests can miss a failure of a life that is not
        exponential, its values under its tests (see ``_hidden_tests``),
        worked out once; None for any other (see ``_tested_kind``), but
        with ``any_kind``, for its spares (whose tests the caller has
        checked)."""
        if not any_kind:
            self._require_tested_exact(node)
            if self._tested_kind(node) != "unit":
                return None
        component = self.components[node]
        inspection = self._inspection[node]
        cache = self.__dict__.setdefault("_tested_units", {})
        key = (
            node,
            id(component.reliability),
            id(component.time_to_replace),
            id(inspection.duration),
            float(inspection.interval),
            float(inspection.coverage),
            inspection.per_full_test,
        )
        if key not in cache:
            cache[key] = TestedUnit(
                component.reliability,
                component.time_to_replace,
                inspection.duration,
                inspection.interval,
                inspection.coverage,
                inspection.per_full_test,
                node,
                _constant_rate(component.reliability),
            )
        return cache[key]

    def _unit_phase(self, node, times: np.ndarray) -> np.ndarray:
        """The time since a tested component's last full test at each of
        ``times`` on its calendar (its full tests at its offset and every
        full test's interval from it): where its long run's profile is
        read (see ``TestedLongRun``)."""
        inspection = self._inspection[node]
        position = times - inspection.offset if inspection.offset else times
        period = inspection.period
        return position - period * np.floor(position / period)

    def _tested_rate(self, node) -> float:
        """How fast a tested component's long-run availability falls
        between tests: its failure rate, or for any other life the rate of
        its profile's mean decay (see ``TestedLife.rate``), or its failures
        per unit of up time (``TestedLongRun.rate``), for the long-run
        grid's spacing."""
        unit = self._tested_unit(node)
        if unit is not None:
            return unit.long_run.rate
        life = self._tested_life(node)
        return self._inspected_rate(node)[0] if life is None else life.rate

    def _tested_scale(self, node) -> float:
        """The rate that sets the scale of a tested component's interval:
        its failure rate, or one over the mean of any other life."""
        self._require_tested_exact(node)
        life = self.components[node].reliability
        rate = _constant_rate(life)
        return float(rate) if rate else 1.0 / float(model_mean(life))

    def _tested_phase(self, node, times: np.ndarray) -> np.ndarray:
        """The time since a tested component's last test at each of
        ``times`` on its calendar (its tests at its offset and every
        interval from it)."""
        inspection = self._inspection[node]
        position = times - inspection.offset if inspection.offset else times
        interval = inspection.interval
        return position - interval * np.floor(position / interval)

    def _tested_intensity(
        self, node, times: np.ndarray, availability
    ) -> np.ndarray:
        """A tested component's long-run rate of failing at each of
        ``times`` (``availability`` its availability there): its constant
        rate while it is up, or for any other life the rate of its profile
        (see ``TestedLife.intensity``)."""
        unit = self._tested_unit(node)
        if unit is not None:
            return unit.long_run.intensity(self._unit_phase(node, times))
        life = self._tested_life(node)
        if life is None:
            rate, _ = self._inspected_rate(node)
            return rate * availability
        return life.intensity(self._tested_phase(node, times))

    def system_failure_frequency(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run failure frequency (failures per unit
        time), by the Birnbaum/Vesely formula.

        In steady state the system failure frequency is

        ```text
        omega_sys = sum_i I_B(i) * omega_i
        ```

        where ``I_B(i)`` is node i's Birnbaum importance evaluated at the
        nodes' long-run availabilities (see ``birnbaum_importance``) and
        ``omega_i`` is node i's long-run failure frequency: ``1 / (MTTF_i +
        MTTR_i)`` for a component, ``F(T) / C`` for one under age
        replacement at ``T`` (``C`` its mean renewal cycle, see
        ``expected_cost_rate``), its mean failures per renewal cycle over
        the cycle's mean length for one under block replacement, and a
        nested ``RepairableRBD``'s own
        ``system_failure_frequency``. Exact for independent repairable
        nodes, with no simulation. Each Birnbaum importance is a sum of
        products of the node availabilities and unavailabilities, so a small
        frequency (a reliable redundant system's, say) keeps its full
        relative precision. Every system failure counts, including
        the zero-length outages an instantly repaired component causes;
        planned outages (preventive maintenance) are not failures.

        With limited ``repair_crews`` the formula is summed over the states
        of the crews' Markov chain (see ``mean_availability``): in each, a
        component that is up fails at its constant rate, and takes the
        system down if it is critical there.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (they contribute no
            failures), by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (they contribute no
            failures), by default None.

        Returns
        -------
        float
            Expected number of system failures per unit time, in the long
            run.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            If a component is under block replacement with models its exact
            values do not cover (see ``node_availability``); or, while a
            component can wait for a repair crew, as for
            ``mean_availability``.

        Examples
        --------
        Two components in series, with MTTFs 5 and 2 and MTTRs of 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.system_failure_frequency(), 4)  # 2/3 * 1/6 + 5/6 * 1/3
        0.3889
        >>> round(rbd.system_failure_frequency(working_nodes=["a"]), 4)
        0.3333
        """
        return _long_run.system_failure_frequency(
            self, working_nodes=working_nodes, broken_nodes=broken_nodes
        )

    def mean_time_between_failures(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Time Between Failures.

        Exact, to full precision however rare the failures:
        ``MTBF = 1 / system_failure_frequency`` — the mean length of
        one full up-down cycle, i.e. ``MTBF = MUT + MDT`` (when there are
        no planned outages: preventive maintenance that takes time also
        ends up periods, so MTBF is then longer). Infinite if the system
        never fails (e.g. a node held working keeps it up, or one held
        broken keeps it down).

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean time between system failures, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        Two components in series, with MTTFs 5 and 2 and MTTRs of 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_up_time(), 4)  # 1 / (0.2 + 0.5)
        1.4286
        >>> round(rbd.mean_down_time(), 4)
        1.1429
        >>> round(rbd.mean_time_between_failures(), 4)  # MUT + MDT
        2.5714
        """
        return _long_run.mean_time_between_failures(
            self, working_nodes=working_nodes, broken_nodes=broken_nodes
        )

    def mean_up_time(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Up Time.

        Exact: the mean duration of an uninterrupted working period of the
        system (sometimes called the repairable-system MTTF),
        ``MUT = mean_availability / system_failure_frequency``, to full
        precision however rare the failures. A planned
        outage (preventive maintenance that takes time) also ends a working
        period, so its frequency is added to the failure frequency. If the
        system never goes down it is infinite when the system is up and
        0.0 when it is always down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean up time, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        For exponential failures in series, MUT is one over the sum of the
        failure rates:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_up_time(), 4)  # 1 / (0.2 + 0.5)
        1.4286
        >>> round(rbd.mean_up_time(working_nodes=["a"]), 4)  # b alone
        2.0
        """
        return _long_run.mean_up_time(
            self, working_nodes=working_nodes, broken_nodes=broken_nodes
        )

    def mean_down_time(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Down Time.

        Exact: the mean duration of a system outage (the repairable-system
        MTTR), ``MDT = mean_unavailability / system_failure_frequency``,
        both worked out in their own right, so a reliable system's keeps
        its precision. Planned outages (preventive maintenance that takes
        time) count as
        outages, so their frequency is added to the failure frequency. If
        the system never goes down it is 0.0 when the system is always up
        and infinite when it is down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean down time, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        A parallel pair is down only while both are, and the outage ends
        when either is repaired:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_down_time(), 4)  # 1 / (1.0 + 1.0)
        0.5
        >>> rbd.mean_down_time(working_nodes=["a"])  # never down
        0.0
        """
        return _long_run.mean_down_time(
            self, working_nodes=working_nodes, broken_nodes=broken_nodes
        )

    @_times_first()
    def birnbaum_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        x=None,
        window=None,
        state=None,
    ) -> dict[Any, Any]:
        """Returns the Birnbaum measure of importance for all nodes,
        evaluated at the nodes' long-run availabilities.

        In the guide's Greeks it is *delta*: how far the system moves with each
        component (see [Sensitivities: the Greeks](../guide/greeks.md)).

        Exact, with no simulation: ``I_B(i) = A_sys(A_i = 1) -
        A_sys(A_i = 0)``, the system's long-run availability with node i
        always working minus that with node i always failed. It is the
        probability that node i is critical (the system works if and only
        if node i does), and how much the system availability changes per
        unit change in node i's availability. The node availabilities are
        those of ``node_availability``, with ``working_nodes`` and
        ``broken_nodes`` held at 1 and 0. It is worked out for every node
        at once, as the derivative of the system availability: a sum of
        products of the node availabilities and unavailabilities (each
        worked out in its own right, see ``mean_unavailability``), so a
        small one keeps its full relative precision, where the difference
        above would cancel in a reliable system.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent. With common-cause groups, a member's values with it
        working and failed are the system's availability *given* its state
        at each long-run time (the groups' Markov chains give the members'
        joint states), averaged over the times as every node's are; a node
        outside the groups is held up and down, as without them. The other
        measures are built on the same values, and ``beta = 0`` gives them
        as without the group.

        With limited ``repair_crews``, under which a component can wait for
        a crew, the components are not independent: node i is then held
        working and failed in the crews' Markov chain, which is solved
        without it (held either way, it needs no crew), and ``I_B(i)`` is
        the difference of the system's long-run availabilities so held. The
        improvement potential, RAW and RRW are built on the same values;
        the criticality and Fussell-Vesely measures are probabilities over
        the chain's states. With a crew for each component they are the
        independent ones.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        x : float or array-like, optional
            Times from new (or from ``state``) to evaluate it at instead of
            the long run, by default None: at each, the nodes are up with
            their point availabilities then (see ``point_availability``),
            and with limited repair crews the crews' chain is followed to
            it.
        window : float or array-like, optional
            The length of a window ``[0, window)`` to evaluate it over
            instead, by default None: a ratio measure is then the ratio of
            the system's means over the window (as ``mission_availability``
            is its mean availability), not the mean of the ratio. Not with
            ``x``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states to
            start from, as for ``point_availability``; by default None,
            every component new at 0.

        Returns
        -------
        dict[Any, float or numpy.ndarray]
            Dictionary with node names as keys and Birnbaum importances as
            values, for every node except the input and output nodes.
            Floats in the long run, at a single time or over a single
            window; else arrays in the shape of ``x`` or ``window``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        In series, a node is critical exactly when the others work:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> importance = rbd.birnbaum_importance()
        >>> {node: round(i, 4) for node, i in importance.items()}
        {'a': 0.6667, 'b': 0.8333}
        >>> round(rbd.birnbaum_importance(working_nodes=["a"])["b"], 4)
        1.0
        """
        return _repairable_importance.birnbaum_importance(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            x=x,
            window=window,
            state=state,
        )

    @_times_first()
    def improvement_potential(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        x=None,
        window=None,
        state=None,
    ) -> dict[Any, Any]:
        """Returns the improvement potential of all nodes, evaluated at the
        nodes' long-run availabilities.

        Exact, with no simulation: ``A_sys(A_i = 1) - A_sys``, the gain in
        the system's long-run availability if node i never failed, worked
        out as ``I_B(i) * (1 - A_i)`` (see ``birnbaum_importance``) so that
        a small one keeps its precision. The node availabilities are those
        of ``node_availability``, with ``working_nodes`` and
        ``broken_nodes`` held at 1 and 0.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        x : float or array-like, optional
            Times from new (or from ``state``) to evaluate it at instead of
            the long run, by default None: at each, the nodes are up with
            their point availabilities then (see ``point_availability``),
            and with limited repair crews the crews' chain is followed to
            it.
        window : float or array-like, optional
            The length of a window ``[0, window)`` to evaluate it over
            instead, by default None: a ratio measure is then the ratio of
            the system's means over the window (as ``mission_availability``
            is its mean availability), not the mean of the ratio. Not with
            ``x``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states to
            start from, as for ``point_availability``; by default None,
            every component new at 0.

        Returns
        -------
        dict[Any, float or numpy.ndarray]
            Dictionary with node names as keys and improvement potentials as
            values, for every node except the input and output nodes.
            Floats in the long run, at a single time or over a single
            window; else arrays in the shape of ``x`` or ``window``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> potential = rbd.improvement_potential()
        >>> {node: round(p, 4) for node, p in potential.items()}
        {'a': 0.1111, 'b': 0.2778}
        """
        return _repairable_importance.improvement_potential(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            x=x,
            window=window,
            state=state,
        )

    @_times_first()
    def risk_achievement_worth(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        x=None,
        window=None,
        state=None,
    ) -> dict[Any, Any]:
        """Returns the RAW importance per Modarres & Kaminskiy, evaluated at
        the nodes' long-run availabilities. That is RAW_i =
        (unavailability of system given i failed) /
        (nominal system unavailability).

        Exact, with no simulation: ``RAW_i = U_sys(A_i = 0) / U_sys``,
        where ``U = 1 - A`` is long-run unavailability: the factor by which
        the system's unavailability grows if node i is always failed. The
        node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0. Where the
        nominal unavailability is 0 (e.g. every component is repaired
        instantly), the ratio is ``inf`` or ``nan``, with a numpy warning.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        x : float or array-like, optional
            Times from new (or from ``state``) to evaluate it at instead of
            the long run, by default None: at each, the nodes are up with
            their point availabilities then (see ``point_availability``),
            and with limited repair crews the crews' chain is followed to
            it.
        window : float or array-like, optional
            The length of a window ``[0, window)`` to evaluate it over
            instead, by default None: a ratio measure is then the ratio of
            the system's means over the window (as ``mission_availability``
            is its mean availability), not the mean of the ratio. Not with
            ``x``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states to
            start from, as for ``point_availability``; by default None,
            every component new at 0.

        Returns
        -------
        dict[Any, float or numpy.ndarray]
            Dictionary with node names as keys and RAW importances as
            values, for every node except the input and output nodes.
            Floats in the long run, at a single time or over a single
            window; else arrays in the shape of ``x`` or ``window``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        In series the system is down whenever a node is, so failing any
        node multiplies the unavailability by ``1 / U_sys``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> raw = rbd.risk_achievement_worth()
        >>> {node: round(r, 4) for node, r in raw.items()}
        {'a': 2.25, 'b': 2.25}
        """
        return _repairable_importance.risk_achievement_worth(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            x=x,
            window=window,
            state=state,
        )

    @_times_first()
    def risk_reduction_worth(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        x=None,
        window=None,
        state=None,
    ) -> dict[Any, Any]:
        """Returns the RRW importance per Modarres & Kaminskiy, evaluated at
        the nodes' long-run availabilities. That is RRW_i =
        (nominal unavailability of system) /
        (unavailability of system given i is working).

        Exact, with no simulation: ``RRW_i = U_sys / U_sys(A_i = 1)``,
        where ``U = 1 - A`` is long-run unavailability: the factor by which
        the system's unavailability would fall if node i never failed. The
        node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0. It is
        ``inf`` (with a numpy warning) where making node i perfect would
        make the system perfect, as for either node of a parallel pair, and
        ``nan`` if the system is already never down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        x : float or array-like, optional
            Times from new (or from ``state``) to evaluate it at instead of
            the long run, by default None: at each, the nodes are up with
            their point availabilities then (see ``point_availability``),
            and with limited repair crews the crews' chain is followed to
            it.
        window : float or array-like, optional
            The length of a window ``[0, window)`` to evaluate it over
            instead, by default None: a ratio measure is then the ratio of
            the system's means over the window (as ``mission_availability``
            is its mean availability), not the mean of the ratio. Not with
            ``x``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states to
            start from, as for ``point_availability``; by default None,
            every component new at 0.

        Returns
        -------
        dict[Any, float or numpy.ndarray]
            Dictionary with node names as keys and RRW importances as
            values, for every node except the input and output nodes.
            Floats in the long run, at a single time or over a single
            window; else arrays in the shape of ``x`` or ``window``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> rrw = rbd.risk_reduction_worth()
        >>> {node: round(r, 4) for node, r in rrw.items()}
        {'a': 1.3333, 'b': 2.6667}
        """
        return _repairable_importance.risk_reduction_worth(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            x=x,
            window=window,
            state=state,
        )

    @_times_first()
    def criticality_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        kind: str = "failure",
        *,
        x=None,
        window=None,
        state=None,
    ) -> dict[Any, Any]:
        """Returns the criticality importance of all nodes, evaluated at the
        nodes' long-run availabilities.

        Exact, with no simulation. With ``I_B(i)`` the Birnbaum importance,
        ``A_i`` node i's availability and ``A_sys`` the system's:

        - ``kind="failure"`` (the default) gives the failure-oriented form
          (Rausand & Høyland), ``I_B(i) * (1 - A_i) / (1 - A_sys)``: the
          probability that node i is down and critical, given that the
          system is down -- the share of the system's downtime that node i
          accounts for. It ranks nodes in series by their unavailability.
          It is computed from the node unavailabilities, through the
          minimal cut sets, so the system unavailability is not lost to
          cancellation in ``1 - A_sys`` however available the system is.
          It is ``nan`` if the system is never down.
        - ``kind="success"`` gives the success-oriented form,
          ``I_B(i) * A_i / A_sys``: the probability that node i is working
          and critical, given that the system is working. It is 1 for every
          node in series with the rest of the system, so it cannot rank
          them. It is ``nan`` if the system is never up.

        The node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.
        kind : str, optional
            ``"failure"`` (the default) or ``"success"``.

        x : float or array-like, optional
            Times from new (or from ``state``) to evaluate it at instead of
            the long run, by default None: at each, the nodes are up with
            their point availabilities then (see ``point_availability``),
            and with limited repair crews the crews' chain is followed to
            it.
        window : float or array-like, optional
            The length of a window ``[0, window)`` to evaluate it over
            instead, by default None: a ratio measure is then the ratio of
            the system's means over the window (as ``mission_availability``
            is its mean availability), not the mean of the ratio. Not with
            ``x``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states to
            start from, as for ``point_availability``; by default None,
            every component new at 0.

        Returns
        -------
        dict[Any, float or numpy.ndarray]
            Dictionary with node names as keys and criticality importances
            as values, for every node except the input and output nodes.
            Floats in the long run, at a single time or over a single
            window; else arrays in the shape of ``x`` or ``window``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if ``kind`` is neither
            ``"failure"`` nor ``"success"``.
        NotImplementedError
            As for ``mean_availability``.

        References
        ----------
        M. Rausand and A. Høyland, System Reliability Theory: Models,
        Statistical Methods, and Applications, 2nd edition, Wiley, 2004.

        Examples
        --------
        Two nodes in series, ``b`` down twice as often as ``a``: ``b``
        accounts for more of the system's downtime (the shares add to less
        than 1, as neither is critical while both are down):

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> criticality = rbd.criticality_importance()
        >>> {node: round(c, 4) for node, c in criticality.items()}
        {'a': 0.25, 'b': 0.625}

        The success-oriented form cannot tell them apart:

        >>> criticality = rbd.criticality_importance(kind="success")
        >>> {node: round(c, 4) for node, c in criticality.items()}
        {'a': 1.0, 'b': 1.0}
        """
        return _repairable_importance.criticality_importance(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            kind=kind,
            x=x,
            window=window,
            state=state,
        )

    @_times_first()
    def fussell_vesely(
        self,
        fv_type: str = "c",
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "exact",
        *,
        x=None,
        window=None,
        state=None,
    ) -> dict[Any, Any]:
        """Calculate Fussell-Vesely importance of all nodes, evaluated at the
        nodes' long-run availabilities.

        The Fussell-Vesely importance of node i is the probability that
        some minimal cut set containing node i has failed (all its nodes
        down), over the probability that the system has failed: the share
        of the system's unavailability that involves node i. Here a node's
        probability of having failed is its long-run unavailability,
        ``1 - A_i``, from ``node_availability`` (with ``working_nodes`` and
        ``broken_nodes`` held at availability 1 and 0), and the system's is
        ``1 - A_sys``; both are exact, with no simulation, and worked out
        in their own right (see ``mean_unavailability``), so small ones
        keep their precision. ``method="exact"`` (the default) works out
        the probability that some cut set containing node i has failed
        exactly, from the exact engine, so the measure is between 0 and 1.
        ``method="rare_event"`` sums the cut sets' probabilities instead,
        the usual rare-event approximation, which with large
        unavailabilities can exceed 1. If the system never fails, the ratio
        is ``nan`` or ``inf``, with a numpy warning.

        Typically this measure is implemented using cut-sets as mentioned
        above, although it can be implemented using path-sets. Both are
        implemented here, selected by ``fv_type``: ``"c"`` takes the
        minimal cut sets containing node i, ``"p"`` the minimal path sets
        containing it. Either way a set has failed when all of its nodes
        have, and the probability that one of them has (or, with
        ``"rare_event"``, the sum of their probabilities) is divided by the
        system's unavailability.

        Parameters
        ----------
        fv_type : str, optional
            Dictates the method of calculation, "c" = cut-set and
            "p" = path-set, by default "c".
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.
        method : str, optional
            ``"exact"`` (the default) or ``"rare_event"``.

        x : float or array-like, optional
            Times from new (or from ``state``) to evaluate it at instead of
            the long run, by default None: at each, the nodes are up with
            their point availabilities then (see ``point_availability``),
            and with limited repair crews the crews' chain is followed to
            it.
        window : float or array-like, optional
            The length of a window ``[0, window)`` to evaluate it over
            instead, by default None: a ratio measure is then the ratio of
            the system's means over the window (as ``mission_availability``
            is its mean availability), not the mean of the ratio. Not with
            ``x``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states to
            start from, as for ``point_availability``; by default None,
            every component new at 0.

        Returns
        -------
        dict[Any, float or numpy.ndarray]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values, for every node except the input and output nodes.
            Floats in the long run, at a single time or over a single
            window; else arrays in the shape of ``x`` or ``window``.

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set), or
            ``method`` not 'exact' or 'rare_event'; if a working/broken node
            is unknown, is the input or output node, or is in both sets.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        In series each node is a cut set on its own, so its importance is
        its unavailability over the system's:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> fv = rbd.fussell_vesely()
        >>> {node: round(v, 4) for node, v in fv.items()}
        {'a': 0.375, 'b': 0.75}
        """
        return _repairable_importance.fussell_vesely(
            self,
            fv_type=fv_type,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            x=x,
            window=window,
            state=state,
        )

    @_times_first()
    def parameter_sensitivity(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        x=None,
        window=None,
        state=None,
        rel_step: Optional[float] = None,
        of="availability",
        unit_costs: Optional[dict] = None,
    ) -> dict:
        """How the system's availability (or its cost rate) moves with each
        lever: the derivative in each of its components' parameters, and
        the change one more standby unit or repair crew makes.

        In the guide's Greeks it is the levers' *deltas* (see [Sensitivities:
        the Greeks](../guide/greeks.md)).

        The levers are each component's life and repair models'
        parameters (``"reliability.alpha"``, ``"repairability.beta"``,
        ... with surpyval's names), its preventive maintenance's
        (``"preventive.interval"``, ``"preventive.threshold"``,
        ``"preventive.opportunity"`` and ``"preventive.duration.<name>"``),
        its tests' (``"inspection.interval"``, ``"inspection.coverage"``,
        ``"inspection.offset"`` and ``"inspection.duration.<name>"``), its
        standby group's (``"standby.dormancy_factor"``,
        ``"standby.switching_probability"``, and ``"standby.units"``, one
        more unit), its imperfect repair's (``"repair.q"``), a common-cause
        group's (its members' parameters, moved together, and its model's
        ``"ccf_beta"``, ``"ccf_gamma"``, ...), and the repair crews
        (``"repair_crews"``, one more, under the key None). ``levers()``
        lists them, with their values and ranges, and ``with_levers``
        builds the RBD with them moved, as they are moved here.

        A continuous lever's derivative is a central difference of the
        system's own value with the lever moved by ``rel_step`` of its
        value either way, the diagram rebuilt with the changed spec:
        one-sided where one side is not a valid value (a coverage past 1,
        say), NaN where neither is. In the long run (the default) that is
        a difference of the exact long-run values, worked out as
        unavailabilities so that a small one keeps its precision. At times
        ``x`` (from new, or from ``state``) or over ``[0, window)``, with
        independent components, a component's lever moves its own curve
        alone, and the system's availability is linear in each
        component's: the derivative is the component's Birnbaum importance
        at each time (see ``birnbaum_importance``) times the difference of
        its own point availability, worked out on the component alone,
        and over a window the same averaged on the window's quadrature
        points (but for a lever that moves its curve's breaks, an interval
        or offset, differenced on the system). With limited repair crews
        or common-cause groups the system itself is differenced.

        In the long run, a block replacement's or a test's interval of a
        component whose schedule shares a calendar with others' (other
        block replacements or tests) would move it off their common
        calendar, where the long-run values jump (they are averages over
        the schedules' common period, and the components' outages fall
        together or apart): its derivative takes the component's schedule
        as apart from the others', its long-run Birnbaum importance times
        the change in its own long-run availability (and its own cost
        rate). A test interval moves the full tests with it, every so many
        tests as before. A common-cause group's interval there is NaN.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Nodes held working, as for ``mean_availability``: a held node's
            levers report 0.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed, likewise.
        x : float or array-like, optional
            Times from new (or from ``state``) to take the point
            availability's sensitivity at, by default None: the long-run
            availability's.
        window : float, optional
            Take the sensitivity of the mean availability over
            ``[0, window)`` (``mission_availability``) instead. Not with
            ``x``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states to
            start from, as for ``point_availability``.
        rel_step : float, optional
            The step of a continuous lever's difference, relative to its
            value (absolute where the value is 0), by default ``1e-5`` in
            the long run and ``1e-2`` over time, whose curves are
            numerical (to about ``1e-7``): a smaller step there would
            difference their error. The derivatives over time are then
            good to about ``1e-4`` of their size (``1e-3`` early on, where
            the curves change fastest).
        of : str or tuple of str, optional
            ``"availability"`` (the default), ``"cost_rate"`` (the long-run
            ``expected_cost_rate``, or over a window the expected cost per
            unit time, ``expected_cost(window).mean / window``; not at
            times), or a tuple of both, for both from the same rebuilt
            diagrams.
        unit_costs : dict, optional
            The cost of a unit change of each lever to rank, by ``(key,
            lever)`` (the key as the result has it): only those levers are
            reported, each as its sensitivity per unit of that cost (the
            availability gained per unit spent, say).

        Returns
        -------
        dict
            ``{key: {lever: sensitivity}}``, the key a node, a
            common-cause group's tuple of members, or None for the repair
            crews; floats, but for an array ``x`` arrays in its shape.
            With a tuple ``of``, a dict of these by quantity.

        Raises
        ------
        ValueError
            For both ``x`` and ``window``, a cost at times, ``state``
            without either, a ``rel_step`` outside ``(0, 0.5)``, an
            unknown ``of``, or a ``unit_costs`` key that is no lever.
        NotImplementedError
            As the value differenced refuses (``mean_unavailability``,
            ``point_availability``, ``mission_availability``,
            ``expected_cost_rate`` or ``expected_cost``).

        Examples
        --------
        A component that fails at the rate 0.01 and is repaired at the
        rate 0.1 is up ``0.1 / 0.11`` of the time in the long run: its
        availability falls by ``0.1 / 0.11 ** 2`` per unit of the failure
        rate, about 8.264, and rises by ``0.01 / 0.11 ** 2``, about
        0.8264, per unit of the repair rate:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": {"reliability": E([0.01]), "repairability": E([0.1])}},
        ... )
        >>> sensitivity = rbd.parameter_sensitivity()["c"]
        >>> round(sensitivity["reliability.failure_rate"], 3)
        -8.264
        >>> round(sensitivity["repairability.failure_rate"], 4)
        0.8264
        """
        return _sensitivity.sensitivity(
            self,
            x=x,
            window=window,
            state=state,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            rel_step=rel_step,
            of=of,
            unit_costs=unit_costs,
        )

    def levers(self) -> List[Lever]:
        """The levers that ``parameter_sensitivity`` moves, in the order it
        reports them: each component's life and repair models'
        parameters, its maintenance's and tests' options, its standby
        group's and its imperfect repair's, each common-cause group's, and
        the repair crews.

        Each is a [`Lever`][repyability.Lever]: whose it is and its name, as
        ``parameter_sensitivity`` keys them, its value, the range of its
        values, whether it is discrete (one more standby unit or repair
        crew), and whether it moves a calendar its component shares with
        others. Nothing is worked out, so a report can name the levers and
        show their values without the sensitivities;
        [`with_levers`][repyability.RepairableRBD.with_levers] sets them.

        Returns
        -------
        list of Lever
            The levers, in ``parameter_sensitivity``'s order.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": {"reliability": E([0.01]), "repairability": E([0.1])}},
        ... )
        >>> for lever in rbd.levers():
        ...     print(lever.key, lever.name, lever.value, lever.bounds)
        c reliability.failure_rate 0.01 (0.0, inf)
        c repairability.failure_rate 0.1 (0.0, inf)
        """
        return _sensitivity.public_levers(self)

    def with_levers(self, values) -> "RepairableRBD":
        """A copy of this RBD with levers at new values, each moved
        as ``parameter_sensitivity`` moves it, so that a what-if agrees
        with the sensitivities: a test interval takes the full tests with it
        (every so many tests, as before), and a common-cause group's
        members' parameter moves for every member. ``levers()`` lists the
        levers.

        Parameters
        ----------
        values : dict
            Each lever's new value, by its [`Lever`][repyability.Lever]
            (from ``levers()``) or by its ``(key, name)``, as
            ``parameter_sensitivity`` reports it; a discrete lever's (the
            standby units, the repair crews) a whole number.

        Returns
        -------
        RepairableRBD
            The new RBD, built as this one was with the levers moved; this
            one is unchanged.

        Raises
        ------
        ValueError
            For a lever the RBD does not have (with the closest name), a
            discrete lever given a fraction, or a value the RBD refuses
            (one outside the lever's ``bounds``, say).
        TypeError
            If ``values`` is not a dict, a lever is neither a ``Lever`` nor
            a ``(key, name)``, or a value is not a number.

        Examples
        --------
        Repairs in 8 rather than 10, on average: the availability rises
        from ``100 / 110`` to ``100 / 108``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {"c": {"reliability": E([0.01]), "repairability": E([0.1])}},
        ... )
        >>> repair = ("c", "repairability.failure_rate")
        >>> faster = rbd.with_levers({repair: 1 / 8})
        >>> round(float(faster.mean_availability()), 4)
        0.9259
        >>> float(faster.repairability["c"].mean())
        8.0
        """
        return _sensitivity.with_levers(self, values)

    def _uncertainty(
        self, of: str, x, uncertainty, n_draws, seed, sampling, state
    ) -> UncertaintyResult:
        """The quantity ``of`` over draws of the uncertain models (see
        ``_repairable_uncertainty``), as an ``UncertaintyResult``."""
        from repyability.rbd import _repairable_uncertainty as drawn

        drawn.check(self, of, x, state)
        # The diagram's own value first: what it refuses, every draw would.
        nominal = drawn.value(self, of, x, state)
        inputs, groups = drawn.sources(self, uncertainty)
        models, group_models = drawn.draws(
            self, inputs, groups, n_draws, seed, sampling
        )
        samples = drawn.samples(
            self, of, x, state, inputs, models, group_models, n_draws
        )
        if x is None or np.ndim(x) == 0:
            return UncertaintyResult(
                samples=samples[:, 0],
                nominal=float(nominal[0]),
                n_draws=n_draws,
            )
        return UncertaintyResult(
            samples=samples, nominal=nominal, n_draws=n_draws
        )

    def mean_availability_uncertainty(
        self,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
    ) -> UncertaintyResult:
        """The long-run availability over plausible models of the components
        (parameter uncertainty).

        A component's models are estimated from data, so their parameters
        are uncertain: *epistemic* uncertainty, about what the models are,
        as opposed to the aleatory variability they describe. Each of the
        ``n_draws`` draws gives every uncertain model a plausible value, and
        the diagram, rebuilt with them, its long-run availability, worked
        out as ``mean_availability`` works it out (exactly, or numerically
        with maintenance): the spread of the draws, and its percentiles
        (``interval``), say how well the availability is known.

        A component has several models: its life (``"reliability"``), its
        repair (``"repairability"``), and the durations of its preventive
        maintenance and of its tests (``"preventive.duration"``,
        ``"inspection.duration"``), named as ``parameter_sensitivity``'s
        levers. ``uncertainty`` maps an input to its uncertainty: a node; a
        tuple of nodes of one population, sharing a model, which each draw
        gives them alike; or a common-cause group (the
        [`CCFGroup`][repyability.CCFGroup] itself), whose model's
        parameters, or alternative models, are drawn as for
        ``NonRepairableRBD.sf_uncertainty``. A node may come under several
        inputs, a role of it under one: a fleet's life fit, shared by two
        pumps whose repairs were recorded apart, is ``{("a", "b"):
        {"reliability": "fit"}, "a": {"repairability": "fit"}, "b":
        {"repairability": "fit"}}``, the life drawn once a draw. An
        input drawn without other nodes that hold the same model object,
        which then keep it as fitted, is warned about. A node's uncertainty
        is

        - ``"fit"``: every one of its models that is a surpyval fit with a
          parameter covariance (``covariance()``), drawn from its normal
          approximation (on the log scale for a positive parameter, the
          logit scale for one in (0, 1));
        - ``{role: uncertainty}``: each model named drawn as its
          uncertainty says (``"fit"``, ``{parameter: distribution}`` or a
          list of models), the others kept;
        - ``{parameter: distribution}`` or a list of models: its life's.

        Parameters
        ----------
        uncertainty : dict, optional
            ``{node, tuple of nodes or CCFGroup: uncertainty}`` (see above).
            By default, every model that is a fit with a parameter
            covariance, drawn once for all the nodes holding that fitted
            object in that role, and for one common-cause group's members
            together.
        n_draws : int, optional
            The number of draws, by default 1000: one evaluation each.
        seed : int, optional
            Seed for the draws.
        sampling : str, optional
            ``"random"`` (the default), or ``"sobol"``: the draws from the
            points of a scrambled Sobol sequence, which cover the
            parameters more evenly and shrink the error of the summaries for
            the same number of draws.

        Returns
        -------
        UncertaintyResult
            The availability of every draw (``samples``), the ``nominal``
            value with the components' own models, and summaries (``mean``,
            ``median``, ``std``, ``percentile``, ``interval``).

        Raises
        ------
        ValueError
            If ``uncertainty`` names an unknown node, a nested RBD or a
            junction, a node twice, nodes of a population with different
            models, a model a node does not have, or a common-cause group's
            members apart; if a model's uncertainty cannot be drawn; if
            ``n_draws`` or ``sampling`` is invalid; or as
            ``mean_availability`` does for a draw.

        Examples
        --------
        A pump whose repair time was fitted to 12 repairs, in series with a
        valve: the repair's uncertainty is the availability's.

        >>> import numpy as np
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> repairs = surv.LogNormal.fit(
        ...     np.exp(1.5 + 0.4 * np.linspace(-1.6, 1.6, 12))
        ... )
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "valve"), ("valve", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([0.01]),
        ...             "repairability": repairs,
        ...         },
        ...         "valve": {
        ...             "reliability": surv.Exponential.from_params([0.002]),
        ...             "repairability": surv.Exponential.from_params([0.5]),
        ...         },
        ...     },
        ... )
        >>> result = rbd.mean_availability_uncertainty(n_draws=2000, seed=0)
        >>> round(result.nominal, 4)
        0.9499
        >>> lower, upper = result.interval(0.9)
        >>> round(lower, 4), round(upper, 4)
        (0.9392, 0.9577)
        """
        return self._uncertainty(
            "mean_availability",
            None,
            uncertainty,
            n_draws,
            seed,
            sampling,
            None,
        )

    def point_availability_uncertainty(
        self,
        x,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
        state=None,
    ) -> UncertaintyResult:
        """The point availability at times ``x`` (from new, or from
        ``state``) over plausible models of the components, each draw
        worked out as ``point_availability`` works it out: the inputs and
        their uncertainty as for ``mean_availability_uncertainty``.

        Parameters
        ----------
        x : float or array-like
            Times.
        uncertainty : dict, optional
            As for ``mean_availability_uncertainty``.
        n_draws : int, optional
            The number of draws, by default 1000: one evaluation of the
            availability over time each.
        seed : int, optional
            Seed for the draws.
        sampling : str, optional
            ``"random"`` (the default) or ``"sobol"`` (see
            ``mean_availability_uncertainty``).
        state : dict or str, optional
            The components' current states, as for ``point_availability``.

        Returns
        -------
        UncertaintyResult
            Per draw, the availability at each time (``samples``: one value
            a draw for a number ``x``, else one row a draw), the
            ``nominal`` values and their summaries.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``mean_availability_uncertainty`` and
            ``point_availability``.
        """
        return self._uncertainty(
            "point_availability",
            x,
            uncertainty,
            n_draws,
            seed,
            sampling,
            state,
        )

    def mission_availability_uncertainty(
        self,
        t,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
        state=None,
    ) -> UncertaintyResult:
        """The mission availability over ``[0, t]`` (from new, or from
        ``state``) over plausible models of the components, each draw
        worked out as ``mission_availability`` works it out: the inputs and
        their uncertainty as for ``mean_availability_uncertainty``.

        Parameters
        ----------
        t : float or array-like
            The missions' lengths.
        uncertainty : dict, optional
            As for ``mean_availability_uncertainty``.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seed for the draws.
        sampling : str, optional
            ``"random"`` (the default) or ``"sobol"``.
        state : dict or str, optional
            The components' current states, as for
            ``mission_availability``.

        Returns
        -------
        UncertaintyResult
            Per draw, the mission availability of each mission, the
            ``nominal`` values and their summaries.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``mean_availability_uncertainty`` and
            ``mission_availability``.
        """
        return self._uncertainty(
            "mission_availability",
            t,
            uncertainty,
            n_draws,
            seed,
            sampling,
            state,
        )

    def expected_cost_rate_uncertainty(
        self,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
    ) -> UncertaintyResult:
        """The long-run cost rate over plausible models of the components,
        each draw worked out as ``expected_cost_rate`` works it
        out: the inputs and their uncertainty as for
        ``mean_availability_uncertainty``.

        Parameters
        ----------
        uncertainty : dict, optional
            As for ``mean_availability_uncertainty``.
        n_draws : int, optional
            The number of draws, by default 1000.
        seed : int, optional
            Seed for the draws.
        sampling : str, optional
            ``"random"`` (the default) or ``"sobol"``.

        Returns
        -------
        UncertaintyResult
            The cost rate of every draw, the ``nominal`` value and their
            summaries.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``mean_availability_uncertainty`` and
            ``expected_cost_rate``.
        """
        return self._uncertainty(
            "expected_cost_rate",
            None,
            uncertainty,
            n_draws,
            seed,
            sampling,
            None,
        )

    def uncertainty_importance(
        self,
        x=None,
        uncertainty: Optional[Dict[Hashable, Any]] = None,
        *,
        of: str = "mean_availability",
        method: str = "delta",
        n_draws: int = 1000,
        seed=None,
        sampling: str = "random",
        rel_step: Optional[float] = None,
        state=None,
    ) -> UncertaintyImportance:
        """Which input's parameter uncertainty makes the availability (or
        the cost rate) uncertain: each uncertain input's share of its
        variance (vega), the inputs as for
        ``mean_availability_uncertainty``.

        In the guide's Greeks it is *vega*: whose uncertainty widens the answer
        (see [Sensitivities: the Greeks](../guide/greeks.md)).

        ``of`` is the quantity: ``"mean_availability"`` (the default, the
        long run), ``"point_availability"`` at the times ``x``,
        ``"mission_availability"`` over ``[0, x]``, or
        ``"expected_cost_rate"`` (or ``"cost_rate"``, as
        ``parameter_sensitivity`` names it).

        ``method="delta"`` (the default) linearises the quantity in the
        inputs' parameters: ``Var(Q) ~ sum_k g_k^T Sigma_k g_k``, ``g_k``
        its derivatives in input ``k``'s parameters (``parameter_
        sensitivity``'s, of every one of its uncertain models; a
        population's summed over its nodes, which move together) and
        ``Sigma_k`` their covariance (a fit's ``covariance()``, or the
        variances of the distributions given). The inputs are independent,
        so each one's part is its own term, and the shares add up to 1. A
        list of models has no parameters to move: it needs
        ``method="sobol"``.

        ``method="sobol"`` draws the inputs as
        ``mean_availability_uncertainty`` does and estimates the first-order
        and total Sobol indices: ``n_draws`` draws of two independent sets,
        and one more set for each input with it taken from the second
        (Jansen's estimators), ``n_draws * (inputs + 2)`` evaluations in
        all, at the cost of sampling error, which ``sampling="sobol"``
        shrinks.

        Parameters
        ----------
        x : float or array-like, optional
            The times for ``"point_availability"``, the missions' lengths
            for ``"mission_availability"``; none for the long run.
        uncertainty : dict, optional
            As for ``mean_availability_uncertainty``.
        of : str, optional
            The quantity (see above).
        method : str, optional
            ``"delta"`` (the default) or ``"sobol"``.
        n_draws : int, optional
            With ``method="sobol"``, the draws of each set, by default
            1000.
        seed : int, optional
            With ``method="sobol"``, the seed of the draws.
        sampling : str, optional
            With ``method="sobol"``, ``"random"`` (the default) or
            ``"sobol"`` (see ``mean_availability_uncertainty``).
        rel_step : float, optional
            With ``method="delta"``, ``parameter_sensitivity``'s step.
        state : dict or str, optional
            For the values over time, the components' current states.

        Returns
        -------
        UncertaintyImportance
            The ``method``, the quantity's ``variance``, and each input's
            ``first_order`` and ``total`` shares, by the key it was given
            under: floats, or arrays for an array ``x``.

        Raises
        ------
        ValueError
            For an unknown ``of`` or ``method``, an ``x`` or ``state`` the
            quantity does not take, a list of models with the delta method,
            or as ``mean_availability_uncertainty`` does.

        Examples
        --------
        A pump whose repair was fitted to 12 repairs, in series with a valve
        failing half as often, whose repair was fitted to 8: the pump's,
        down twice as often, makes three quarters of the availability's
        uncertainty.

        >>> import numpy as np
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def fitted(median, n):
        ...     spread = 0.4 * np.linspace(-1.6, 1.6, n)
        ...     return surv.LogNormal.fit(median * np.exp(spread))
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "valve"), ("valve", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([0.01]),
        ...             "repairability": fitted(4.5, 12),
        ...         },
        ...         "valve": {
        ...             "reliability": surv.Exponential.from_params([0.005]),
        ...             "repairability": fitted(4.0, 8),
        ...         },
        ...     },
        ... )
        >>> parts = rbd.uncertainty_importance()
        >>> {n: round(s, 2) for n, s in parts.first_order.items()}
        {'pump': 0.74, 'valve': 0.26}
        """
        from repyability.rbd import _repairable_uncertainty as drawn

        # parameter_sensitivity's name for the cost rate is taken too (#232).
        of = drawn.QUANTITY_NAMES.get(of, of) if isinstance(of, str) else of
        drawn.check(self, of, x, state)
        one_of("method", method, ("delta", "sobol"))
        inputs, groups = drawn.sources(self, uncertainty)
        keys = [item.key for item in inputs] + [
            groups[i][0] for i in sorted(groups)
        ]
        if method == "delta":
            parts = drawn.delta(self, of, x, state, inputs, groups, rel_step)
            variance = np.sum(parts, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                shares = [
                    np.where(variance > 0.0, part / variance, np.nan)
                    for part in parts
                ]
            first = total = dict(zip(keys, shares))
        else:
            firsts, totals, variance = drawn.sobol(
                self, of, x, state, inputs, groups, n_draws, seed, sampling
            )
            first, total = dict(zip(keys, firsts)), dict(zip(keys, totals))
        scalar = x is None or np.ndim(x) == 0

        def shaped(values):
            values = np.asarray(values, dtype=float).reshape(-1)
            if scalar:
                return float(values[0])
            return values.reshape(np.shape(x))

        return UncertaintyImportance(
            method=method,
            variance=shaped(variance),
            first_order={key: shaped(v) for key, v in first.items()},
            total={key: shaped(v) for key, v in total.items()},
        )

    @_times_first()
    def differential_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        x=None,
        window=None,
        state=None,
        over: str = "components",
        change: str = "uniform",
        kind: str = "failure",
        improving: bool = False,
        groups: Optional[Mapping[Hashable, Collection[Hashable]]] = None,
        rel_step: Optional[float] = None,
    ) -> dict:
        """Each node's (or lever's) share of the change in the system's
        availability when they all change together: the differential
        importance measure (DIM, Borgonovo & Apostolakis, 2001), in
        the long run, at times ``x`` or over a window.

        In the guide's Greeks it is *DIM*, the shares of a change (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        ``DIM_i = dA/dtheta_i dtheta_i / sum_j dA/dtheta_j dtheta_j``, so
        the shares add up to 1, and a group's share is the sum of its
        members' (``groups``): what share of a possible gain lies in the
        repair times, say, or in the maintenance intervals.
        ``change="uniform"`` moves every ``theta`` by as much;
        ``"proportional"`` each by the same fraction of itself.

        Over the nodes (``over="components"``), ``theta`` is each node's
        unavailability (``kind="failure"``) or availability
        (``kind="success"``). A uniform change shares out the Birnbaum
        importance, either way; a proportional one the criticality
        importance of that ``kind`` (the failure-oriented one is the
        improvement potential over the system's unavailability). Each is
        as ``birnbaum_importance`` and ``criticality_importance`` work it
        out, with ``x``, ``window`` and ``state``. A node held working or
        failed takes no part (its share is 0).

        Over the levers (``over="parameters"``), the derivatives are
        ``parameter_sensitivity``'s, keyed ``(key, lever)``, and a
        proportional change moves each lever by the same fraction of its
        value. One more standby unit or repair crew is no derivative, and
        takes no part.

        Effects that oppose (a failure rate that lowers the
        availability, a repair rate that raises it) give shares of either
        sign, and some beyond 1; where they cancel (an exponential unit's
        failure and repair rates, moved in proportion), there is nothing
        to share out, and the shares are NaN. ``improving=True`` moves
        each lever instead the way that raises the availability (a
        failure rate down, a repair rate up), so that every share is of a
        gain, between 0 and 1: what share of the gain from improving
        every lever by the same fraction lies in each. A uniform change
        adds the same amount to levers of different units (an interval in
        hours, a rate per hour): over levers, the proportional change is
        usually the one to ask for.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Nodes held working, as for ``birnbaum_importance``.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed, likewise.
        x : float or array-like, optional
            Times from new (or from ``state``), as for
            ``birnbaum_importance``; by default the long run.
        window : float, optional
            Over ``[0, window)`` instead, as for ``birnbaum_importance``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states.
        over : str, optional
            ``"components"`` (the default) or ``"parameters"``.
        change : str, optional
            ``"uniform"`` (the default) or ``"proportional"``.
        kind : str, optional
            Over the nodes, with a proportional change, which of each
            node's probabilities moves in proportion: ``"failure"`` (the
            default), of being down, or ``"success"``, of being up.
        improving : bool, optional
            Move each lever the way that raises the availability, so that
            the shares are of a gain (each one's size, shared out),
            instead of every one up together (the default, the measure as
            defined). The nodes' shares are the same either way.
        groups : dict, optional
            ``{name: keys}``: each group's share, the sum of its keys'
            (nodes, or ``(key, lever)`` pairs), instead of each key's.
        rel_step : float, optional
            The levers' step, as for ``parameter_sensitivity``.

        Returns
        -------
        dict
            The shares, by node, ``(key, lever)`` or group name: floats
            but for an array ``x`` (NaN where the total is 0).

        Raises
        ------
        ValueError
            For an unknown ``over``, ``change`` or ``kind``, a group naming
            an unknown key, or as ``birnbaum_importance`` and
            ``parameter_sensitivity`` do.
        NotImplementedError
            As they do.

        Examples
        --------
        Two units in series, up 10/11 and 4/5 of the time: a uniform change
        of their unavailabilities moves the system's by ``0.8 dU_a +
        (10/11) dU_b`` (the Birnbaum importances):

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.1]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.25]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> shares = rbd.differential_importance()
        >>> {node: round(share, 4) for node, share in shares.items()}
        {'a': 0.4681, 'b': 0.5319}

        A proportional change weighs each by its unavailability, ``0.8 /
        11`` against ``(10/11) / 5``:

        >>> shares = rbd.differential_importance(change="proportional")
        >>> round(shares["b"], 4)
        0.7143
        """
        return _repairable_importance.differential_importance(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            x=x,
            window=window,
            state=state,
            over=over,
            change=change,
            kind=kind,
            improving=improving,
            groups=groups,
            rel_step=rel_step,
        )

    _MAX_PLAYERS = _repairable_importance._MAX_PLAYERS

    def availability_rate(
        self,
        x,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        state=None,
    ) -> RateBreakdown:
        """How fast the system's availability is changing at each time
        ``x`` from new (or from the components' ``state``), and which
        components are moving it.

        In the guide's Greeks it is *theta*: what is moving the system now (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        The components failing and recovering independently, the system's
        point availability is multilinear in theirs, so

        ``dA/dt = sum_i I_B^i(t) dA_i/dt``

        exactly, ``I_B^i(t)`` component ``i``'s Birnbaum importance at the
        components' point availabilities then: each term is what that
        component is doing to the system, pulling it down (negative) as it
        wears or fails, or up as it is restored, and the terms add up to
        the system's rate. A component's ``dA_i/dt`` is the rate of change
        of its point availability (see ``point_availability``), by
        differences on the grid that is worked out on, so the rates are
        numerical: to about ``1e-5`` of their size, less at 0 where a
        life's density is not smooth there. The down times kept off the
        grid (a repair or maintenance that starts at a known time) are
        differentiated on their own scale, however short, so the
        rates just after a scheduled event are as close.

        At a scheduled event a component's availability can jump: a block
        replacement or test that takes it off line, maintenance due at a
        fixed age from new. The system's jumps there are reported apart,
        each split among the components that jump together along the
        straight path between their values before and after, which, the
        system being multilinear, gives parts that add up to the jump. The
        rate at the time of a jump is the rate just after it.

        With limited repair crews or common-cause groups the components no
        longer fail and recover independently, and the parts come from
        the crews' or the groups' Markov chains. The system's rate
        is ``p(t) Q u``, ``p(t)`` the chain's states' probabilities,
        ``Q`` its generator and ``u`` the system up in each state; each
        transition is one component's failure or repair (a crew finishing
        a repair and taking the next job belongs to the repair that freed
        it), so ``Q`` splits into each component's transitions, and each
        part, ``p(t) Q_i u``, is exact. A nested RBD, with crews of its
        own, takes its part as an independent component does. In a
        common-cause group, a member's own cause and its repairs are its
        part, and the causes that strike more than one member are the
        group's, under the tuple of its members. A hidden group's members
        are found at their tests, where the system jumps: as tests change
        the members' joint states, which are not independent, a jump with
        a test in it is split by the Shapley value of each test and node
        that changes then, worked out exactly (for independent nodes alone,
        the same as the path).

        Parameters
        ----------
        x : float or array-like
            Times from new (or from ``state``).
        working_nodes : Collection[Hashable], optional
            Nodes held working: their part is 0.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed, likewise.
        state : dict or str, optional
            The components' current states, as for ``point_availability``.

        Returns
        -------
        RateBreakdown
            The system's ``rate`` (a float for a number ``x``, else an array
            in its shape) and each component's part in it (``node_rate``;
            a common-cause group's shared causes' under the tuple of its
            members), and the jumps after 0 and up to the last of ``x``
            (``jump_times``, ``jumps``, ``node_jumps``).

        Raises
        ------
        ValueError
            For a negative or non-finite time, or as
            ``point_availability`` does.
        NotImplementedError
            As ``point_availability`` does, or with more than 12 tests and
            nodes changing at one instant, a common-cause group's tests
            among them.

        Examples
        --------
        Two pumps in parallel, wearing out, and a valve after them that
        fails at a constant rate. Early on it is the valve that pulls the
        system down; by 8, it has settled, and the pumps' wear does:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def unit(life):
        ...     return {
        ...         "reliability": life,
        ...         "repairability": surv.Exponential.from_params([1.0]),
        ...     }
        >>> rbd = RepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {
        ...         "p1": unit(surv.Weibull.from_params([10, 3])),
        ...         "p2": unit(surv.Weibull.from_params([10, 3])),
        ...         "v": unit(surv.Exponential.from_params([0.02])),
        ...     },
        ... )
        >>> rate = rbd.availability_rate([2.0, 8.0])
        >>> rate.node_rate["v"].round(5).tolist()
        [-0.0026, -1e-05]
        >>> rate.node_rate["p1"].round(5).tolist()
        [-3e-05, -0.00156]
        """
        return _repairable_importance.availability_rate(
            self,
            x=x,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            state=state,
        )

    @_times_first("window")
    def barlow_proschan_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        window=None,
        state=None,
    ) -> dict:
        """Each component's share of the system's failures: the probability
        that a system failure is caused by the component's (Barlow &
        Proschan, 1975), in the long run or over a window from new
        (or from the components' ``state``).

        In the guide's Greeks it is *theta*, integrated: who caused the
        failures (see [Sensitivities: the Greeks](../guide/greeks.md)).

        A component's failure fails the system when the component is
        critical then, which it is with probability ``I_B^i(t)``, so the
        system's failures are ``sum_i integral I_B^i(t) dM_i(t)``,
        ``M_i(t)`` the component's expected failures by ``t``: each term is
        the system failures that component causes, and its share of the sum
        is its Barlow-Proschan importance. In the long run it is its share
        of ``system_failure_frequency``; over a window, of
        ``expected_failures`` (the exact counterpart of the simulated
        ``failure_criticality_index.per_system_failure``). Failures of
        several components at one instant (exact lifetimes, units dead on
        arrival) are split among them as ``availability_rate`` splits a
        jump.

        With common-cause groups, a cause that strikes several members at
        once is counted for the group, under the tuple of its members, and
        one that strikes a member alone for that member; with limited
        repair crews, the shares come from the crews' chain, its
        components' failures at their rates in each state, over a window
        from its transient probabilities.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Nodes held working (they cause nothing).
        broken_nodes : Collection[Hashable], optional
            Nodes held failed (likewise).
        window : float or array-like, optional
            Over ``[0, window)`` (one share for each window); by default
            the long run.
        state : dict or str, optional
            With ``window``, the components' current states, as for
            ``expected_failures``.

        Returns
        -------
        dict
            Each component's share (and each common-cause group's, under
            the tuple of its members), adding up to 1: floats in the long
            run or for one window, else arrays in the windows' shape. NaN
            where the system cannot fail.

        Raises
        ------
        ValueError
            For ``state`` without a window, a window that is not a positive
            length, or as ``system_failure_frequency`` and
            ``expected_failures`` do.
        NotImplementedError
            As ``system_failure_frequency`` and ``expected_failures`` do.

        Examples
        --------
        Two units in series, failing at rates 0.2 and 0.5 and repaired at
        rate 1: in the long run a share of the system's failures in
        proportion to each one's rate, while the system is up:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> shares = rbd.barlow_proschan_importance()
        >>> {node: round(share, 4) for node, share in shares.items()}
        {'a': 0.2857, 'b': 0.7143}
        """
        return _repairable_importance.barlow_proschan_importance(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            window=window,
            state=state,
        )

    @_times_first()
    def joint_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        *,
        x=None,
        window=None,
        state=None,
    ) -> dict:
        """The joint (second-order) importance of each pair of components:
        whether improving the two together is worth more than improving
        each, in the long run, at times ``x`` or over a window.

        In the guide's Greeks it is *gamma*: complements or substitutes (see
        [Sensitivities: the Greeks](../guide/greeks.md)).

        ``JRI(i, j) = d2A / dA_i dA_j = A(1_i, 1_j) - A(1_i, 0_j) -
        A(0_i, 1_j) + A(0_i, 0_j)``, the system's availability with
        component ``i`` up or down and ``j`` up or down: how much ``j``'s
        Birnbaum importance rises when ``i`` goes from down to up. Positive,
        the two are complements, as in series: improving either makes
        improving the other worth more. Negative, they are substitutes, as
        in parallel. It is each component's Birnbaum importance with the
        other held up, less that with it held down, as
        ``birnbaum_importance`` works them out, with ``x``, ``window`` and
        ``state``; with limited repair crews, each pair held in the crews'
        chain.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Components held up: their pairs are 0.
        broken_nodes : Collection[Hashable], optional
            Components held down, likewise.
        x : float or array-like, optional
            Times from new (or from ``state``), as for
            ``birnbaum_importance``; by default the long run.
        window : float, optional
            Over ``[0, window)`` instead, as for ``birnbaum_importance``.
        state : dict or str, optional
            With ``x`` or ``window``, the components' current states.

        Returns
        -------
        dict
            ``{(i, j): JRI}`` for each pair once, its components' names
            in order as text, and found either way round (the measure is
            symmetric): floats but for an array ``x``.

        Raises
        ------
        ValueError
            As ``birnbaum_importance`` does.
        NotImplementedError
            With common-cause groups, whose members cannot be held, or as
            ``birnbaum_importance`` does.

        Examples
        --------
        Two units in parallel, then a third, each up 10/11 of the time: the
        pair are substitutes, and each is a complement of the third.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> unit = {"reliability": E([0.1]), "repairability": E([1.0])}
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        ...     {"a": unit, "b": unit, "c": unit},
        ... )
        >>> {pair: round(v, 4) for pair, v in rbd.joint_importance().items()}
        {('a', 'b'): -0.9091, ('a', 'c'): 0.0909, ('b', 'c'): 0.0909}
        """
        return _repairable_importance.joint_importance(
            self,
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            x=x,
            window=window,
            state=state,
        )
