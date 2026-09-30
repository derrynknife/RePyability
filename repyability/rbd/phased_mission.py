"""Phased missions (#100): a mission that runs through phases (take-off,
cruise, landing), each with its own duration and its own diagram over the
same non-repairable components.

A component that fails in one phase stays failed in the later ones, so the
phases are not independent, and the mission's reliability is not the
product of theirs. The mission succeeds if each phase's diagram works
throughout the phase; as the components only ever fail, that is its
working at the phase's end, ``T_j``.

The exact values use the classic transformation of Esary and Ziehms (1975):
a component's life through the mission is a chain of independent segments,
one per phase, the ``k``-th survived with probability ``R(T_k) /
R(T_{k-1})``, and the component works at ``T_j`` if it survives its first
``j`` segments. Phase ``j`` then works if some minimal path set of its
diagram has every segment up to ``j`` of each of its components survived:
the mission succeeds when every phase has such a set, a condition on
independent segments that a Shannon decomposition (as in ``shannon.py``,
but over the phases' families of sets at once) works out exactly. The
simulation draws each component's life once per mission instead.
"""

from typing import (
    Any,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd._model_utils import is_fixed_probability
from repyability.rbd._sampling import RowSampler, row_sampler
from repyability.rbd.results import ConfidenceInterval
from repyability.rbd.shannon import _shannon_value_and_gradient
from repyability.utils.wrappers import numpy_seed

#: The most distinct sub-problems the exact decomposition solves before it
#: refuses, pointing to the simulation.
MAX_STATES = 200_000

# Value slots 0 and 1 of a plan hold the mission failing and succeeding.
_FAIL, _SUCCEED = 0, 1


class _Phase(NamedTuple):
    name: Hashable
    duration: float
    rbd: Any
    #: When the phase ends, from the mission's start.
    end: float
    #: The components its diagram uses.
    components: Tuple[Hashable, ...]


def _mission_plan(families: Sequence[Sequence[frozenset]]) -> tuple:
    """The Shannon decomposition of the probability that each of
    ``families`` has a set whose elements are all active (the elements
    independent), as a plan ``_shannon_value_and_gradient`` replays: step
    ``i`` fills value slot ``i + 2`` from its pivot's two branches.

    A state is the families still to be satisfied, each with the elements
    already active taken out of its sets: a family with an empty set is
    satisfied and drops out, and a state with a family that has no set left
    fails. Each distinct state is solved once.

    Raises
    ------
    NotImplementedError
        If more than ``MAX_STATES`` states are needed.
    """
    slots: Dict[frozenset, int] = {}
    steps: List[tuple] = []

    def settle(families) -> Optional[frozenset]:
        """The state of ``families``: None if one cannot be satisfied."""
        state = []
        for family in families:
            if frozenset() in family:
                continue
            if not family:
                return None
            state.append(family)
        return frozenset(state)

    def known(state: Optional[frozenset]) -> Optional[int]:
        if state is None:
            return _FAIL
        if not state:
            return _SUCCEED
        return slots.get(state)

    def split(state: frozenset) -> list:
        # Pivot on the element in the most sets, the lowest on a tie (the
        # elements are numbered, so the plan is the same in every run).
        counts: Dict[int, int] = {}
        for family in state:
            for elements in family:
                for element in elements:
                    counts[element] = counts.get(element, 0) + 1
        pivot = max(counts, key=lambda e: (counts[e], -e))
        active = settle(
            frozenset(s - {pivot} for s in family) for family in state
        )
        inactive = settle(
            frozenset(s for s in family if pivot not in s) for family in state
        )
        return [state, pivot, [active, inactive], []]

    root_state = settle(frozenset(family) for family in families)
    root = known(root_state)
    stack = []
    if root is None:
        assert root_state is not None
        stack.append(split(root_state))
    while stack:
        state, pivot, branches, solved = stack[-1]
        if len(solved) < 2:
            branch = branches[len(solved)]
            slot = known(branch)
            if slot is None:
                if len(slots) + len(stack) > MAX_STATES:
                    raise NotImplementedError(
                        "The mission is too large to solve exactly (more "
                        f"than {MAX_STATES:,} states of its decomposition). "
                        "Simulate it: method='simulate'."
                    )
                stack.append(split(branch))
            else:
                solved.append(slot)
            continue
        steps.append((pivot, solved[0], solved[1]))
        slots[state] = len(steps) + 1
        stack.pop()
        if stack:
            stack[-1][3].append(slots[state])
        else:
            root = slots[state]
    assert root is not None
    return steps, root


def _lifetime_sampler(model) -> Optional[RowSampler]:
    """A component's lives as a ``RowSampler``, or None if they cannot be
    drawn in a block. A fixed probability fails at the start or never."""
    if is_fixed_probability(model):
        failure = float(np.ravel(model.ff(1.0))[0])
        return RowSampler(
            1, lambda u: np.where(u[:, 0] < failure, 0.0, np.inf)
        )
    return row_sampler(model)


class PhasedMission:
    """A mission through phases, each with its own duration and diagram over
    the same non-repairable components.

    A mission (take-off, cruise, landing) runs through its phases in order.
    Each phase is a ``NonRepairableRBD`` over some of the mission's
    components, which it must keep working for the phase's duration. A
    component is the same in every phase that uses it (the same node name,
    with the same model), and its life runs through the whole mission: once
    failed, it stays failed in the later phases, even those whose diagrams
    tolerate it. So the phases are not independent, and the mission's
    reliability is not the product of theirs. The mission succeeds if every
    phase's diagram works to the phase's end.

    ``reliability`` and ``phase_failure_probabilities`` are exact by
    default: each component's life through the mission is a chain of
    independent segments, one per phase (Esary and Ziehms, 1975), and the
    phases' minimal path sets over those segments are decomposed together
    (see ``_mission_plan``), up to ``MAX_STATES`` sub-problems.
    ``method="simulate"`` draws each component's life once per mission
    instead, and ``reliability_interval`` gives the simulated reliability
    with a confidence interval, to a tolerance if asked.

    Parameters
    ----------
    phases : Sequence[tuple[Hashable, float, NonRepairableRBD]]
        The phases in order, each a ``(name, duration, rbd)``: a distinct
        name, a duration (finite, and not negative: a phase of no duration
        needs its diagram working at that instant), and the phase's
        diagram, whose node names are the mission's components.

    Attributes
    ----------
    phases : list
        The phases, each with its ``name``, ``duration``, ``rbd``, ``end``
        (from the mission's start) and the ``components`` it uses.
    components : dict
        Each component's model, by node name, in order of first use.
    duration : float
        The mission's length: the sum of the phases' durations.

    Raises
    ------
    ValueError
        If ``phases`` is empty, a phase is not a ``(name, duration, rbd)``
        with a distinct name, a duration that is finite and not negative
        and a ``NonRepairableRBD``, a phase has common-cause groups, or a
        node name has different models in different phases.

    Examples
    --------
    Two engines, either of which is enough to cruise, but both needed to
    take off, for 0.1 hours then 5:

    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD, PhasedMission
    >>> engine = surv.Exponential.from_params([0.01])
    >>> both = NonRepairableRBD(
    ...     [("s", "e1"), ("e1", "e2"), ("e2", "t")],
    ...     {"e1": engine, "e2": engine},
    ... )
    >>> either = NonRepairableRBD(
    ...     [("s", "e1"), ("s", "e2"), ("e1", "t"), ("e2", "t")],
    ...     {"e1": engine, "e2": engine},
    ... )
    >>> mission = PhasedMission(
    ...     [("take-off", 0.1, both), ("cruise", 5.0, either)]
    ... )
    >>> round(mission.reliability(), 6)
    0.995628
    """

    def __init__(self, phases: Sequence[Tuple[Hashable, float, Any]]):
        from repyability.rbd.non_repairable_rbd import NonRepairableRBD

        phases = list(phases)
        if not phases:
            raise ValueError("A mission needs at least one phase.")
        self.components: Dict[Hashable, Any] = {}
        self.phases: List[_Phase] = []
        names: set = set()
        end = 0.0
        for phase in phases:
            if not (isinstance(phase, (tuple, list)) and len(phase) == 3):
                raise ValueError(
                    "Each phase must be a (name, duration, rbd), got "
                    f"{phase!r}."
                )
            name, duration, rbd = phase
            if name in names:
                raise ValueError(f"Two phases are named {name!r}.")
            names.add(name)
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float, np.number))
                or not 0.0 <= float(duration) < np.inf
            ):
                raise ValueError(
                    f"Phase {name!r}: the duration must be a finite number, "
                    f"not negative, got {duration!r}."
                )
            if not isinstance(rbd, NonRepairableRBD):
                raise ValueError(
                    f"Phase {name!r}: the diagram must be a NonRepairableRBD, "
                    f"got {type(rbd).__name__}."
                )
            if rbd.ccf_groups:
                raise ValueError(
                    f"Phase {name!r} has common-cause groups, which a phased "
                    "mission does not model."
                )
            ends = {rbd.input_node, rbd.output_node}
            used = tuple(n for n in rbd._components() if n not in ends)
            for node in used:
                model = rbd.reliabilities[node]
                known = self.components.setdefault(node, model)
                if not NonRepairableRBD._same_model(known, model):
                    raise ValueError(
                        f"Node {node!r} has a different model in phase "
                        f"{name!r} than before: a component keeps one "
                        "lifetime model through the mission."
                    )
            end += float(duration)
            self.phases.append(_Phase(name, float(duration), rbd, end, used))
        self.duration = end
        # The decomposition of the first so many phases, once worked out.
        self._plans: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Exact
    # ------------------------------------------------------------------

    def _segments(self) -> Tuple[Dict[int, float], Dict[int, float]]:
        """Each component's segments' survival probabilities, and their
        complements, by element number (``component * phases + k``): the
        ``k``-th survived with ``R(T_k) / R(T_{k-1})``, the first from the
        mission's start (so a unit dead on arrival fails in it)."""
        ends = np.array([phase.end for phase in self.phases])
        count = len(ends)
        p: Dict[int, float] = {}
        q: Dict[int, float] = {}
        for index, model in enumerate(self.components.values()):
            with np.errstate(all="ignore"):
                sf = np.clip(np.ravel(model.sf(ends)).astype(float), 0, 1)
                ff = np.clip(np.ravel(model.ff(ends)).astype(float), 0, 1)
            before_sf, before_ff = 1.0, 0.0
            for k in range(count):
                element = index * count + k
                if before_sf > 0.0:
                    p[element] = min(sf[k] / before_sf, 1.0)
                    q[element] = min(
                        max(ff[k] - before_ff, 0.0) / before_sf, 1.0
                    )
                else:
                    p[element], q[element] = 0.0, 1.0
                before_sf, before_ff = sf[k], ff[k]
        return p, q

    def _families(self) -> List[List[frozenset]]:
        """Each phase's minimal path sets over the segments: a path of
        phase ``j`` needs each of its components' first ``j`` segments."""
        index = {node: i for i, node in enumerate(self.components)}
        count = len(self.phases)
        families = []
        for j, phase in enumerate(self.phases):
            try:
                paths = phase.rbd.get_min_path_sets(include_in_out_nodes=False)
            except ValueError:
                paths = set()  # no set of working nodes makes it work
            families.append(
                [
                    frozenset(
                        index[node] * count + k
                        for node in path
                        for k in range(j + 1)
                    )
                    for path in paths
                ]
            )
        return families

    def _exact(self, phases: int) -> Tuple[float, float]:
        """The probability that the mission gets through its first
        ``phases`` phases, and that it fails in them, each to its own
        relative precision."""
        plan = self._plans.get(phases)
        if plan is None:
            plan = _mission_plan(self._families()[:phases])
            self._plans[phases] = plan
        p, q = self._segments()
        success, _ = _shannon_value_and_gradient(plan, p, q, (0.0, 1.0))
        failure, _ = _shannon_value_and_gradient(plan, p, q, (1.0, 0.0))
        return float(success), float(failure)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _failure_phases(self, size: int, antithetic: bool) -> np.ndarray:
        """The phase in which each of ``size`` simulated missions fails
        (its number, ``len(phases)`` for none), each component's life drawn
        once per mission from numpy's global RNG as it stands."""
        samplers = {
            node: _lifetime_sampler(model)
            for node, model in self.components.items()
        }
        if antithetic and any(s is None for s in samplers.values()):
            raise NotImplementedError(
                "Antithetic sampling needs every component's lives to be "
                "drawn from uniforms (surpyval parametric distributions and "
                "the composite nodes built from them)."
            )
        width = sum(s.width for s in samplers.values() if s is not None)
        rows = size // 2 if antithetic else size
        u = np.random.random_sample((rows, width))
        lives: Dict[Hashable, np.ndarray] = {}
        start = 0
        for node, sampler in samplers.items():
            if sampler is None:
                model = self.components[node]
                lives[node] = np.asarray(model.random(size), dtype=float)
                continue
            block = u[:, start : start + sampler.width]
            start += sampler.width
            if antithetic:
                both = np.empty(size)
                both[0::2] = sampler.draw(block)
                both[1::2] = sampler.draw(1.0 - block)
                lives[node] = both
            else:
                lives[node] = np.asarray(sampler.draw(block), dtype=float)
        failed = np.full(size, len(self.phases))
        for j in range(len(self.phases) - 1, -1, -1):
            phase = self.phases[j]
            rbd = phase.rbd
            nodes = {
                n: lives.get(n, np.full(size, np.inf))
                for n in rbd._components()
            }
            lifetime = np.asarray(
                rbd._decomposition().lifetime(nodes, size), dtype=float
            )
            failed[~(lifetime > phase.end)] = j
        return failed

    def _simulated(
        self,
        mc_samples: int,
        seed,
        antithetic: bool = False,
        tolerance=None,
        confidence: float = 0.95,
        max_samples=None,
    ) -> np.ndarray:
        """The failure phases of ``mc_samples`` simulated missions, then
        more while a run to ``tolerance`` needs them."""
        montecarlo.check_count(mc_samples, antithetic, "mc_samples")
        montecarlo.check_confidence(confidence)
        limit = montecarlo.sample_limit(
            mc_samples,
            tolerance,
            max_samples,
            antithetic,
            ("mc_samples", "max_samples"),
        )
        parts: List[np.ndarray] = []
        with numpy_seed(seed):
            batch = mc_samples
            while batch:
                parts.append(self._failure_phases(batch, antithetic))
                batch = 0
                if limit is not None:
                    succeeded = np.concatenate(parts) == len(self.phases)
                    batch = montecarlo.more_samples(
                        succeeded.astype(float),
                        mc_samples,
                        tolerance,
                        confidence,
                        limit,
                        antithetic,
                        "mission reliability",
                        "max_samples",
                    )
        return np.concatenate(parts)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reliability(
        self,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> float:
        """The probability that the mission succeeds: that every phase's
        diagram works to the phase's end.

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default): the Esary-Ziehms decomposition (see
            the class docstring). ``"simulate"``: the fraction of
            ``mc_samples`` simulated missions that succeed.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000
            (``method="simulate"`` only).
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float
            The mission's reliability.

        Raises
        ------
        ValueError
            If ``method`` is unknown, or ``mc_samples`` is given with
            ``method="exact"`` or is not a positive integer.
        NotImplementedError
            If the exact decomposition needs more than ``MAX_STATES``
            sub-problems: simulate instead.

        Examples
        --------
        Two phases with the same diagram are one phase as long as both:

        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Weibull.from_params([100, 2])
        >>> pair = NonRepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> mission = PhasedMission(
        ...     [("first", 30.0, pair), ("second", 20.0, pair)]
        ... )
        >>> round(mission.reliability(), 10) == round(float(pair.sf(50.0)), 10)
        True
        """
        return self._outcome(method, mc_samples, seed)[0]

    def unreliability(
        self,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> float:
        """The probability that the mission fails, ``1 - reliability()``,
        computed to its own relative precision when it is small (see
        ``reliability`` for the parameters).

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000.
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        float
            The mission's unreliability.
        """
        return self._outcome(method, mc_samples, seed)[1]

    def phase_failure_probabilities(
        self,
        method: str = "exact",
        *,
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> Dict[Hashable, float]:
        """The probability that the mission fails in each phase: that the
        earlier phases get through and this one does not. They add up to
        the mission's unreliability.

        Parameters
        ----------
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``: the fraction of
            ``mc_samples`` simulated missions that fail in each phase.
        mc_samples : int, optional
            The number of simulated missions, by default 10_000.
        seed : int, optional
            Seeds the simulation, by default None.

        Returns
        -------
        dict[Hashable, float]
            Per phase, by name, the probability of failing in it.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> failing = mission.phase_failure_probabilities()
        >>> {name: round(chance, 5) for name, chance in failing.items()}
        {'out': 0.09516, 'back': 0.08611}
        """
        method = self._method(method, mc_samples)
        names = [phase.name for phase in self.phases]
        if method == "simulate":
            failed = self._simulated(
                10_000 if mc_samples is None else mc_samples, seed
            )
            counts = np.bincount(failed, minlength=len(names) + 1)
            return {
                name: float(counts[j] / len(failed))
                for j, name in enumerate(names)
            }
        out: Dict[Hashable, float] = {}
        before = 0.0
        for j, name in enumerate(names):
            _, failure = self._exact(j + 1)
            out[name] = max(failure - before, 0.0)
            before = failure
        return out

    def reliability_interval(
        self,
        mc_samples: int = 10_000,
        seed=None,
        *,
        confidence: float = 0.95,
        tolerance: Optional[float] = None,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
    ) -> ConfidenceInterval:
        """The mission's reliability by simulation, with a confidence
        interval: each component's life drawn once per mission.

        Parameters
        ----------
        mc_samples : int, optional
            The number of simulated missions, by default 10_000; with a
            ``tolerance``, the size of each batch.
        seed : int, optional
            Seeds the simulation, by default None.
        confidence : float, optional
            The interval's confidence level, by default 0.95.
        tolerance : float, optional
            Simulate until the interval is at most ``tolerance`` either side
            of the estimate, in batches of ``mc_samples``, or until
            ``max_samples`` missions (then with a RuntimeWarning). By
            default None: ``mc_samples`` missions.
        max_samples : int, optional
            The most missions a run to ``tolerance`` simulates, by default
            ``100 * mc_samples``.
        antithetic : bool, optional
            Draw the missions in antithetic pairs, the second of each from
            ``1 - u`` for the first's uniforms ``u``, by default False
            (``mc_samples`` must then be even).

        Returns
        -------
        ConfidenceInterval
            The estimate, its standard error and a normal interval, clipped
            to ``[0, 1]``.

        Raises
        ------
        ValueError
            If ``mc_samples``, ``confidence``, ``tolerance`` or
            ``max_samples`` is invalid.
        NotImplementedError
            If ``antithetic`` is asked for and a component's lives cannot
            be drawn from uniforms.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import NonRepairableRBD, PhasedMission
        >>> unit = surv.Exponential.from_params([0.01])
        >>> one = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": unit})
        >>> mission = PhasedMission([("out", 10.0, one), ("back", 10.0, one)])
        >>> interval = mission.reliability_interval(20_000, seed=1)
        >>> bool(interval.lower < mission.reliability() < interval.upper)
        True
        """
        failed = self._simulated(
            mc_samples, seed, antithetic, tolerance, confidence, max_samples
        )
        succeeded = (failed == len(self.phases)).astype(float)
        estimate = float(succeeded.mean())
        error = montecarlo.standard_error(succeeded, antithetic)
        half = montecarlo.z_value(confidence) * error
        return ConfidenceInterval(
            estimate=estimate,
            lower=max(estimate - half, 0.0),
            upper=min(estimate + half, 1.0),
            confidence=confidence,
            standard_error=error,
            n_samples=len(succeeded),
        )

    def _method(self, method: str, mc_samples) -> str:
        if method not in ("exact", "simulate"):
            raise ValueError(
                f"method must be 'exact' or 'simulate', got {method!r}."
            )
        if method == "exact" and mc_samples is not None:
            raise ValueError("mc_samples applies only to method='simulate'.")
        return method

    def _outcome(self, method: str, mc_samples, seed) -> Tuple[float, float]:
        """The mission's reliability and unreliability."""
        method = self._method(method, mc_samples)
        if method == "simulate":
            failed = self._simulated(
                10_000 if mc_samples is None else mc_samples, seed
            )
            success = float(np.mean(failed == len(self.phases)))
            return success, 1.0 - success
        return self._exact(len(self.phases))

    def __repr__(self) -> str:
        phases = ", ".join(
            f"{phase.name!r} ({phase.duration:g})" for phase in self.phases
        )
        return f"PhasedMission({phases})"
