"""Reliability and mean time to the first system failure of a
``RepairableRBD``.

The probability that the system runs to ``t`` without a failure, and the
mean time until the first one, differ from a non-repairable diagram's: a
component can fail and be repaired while its partner in a redundant pair
keeps the system up. Two exact routes cover what they can:

- **Series structures** (every component a cut set of one). No repair
  can happen before the first system failure, since the first failure of
  any component is one of the system. So ``R(t) = prod_i R_i(t)``, each
  from the component's state at 0, whatever its life distribution.
- **Exponential lives and repairs** (or instant repairs), with any
  number of repair crews. The components' joint states form a Markov
  chain (the crews' chain of ``_crew_chain``, which has one crew per
  component when the diagram has no limit). Making the states where the
  system is down absorbing, ``R(t)`` is the chain's probability of not
  having been absorbed by ``t``, followed by uniformization
  (``p(t) 1 = E[s_N]``, ``s_k = p(0) P^k 1`` and ``N`` Poisson with mean
  ``L t``, as in ``_chain_transient``), and the mean time to absorption is
  one linear solve over the states where the system is up.

Everything else (a life or repair that is not exponential in a redundant
structure, scheduled maintenance, tests, standby groups, common-cause
groups, nested diagrams) is simulated on request:
``method="simulate"`` reads each history's first failure off
``simulate_timelines``.
"""

import math
import warnings
from typing import Any, Dict, NoReturn, Optional, Tuple

import numpy as np

from repyability.rbd import _crew_chain as crew_chain
from repyability.rbd import _curves, _overrides
from repyability.rbd._common import _constant_rate, _repair_rate
from repyability.rbd._mean_lifetime import mean_lifetime, model_knots
from repyability.rbd.modular import NODE, SERIES, GraphStructure
from repyability.rbd.results import ConfidenceInterval
from repyability.utils.checks import (
    nonnegative_times,
    one_of,
    simulation_options,
)

#: The most steps uniformization takes, and the most work (steps times the
#: chain's transitions) it does before the chain is given up on.
MAX_STEPS = 2_000_000
MAX_WORK = 3e9
#: A step's survival probability below this has ended: the rest of the
#: chain's survival is within it.
_ENDED = 1e-18
_SPREAD, _MARGIN = 12.0, 20.0

_SIMULATE = (
    " Estimate it with method='simulate' (with t_simulation and mc_samples)."
)


def _refuse(why: str) -> NoReturn:
    raise NotImplementedError(why + _SIMULATE)


def _components_are_plain(rbd) -> None:
    """Raise if the first failure's exact values do not cover the
    diagram's kinds of component or maintenance."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    if rbd.ccf_groups:
        _refuse(
            "The exact values do not cover common-cause groups, whose "
            "members fail together"
        )
    if rbd._maintenance:
        _refuse(
            "The exact values do not cover maintenance groups, which "
            "stop their members together"
        )
    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            _refuse(
                "The exact values do not cover a nested diagram "
                f"(component {node!r}), which has repairs of its own"
            )
        for table, what in (
            (rbd._standby, "a standby group"),
            (rbd._preventive, "scheduled maintenance"),
            (rbd._inspection, "inspections"),
            (rbd._imperfect, "imperfect repair"),
        ):
            if node in table:
                _refuse(
                    f"The exact values do not cover {what}, which "
                    f"component {node!r} has"
                )


def _in_series(structure, term: int) -> bool:
    """Whether the term is a node or a series of terms like it."""
    kind = structure.terms[term][0]
    if kind == NODE:
        return True
    return kind == SERIES and all(
        _in_series(structure, child) for child in structure.terms[term][1]
    )


def _is_series(rbd) -> bool:
    """Whether every component is a cut set of one: the system works only
    while all its relevant components do."""
    structure = rbd._decomposition()
    if isinstance(structure, GraphStructure):
        return False
    if structure.always_works:
        return True
    return structure.root is not None and _in_series(structure, structure.root)


def _markov_rates(rbd) -> Tuple[Dict[Any, Tuple[float, float]], int]:
    """Each component's failure and repair rates (``inf`` for an instant
    repair), and the number of crews, for the Markov chain; raise if the
    components are not exponential or the chain is too large."""
    rates: Dict[Any, Tuple[float, float]] = {}
    for node, component in rbd.components.items():
        life = _constant_rate(component.reliability)
        if life is None:
            _refuse(
                "A structure that is not a series needs exponential lives "
                f"(a constant failure rate), and the life of component "
                f"{node!r} is not one"
            )
        repair = _repair_rate(component.time_to_replace)
        if repair is None:
            _refuse(
                "A structure that is not a series needs exponential repair "
                "times (or instant repair), and the repair times of "
                f"component {node!r} are not"
            )
        rates[node] = (life, repair)
    crews = rbd.repair_crews or max(len(rates), 1)
    count = crew_chain.state_count(
        [rbd._priority.get(node, 0.0) for node in rates],
        [math.isinf(repair) for _, repair in rates.values()],
        crews,
    )
    if count > crew_chain.MAX_STATES:
        _refuse(
            f"Its Markov chain has {count:,} states, more than the "
            f"{crew_chain.MAX_STATES:,} it is solved for"
        )
    return rates, crews


def _require_exact(rbd) -> bool:
    """Raise if the first failure's exact values do not cover the diagram;
    otherwise whether it is worked out as a series (True) or by the
    Markov chain (False)."""
    structure = rbd._decomposition()
    if isinstance(structure, GraphStructure):
        raise NotImplementedError(structure.reason)
    _components_are_plain(rbd)
    if _is_series(rbd):
        return True
    _markov_rates(rbd)
    return False


def exact_route(rbd) -> str:
    """How the exact values are worked out, for ``analysis_routes``."""
    return (
        "A series structure: the product of each component's survival "
        "from its state at 0, as no repair can come before the first "
        "failure."
        if _require_exact(rbd)
        else "The Markov chain of the components' states (with the repair "
        "crews' queue), the states the system is down in made absorbing: "
        "the survival probability by uniformization (to about 1e-13) and "
        "the mean time to absorption from one linear solve."
    )


# -- a series structure ------------------------------------------------------


def _series_parts(rbd, states: dict) -> list:
    """For each component the system needs: ``(sf, age)``, its survival
    function and age at 0; a component down at 0 makes the system's
    reliability 0 (an empty list is never returned for it: see
    ``_down_at_start``)."""
    structure = rbd._decomposition()
    needed = [node for node in rbd.components if node in structure.nodes]
    parts = []
    for node in needed:
        state = states.get(node)
        age = 0.0 if state is None else float(state.age)
        life = rbd.components[node].reliability
        if state is not None and state.alive and state.own_life:
            # A life of its own (see _overrides), from now or at its age.
            life, at = _overrides.life_now(node, rbd.components[node], state)
            age = 0.0 if at is None else at
        parts.append((life, age))
    return parts


def _down_at_start(rbd, states: dict) -> bool:
    structure = rbd._decomposition()
    return any(
        not states[node].alive
        for node in states
        if node in structure.nodes and not structure.always_works
    )


def _series_sf(parts: list, x: np.ndarray) -> np.ndarray:
    out = np.ones(x.shape)
    for model, age in parts:
        if age > 0.0:
            out = out * (
                np.asarray(model.sf(age + x), dtype=float)
                / float(np.asarray(model.sf(np.array([age])))[0])
            )
        else:
            out = out * np.asarray(model.sf(x), dtype=float)
    return out


def _series_mean(parts: list) -> float:
    knots = [np.array([1.0])]
    for model, age in parts:
        shifted = np.asarray(model_knots(model), dtype=float) - age
        knots.append(shifted[shifted > 0.0])
    return mean_lifetime(
        lambda t: _series_sf(parts, np.asarray(t, dtype=float)),
        np.concatenate(knots),
    )


# -- the Markov chain --------------------------------------------------------


class _Absorbing:
    """The crews' chain with the system-down states made absorbing, from
    the start distribution ``start`` over its states."""

    def __init__(self, rbd, states: dict):
        from scipy import sparse

        rates, crews = _markov_rates(rbd)
        nodes = list(rates)
        chain = crew_chain.solve(
            nodes,
            [rates[node][0] for node in nodes],
            [rates[node][1] for node in nodes],
            [rbd._priority.get(node, 0.0) for node in nodes],
            crews,
        )
        structure = rbd._decomposition()
        size = len(chain.probabilities)
        node_up = {
            node: chain.up[:, k].astype(float)
            for k, node in enumerate(chain.nodes)
            if node in structure.nodes
        }
        works, _ = structure.probabilities(node_up, shape=(size,))
        self.up = np.asarray(works, dtype=float) > 0.5
        self.start = _start(rbd, chain, states)[self.up]
        generator = sparse.csr_matrix(chain.generator)
        self.generator = generator[self.up][:, self.up].tocsr()
        # The system down at the start has failed already.
        self.at_start = float(self.start.sum())

    def sf(self, x: np.ndarray) -> np.ndarray:
        """``p(0) exp(T t) 1`` at each time, by uniformization."""
        from scipy import sparse
        from scipy.stats import poisson

        matrix = self.generator
        size = matrix.shape[0]
        if size == 0 or self.at_start == 0.0:
            return np.zeros(x.shape)
        top = float(np.max(-matrix.diagonal()))
        if top == 0.0:
            return np.full(x.shape, self.at_start)
        rate = 1.02 * top
        jump = sparse.identity(size, format="csr") + matrix / rate
        mean = rate * float(x.max())
        needed = int(mean + _SPREAD * math.sqrt(mean) + _MARGIN) + 1
        steps = min(needed, MAX_STEPS)
        if needed * max(matrix.nnz, 1) > MAX_WORK:
            _refuse(
                "Its Markov chain's rates are too far apart to follow "
                "over this long"
            )
        survivals = []
        vector = self.start.copy()
        ones = np.ones(size)
        for _ in range(steps):
            value = float(vector @ ones)
            survivals.append(value)
            if value < _ENDED:
                break
            vector = jump.T @ vector
        s = np.array(survivals)
        if len(s) == steps and s[-1] >= _ENDED and steps < needed:
            _refuse(
                "Its Markov chain's rates are too far apart to follow "
                "over this long"
            )
        k = np.arange(len(s))
        out = np.empty(x.shape)
        for i, t in enumerate(x.ravel()):
            out.ravel()[i] = float(poisson.pmf(k, rate * t) @ s)
        return np.minimum(out, self.at_start)

    def _ends(self) -> bool:
        """Whether the system can fail from every state the chain may start
        in: from a state where nothing fails (every repair instant, say) it
        never does, and the mean time is infinite."""
        matrix = self.generator
        leak = np.asarray(matrix.sum(axis=1)).ravel() < -1e-12 * np.maximum(
            -matrix.diagonal(), 1.0
        )
        reach = matrix.copy()
        reach.setdiag(0.0)
        reach.eliminate_zeros()
        reach.data[:] = 1.0
        can = leak.copy()
        while True:
            more = can | ((reach @ can.astype(float)) > 0.0)
            if (more == can).all():
                break
            can = more
        return bool(np.all(can[self.start > 0.0]))

    def mean(self) -> float:
        """The mean time to absorption from the start distribution."""
        from scipy.sparse.linalg import spsolve

        if self.generator.shape[0] == 0 or self.at_start == 0.0:
            return 0.0
        if not self._ends():
            return math.inf
        time = spsolve(
            (-self.generator).tocsc(), np.ones(self.generator.shape[0])
        )
        return float(self.start @ np.atleast_1d(time))


def _start(rbd, chain, states: dict) -> np.ndarray:
    """The chain's states' probabilities at 0 (see ``_curves._crew_start``,
    with no limit on the crews when the diagram has none)."""
    named = {node: states[node] for node in chain.nodes if node in states}
    _overrides.refuse_in_chain(named)
    stationary = [node for node, s in named.items() if s.stationary]
    if stationary:
        if len(stationary) < len(chain.nodes):
            raise NotImplementedError(
                "The components' long-run states are tied together by the "
                "repair queue: start every component in its long-run state "
                "(state='stationary'), or give each its state."
            )
        return np.array(chain.probabilities, dtype=float)
    down = tuple(
        k
        for k, node in enumerate(chain.nodes)
        if node in named and not named[node].alive
    )
    crews = rbd.repair_crews or len(chain.nodes)
    if len(down) > crews:
        raise ValueError(
            f"{len(down)} components are down at the start, in repairs "
            f"going on, which takes more than the {crews} repair crew(s): "
            "a state does not say which jobs wait."
        )
    start = np.zeros(len(chain.probabilities))
    start[chain.states.index((down, ()))] = 1.0
    return start


# -- the methods -------------------------------------------------------------


def _exact_sf(rbd, x: np.ndarray, state) -> np.ndarray:
    states = _curves._states(rbd, state)
    if _require_exact(rbd) and not any(s.stationary for s in states.values()):
        if _down_at_start(rbd, states):
            return np.zeros(x.shape)
        return _series_sf(_series_parts(rbd, states), x)
    return _Absorbing(rbd, states).sf(x)


def first_failure_sf(
    rbd,
    x,
    state=None,
    method: str = "exact",
    t_simulation: Optional[float] = None,
    mc_samples: Optional[int] = None,
    seed: Optional[int] = None,
    engine: str = "auto",
):
    """See ``RepairableRBD.first_failure_sf``."""
    one_of("method", method, ("exact", "simulate"))
    times = nonnegative_times(x)
    scalar = np.ndim(x) == 0
    if method == "exact":
        simulation_options(
            "first_failure_sf()",
            {
                "t_simulation": t_simulation,
                "mc_samples": mc_samples,
                "seed": seed,
                "engine": None if engine == "auto" else engine,
            },
        )
        out = _exact_sf(rbd, times, state)
    else:
        runs = _simulated(
            rbd,
            float(times.max()) if t_simulation is None else t_simulation,
            state,
            mc_samples,
            seed,
            engine,
        )
        first = runs.system.first_failure
        out = np.array([float(np.mean(first > t)) for t in times])
    return float(out[0]) if scalar else out.reshape(np.shape(x))


def _simulated(rbd, t_simulation, state, mc_samples, seed, engine):
    return rbd.simulate_timelines(
        t_simulation,
        mc_samples=mc_samples,
        seed=seed,
        engine=engine,
        state=state,
    )


def mean_time_to_first_failure(
    rbd,
    state=None,
    method: str = "exact",
    t_simulation: Optional[float] = None,
    mc_samples: Optional[int] = None,
    seed: Optional[int] = None,
    engine: str = "auto",
    confidence: float = 0.95,
):
    """See ``RepairableRBD.mean_time_to_first_failure``."""
    one_of("method", method, ("exact", "simulate"))
    if method == "exact":
        simulation_options(
            "mean_time_to_first_failure()",
            {
                "t_simulation": t_simulation,
                "mc_samples": mc_samples,
                "seed": seed,
                "engine": None if engine == "auto" else engine,
            },
        )
        states = _curves._states(rbd, state)
        if _require_exact(rbd) and not any(
            s.stationary for s in states.values()
        ):
            if _down_at_start(rbd, states):
                return 0.0
            return _series_mean(_series_parts(rbd, states))
        return _Absorbing(rbd, states).mean()
    if t_simulation is None:
        raise ValueError(
            "method='simulate' needs t_simulation, the time each "
            "simulation runs to."
        )
    from scipy.stats import norm

    runs = _simulated(rbd, t_simulation, state, mc_samples, seed, engine)
    first = runs.system.first_failure
    limited = np.minimum(first, float(t_simulation))
    cut = int(np.sum(~np.isfinite(first)))
    if cut:
        warnings.warn(
            f"{cut} of {len(first)} simulations had no system failure by "
            f"t_simulation={t_simulation}: the estimate is their mean "
            "time to the first failure or t_simulation, a lower bound of "
            "the mean. Simulate for longer for the mean itself.",
            stacklevel=2,
        )
    n = len(limited)
    error = float(np.std(limited, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    z = float(norm.ppf(0.5 + confidence / 2.0))
    estimate = float(np.mean(limited))
    return ConfidenceInterval(
        estimate=estimate,
        lower=max(0.0, estimate - z * error),
        upper=estimate + z * error,
        confidence=confidence,
        standard_error=error,
        n_samples=n,
        method="simulated",
    )
