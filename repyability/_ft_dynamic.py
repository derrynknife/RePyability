"""The time to the top event of a fault tree with dynamic gates.

A dynamic gate's output depends on the order in which its inputs occur,
so the tree is worked out over time, from the times its events occur:

- **PAND** (priority AND) occurs when all its inputs have occurred, in
  their order, left to right (inclusive: inputs occurring at one instant,
  as an FDEP gate can make them, are in order);
- **SPARE** gives its primary (the first input, an event or a gate) the
  spares after it (basic events), each switched in, in turn, when the one
  working fails; it occurs when the primary and every spare have failed. A
  spare waiting in reserve ages at the gate's dormancy factor ``a`` (0 a
  cold spare that cannot fail in reserve, 1 a hot one that ages as if
  working, between them a warm one): it accumulates ``a`` units of age per
  unit of time, and a spare found failed when needed is passed over;
- **SEQ** (sequence enforcing) makes its inputs, basic events, occur in
  their order: each starts ageing when the one before it occurs, so the
  gate occurs at the sum of their lives;
- **FDEP** (functional dependency) is written out before anything here
  (see ``FaultTree``): an event it forces is an OR of itself and its
  triggers.

Where a dynamic gate is an independent module (nothing below it is shared
with the rest of the tree, and its inputs share nothing with each other)
its probability of having occurred by ``t`` is integrated over its
inputs' distributions (``Integrated``), on a grid halved until two
Richardson-extrapolated passes agree:

- PAND: ``G_k(t) = integral of G_{k-1}(s) dF_k(s)`` over ``[0, t]``, by
  Stieltjes sums, ``G_1 = F_1``;
- SPARE: the measure of the times the unit at work fails, carried from
  one unit to the next: unit ``j`` switched in at ``s`` has age ``a s``
  and fails by ``t`` with ``F_j(a s + t - s)``, and a spare failed in
  reserve by ``s``, with ``F_j(a s)``, is passed over (for one cold spare
  this is the convolution ``integral of F_2(t - s) dF_1(s)``, for a hot
  one the product ``F_1(t) F_2(t)``);
- SEQ: the cold SPARE of its inputs.

Its inputs' distributions are the events' models, or a gate's own: an
independent static gate's from the tree's exact engine (the gate's subtree
as a tree, whose integrated dynamic gates are events of it), a dynamic
gate's by the same integration. The static part of the tree above the
dynamic gates is exact in the same way, the dynamic gates its events. A
dynamic gate that is not an independent module is simulated (``method=
"simulate"``), with the smallest independent part of the tree that holds
it (``simulate``), and the rest stays exact.
"""

import warnings
from typing import Any, Callable, Dict, Hashable, List, Optional, Set

import numpy as np

from repyability.utils.wrappers import outside_level

#: The integration stops halving its grid once two passes agree to this,
#: absolutely, in probability.
TOLERANCE = 1e-10
#: The most points a grid of a PAND (a sum per step) and of a SPARE or SEQ
#: (a sum per pair of steps) has.
FINEST = {"pand": 2**17, "spare": 2**12, "seq": 2**12}
#: The rows of a SPARE's pairs of steps worked out at once.
_ROWS = 256


def _event_cdf(model) -> Callable[[np.ndarray], np.ndarray]:
    """``t -> P(the event has occurred by t)``: a probability is one at
    the start, for good."""
    if isinstance(model, float):
        return lambda t: np.full(np.shape(t), model)
    return lambda t: np.broadcast_to(
        np.asarray(model.ff(t), dtype=float), np.shape(t)
    ).copy()


def _base(end: float) -> np.ndarray:
    """The grid the integration refines over ``[0, end]``: halvings of
    ``end`` down to ``2**-30`` of it (a step for every scale) and 16
    equal steps."""
    halvings = end * 2.0 ** -np.arange(30, 0, -1)
    return np.unique(
        np.concatenate(([0.0], halvings, np.linspace(0.0, end, 17)))
    )


def _refined(base: np.ndarray, level: int) -> np.ndarray:
    """``base`` with each step cut into ``2**level`` equal ones."""
    parts = 2**level
    steps = np.diff(base)
    fractions = np.arange(parts) / parts
    inner = base[:-1, None] + steps[:, None] * fractions[None, :]
    return np.concatenate((inner.reshape(-1), base[-1:]))


class Simulated:
    """A part of the tree as an event of the rest: its simulated times to
    occur, ``ff`` and ``sf`` their shares by and after ``t``."""

    def __init__(self, times: np.ndarray):
        self._sorted = np.sort(times)

    def ff(self, t):
        t = np.asarray(t, dtype=float)
        n = len(self._sorted)
        return np.searchsorted(self._sorted, t, side="right") / n

    def sf(self, t):
        t = np.asarray(t, dtype=float)
        n = len(self._sorted)
        return (n - np.searchsorted(self._sorted, t, side="right")) / n


class Integrated:
    """An independent dynamic gate as an event of the rest of the tree:
    its probability of having occurred by ``t``, integrated (see the
    module's notes). ``first`` are the distributions of the inputs taken
    on the grid (every PAND input; a SPARE's primary or a SEQ's first),
    ``spares`` those of a SPARE's spares or a SEQ's later inputs, taken
    at any age."""

    def __init__(
        self,
        name: Hashable,
        kind: str,
        dormancy: float,
        first: List[Callable],
        spares: List[Callable],
    ):
        self.name, self.kind = name, kind
        self.dormancy = 0.0 if kind == "seq" else dormancy
        self._first, self._spares = first, spares
        self._curves: Dict[float, Any] = {}

    def ff(self, t):
        t = np.asarray(t, dtype=float)
        flat = t.reshape(-1)
        end = float(flat.max(initial=0.0))
        if end <= 0.0:
            at_zero = self._values(np.zeros(1))[0]
            return np.full(t.shape, at_zero)
        curve = self._curve(end)
        return np.clip(curve(flat), 0.0, 1.0).reshape(t.shape)

    def sf(self, t):
        return 1.0 - self.ff(t)

    def _curve(self, end: float):
        """The gate's probability over ``[0, end]``: a cubic spline through
        Richardson-extrapolated values on a grid halved until the spline
        of one halving agrees with the next's values, at every point of
        its grid, within ``TOLERANCE`` (or the next grid would pass
        ``FINEST`` points), so both the sums and the spline between their
        points are checked. The grid starts from ``_base(end)``, fine at
        every scale below ``end``, so the answer does not depend on how
        far ``end`` lies past the lives."""
        from scipy.interpolate import CubicSpline

        for kept, curve in self._curves.items():
            if kept >= end:
                return curve
        base = _base(end)
        level = 1
        coarse = self._values(_refined(base, level))
        curve, change = None, np.inf
        while len(_refined(base, level + 1)) <= FINEST[self.kind]:
            fine = self._values(_refined(base, level + 1))
            # Second order: the error of a pass falls four-fold a halving.
            extrapolated = fine[::2] + (fine[::2] - coarse) / 3.0
            grid = _refined(base, level)
            if curve is not None:
                change = float(np.max(np.abs(curve(grid) - extrapolated)))
            curve = CubicSpline(grid, extrapolated)
            if change <= TOLERANCE:
                break
            coarse, level = fine, level + 1
        if change > 1e4 * TOLERANCE:
            warnings.warn(
                f"The {self.kind.upper()} gate {self.name!r} was integrated "
                f"to about {change:.1g} only (its inputs' distributions are "
                "not smooth enough for its grid).",
                stacklevel=outside_level(),
            )
        self._curves[end] = curve
        return curve

    def _values(self, u: np.ndarray) -> np.ndarray:
        """The gate's probability at each point of the grid ``u`` (from 0,
        increasing)."""
        if self.kind == "pand":
            return self._pand(u)
        return self._spare(u)

    def _pand(self, u: np.ndarray) -> np.ndarray:
        G = self._first[0](u)
        for cdf in self._first[1:]:
            F = cdf(u)
            # Inclusive: inputs that occur together, at 0, are in order.
            start = G[0] * F[0]
            G = start + np.concatenate(
                ([0.0], np.cumsum(0.5 * (G[:-1] + G[1:]) * np.diff(F)))
            )
        return G

    def _spare(self, u: np.ndarray) -> np.ndarray:
        a = self.dormancy
        F0 = self._first[0](u)
        # Each step's measure sits at its middle; the start's at 0.
        where = np.concatenate((u[:1], 0.5 * (u[:-1] + u[1:])))
        failed: List[np.ndarray] = [cdf(a * where) for cdf in self._spares]
        m = len(self._spares)

        def reserve_failed(lo: int, hi: int) -> np.ndarray:
            # Spares lo..hi-1 (counted from 1) all failed in reserve.
            out = np.ones(len(u))
            for i in range(lo, hi):
                out = out * failed[i - 1]
            return out

        units = [np.concatenate((F0[:1], np.diff(F0)))]
        occurs = units[0] * reserve_failed(1, m + 1)
        for j in range(1, m + 1):
            switched = sum(
                units[k] * reserve_failed(k + 1, j) for k in range(j)
            )
            unit = self._carried(switched, self._spares[j - 1], a, where, u)
            units.append(unit)
            occurs = occurs + unit * reserve_failed(j + 1, m + 1)
        return np.cumsum(occurs)

    @staticmethod
    def _carried(
        switched: np.ndarray,
        cdf: Callable,
        a: float,
        where: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        """The measure of the times a unit fails, by step of ``u``, given
        the measure of the times it is switched in (``switched``, at
        ``where``): switched in at ``s`` with age ``a s``, it has failed by
        ``t`` with ``F(a s + t - s)``."""
        out = np.zeros(len(u))
        rows = np.flatnonzero(switched)
        for start in range(0, len(rows), _ROWS):
            block = rows[start : start + _ROWS]
            s = where[block, None]
            age = a * s + np.maximum(u[None, :] - s, 0.0)
            C = cdf(age.reshape(-1)).reshape(age.shape)
            out[1:] += switched[block] @ np.diff(C, axis=1)
        return out


class Dynamic:
    """The dynamic gates of a ``FaultTree`` (its ``_gates``, FDEP written
    out), checked; how its top event is worked out (``plan``); and its
    simulation (``simulate``)."""

    def __init__(self, tree):
        self.tree = tree
        gates, events = tree._gates, tree.events
        role: Dict[Hashable, Hashable] = {}
        for name, gate in gates.items():
            if gate.kind not in ("spare", "seq"):
                continue
            held = gate.inputs[1:] if gate.kind == "spare" else gate.inputs
            for x in held:
                what = "spare" if gate.kind == "spare" else "input"
                if x not in events:
                    raise ValueError(
                        f"{gate.kind.upper()} gate {name!r}: its {what} {x!r} "
                        "must be a basic event"
                        + (
                            " (the primary, first, may be a gate)."
                            if gate.kind == "spare"
                            else "."
                        )
                    )
                if x in role:
                    raise NotImplementedError(
                        f"Event {x!r} is a spare or sequence member of gates "
                        f"{role[x]!r} and {name!r}: an event shared between "
                        "SPARE or SEQ gates is not supported."
                    )
                role[x] = name
        self.role = role
        self.order = self._ordered()
        self._below = self._descendants()
        self._exact: Optional[dict] = None

    # -- structure ---------------------------------------------------------

    def _depends(self, x: Hashable) -> tuple:
        """What the time ``x`` occurs depends on."""
        gates = self.tree._gates
        if x in gates:
            return gates[x].inputs
        owner = self.role.get(x)
        if owner is None:
            return ()
        gate = gates[owner]
        if gate.kind == "spare":
            return gate.inputs[:1]
        i = gate.inputs.index(x)
        return gate.inputs[i - 1 : i]

    def _ordered(self) -> list:
        """Every gate and event, each after what its time depends on; a
        ValueError if they depend on each other in a loop."""
        order: list = []
        state: Dict[Hashable, int] = {}
        for start in list(self.tree._gates) + list(self.tree.events):
            stack = [(start, False)]
            while stack:
                x, done = stack.pop()
                if done:
                    state[x] = 2
                    order.append(x)
                    continue
                if state.get(x) == 2:
                    continue
                if state.get(x) == 1:
                    raise ValueError(
                        "The dynamic gates make the times events occur "
                        f"depend on each other in a loop, through {x!r}."
                    )
                state[x] = 1
                stack.append((x, True))
                for c in self._depends(x):
                    if state.get(c) != 2:
                        stack.append((c, False))
        return order

    def _descendants(self) -> Dict[Hashable, Set[Hashable]]:
        """Each gate's subtree, the gate included."""
        below: Dict[Hashable, Set[Hashable]] = {}
        gates = self.tree._gates
        for x in self.tree._post_order():
            if x in gates:
                found = {x}
                for c in gates[x].inputs:
                    found |= below.get(c, {c})
                below[x] = found
        return below

    def _is_module(self, g: Hashable) -> bool:
        """Whether nothing in ``g``'s subtree is an input outside it, and
        every SPARE or SEQ gate an event of it belongs to is in it."""
        inside = self._below[g]
        parents = self.tree._parents
        for x in inside:
            if x != g and not set(parents.get(x, ())) <= inside:
                return False
            if x in self.role and self.role[x] not in inside:
                return False
        return True

    def plan(self, simulate: bool) -> Dict[Hashable, str]:
        """The parts of the tree worked out on their own, each an event of
        the rest, by gate: ``"integrated"`` or ``"simulated"``. With
        ``simulate``, every dynamic gate's part is simulated."""
        gates = self.tree._gates
        dynamic = [g for g in self.tree._post_order() if g in gates]
        dynamic = [
            g for g in dynamic if gates[g].kind in ("pand", "spare", "seq")
        ]
        integrable: Dict[Hashable, bool] = {}
        for g in dynamic:  # inner gates first
            parts = [self._below.get(c, {c}) for c in gates[g].inputs]
            disjoint = sum(map(len, parts)) == len(set().union(*parts))
            inner = [d for d in dynamic if d != g and d in self._below[g]]
            integrable[g] = (
                self._is_module(g)
                and disjoint
                and all(integrable[d] for d in inner)
            )
        units: Dict[Hashable, str] = {}
        for g in dynamic:
            if integrable[g]:
                units[g] = "simulated" if simulate else "integrated"
                continue
            holding = [
                h for h in gates if g in self._below[h] and self._is_module(h)
            ]
            unit = min(holding, key=lambda h: len(self._below[h]))
            units[unit] = "simulated"
        # A part inside a simulated one is simulated with it.
        simulated = [u for u, how in units.items() if how == "simulated"]
        return {
            u: how
            for u, how in units.items()
            if not any(s != u and u in self._below[s] for s in simulated)
        }

    def needs_simulation(self) -> List[Hashable]:
        """The dynamic gates that are not independent modules, so cannot
        be integrated."""
        units = self.plan(simulate=False)
        return sorted(
            (u for u, how in units.items() if how == "simulated"), key=str
        )

    # -- evaluation --------------------------------------------------------

    def _reduced(self, root: Hashable, parts: Dict[Hashable, Any]):
        """``root``'s subtree as a static tree, each part in ``parts``
        below it an event (its model there)."""
        from repyability.fault_tree import FaultTree

        gates, events = self.tree._gates, self.tree.events
        kept_gates: Dict[Hashable, tuple] = {}
        kept_events: Dict[Hashable, Any] = {}
        stack = [root]
        while stack:
            x = stack.pop()
            if x in kept_gates or x in kept_events:
                continue
            if x in events:
                kept_events[x] = events[x]
            elif x != root and x in parts:
                kept_events[x] = parts[x]
            else:
                kept_gates[x] = gates[x].written()
                stack.extend(gates[x].inputs)
        return FaultTree(kept_gates, kept_events, top=root)

    def _cdf(self, x: Hashable, parts: Dict[Hashable, Any]) -> Callable:
        events = self.tree.events
        if x in events:
            return _event_cdf(events[x])
        if x in parts:
            return parts[x].ff
        reduced = self._reduced(x, parts)
        return lambda t: np.asarray(
            reduced.top_event_probability(t), dtype=float
        )

    def _parts(
        self, simulate: bool, mc_samples: int, entropy: Optional[int]
    ) -> Dict[Hashable, Any]:
        """Each part worked out on its own, inner first, as a model with
        ``ff`` and ``sf``."""
        if not simulate and self._exact is not None:
            return self._exact
        units = self.plan(simulate)
        gates = self.tree._gates
        parts: Dict[Hashable, Any] = {}
        for u in sorted(units, key=lambda g: len(self._below[g])):
            if units[u] == "simulated":
                parts[u] = Simulated(self.simulate(u, mc_samples, entropy))
                continue
            gate = gates[u]
            split = 1 if gate.kind in ("spare", "seq") else len(gate.inputs)
            parts[u] = Integrated(
                u,
                gate.kind,
                gate.dormancy,
                [self._cdf(c, parts) for c in gate.inputs[:split]],
                [self._cdf(c, parts) for c in gate.inputs[split:]],
            )
        if not simulate:
            self._exact = parts
        return parts

    def probabilities(
        self,
        t: np.ndarray,
        simulate: bool,
        mc_samples: int,
        entropy: Optional[int],
    ):
        """The probabilities that the top event has not occurred and that
        it has, by each of the times ``t``."""
        parts = self._parts(simulate, mc_samples, entropy)
        top = self.tree.top
        if top in parts:
            model = parts[top]
        else:
            model = self._reduced(top, parts)
            return model.sf(t), model.ff(t)
        return (
            np.asarray(model.sf(t), dtype=float),
            np.asarray(model.ff(t), dtype=float),
        )

    # -- simulation --------------------------------------------------------

    def _lives(
        self, names: Set[Hashable], size: int, entropy: Optional[int]
    ) -> Dict[Hashable, np.ndarray]:
        """Each event's life (its age at occurring), from a stream of its
        own: the event's place among the tree's events and ``entropy``
        key it, so a part simulated alone draws what the whole tree
        draws."""
        from repyability.rbd._model_utils import is_fixed_probability
        from repyability.rbd._sampling import stream_sampler

        events = list(self.tree.events)
        streams = np.random.SeedSequence(entropy).spawn(len(events))
        out: Dict[Hashable, np.ndarray] = {}
        for i, e in enumerate(events):
            if e not in names:
                continue
            u = np.random.Generator(np.random.PCG64(streams[i])).random(size)
            model = self.tree.events[e]
            if isinstance(model, float) or is_fixed_probability(model):
                p = (
                    model
                    if isinstance(model, float)
                    else float(np.ravel(model.ff(1.0))[0])
                )
                out[e] = np.where(u < p, 0.0, np.inf)
                continue
            sampler = stream_sampler(model)
            if sampler is None:
                raise NotImplementedError(
                    f"The model of event {e!r} cannot be drawn from its "
                    "quantile function, which the simulation of a tree with "
                    "dynamic gates takes."
                )
            out[e] = np.asarray(sampler(u), dtype=float)
        return out

    def simulate(
        self, root: Hashable, size: int, entropy: Optional[int]
    ) -> np.ndarray:
        """``size`` simulated times at which ``root`` occurs (``inf``
        where it never does), its subtree simulated alone."""
        gates = self.tree._gates
        needed: Set[Hashable] = set()
        stack = [root]
        while stack:
            x = stack.pop()
            if x in needed:
                continue
            needed.add(x)
            stack.extend(self._depends(x))
            if x in self.role:
                stack.append(self.role[x])
        lives = self._lives(
            {x for x in needed if x in self.tree.events}, size, entropy
        )
        times: Dict[Hashable, np.ndarray] = {}
        done: Set[Hashable] = set()
        for x in self.order:
            if x not in needed or x in times:
                continue
            if x in gates:
                gate = gates[x]
                if gate.kind in ("spare", "seq"):
                    self._run(x, lives, times)
                    continue
                stack_ = np.vstack([times[c] for c in gate.inputs])
                if gate.kind == "pand":
                    ordered = np.all(stack_[:-1] <= stack_[1:], axis=0)
                    times[x] = np.where(ordered, stack_[-1], np.inf)
                else:
                    times[x] = np.sort(stack_, axis=0)[gate.k - 1]
            elif x in self.role:
                owner = self.role[x]
                if owner not in done:
                    self._run(owner, lives, times)
                    done.add(owner)
            else:
                times[x] = lives[x]
        return times[root]

    def _run(
        self,
        name: Hashable,
        lives: Dict[Hashable, np.ndarray],
        times: Dict[Hashable, np.ndarray],
    ) -> None:
        """A SPARE or SEQ gate's time, and those of its spares or
        members, into ``times`` (once)."""
        if name in times:
            return
        gate = self.tree._gates[name]
        with np.errstate(invalid="ignore"):
            if gate.kind == "seq":
                clock = np.zeros_like(lives[gate.inputs[0]])
                for x in gate.inputs:
                    clock = clock + lives[x]
                    times[x] = clock
                times[name] = clock
                return
            a = gate.dormancy
            clock = times[gate.inputs[0]]
            for x in gate.inputs[1:]:
                life = lives[x]
                if a > 0.0:
                    in_reserve = life / a
                else:
                    in_reserve = np.where(life == 0.0, 0.0, np.inf)
                passed = in_reserve <= clock
                ends = np.where(passed, clock, clock + life - a * clock)
                times[x] = np.where(passed, in_reserve, ends)
                clock = ends
        times[name] = clock
