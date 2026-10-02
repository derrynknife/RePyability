"""The modular (series-parallel) decomposition behind the exact engine.

Before anything is evaluated, an RBD's diagram is reduced by collapsing the
parts that have a closed form into *modules*:

- a **series** chain: a node whose only successor has it as its only
  predecessor. The module works when all of its members work.
- a **parallel** group: nodes with the same predecessors, the same
  successors and the same ``k``, where every successor needs only one
  working predecessor. The module works when any member works.
- a **k-out-of-n** group: all the predecessors of a node that needs
  ``k >= 2`` of them, when they share their predecessors and ``k`` and feed
  only that node. The module works when at least ``k`` members work.
- a **bypassed** node: one whose only predecessor also feeds each of its
  successors directly, each needing only one working predecessor. Whether
  it works can never matter, so it is left out.

The rules are applied until none applies. Each is exact: the reduced
diagram works exactly when the original does. Every node ends up in at most
one module, so the members of a module are independent of everything
outside it, and a module's probability has a closed form in its members'.
What is left is either one module (or node) joining the input to the
output, which is the case for any series-parallel diagram, or a *core*: the
part of the diagram that is not series-parallel (e.g. a bridge), over
modules and single nodes. Only the core is worked out from its minimal path
sets, with the Shannon decomposition of ``shannon.py``; the core is usually
small, so large redundant diagrams stay fast. A diagram with no module at
all (e.g. a bridge of single nodes) is evaluated exactly as without the
reduction.

The decomposition is a tree of *terms*, stored in post-order (children
before their parents). A term is one of

- ``(NODE, name)``: an RBD node;
- ``(SERIES, children)`` and ``(PARALLEL, children)``;
- ``(KOON, children, k)``, with ``1 < k < len(children)``.

The children are the positions of other terms. Every probability is
computed together with its complement, both as sums of products of the
nodes' probabilities and complements, so neither loses precision however
close to 0 or 1 it is.

Each rule also keeps how much the diagram can carry, when its nodes have
capacities (see ``capacity.py``): a series chain carries the least of its
members' capacities, a parallel group the sum, and a bypassed node never
adds to what its predecessor already sends on directly. So the capacity
analysis works on the reduced diagram too, kept whole as a
:class:`FlowGraph`.
"""

from itertools import combinations, product
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
)

import numpy as np

from repyability.rbd import bdd
from repyability.rbd.min_path_sets import min_path_sets as find_min_path_sets
from repyability.rbd.rbd_graph import RBDGraph
from repyability.rbd.shannon import (
    _minimal_cut_sets,
    _shannon_plan,
    _shannon_value_and_gradient,
    _union_plan,
)

# The kinds of term.
NODE, SERIES, PARALLEL, KOON = 0, 1, 2, 3

# The reduced diagram's input and output vertices; its other vertices are
# term positions, from 0.
_SOURCE, _SINK = -1, -2

NO_PATHS = "RBD has no paths through! Need to re-evaluate the KooN nodes."

# The compiled structure function writes out the core's path (or cut) sets
# when they hold at most this many nodes in all, and loops over them if not.
_WRITTEN_OUT = 5000

#: How the core is decided: ``"paths"``, by the Shannon decomposition of
#: its minimal path sets, ``"bdd"``, by a binary decision diagram built
#: from its graph without listing them (see ``bdd.py``; #102), or
#: ``"auto"`` (the default, #103): the decision diagram for a core that may
#: have more than ``AUTO_PATHS`` minimal path sets, which multiply in a
#: meshed core while the diagram grows with the mesh's width, and the path
#: sets for a smaller one. Read when a diagram is decomposed (on
#: construction).
CORE_METHOD = "auto"
#: The most minimal path sets (as ``_path_count`` bounds them) for which
#: ``"auto"`` lists them: beyond, the decision diagram is several times
#: smaller and the listing slows (a hundred take about a tenth of a second
#: to decompose, four thousand about nine).
AUTO_PATHS = 100


class FlowGraph:
    """The reduced diagram, whole, for the capacity analysis.

    The structure function needs only the minimal path sets of what the
    reduction leaves, but the capacity needs the diagram itself: a
    component that can never decide whether the system works can still add
    to what it carries (through a k-out-of-n node that the system can also
    bypass). ``terms`` is the reduction's list of terms, in which each
    module comes after its members, with each node named by the component
    it stands for; a group that a node needs all of stays a k-out-of-n
    term here, with ``k`` its size, as it carries the sum of its members'
    capacities (the tree makes it a series module). ``vertices`` are the
    terms left, in topological order;
    two of them can stand for one component (a repeated node, drawn in two
    places). ``preds`` and ``k`` hold each vertex's predecessors
    (``_SOURCE`` for the input) and k, and ``sink_preds`` and ``sink_k``
    the output's.
    """

    def __init__(
        self,
        terms: Sequence[tuple],
        vertices: Sequence[int],
        preds: Sequence[tuple],
        k: Sequence[int],
        sink_preds: tuple,
        sink_k: int,
    ):
        self.terms = tuple(terms)
        self.vertices = tuple(vertices)
        self.preds = tuple(preds)
        self.k = tuple(k)
        self.sink_preds = sink_preds
        self.sink_k = sink_k
        self._cuts: Optional[List[tuple]] = None

    def cuts(self) -> List[tuple]:
        """The minimal sets of vertices whose removal leaves no path from
        the input to the output, taking every k as 1, each in topological
        order; none if an edge joins the input to the output directly. By
        the max-flow min-cut theorem the most the vertices can carry is the
        least total capacity of one of these sets."""
        if self._cuts is None:
            # Each vertex's minimal paths from the input, as vertex sets: a
            # path that holds another cannot be part of a minimal one.
            paths: Dict[int, list] = {_SOURCE: [frozenset()]}
            for v, preds in zip(self.vertices, self.preds):
                paths[v] = _minimal_sets(
                    p | {v} for u in preds for p in paths[u]
                )
            ends = _minimal_sets(p for u in self.sink_preds for p in paths[u])
            order = {v: i for i, v in enumerate(self.vertices)}
            self._cuts = (
                []
                if frozenset() in ends
                else sorted(
                    tuple(sorted(cut, key=order.__getitem__))
                    for cut in _minimal_cut_sets(_shannon_plan(ends))
                )
            )
        return self._cuts


class Decomposition:
    """An RBD's structure function as a tree of modules over a core.

    Built by :func:`decompose`. ``terms`` is the tree, in post-order. The
    system is one of: ``root``, the position of the term joining the input
    to the output; the ``core``, given by its minimal path sets over the
    positions of its terms; or neither, when the system always works (e.g.
    an edge joins the input to the output directly). ``nodes`` are the
    nodes in the tree: the relevant ones. Every other node is in no minimal
    path set. ``flow`` is the reduced diagram whole, for the capacity
    analysis (None for a structure that is not reduced).
    """

    def __init__(
        self,
        terms: Sequence[tuple],
        root: Optional[int] = None,
        core: Optional[Iterable[Iterable[int]]] = None,
        flow: Optional[FlowGraph] = None,
        plan: Optional[tuple] = None,
    ):
        self.terms = list(terms)
        self.root = root
        self._core: Optional[List[tuple]] = (
            None if core is None else [tuple(ps) for ps in core]
        )
        self.always_works = root is None and core is None and plan is None
        self.nodes = frozenset(t[1] for t in self.terms if t[0] == NODE)
        self.flow = flow
        # A core given as a decision diagram built from its graph (see
        # bdd.py) rather than by its path sets, which are then found from
        # it only when asked for.
        self.from_graph = plan is not None
        self._core_plan: Optional[tuple] = plan
        self._core_cut_sets: Optional[List[tuple]] = None
        self._functions: Dict[str, Callable] = {}
        # The Shannon decomposition of each core term's cut sets less the
        # term (see ``failed_cut_sets``), and the dual decomposition.
        self._cut_plan: Optional[tuple] = None
        self._dual: Optional["Decomposition"] = None

    @property
    def core(self) -> Optional[List[tuple]]:
        """The core's minimal path sets, over term positions (None without
        a core); for a core decided from its graph, found from its decision
        diagram on first use."""
        if self._core is None and self.from_graph:
            assert self._core_plan is not None
            self._core = sorted(
                tuple(sorted(s)) for s in bdd.path_sets(self._core_plan)
            )
        return self._core

    def __getstate__(self) -> dict:
        # The compiled structure functions cannot be pickled; they are
        # compiled again on first use.
        state = dict(self.__dict__)
        state["_functions"] = {}
        return state

    # -- the core ----------------------------------------------------------

    def core_plan(self) -> tuple:
        """The Shannon decomposition of the core, over term positions."""
        if self._core_plan is None:
            assert self.core is not None
            self._core_plan = _shannon_plan(frozenset(s) for s in self.core)
        return self._core_plan

    def core_cut_sets(self) -> List[tuple]:
        """The core's minimal cut sets, over term positions."""
        if self._core_cut_sets is None:
            self._core_cut_sets = [
                tuple(sorted(cut))
                for cut in _minimal_cut_sets(self.core_plan())
            ]
        return self._core_cut_sets

    # -- probabilities -----------------------------------------------------

    def _forward(self, p: Dict, q: Optional[Dict]) -> tuple[list, list]:
        """Every term's probability of working and of failing."""
        R: list = [None] * len(self.terms)
        Q: list = [None] * len(self.terms)
        for i, term in enumerate(self.terms):
            kind = term[0]
            if kind == NODE:
                R[i] = p[term[1]]
                Q[i] = 1 - R[i] if q is None else q[term[1]]
            elif kind == SERIES:
                # Fails at the first failed member: q1 + r1 q2 + r1 r2 q3...
                first, *rest = term[1]
                works, fails = R[first], Q[first]
                for c in rest:
                    fails = fails + works * Q[c]
                    works = works * R[c]
                R[i], Q[i] = works, fails
            elif kind == PARALLEL:
                # Works at the first working member: r1 + q1 r2 + q1 q2 r3...
                first, *rest = term[1]
                works, fails = R[first], Q[first]
                for c in rest:
                    works = works + fails * R[c]
                    fails = fails * Q[c]
                R[i], Q[i] = works, fails
            else:
                # below[j]: the probability that exactly j of the members so
                # far work (j < k); above: that at least k do.
                children, k = term[1], term[2]
                below: list = [1.0] + [0.0] * (k - 1)
                above: Any = 0.0
                for c in children:
                    above = above + below[k - 1] * R[c]
                    for j in range(k - 1, 0, -1):
                        below[j] = below[j] * Q[c] + below[j - 1] * R[c]
                    below[0] = below[0] * Q[c]
                R[i], Q[i] = above, sum(below)
        return R, Q

    def _core_value(self, R: list, Q: list, fails: bool, shape) -> Any:
        """The core's probability of working (or, with ``fails``, of
        failing), from its terms' probabilities. The complement uses the
        same decomposition with the outcomes swapped, so it too is a sum of
        products and keeps its relative precision."""
        steps, root = self.core_plan()
        zero: Any = 0.0 if shape is None else np.zeros(shape)
        one: Any = 1.0 if shape is None else np.ones(shape)
        values = [one, zero] if fails else [zero, one]
        for pivot, active, inactive in steps:
            values.append(
                R[pivot] * values[active] + Q[pivot] * values[inactive]
            )
        return values[root]

    def probabilities(
        self,
        p: Dict,
        q: Optional[Dict] = None,
        shape=None,
        works: bool = True,
        fails: bool = True,
    ) -> tuple[Any, Any]:
        """The probabilities that the system works and that it fails.

        ``p`` holds each node's probability of working (floats, or arrays
        of shape ``shape``) and ``q`` their complements, by default
        ``1 - p``; pass them when they are known more precisely. Only the
        nodes in ``nodes`` are used. ``works`` and ``fails`` say which of
        the two are needed: the other may come back as None.
        """
        if self.always_works:
            if shape is None:
                return 1.0, 0.0
            return np.ones(shape), np.zeros(shape)
        R, Q = self._forward(p, q)
        if self.root is not None:
            return R[self.root], Q[self.root]
        return (
            self._core_value(R, Q, False, shape) if works else None,
            self._core_value(R, Q, True, shape) if fails else None,
        )

    def value_and_gradient(
        self, p: Dict[Any, Any], q: Dict[Any, Any], shape=None
    ) -> tuple[Any, Any, Dict[Any, Any]]:
        """For the nodes' probabilities ``p`` and their complements ``q``
        (single probabilities, or arrays of shape ``shape``): the
        probability that the system works, that it fails, and the
        derivative of the first with respect to each node's probability
        (its Birnbaum importance; nodes it does not depend on may be
        missing). All three keep their full relative precision: through
        the modules, the derivative is a product of the modules' own
        derivatives, each a product or a sum of products of their members'
        probabilities; in the core, it is taken from whichever of the
        probabilities of working and of failing is the smaller (element by
        element), so that it is a difference of small values, not of
        values near 1."""
        if self.always_works:
            if shape is None:
                return 1.0, 0.0, {}
            return np.ones(shape), np.zeros(shape), {}
        R, Q = self._forward(p, q)
        # None: a term the root does not reach.
        adjoint: list = [None] * len(self.terms)
        if self.root is not None:
            works, fails = R[self.root], Q[self.root]
            adjoint[self.root] = 1.0
        else:
            plan = self.core_plan()
            works, d_works = _shannon_value_and_gradient(plan, R, Q)
            fails, d_fails = _shannon_value_and_gradient(
                plan, R, Q, (1.0, 0.0)
            )
            # From whichever end is accurate: a difference of two values
            # near 1 would cancel.
            if np.ndim(works) == 0 and np.ndim(fails) == 0:
                if works <= fails:
                    for c, d in d_works.items():
                        adjoint[c] = d
                else:
                    for c, d in d_fails.items():
                        adjoint[c] = -d
            else:
                from_works = np.asarray(works) <= np.asarray(fails)
                for c, d in d_works.items():
                    adjoint[c] = np.where(from_works, d, -d_fails[c])
        gradient: Dict[Any, Any] = {}
        for i in range(len(self.terms) - 1, -1, -1):
            a = adjoint[i]
            if a is None:
                continue
            term = self.terms[i]
            kind = term[0]
            if kind == NODE:
                gradient[term[1]] = a
                continue
            children = term[1]
            if kind == SERIES:
                partials = _products_of_others([R[c] for c in children])
            elif kind == PARALLEL:
                partials = _products_of_others([Q[c] for c in children])
            else:
                partials = _koon_partials(
                    [R[c] for c in children], [Q[c] for c in children], term[2]
                )
            for c, d in zip(children, partials):
                adjoint[c] = a * d
        return works, fails, gradient

    def failed_cut_sets(
        self, p: Dict[Any, Any], q: Dict[Any, Any], shape=None
    ) -> Dict[Any, Any]:
        """For the nodes' probabilities of working ``p`` and of failing
        ``q`` (as for ``value_and_gradient``): for each node, the
        probability that some minimal cut set containing it has failed
        (every node in it), the numerator of its exact Fussell-Vesely
        importance. Nodes in no minimal cut set are missing.

        A module's members share no node, so its minimal cut sets join
        its members' (see ``_families``): a series module's are its
        members', so one containing node ``i`` is one of the member's that
        contains it; a parallel module's join one of each member's, so one
        containing ``i`` has failed when the member's has and every other
        member has failed; a k-out-of-n module's join those of
        ``n - k + 1`` members, so when the member's has and at least
        ``n - k`` of the others have failed. Down the tree, the probability
        is the node's probability of failing times such factors, as its
        Birnbaum importance is a product of the modules' derivatives. In a
        core, a term's factor is the probability that, for some minimal
        cut set of the core containing the term, every other term in it has
        failed: a union, worked out by the Shannon decomposition of those
        sets, failing for working. Every factor is a product or a sum of
        products, so a small probability keeps its precision."""
        if self.always_works:
            return {}
        R, Q = self._forward(p, q)
        # None: a term in no minimal cut set.
        factor: list = [None] * len(self.terms)
        if self.root is not None:
            factor[self.root] = 1.0
        else:
            terms, steps, roots = self._core_cut_plan()
            values: list = [
                0.0 if shape is None else np.zeros(shape),
                1.0 if shape is None else np.ones(shape),
            ]
            for pivot, failed, works in steps:
                values.append(
                    Q[pivot] * values[failed] + R[pivot] * values[works]
                )
            for t, root in zip(terms, roots):
                factor[t] = values[root]
        out: Dict[Any, Any] = {}
        for i in range(len(self.terms) - 1, -1, -1):
            a = factor[i]
            if a is None:
                continue
            term = self.terms[i]
            if term[0] == NODE:
                out[term[1]] = a * Q[i]
                continue
            children = term[1]
            if term[0] == SERIES:
                shares: list = [1.0] * len(children)
            elif term[0] == PARALLEL:
                shares = _products_of_others([Q[c] for c in children])
            else:
                shares = _koon_cut_shares(
                    [R[c] for c in children], [Q[c] for c in children], term[2]
                )
            for c, d in zip(children, shares):
                factor[c] = a * d
        return out

    def _core_cut_plan(self) -> tuple[list, list, list]:
        """The core's terms in some minimal cut set, and one Shannon
        decomposition of, for each, the probability that for some minimal
        cut set of the core containing it every other term in it has
        failed (a set's terms *satisfied* when failed): ``(terms, steps,
        roots)``, as :func:`shannon._union_plan` gives them."""
        if self._cut_plan is None:
            cuts = [frozenset(cut) for cut in self.core_cut_sets()]
            terms = sorted({c for cut in cuts for c in cut})
            steps, roots = _union_plan(
                [[cut - {t} for cut in cuts if t in cut] for t in terms]
            )
            self._cut_plan = (terms, steps, roots)
        return self._cut_plan

    def dual(self) -> "Decomposition":
        """The decomposition of the dual structure, which works unless every
        node of some minimal path set of this one has failed: series and
        parallel modules swap, a k-out-of-n module needs ``n - k + 1``, and
        the core's minimal path and cut sets swap. Its minimal cut sets are
        this structure's minimal path sets. Not for a structure that always
        works (whose dual never does)."""
        if self.always_works:
            raise ValueError("A structure that always works has no dual.")
        if self._dual is None:
            terms: list = []
            for term in self.terms:
                if term[0] == SERIES:
                    terms.append((PARALLEL, term[1]))
                elif term[0] == PARALLEL:
                    terms.append((SERIES, term[1]))
                elif term[0] == KOON:
                    terms.append((KOON, term[1], len(term[1]) - term[2] + 1))
                else:
                    terms.append(term)
            if self.root is not None:
                self._dual = Decomposition(terms, root=self.root)
            else:
                self._dual = Decomposition(terms, core=self.core_cut_sets())
                self._dual._core_cut_sets = list(self.core or [])
        return self._dual

    # -- the structure function --------------------------------------------

    def works(self, status, method: str = "p") -> bool:
        """Whether the system works, given whether each node works (truthy)
        or has failed (falsy). ``method`` says whether the core is checked
        through its minimal path sets (``"p"``) or cut sets (``"c"``)."""
        return self.structure_function(method)(status)

    def structure_function(self, method: str = "p") -> Callable:
        """The compiled structure function behind :meth:`works`, for callers
        (the simulations) that evaluate it at every event: ``function(status)``
        is ``works(status, method)``."""
        function = self._functions.get(method)
        if function is None:
            function = self._functions[method] = self._structure_function(
                method
            )
        return function

    def _structure_function(self, method: str) -> Callable:
        """The structure function compiled to a Python function: one line
        per module, and the core's path (or cut) sets written out as ``or``
        of ``and`` (or ``and`` of ``or``). The simulations call it at every
        event, and this is several times faster than walking the tree."""
        names: list = []
        slot: Dict[Hashable, int] = {}
        lines: list = []
        value: Dict[int, str] = {}
        for i, term in enumerate(self.terms):
            if term[0] == NODE:
                if term[1] not in slot:
                    slot[term[1]] = len(names)
                    names.append(term[1])
                value[i] = f"s[n{slot[term[1]]}]"
                continue
            members = [value[c] for c in term[1]]
            if term[0] == SERIES:
                lines.append(f"m{i} = " + " and ".join(members))
            elif term[0] == PARALLEL:
                lines.append(f"m{i} = " + " or ".join(members))
            else:
                lines.append(
                    f"m{i} = sum(map(bool, ({', '.join(members)},)))"
                    f" >= {term[2]}"
                )
            value[i] = f"m{i}"
        namespace: Dict[str, Any] = {"names": tuple(names)}
        if self.always_works:
            result = "True"
        elif self.root is not None:
            result = value[self.root]
        elif self.from_graph:
            # One decision per variable on the way from the root.
            steps, root = self.core_plan()
            used = sorted({pivot for pivot, _, _ in steps})
            index = {c: j for j, c in enumerate(used)}
            namespace["walk"] = bdd.walk
            namespace["plan"] = (
                [(index[p], a, i) for p, a, i in steps],
                root,
            )
            lines.append(
                "v = (" + "".join(value[c] + ", " for c in used) + ")"
            )
            result = "walk(plan, v)"
        else:
            sets = self.core if method == "p" else self.core_cut_sets()
            inner, outer = (
                (" and ", " or ") if method == "p" else (" or ", " and ")
            )
            if sum(map(len, sets or [])) <= _WRITTEN_OUT:
                result = outer.join(
                    "(" + inner.join(value[c] for c in group) + ")"
                    for group in sets or []
                )
            else:
                # Too many to write out: loop over them instead.
                terms = sorted(set().union(*sets or []))
                index = {c: j for j, c in enumerate(terms)}
                namespace["groups"] = [
                    tuple(index[c] for c in group) for group in sets or []
                ]
                lines.append(
                    "v = (" + "".join(value[c] + ", " for c in terms) + ")"
                )
                if method == "p":
                    result = "any(all(map(v.__getitem__, g)) for g in groups)"
                else:
                    result = "all(any(map(v.__getitem__, g)) for g in groups)"
        lines.append(f"return True if {result} else False")
        arguments = "".join(f", n{j}=names[{j}]" for j in range(len(names)))
        source = f"def works(s{arguments}):\n" + "".join(
            f"    {line}\n" for line in lines
        )
        exec(source, namespace)  # the source is built from the tree alone
        return namespace["works"]

    def lifetime(self, lifetimes: Dict, size: int) -> np.ndarray:
        """The system's lifetime in each sample, from each node's: a series
        module fails at its first failure, a parallel one at its last, a
        k-out-of-n one at the failure that leaves fewer than k working, and
        the core when its last minimal path set breaks."""
        if self.always_works:
            return np.full(size, np.inf)
        values: list = [None] * len(self.terms)
        for i, term in enumerate(self.terms):
            kind = term[0]
            if kind == NODE:
                values[i] = lifetimes[term[1]]
                continue
            members = [values[c] for c in term[1]]
            if kind == SERIES:
                values[i] = np.min(members, axis=0)
            elif kind == PARALLEL:
                values[i] = np.max(members, axis=0)
            else:
                # The k-th longest of the members' lifetimes.
                values[i] = np.sort(members, axis=0)[len(members) - term[2]]
        if self.root is not None:
            return np.asarray(values[self.root], dtype=float)
        if self.from_graph:
            return bdd.lifetime(self.core_plan(), values, size)
        out = np.full(size, -np.inf)
        for path_set in self.core or []:
            path_life = np.full(size, np.inf)
            for c in path_set:
                path_life = np.minimum(path_life, values[c])
            out = np.maximum(out, path_life)
        return out

    # -- path and cut sets -------------------------------------------------

    def _families(self, paths: bool) -> list:
        """Every term's minimal path sets (or cut sets), over nodes. The
        members of a module share no node, so combining one minimal set of
        each needed member gives a minimal set of the module."""
        families: list = [None] * len(self.terms)
        for i, term in enumerate(self.terms):
            kind = term[0]
            if kind == NODE:
                families[i] = [frozenset([term[1]])]
                continue
            members = [families[c] for c in term[1]]
            if kind == KOON:
                # k working members, or n - k + 1 failed ones.
                size = term[2] if paths else len(members) - term[2] + 1
                families[i] = [
                    s
                    for chosen in combinations(members, size)
                    for s in _joint(chosen)
                ]
            elif (kind == SERIES) == paths:
                # Every member is needed.
                families[i] = _joint(members)
            else:
                # Any one member will do.
                families[i] = [s for family in members for s in family]
        return families

    def _sets(self, paths: bool) -> set[frozenset]:
        if self.always_works:
            return {frozenset()} if paths else set()
        families = self._families(paths)
        if self.root is not None:
            return set(families[self.root])
        core = self.core if paths else self.core_cut_sets()
        return {
            s
            for chosen in core or []
            for s in _joint([families[c] for c in chosen])
        }

    def path_sets(self) -> set[frozenset]:
        """The minimal path sets, over nodes."""
        return self._sets(True)

    def cut_sets(self) -> set[frozenset]:
        """The minimal cut sets, over nodes."""
        return self._sets(False)


def _minimal_sets(sets: Iterable[frozenset]) -> list:
    """The distinct sets that contain no other."""
    kept: list = []
    for s in sorted(set(sets), key=len):
        if not any(k <= s for k in kept):
            kept.append(s)
    return kept


def _joint(families: Sequence[Sequence[frozenset]]) -> list:
    """One set from each family, joined, in every combination."""
    return [frozenset().union(*chosen) for chosen in product(*families)]


def _products_of_others(values: Sequence[Any]) -> list:
    """For each value, the product of all the others (with no division, so
    zeros are fine)."""
    n = len(values)
    before: list = [1.0] * n
    after: list = [1.0] * n
    for i in range(1, n):
        before[i] = before[i - 1] * values[i - 1]
    for i in range(n - 2, -1, -1):
        after[i] = after[i + 1] * values[i + 1]
    return [b * a for b, a in zip(before, after)]


def _others_working(
    R: Sequence[Any], Q: Sequence[Any], counts: Iterable[int]
) -> list:
    """For each of ``n`` independent members working with probabilities
    ``R`` (failing with ``Q``): the probabilities that exactly ``j`` of the
    others work, for each ``j`` in ``counts``, each a sum of products."""
    counts = list(counts)
    n = len(R)

    def distributions(order: Iterable[int]) -> list:
        # dists[m][j]: the probability that exactly j of the first m
        # members taken in ``order`` work.
        dists = [[1.0]]
        for i in order:
            last = dists[-1]
            new = [0.0] * (len(last) + 1)
            for j, value in enumerate(last):
                new[j] += value * Q[i]
                new[j + 1] += value * R[i]
            dists.append(new)
        return dists

    before = distributions(range(n))
    after = distributions(range(n - 1, -1, -1))
    out = []
    for i in range(n):
        b, a = before[i], after[n - 1 - i]
        out.append(
            [
                sum(
                    b[x] * a[j - x]
                    for x in range(len(b))
                    if 0 <= j - x < len(a)
                )
                for j in counts
            ]
        )
    return out


def _koon_partials(R: Sequence[float], Q: Sequence[float], k: int) -> list:
    """For each member of a k-out-of-n module, the derivative of the
    module's probability of working with respect to the member's: the
    probability that exactly ``k - 1`` of the others work."""
    return [exactly[0] for exactly in _others_working(R, Q, [k - 1])]


def _koon_cut_shares(R: Sequence[Any], Q: Sequence[Any], k: int) -> list:
    """For each member of a k-out-of-n module, the probability that at most
    ``k - 1`` of the others work (at least ``n - k`` have failed): with a
    cut set of the member failed, one of the module's has then failed."""
    return [sum(exactly) for exactly in _others_working(R, Q, range(k))]


class _Reduction:
    """The reduction rules, applied to a working copy of the diagram whose
    vertices are term positions (and the input and output).

    The ``pinned`` nodes (those that stand for a component drawn in more
    than one place) never become members of a module: a module's closed
    form assumes its members are independent of everything outside it.
    They may still be bypassed: an appearance that can never make a
    difference is dropped, whichever component it stands for."""

    def __init__(
        self,
        graph: RBDGraph,
        input_node,
        output_node,
        pinned: Iterable[Hashable] = (),
    ):
        self.terms: list = []
        vertex: Dict[Hashable, int] = {input_node: _SOURCE, output_node: _SINK}
        for node in graph.nodes:
            if node not in vertex:
                vertex[node] = len(self.terms)
                self.terms.append((NODE, node))
        self.pinned = {vertex[n] for n in pinned if n in vertex}
        self.pred: Dict[int, set] = {v: set() for v in vertex.values()}
        self.succ: Dict[int, set] = {v: set() for v in vertex.values()}
        for a, b in graph.edges:
            self.succ[vertex[a]].add(vertex[b])
            self.pred[vertex[b]].add(vertex[a])
        self.k = {vertex[n]: graph.nodes[n]["k"] for n in graph.nodes}
        self.alive = set(range(len(self.terms)))

    def run(self) -> None:
        work = [_SINK] + sorted(self.alive, reverse=True)
        while work:
            v = work.pop()
            if v != _SINK and v not in self.alive:
                continue
            changed = (
                self._bypass(v)
                or self._series(v)
                or self._parallel(v)
                or self._koon(v)
            )
            if changed:
                work.extend(changed)

    def _module(self, kind: int, members: Sequence[int], k: int = 0) -> int:
        # Members of the same kind of series or parallel module are merged
        # into it, keeping the tree shallow.
        children: list = []
        for v in members:
            term = self.terms[v]
            if kind in (SERIES, PARALLEL) and term[0] == kind:
                children.extend(term[1])
            else:
                children.append(v)
        if kind == KOON:
            self.terms.append((KOON, tuple(children), k))
        else:
            self.terms.append((kind, tuple(children)))
        m = len(self.terms) - 1
        self.alive.difference_update(members)
        self.alive.add(m)
        return m

    def _replace(
        self, members: Sequence[int], m: int, pred: set, succ: set, k: int
    ) -> list:
        """Put module ``m`` in place of ``members``, between ``pred`` and
        ``succ``; returns the vertices to look at again."""
        pred, succ, gone = set(pred), set(succ), set(members)
        for v in gone:
            del self.pred[v], self.succ[v], self.k[v]
        self.pred[m], self.succ[m], self.k[m] = pred, succ, k
        for v in pred:
            self.succ[v] -= gone
            self.succ[v].add(m)
        for v in succ:
            self.pred[v] -= gone
            self.pred[v].add(m)
        return [m, *pred, *succ]

    def _bypass(self, v: int) -> list:
        # Its only predecessor reaches each of its successors directly, and
        # each needs one working predecessor: v can never make a difference.
        # (A node with one predecessor has k = 1: the diagram is valid, and
        # the rules keep every k within its node's number of predecessors.)
        if v < 0 or len(self.pred[v]) != 1:
            return []
        (p,) = self.pred[v]
        succ = self.succ[v]
        if not all(self.k[w] == 1 and p in self.pred[w] for w in succ):
            return []
        self.succ[p].discard(v)
        for w in succ:
            self.pred[w].discard(v)
        del self.pred[v], self.succ[v], self.k[v]
        self.alive.discard(v)
        return [p, *succ]

    def _series(self, v: int) -> list:
        # The longest chain through v in which each node's only successor
        # has it as its only predecessor.
        if v < 0 or v in self.pinned:
            return []
        before: list = []
        u = v
        while len(self.pred[u]) == 1:
            (p,) = self.pred[u]
            if p < 0 or len(self.succ[p]) != 1 or p in self.pinned:
                break
            before.append(p)
            u = p
        chain = before[::-1] + [v]
        w = v
        while len(self.succ[w]) == 1:
            (x,) = self.succ[w]
            if x < 0 or len(self.pred[x]) != 1 or x in self.pinned:
                break
            chain.append(x)
            w = x
        if len(chain) < 2:
            return []
        pred, succ = self.pred[chain[0]], self.succ[chain[-1]]
        m = self._module(SERIES, chain)
        return self._replace(chain, m, pred, succ, self.k[chain[0]])

    def _parallel(self, v: int) -> list:
        # The nodes sharing v's predecessors, successors and k, when each of
        # the successors needs just one working predecessor.
        if v < 0:
            return []
        pred, succ, k = self.pred[v], self.succ[v], self.k[v]
        if not all(self.k[w] == 1 for w in succ):
            return []
        group = [
            u
            for u in self.succ[next(iter(pred))]
            if u >= 0
            and u not in self.pinned
            and self.k[u] == k
            and self.pred[u] == pred
            and self.succ[u] == succ
        ]
        if len(group) < 2:
            return []
        m = self._module(PARALLEL, group)
        return self._replace(group, m, pred, succ, k)

    def _koon(self, w: int) -> list:
        # All of w's predecessors, when w needs k >= 2 of them and they
        # share their predecessors and k and feed only w.
        if self.k[w] < 2:
            return []
        # (The input cannot be a member: it has no predecessors to share.)
        members, k = list(self.pred[w]), self.k[w]
        pred, member_k = self.pred[members[0]], self.k[members[0]]
        if not all(
            u not in self.pinned
            and self.succ[u] == {w}
            and self.k[u] == member_k
            and self.pred[u] == pred
            for u in members
        ):
            return []
        # Needing all of them works as a series chain would, but carries
        # their sum, not the least of them (see FlowGraph): the tree makes
        # it a series module (see _tree).
        m = self._module(KOON, members, k)
        changed = self._replace(members, m, pred, {w}, member_k)
        self.k[w] = 1
        return changed

    def flow_graph(self, aliases: Dict[Hashable, Hashable]) -> FlowGraph:
        """The reduced diagram as the capacity analysis needs it (see
        :class:`FlowGraph`), its vertices in topological order."""
        terms = [
            (NODE, aliases.get(t[1], t[1])) if t[0] == NODE else t
            for t in self.terms
        ]
        waiting = {
            v: sum(1 for u in self.pred[v] if u != _SOURCE) for v in self.alive
        }
        ready = sorted(v for v, count in waiting.items() if count == 0)
        order: list = []
        while ready:
            v = ready.pop()
            order.append(v)
            for w in sorted(self.succ[v], reverse=True):
                if w >= 0:
                    waiting[w] -= 1
                    if waiting[w] == 0:
                        ready.append(w)
        return FlowGraph(
            terms,
            order,
            [tuple(sorted(self.pred[v])) for v in order],
            [self.k[v] for v in order],
            tuple(sorted(self.pred[_SINK])),
            self.k[_SINK],
        )

    def core_graph(self) -> RBDGraph:
        """The reduced diagram, with each vertex's k."""
        graph = RBDGraph()
        for v in (_SOURCE, _SINK, *sorted(self.alive)):
            graph.add_node(v)
            graph.nodes[v]["k"] = self.k[v]
        for v, successors in self.succ.items():
            for w in successors:
                graph.add_edge(v, w)
        return graph


def _tree(terms: list, roots: Iterable[int]) -> tuple[list, Dict[int, int]]:
    """The terms under ``roots``, renumbered in post-order, and each kept
    term's new position."""
    position: Dict[int, int] = {}
    tree: list = []
    stack = [(v, False) for v in reversed(list(roots))]
    while stack:
        # Each term has one parent, so each is reached once.
        v, expanded = stack.pop()
        term = terms[v]
        if term[0] == NODE:
            position[v] = len(tree)
            tree.append(term)
        elif expanded:
            children = tuple(position[c] for c in term[1])
            position[v] = len(tree)
            if term[0] == KOON and term[2] == len(children):
                # All k of them: whether it works is a series chain's.
                tree.append((SERIES, children))
            else:
                tree.append((term[0], children, *term[2:]))
        else:
            stack.append((v, True))
            stack.extend((c, False) for c in reversed(term[1]))
    return tree, position


def _from_path_sets(
    terms: list,
    path_sets: Sequence[set],
    ends: tuple,
    aliases: Optional[Dict[Hashable, Hashable]] = None,
) -> Decomposition:
    """The decomposition whose core has the given minimal path sets (over
    term positions, including the input and output ``ends``).

    With ``aliases`` (``{node: component}`` for the nodes that stand for a
    component drawn in several places), every appearance of a component
    becomes one term, named after the component, and the path sets are
    made minimal again: a component is one random variable, wherever it is
    drawn."""
    if not path_sets:
        raise ValueError(NO_PATHS)
    core = [frozenset(ps) - set(ends) for ps in path_sets]
    if aliases:
        terms = list(terms)
        appearing = set().union(*core)
        canonical: Dict[Hashable, int] = {}
        rename: Dict[int, int] = {}
        for v in sorted(appearing):
            if terms[v][0] != NODE:
                continue
            node = terms[v][1]
            component = aliases.get(node, node)
            if component not in canonical:
                canonical[component] = v
                terms[v] = (NODE, component)
            else:
                rename[v] = canonical[component]
        core = _minimal_sets(
            [frozenset(rename.get(v, v) for v in ps) for ps in core]
        )
    if frozenset() in core:
        # A path set of the input and output alone: always works.
        return Decomposition([])
    used = sorted(set().union(*core))
    tree, position = _tree(terms, used)
    return Decomposition(
        tree, core=[sorted(position[c] for c in ps) for ps in core]
    )


def decompose(
    graph: RBDGraph,
    input_node,
    output_node,
    reduce: bool = True,
    aliases: Optional[Dict[Hashable, Hashable]] = None,
    core: Optional[str] = None,
) -> Decomposition:
    """The modular decomposition of the RBD diagram ``graph``.

    With ``reduce=False`` no module is formed: the core is the whole
    diagram, with the minimal path sets found by the memoised search (for
    structures that are not valid RBDs, whose semantics are those of that
    search).

    ``core`` is how what the reduction leaves is decided: ``"paths"``,
    ``"bdd"`` or ``"auto"`` (see ``CORE_METHOD``, the default).

    ``aliases`` maps each node that stands for a component drawn in more
    than one place (a repeated node) to that component. Every appearance
    of such a component stays out of the modules, and the core treats all
    its appearances as one component, named after it.

    Raises
    ------
    ValueError
        If no set of working nodes can reach the output node, or ``core``
        is unknown.
    """
    method = CORE_METHOD if core is None else core
    if method not in ("paths", "bdd", "auto"):
        raise ValueError(
            f"core must be 'paths', 'bdd' or 'auto', got {method!r}."
        )
    if not reduce:
        terms = [(NODE, n) for n in graph.nodes]
        position = {n: i for i, (_, n) in enumerate(terms)}
        found = find_min_path_sets(
            rbd_graph=graph, curr_node=output_node, solns={}
        )
        ends = tuple(
            position[n] for n in (input_node, output_node) if n in position
        )
        return _from_path_sets(
            terms, [{position[n] for n in ps} for ps in found], ends, aliases
        )

    aliases = aliases or {}
    shared = set(aliases) | set(aliases.values())
    reduction = _Reduction(graph, input_node, output_node, pinned=shared)
    reduction.run()
    flow = reduction.flow_graph(aliases)
    alive = reduction.alive
    if not alive:
        # Only a direct edge from the input to the output is left.
        return Decomposition([], flow=flow)
    if len(alive) == 1:
        # One module is left, fed by the input alone. It feeds the output,
        # which may also have a direct edge from the input: then the output
        # needs both (with one, the module would have been bypassed).
        (v,) = alive
        terms = list(reduction.terms)
        if terms[v][0] == NODE:
            # One appearance of a component is left: it is the system.
            terms[v] = (NODE, aliases.get(terms[v][1], terms[v][1]))
        tree, position = _tree(terms, [v])
        return Decomposition(tree, root=position[v], flow=flow)
    if method == "auto":
        many = _path_count(reduction) > AUTO_PATHS
        method = "bdd" if many else "paths"
    if method == "bdd":
        decomposition = _from_graph(reduction, aliases)
    else:
        path_sets = find_min_path_sets(
            rbd_graph=reduction.core_graph(), curr_node=_SINK, solns={}
        )
        decomposition = _from_path_sets(
            reduction.terms, path_sets, (_SOURCE, _SINK), aliases
        )
    decomposition.flow = flow
    return decomposition


def _path_count(reduction: "_Reduction", cap: int = 10**15) -> int:
    """At least as many as the minimal path sets of what ``reduction``
    leaves (capped at ``cap``): the ways of reaching each vertex, from its
    predecessors' in topological order, a vertex that needs ``k`` of them
    combining ``k`` of their ways (the elementary symmetric sum of degree
    ``k``). Each path set is one of these ways; a way may hold another, or
    a component twice, so there may be fewer."""
    waiting = {
        v: sum(1 for u in reduction.pred[v] if u != _SOURCE)
        for v in reduction.alive
    }
    ready = [v for v, n in waiting.items() if n == 0]
    ways: Dict[int, int] = {_SOURCE: 1}

    def combined(v: int) -> int:
        k = reduction.k[v]
        sums = [1] + [0] * k
        for u in reduction.pred[v]:
            x = ways.get(u, 0)
            for j in range(k, 0, -1):
                sums[j] = min(sums[j] + sums[j - 1] * x, cap)
        return sums[k]

    while ready:
        v = ready.pop()
        ways[v] = combined(v)
        for w in reduction.succ[v]:
            if w in waiting:
                waiting[w] -= 1
                if waiting[w] == 0:
                    ready.append(w)
    return combined(_SINK)


def _from_graph(
    reduction: "_Reduction", aliases: Dict[Hashable, Hashable]
) -> Decomposition:
    """The decomposition whose core is the binary decision diagram of what
    ``reduction`` leaves, built from its graph (see ``bdd.py``). Every
    appearance of a component drawn in several places stands for one
    variable, named after the component, as in ``_from_path_sets``."""
    terms = list(reduction.terms)
    alive = sorted(reduction.alive)
    canonical: Dict[Hashable, int] = {}
    variable: Dict[int, int] = {}
    for v in alive:
        if terms[v][0] != NODE:
            variable[v] = v
            continue
        node = terms[v][1]
        component = aliases.get(node, node)
        if component not in canonical:
            canonical[component] = v
            terms[v] = (NODE, component)
        variable[v] = canonical[component]
    sequence = bdd.order(alive, reduction.pred, reduction.succ, _SOURCE, _SINK)
    steps, root = bdd.build(
        sequence,
        reduction.pred,
        reduction.succ,
        reduction.k,
        _SOURCE,
        _SINK,
        variable,
    )
    if root == bdd.FAIL:
        raise ValueError(NO_PATHS)
    if root == bdd.WORK:
        return Decomposition([])
    tree, position = _tree(terms, sorted({p for p, _, _ in steps}))
    return Decomposition(
        tree, plan=([(position[p], a, i) for p, a, i in steps], root)
    )
