"""The logic of a fault tree with NOT gates (a non-coherent tree).

A NOT gate occurs while its input has not occurred, so an event's
occurring can make the top event *less* likely: the tree is not coherent,
and the exact engine of the coherent trees (``rbd/modular.py``), which
works on series, parallel and k-out-of-n blocks, does not apply. Such a
tree is evaluated here through its reduced ordered binary decision diagram
(a ROBDD), built from the gates by Bryant's ``ite`` (if-then-else)
operation, with complemented inputs where a NOT gate stands:

- the top event probability, and the probability that it has not occurred,
  each a sum of products of the events' probabilities along the diagram's
  paths (both kept in their own right, so a small one keeps its precision);
- whether the top event occurs for a given set of occurred events;
- the prime implicants (the non-coherent counterpart of the minimal cut
  sets: the smallest conjunctions of events occurring *and not occurring*
  that make the top event occur), by the recursion of Morreale and of
  Coudert and Madre over the diagram:
  ``PI(f) = PI(f0 f1) + x PI(f1) \\ PI(f0 f1) + x' PI(f0) \\ PI(f0 f1)``,
  ``x`` the diagram's top variable and ``f1``, ``f0`` its two branches.

The object built here stands in for a coherent tree's decomposition where
``FaultTree`` evaluates the top event (``probabilities`` and ``works``), so
the top event probability, the conditional probabilities and the Birnbaum
importance are worked out by the same code for both.
"""

from typing import Collection, Dict, Hashable, List, Mapping, Tuple

import numpy as np

#: The terminals of a diagram: the function that is never true (0) and the
#: one that always is (1).
FALSE, TRUE = 0, 1


class Diagram:
    """A reduced ordered binary decision diagram over variables ``0, 1,
    ...`` (a lower number nearer the root), its nodes numbered from 2
    (0 and 1 are the terminals). ``limit`` bounds its nodes, past which
    ``NotImplementedError(message)`` is raised."""

    def __init__(self, limit: int, message: str):
        self._nodes: List[Tuple[int, int, int]] = [(-1, 0, 0), (-1, 1, 1)]
        self._unique: Dict[Tuple[int, int, int], int] = {}
        self._ite: Dict[Tuple[int, int, int], int] = {}
        self._limit, self._message = limit, message

    def variable(self, node: int) -> int:
        """The variable a node decides (a terminal's is ``-1``)."""
        return self._nodes[node][0]

    def branches(self, node: int) -> Tuple[int, int]:
        """A node's branches: where it goes when its variable is false,
        and when it is true."""
        _, low, high = self._nodes[node]
        return low, high

    def node(self, variable: int, low: int, high: int) -> int:
        """The node deciding ``variable``, going to ``low`` when it is
        false and ``high`` when it is true (``low`` itself if the two are
        one)."""
        if low == high:
            return low
        key = (variable, low, high)
        found = self._unique.get(key)
        if found is None:
            found = len(self._nodes)
            if found > self._limit:
                raise NotImplementedError(self._message)
            self._nodes.append(key)
            self._unique[key] = found
        return found

    def _top_variable(self, *nodes: int) -> int:
        decided = [self.variable(n) for n in nodes if n > TRUE]
        return min(decided)

    def _cofactors(self, node: int, variable: int) -> Tuple[int, int]:
        if node > TRUE and self.variable(node) == variable:
            return self.branches(node)
        return node, node

    def ite(self, f: int, g: int, h: int) -> int:
        """If ``f`` then ``g`` else ``h``, reduced."""
        if f == TRUE:
            return g
        if f == FALSE:
            return h
        if g == h:
            return g
        if g == TRUE and h == FALSE:
            return f
        key = (f, g, h)
        found = self._ite.get(key)
        if found is not None:
            return found
        v = self._top_variable(f, g, h)
        f0, f1 = self._cofactors(f, v)
        g0, g1 = self._cofactors(g, v)
        h0, h1 = self._cofactors(h, v)
        found = self.node(v, self.ite(f0, g0, h0), self.ite(f1, g1, h1))
        self._ite[key] = found
        return found

    def negation(self, f: int) -> int:
        return self.ite(f, FALSE, TRUE)

    def conjunction(self, f: int, g: int) -> int:
        return self.ite(f, g, FALSE)

    def disjunction(self, f: int, g: int) -> int:
        return self.ite(f, TRUE, g)

    def at_least(self, inputs: List[int], k: int) -> int:
        """True when at least ``k`` of ``inputs`` are."""
        memo: Dict[Tuple[int, int], int] = {}

        def needs(i: int, k: int) -> int:
            if k <= 0:
                return TRUE
            if len(inputs) - i < k:
                return FALSE
            key = (i, k)
            if key not in memo:
                memo[key] = self.ite(
                    inputs[i], needs(i + 1, k - 1), needs(i + 1, k)
                )
            return memo[key]

        return needs(0, k)


class NonCoherent:
    """The top event of a static tree with NOT gates, as its decision
    diagram: built from the tree's gates (``{name: (kind, k, inputs)}``
    for the kinds ``or``, ``and``, ``vote`` and ``not``), the events
    decided in the order a walk down from the top first meets them."""

    def __init__(
        self,
        gates: Mapping[Hashable, Tuple[str, int, tuple]],
        events: Collection,
        top: Hashable,
        post_order: list,
        limit: int,
    ):
        order: List[Hashable] = []
        seen, stack = set(), [top]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            if x in events:
                order.append(x)
                continue
            stack.extend(reversed(gates[x][2]))
        self.events = order
        self._index = {e: i for i, e in enumerate(order)}
        self._diagram = Diagram(
            limit,
            "The fault tree's NOT gates and repeated events tie too much of "
            "it together to work it out exactly (its decision diagram has "
            f"more than {limit:,} nodes).",
        )
        d = self._diagram
        made: Dict[Hashable, int] = {}
        for x in post_order:
            if x in events:
                made[x] = d.node(self._index[x], FALSE, TRUE)
                continue
            kind, k, inputs = gates[x]
            parts = [made[c] for c in inputs]
            if kind == "not":
                made[x] = d.negation(parts[0])
            elif k == 1:
                out = FALSE
                for part in parts:
                    out = d.disjunction(out, part)
                made[x] = out
            elif k == len(parts):
                out = TRUE
                for part in parts:
                    out = d.conjunction(out, part)
                made[x] = out
            else:
                made[x] = d.at_least(parts, k)
        self.root = made[top]
        self._implicants: Dict[int, frozenset] = {}

    # -- the decomposition's interface (see FaultTree) ---------------------

    def probabilities(
        self,
        p: Mapping,
        q: Mapping,
        shape=None,
        works: bool = True,
        fails: bool = True,
    ) -> tuple:
        """The probabilities that the top event has not occurred and that
        it has, from each event's probabilities of not having occurred
        (``p``) and of having occurred (``q``): each a sum over the
        diagram's paths to its terminal, so neither is one less the
        other."""
        d = self._diagram
        size = 1 if shape is None else shape
        values: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
            FALSE: (np.ones(size), np.zeros(size)),
            TRUE: (np.zeros(size), np.ones(size)),
        }
        stack = [self.root]
        while stack:
            n = stack[-1]
            if n in values:
                stack.pop()
                continue
            low, high = d.branches(n)
            pending = [c for c in (low, high) if c not in values]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            e = self.events[d.variable(n)]
            p_e = np.asarray(p[e], dtype=float)
            q_e = np.asarray(q[e], dtype=float)
            values[n] = (
                p_e * values[low][0] + q_e * values[high][0],
                p_e * values[low][1] + q_e * values[high][1],
            )
        not_occurred, occurred = values[self.root]
        if shape is None:
            return float(not_occurred[0]), float(occurred[0])
        return not_occurred, occurred

    def works(self, status: Mapping) -> bool:
        """Whether the top event has not occurred, given whether each
        event has not occurred (truthy) or has (falsy)."""
        d, n = self._diagram, self.root
        while n > TRUE:
            low, high = d.branches(n)
            n = low if status[self.events[d.variable(n)]] else high
        return n == FALSE

    # -- prime implicants --------------------------------------------------

    def prime_implicants(self) -> List[Tuple[frozenset, frozenset]]:
        """The prime implicants of the top event: each the events that
        occur and the events that do not, in a smallest conjunction of
        them that makes the top event occur."""
        found = self._prime(self.root)
        out = []
        for implicant in found:
            occurred = frozenset(
                self.events[v] for v, value in implicant if value
            )
            not_occurred = frozenset(
                self.events[v] for v, value in implicant if not value
            )
            out.append((occurred, not_occurred))
        return out

    def _prime(self, node: int) -> frozenset:
        """The prime implicants of the function at ``node``, as sets of
        ``(variable, value)`` literals (Coudert and Madre's recursion)."""
        if node == FALSE:
            return frozenset()
        if node == TRUE:
            return frozenset({frozenset()})
        found = self._implicants.get(node)
        if found is not None:
            return found
        d = self._diagram
        v = d.variable(node)
        low, high = d.branches(node)
        both = self._prime(d.conjunction(low, high))
        found = (
            both
            | frozenset(p | {(v, True)} for p in self._prime(high) - both)
            | frozenset(p | {(v, False)} for p in self._prime(low) - both)
        )
        self._implicants[node] = found
        return found
