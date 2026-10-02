"""Reduced ordered binary decision diagrams, for phased missions (#142).

A diagram is a node of ``OrderedBDD``: a variable (an integer, smaller ones
decided first), the diagram when it is false (``low``) and when it is true
(``high``), or one of the two constants, ``FALSE`` and ``TRUE``. Equal
nodes are made once (the unique table), and a node whose two branches are
equal is never made, so every function has one diagram for the order (it
is canonical). ``ite(f, g, h)``, "if f then g else h", combines diagrams,
each combination worked out once (the computed table); conjunction,
disjunction and an at-least-k of several diagrams are built on it.
Everything iterates, with no recursion, so the order may hold any number
of variables.
"""

from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

#: The two constant diagrams.
FALSE, TRUE = 0, 1


class OrderedBDD:
    """The nodes of reduced ordered binary decision diagrams over integer
    variables, smaller ones first.

    Parameters
    ----------
    limit : int, optional
        The most nodes made; one more raises ``NotImplementedError`` with
        ``message``.
    message : str, optional
        What that error says.
    """

    def __init__(
        self, limit: Optional[int] = None, message: str = "Too many nodes."
    ):
        # Each node's variable and branches; the constants' variable comes
        # after every other.
        self.var: List[float] = [float("inf"), float("inf")]
        self.low: List[int] = [FALSE, TRUE]
        self.high: List[int] = [FALSE, TRUE]
        self._unique: Dict[Tuple[float, int, int], int] = {}
        self._computed: Dict[Tuple[int, int, int], int] = {}
        self._limit = limit
        self._message = message

    def __len__(self) -> int:
        return len(self.var)

    def node(self, variable: int, low: int, high: int) -> int:
        """The diagram that is ``high`` when ``variable`` is true and
        ``low`` when it is false (``variable`` must come before both's)."""
        if low == high:
            return low
        key = (variable, low, high)
        found = self._unique.get(key)
        if found is not None:
            return found
        if self._limit is not None and len(self.var) >= self._limit:
            raise NotImplementedError(self._message)
        self.var.append(variable)
        self.low.append(low)
        self.high.append(high)
        made = self._unique[key] = len(self.var) - 1
        return made

    def ite(self, f: int, g: int, h: int) -> int:
        """If ``f`` then ``g`` else ``h``."""
        results: List[int] = []
        work: List[tuple] = [(f, g, h, None)]
        var, low, high = self.var, self.low, self.high
        while work:
            f, g, h, key = work.pop()
            if key is not None:
                # Both branches are done: make the node.
                otherwise, then = results.pop(), results.pop()
                made = self.node(f, otherwise, then)
                self._computed[key] = made
                results.append(made)
                continue
            if f == TRUE or g == h:
                results.append(g)
                continue
            if f == FALSE:
                results.append(h)
                continue
            if g == TRUE and h == FALSE:
                results.append(f)
                continue
            key = (f, g, h)
            known = self._computed.get(key)
            if known is not None:
                results.append(known)
                continue
            top = min(var[f], var[g], var[h])
            f1, f0 = (high[f], low[f]) if var[f] == top else (f, f)
            g1, g0 = (high[g], low[g]) if var[g] == top else (g, g)
            h1, h0 = (high[h], low[h]) if var[h] == top else (h, h)
            # The node is made once both branches are: the true one is
            # worked out first, so its result lies under the false one's.
            work.append((top, 0, 0, key))
            work.append((f0, g0, h0, None))
            work.append((f1, g1, h1, None))
        return results.pop()

    def conjunction(self, diagrams: Sequence[int]) -> int:
        """All of ``diagrams``."""
        out = TRUE
        for d in diagrams:
            out = self.ite(out, d, FALSE)
        return out

    def disjunction(self, diagrams: Sequence[int]) -> int:
        """Any of ``diagrams``."""
        out = FALSE
        for d in diagrams:
            out = self.ite(out, TRUE, d)
        return out

    def at_least(self, diagrams: Sequence[int], k: int) -> int:
        """At least ``k`` of ``diagrams``: the first true and ``k - 1`` of
        the rest, or it false and ``k`` of the rest."""
        n = len(diagrams)
        # after[m]: at least m of the diagrams from the i-th on.
        after = [TRUE] + [FALSE] * k
        for i in range(n - 1, -1, -1):
            now = [TRUE]
            for m in range(1, k + 1):
                if m > n - i:
                    now.append(FALSE)
                else:
                    now.append(self.ite(diagrams[i], after[m - 1], after[m]))
            after = now
        return after[k]

    def plan(
        self, roots: Sequence[int], pivot: Callable[[int], Hashable]
    ) -> Tuple[list, List[int]]:
        """The diagrams ``roots`` as one plan in the format of
        ``shannon._shannon_plan``'s: step ``i`` fills value slot ``i + 2``
        from its pivot's true and false branches (``pivot`` names each
        variable's), slots 0 and 1 being ``FALSE`` and ``TRUE``; and each
        root's slot."""
        slot: Dict[int, int] = {FALSE: 0, TRUE: 1}
        steps: List[tuple] = []
        for root in roots:
            stack = [root]
            while stack:
                n = stack[-1]
                if n in slot:
                    stack.pop()
                    continue
                pending = [
                    c for c in (self.high[n], self.low[n]) if c not in slot
                ]
                if pending:
                    stack.extend(pending)
                    continue
                stack.pop()
                steps.append(
                    (
                        pivot(int(self.var[n])),
                        slot[self.high[n]],
                        slot[self.low[n]],
                    )
                )
                slot[n] = len(steps) + 1
        return steps, [slot[r] for r in roots]
