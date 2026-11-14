"""The documentation's examples run, and the values they quote are right.

Every ```python block of each page in ``docs/`` (and the README) is run in
order, in one namespace per page, as a reader pasting them would. A statement
whose last line ends in a comment of the form ``# -> <number>`` is evaluated,
and its value must equal the quoted number to the precision it is quoted at
(``0.9091`` to four decimals, ``2884.2`` to one). ``# ~> <number>`` quotes a
simulated value that varies with the simulator (surpyval's, for imperfect
repair), and must hold to within 10%. A comment that opens with a Python
literal quotes the value too (#223): ``# True`` or ``# False``, a string
(``# 'exact'``), or a tuple or list (``# (0.9396, 0.9596)``), whose numbers
hold to the precision quoted (``15.06...`` to the digits shown) and whose
other items exactly; what follows the literal is the page's remark. Blocks
indented inside lists run too. Tracebacks point at the page and line of the
failing example.
"""

import ast
import io
import re
import textwrap
import tokenize

import numpy as np
import pytest

from repyability.tests.repository import CHECKOUT, ROOT

PAGES = (
    sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
    if CHECKOUT
    else []
)
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
QUOTED = re.compile(r"#\s*(->|~>)\s*(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)")
#: A comment opening with a literal: True, False, a string, a tuple or a list.
LITERAL = re.compile(r"#\s*(True\b|False\b|'|\"|\(|\[)")
#: A number quoted to the digits shown, ``15.06...``.
TRUNCATED = re.compile(r"(\d+\.\d+)\.\.\.")

#: The relative tolerance of a ``~>`` (about) quote.
ABOUT = 0.10


def tolerance(token: str) -> float:
    """Half a unit in the last quoted digit (``0.9091`` -> 5e-5)."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    scale = 10.0 ** int(exponent) if exponent else 1.0
    return 0.5 * 10.0 ** (-decimals) * scale * (1 + 1e-9)


class Number:
    """A quoted number (``# -> 0.9091``), to its precision."""

    def __init__(self, token: str, about: bool = False):
        self.token = token
        self.allowed = tolerance(token)
        if about:
            self.allowed = max(self.allowed, ABOUT * abs(float(token)))

    def holds(self, value) -> bool:
        got = float(np.asarray(value, dtype=float).reshape(-1)[0])
        return abs(got - float(self.token)) <= self.allowed

    def __str__(self) -> str:
        return self.token


class Literal:
    """A quoted literal (``# True``, ``# 'exact'``, ``# (0.93, 0.96)``):
    its numbers to the precision quoted, its other items exactly."""

    def __init__(self, source: str, tree: ast.expr, truncated: set):
        self.source, self.tree, self.truncated = source, tree, truncated

    def holds(self, value) -> bool:
        return self._matches(value, self.tree)

    def _matches(self, value, node) -> bool:
        if isinstance(node, (ast.Tuple, ast.List)):
            if isinstance(value, (str, bytes, dict)) or not hasattr(
                value, "__len__"
            ):
                return False
            items = list(value)
            return len(items) == len(node.elts) and all(
                self._matches(item, element)
                for item, element in zip(items, node.elts)
            )
        constant = node.operand if isinstance(node, ast.UnaryOp) else node
        assert isinstance(constant, ast.Constant)
        quoted = constant.value
        if isinstance(quoted, bool):
            flag = np.asarray(value)
            return (
                flag.dtype == np.bool_
                and flag.size == 1
                and (bool(flag.reshape(-1)[0]) is quoted)
            )
        if isinstance(quoted, str):
            return isinstance(value, str) and value == quoted
        if isinstance(value, (bool, np.bool_, str)):
            return False
        token = ast.get_source_segment(self.source, node)
        assert token is not None
        allowed = tolerance(token.lstrip("-"))
        if constant.col_offset in self.truncated:
            allowed *= 2.0  # the digits shown, not rounded
        number = float(token)
        try:
            return abs(float(value) - number) <= allowed
        except (TypeError, ValueError):
            return False

    def __str__(self) -> str:
        return self.source


def literal(comment: str):
    """The literal ``comment`` opens with, or None: the comment is a
    remark."""
    start = LITERAL.search(comment)
    if start is None:
        return None
    text = comment[start.start(1) :]
    truncated: set = set()
    cleaned, last = [], 0
    for match in TRUNCATED.finditer(text):
        cleaned.append(text[last : match.start()])
        truncated.add(sum(map(len, cleaned)))
        cleaned.append(match.group(1))
        last = match.end()
    text = "".join(cleaned) + text[last:]
    # The shortest opening of the comment that parses as a literal.
    for end in range(1, len(text) + 1):
        if text[end - 1] not in ")]'\"e":
            continue
        source = text[:end]
        try:
            tree = ast.parse(source, mode="eval").body
            ast.literal_eval(tree)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(tree, (ast.Constant, ast.Tuple, ast.List)):
            return None
        if isinstance(tree, ast.Constant) and not isinstance(
            tree.value, (bool, str)
        ):
            return None
        if not all(
            isinstance(node, (ast.Constant, ast.Tuple, ast.List, ast.UnaryOp))
            or isinstance(node, (ast.USub, ast.Load))
            for node in ast.walk(tree)
        ):
            return None
        return Literal(source, tree, truncated)
    return None


def examples(text: str):
    """Each top-level statement of each python block, in page order: its
    first line in the page, its AST node (line numbers set to the page's),
    and what its last line quotes, if anything (a ``Number`` or a
    ``Literal``)."""
    for block in BLOCK.finditer(text):
        code = textwrap.dedent(block.group(1))
        offset = text.count("\n", 0, block.start()) + 1
        lines = code.splitlines()
        for node in ast.parse(code).body:
            last = lines[node.end_lineno - 1]
            quoted = QUOTED.search(last)
            ast.increment_lineno(node, offset)
            if quoted is not None:
                arrow, number = quoted.groups()
                yield node.lineno, node, Number(number, arrow == "~>")
                continue
            comment = comment_of(last)
            check = None if comment is None else literal(comment)
            yield node.lineno, node, check


def comment_of(line: str):
    """The comment ending ``line``, or None (a ``#`` in a string is not
    one)."""
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(line).readline):
            tokens.append(token)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    for token in tokens:
        if token.type == tokenize.COMMENT:
            return token.string
    return None


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_examples_run_and_quoted_numbers_hold(page):
    if not page.exists():
        pytest.skip("the documentation sources are not available")
    filename = str(page)
    namespace: dict = {"__name__": "__docs__"}
    for line, node, quoted in examples(page.read_text()):
        if quoted is None or not isinstance(node, ast.Expr):
            module = ast.Module(body=[node], type_ignores=[])
            exec(compile(module, filename, "exec"), namespace)
            continue
        expression = ast.Expression(node.value)
        value = eval(compile(expression, filename, "eval"), namespace)
        assert quoted.holds(value), (
            f"{page.relative_to(ROOT)}:{line} gives {value!r}, but the page "
            f"quotes {quoted}"
        )


def test_every_page_is_covered():
    # Guard against the page list silently going empty (e.g. a moved docs/).
    # An installed copy has no docs: its ROOT is site-packages, which may
    # hold another package's docs/ (#167).
    if not CHECKOUT:
        pytest.skip("the documentation sources are not available")
    assert len(PAGES) > 10
    quoted = [
        check
        for page in PAGES
        for _, _, check in examples(page.read_text())
        if check is not None
    ]
    assert sum(isinstance(check, Number) for check in quoted) > 100
    assert sum(isinstance(check, Literal) for check in quoted) > 50


def test_literals_are_read_as_quoted():
    assert literal("# True: exact") is not None
    assert literal("# True").holds(np.bool_(True))
    assert not literal("# True").holds(1.0)
    assert not literal("# False").holds(True)
    assert literal("# 'exact'").holds("exact")
    assert not literal("# 'exact'").holds("simulated")
    pair = literal("# (0.9396, 0.9596)   the 5th to 95th percentile")
    assert pair.holds((0.93962, 0.95955))
    assert not pair.holds((0.9397, 0.9596))
    assert not pair.holds((0.9396,))
    assert literal("# (15.06..., False): down at 15.06").holds(
        (15.0699, False)
    )
    assert literal("# (128, 8)   128 units").holds((128, 8))
    assert literal("# [['a', 'c'], ['b']]").holds([["a", "c"], ["b"]])
    assert literal("# ('pumps',)").holds(("pumps",))
    # Remarks are not quotes.
    for remark in ("# (the default)", "# a remark", "# (a, b)", "# (1)"):
        assert literal(remark) is None
