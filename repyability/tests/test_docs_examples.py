"""The documentation's examples run, and the numbers they quote are right.

Every ```python block of each page in ``docs/`` (and the README) is run in
order, in one namespace per page, as a reader pasting them would. A statement
whose last line ends in a comment of the form ``# -> <number>`` is evaluated,
and its value must equal the quoted number to the precision it is quoted at
(``0.9091`` to four decimals, ``2884.2`` to one). Blocks indented inside lists
run too. Tracebacks point at the page and line of the failing example.
"""

import ast
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGES = sorted((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md"]
BLOCK = re.compile(r"```python\n(.*?)```", re.S)
QUOTED = re.compile(r"#\s*->\s*(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)")


def tolerance(token: str) -> float:
    """Half a unit in the last quoted digit (``0.9091`` -> 5e-5)."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    scale = 10.0 ** int(exponent) if exponent else 1.0
    return 0.5 * 10.0 ** (-decimals) * scale * (1 + 1e-9)


def examples(text: str):
    """Each top-level statement of each python block, in page order: its
    first line in the page, its AST node (line numbers set to the page's),
    and the number its last line quotes, if any."""
    for block in BLOCK.finditer(text):
        code = textwrap.dedent(block.group(1))
        offset = text.count("\n", 0, block.start()) + 1
        lines = code.splitlines()
        for node in ast.parse(code).body:
            quoted = QUOTED.search(lines[node.end_lineno - 1])
            ast.increment_lineno(node, offset)
            yield node.lineno, node, quoted.group(1) if quoted else None


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
        got = float(np.asarray(value, dtype=float).reshape(-1)[0])
        assert abs(got - float(quoted)) <= tolerance(quoted), (
            f"{page.relative_to(ROOT)}:{line} gives {got!r}, but the page "
            f"quotes {quoted}"
        )


def test_every_page_is_covered():
    # Guard against the page list silently going empty (e.g. a moved docs/).
    if not (ROOT / "docs").exists():
        pytest.skip("the documentation sources are not available")
    assert len(PAGES) > 10
    quoted = sum(
        1
        for page in PAGES
        for _, _, number in examples(page.read_text())
        if number is not None
    )
    assert quoted > 100
