"""RePyability uses surpyval's public interface only.

surpyval's deprecation policy ("What is public") promises the top-level
names, a list of subpackages (with their ``__all__`` where they have one)
and ``surpyval.univariate.regression.kinds``; anything else, and any name
starting with an underscore, can change in any release without a warning.
This scans RePyability's source, tests and docstring examples for an
import from surpyval, or an attribute of the ``surpyval`` module, outside
that, so that one cannot creep back in unnoticed.
"""

import ast
import importlib
import pathlib
import re

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1]

#: surpyval's public modules (its deprecation policy's list).
PUBLIC = {
    "surpyval",
    "surpyval.recurrent",
    "surpyval.degradation",
    "surpyval.multivariate",
    "surpyval.ml",
    "surpyval.life_models",
    "surpyval.metrics",
    "surpyval.diagnostics",
    "surpyval.bayes",
    "surpyval.prior",
    "surpyval.test_planning",
    "surpyval.datasets",
    "surpyval.pymc",
    "surpyval.utils",
    "surpyval.univariate.parametric",
    "surpyval.univariate.nonparametric",
    "surpyval.univariate.regression",
    "surpyval.univariate.competing_risks",
    "surpyval.univariate.regression.kinds",
}

#: Names RePyability needs that surpyval has no public home for yet, each
#: with why. Remove an entry once surpyval gives the name a public home.
NO_PUBLIC_HOME = {
    # A joint model's prediction for one unit, which a diagram takes as a
    # block (``rbd/joint_node.py``).
    ("surpyval.univariate.joint", "DynamicPrediction"),
    # A lifetime's life left at a virtual age, from a uniform: what
    # ``_event_loop._aged_life`` draws for any model it has no closed form
    # for (and the tests check ``MixtureLife`` against).
    ("surpyval.recurrent.renewal.renewal_model", "conditional_gaps"),
}

#: The names RePyability's code binds the surpyval module to.
ALIASES = {"surpyval", "surv"}

DOCTEST_IMPORT = re.compile(
    r"^\s*(?:>>>|\.\.\.)\s*from\s+(surpyval[\w.]*)\s+import\s+(.+)$"
)


def _sources():
    return sorted(PACKAGE.rglob("*.py"))


def _all(module: str):
    """``module``'s ``__all__``, or None where it has none or cannot be
    imported here (``surpyval.pymc`` needs PyMC)."""
    try:
        return getattr(importlib.import_module(module), "__all__", None)
    except ImportError:
        return None


def _private(name: str) -> bool:
    return name.startswith("_") and not (
        name.startswith("__") and name.endswith("__")
    )


def _promised(module: str, name: str) -> bool:
    """Whether ``from module import name`` is surpyval's public
    interface: ``module`` on the public list, and ``name`` in its
    ``__all__`` where it has one (else not underscored), or a public
    module itself."""
    if (module, name) in NO_PUBLIC_HOME:
        return True
    if module not in PUBLIC or name.startswith("_"):
        return False
    if f"{module}.{name}" in PUBLIC:
        return True
    names = _all(module)
    return names is None or name in names


def _imports(path: pathlib.Path):
    """``(line, module, name)`` for each name ``path`` imports from
    surpyval (``name`` None for ``import surpyval.x``), in its code and
    its docstring examples."""
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            module = node.module or ""
            if module == "surpyval" or module.startswith("surpyval."):
                for alias in node.names:
                    yield node.lineno, module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("surpyval."):
                    yield node.lineno, alias.name, None
    for number, line in enumerate(text.splitlines(), 1):
        match = DOCTEST_IMPORT.match(line)
        if match:
            names = match.group(2).strip("() \\")
            for name in names.split(","):
                name = name.split(" as ")[0].strip()
                if name:
                    yield number, match.group(1), name


def _private_attributes(path: pathlib.Path):
    """``(line, dotted)`` for each attribute chain from the surpyval
    module (``surv.x._y``) that reaches an underscored name."""
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Attribute):
            continue
        chain, base = [node.attr], node.value
        while isinstance(base, ast.Attribute):
            chain.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name) and base.id in ALIASES:
            if any(_private(part) for part in chain):
                yield node.lineno, ".".join([base.id, *reversed(chain)])


def test_only_public_surpyval_is_imported():
    found = []
    for path in _sources():
        for line, module, name in _imports(path):
            where = f"{path.relative_to(PACKAGE.parent)}:{line}"
            if name is None:
                if module not in PUBLIC:
                    found.append(f"{where}: import {module}")
            elif not _promised(module, name):
                found.append(f"{where}: from {module} import {name}")
    assert not found, (
        "Imports from outside surpyval's public interface (see its "
        "deprecation policy, 'What is public'):\n" + "\n".join(found)
    )


def test_no_private_surpyval_attribute_is_used():
    found = [
        f"{path.relative_to(PACKAGE.parent)}:{line}: {dotted}"
        for path in _sources()
        for line, dotted in _private_attributes(path)
    ]
    assert not found, "Private surpyval names used:\n" + "\n".join(found)


@pytest.mark.parametrize("module, name", sorted(NO_PUBLIC_HOME))
def test_the_exceptions_still_have_no_public_home(module, name):
    # Once surpyval makes one public, take it from there and drop it from
    # NO_PUBLIC_HOME.
    import surpyval

    assert name not in surpyval.__all__
    for public in PUBLIC:
        assert name not in (_all(public) or ()), f"{name} is in {public} now"


def test_the_check_catches_a_private_import(tmp_path):
    source = tmp_path / "bad.py"
    source.write_text(
        "from surpyval.univariate.regression._kinds import PH\n"
        "from surpyval.degradation.process_models import (\n"
        "    WienerProcessModel,\n"
        ")\n"
        "import surpyval as surv\n"
        "surv.utils._private\n"
        '"""\n>>> from surpyval.univariate.joint import JointModel\n"""\n'
    )
    imports = [
        (module, name)
        for _, module, name in _imports(source)
        if not _promised(module, name)
    ]
    assert imports == [
        ("surpyval.univariate.regression._kinds", "PH"),
        ("surpyval.degradation.process_models", "WienerProcessModel"),
        ("surpyval.univariate.joint", "JointModel"),
    ]
    assert [d for _, d in _private_attributes(source)] == [
        "surv.utils._private"
    ]
