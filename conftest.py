"""Collection settings for the repository's tests."""

import importlib.util

# The compiled simulation loop, the compiled decision diagram, and the
# compiled sorting of a run's changes (#201) import numba, an optional
# dependency: without it the modules cannot be imported, so their
# docstrings are not collected.
collect_ignore = (
    []
    if importlib.util.find_spec("numba")
    else [
        "repyability/rbd/_kernel.py",
        "repyability/rbd/_bdd_kernel.py",
        "repyability/rbd/_time_order.py",
    ]
)
