"""Collection settings for the repository's tests."""

import importlib.util

# The compiled simulation loop, decision diagram and random streams' Philox
# import numba, an optional dependency: without it the modules cannot be
# imported, so their docstrings are not collected.
collect_ignore = (
    []
    if importlib.util.find_spec("numba")
    else [
        "repyability/rbd/_kernel.py",
        "repyability/rbd/_bdd_kernel.py",
        "repyability/rbd/_philox_kernel.py",
    ]
)
