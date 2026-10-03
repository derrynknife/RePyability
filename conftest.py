"""Collection settings for the repository's tests."""

import importlib.util

# The compiled simulation loop and decision diagram import numba, an
# optional dependency: without it the modules cannot be imported, so their
# docstrings are not collected.
collect_ignore = (
    []
    if importlib.util.find_spec("numba")
    else ["repyability/rbd/_kernel.py", "repyability/rbd/_bdd_kernel.py"]
)
