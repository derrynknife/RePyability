"""Collection settings for the repository's tests."""

import importlib.util

# The compiled simulation loop imports numba, an optional dependency: without
# it the module cannot be imported, so its docstrings are not collected.
collect_ignore = (
    [] if importlib.util.find_spec("numba") else ["repyability/rbd/_kernel.py"]
)
