"""Single source of truth for the package version.

Kept deliberately free of imports so the build backend can read
``__version__`` statically (without importing the package and its
dependencies) via ``[tool.setuptools.dynamic]`` in ``pyproject.toml``.
"""

__version__ = "0.5.0"
