"""``_philox.uniforms`` compiled (numba): ``_philox.philox``, compiled as it
is written, run a counter at a time, so it takes no temporary arrays. The
same function, so the same bits.

Importing this module imports numba, which compiles it on first use (or
loads it from numba's cache).
"""

import types

import numpy as np
from numba import njit

from repyability.rbd import _philox

# ``philox`` calls ``_mulhilo``: compiled, with the compiled one.
_mulhilo = njit(cache=True)(_philox._mulhilo)
_rounds = njit(cache=True)(
    types.FunctionType(
        _philox.philox.__code__, {**vars(_philox), "_mulhilo": _mulhilo}
    )
)


@njit(cache=True, nogil=True)
def _fill(k0, k1, sims, first, rows, out):
    """``_philox.uniforms`` into ``out``, a counter at a time."""
    zero = np.uint64(0)
    for j in range(sims.size):
        for q in range(first // 4, (first + rows - 1) // 4 + 1):
            words = _rounds(np.uint64(q + 1), sims[j], zero, zero, k0, k1)
            for i in range(4):
                k = 4 * q + i - first
                if 0 <= k < rows:
                    out[j, k] = (words[i] >> _philox._TOP) * _philox._SCALE


def uniforms(key, simulations, first: int, rows: int) -> np.ndarray:
    """``_philox.uniforms``, compiled."""
    sims = np.asarray(simulations, dtype=np.uint64)
    out = np.empty((sims.size, rows))
    if rows:
        _fill(np.uint64(key[0]), np.uint64(key[1]), sims, first, rows, out)
    return out
