"""The uniforms of a run's random streams (#209), counter-based: the
``k``-th uniform of simulation ``r`` in a stream with key ``key`` is the
``k``-th of numpy's ``Generator(Philox(key=key, counter=(0, r, 0, 0)))
.random()``, a function of the key, ``r`` and ``k`` alone.

Philox4x64-10 (Salmon et al., 2011) turns a counter of four 64-bit words
and a key of two into four 64-bit outputs, so uniform ``k`` of simulation
``r`` is output ``k % 4`` of the counter ``(k // 4 + 1, r, 0, 0)`` (numpy
steps the counter before each four), its top 53 bits times ``2**-53``.
``philox`` is the function, written once: ``uniforms`` runs it on numpy's
arrays, and ``_philox_kernel`` compiles it with numba, an output at a
time, where numba is installed. The two give the same bits, which are
numpy's own (``test_streams.py`` checks both against numpy's ``Philox``).
"""

import numpy as np

_U = np.uint64
_LOW, _HALF = _U(0xFFFFFFFF), _U(32)
_M0, _M1 = _U(0xD2E7470EE14C6C93), _U(0xCA5A826395121157)
_W0, _W1 = _U(0x9E3779B97F4A7C15), _U(0xBB67AE8584CAA73B)
_TOP = _U(11)
#: ``2**-53``: an output's top 53 bits times it is a uniform in [0, 1).
_SCALE = 1.0 / 9007199254740992.0


def _mulhilo(a, m):
    """The high and low 64 bits of the 128-bit product ``a * m``, from
    32-bit halves (numpy has no 128-bit product)."""
    mh, ml = m >> _HALF, m & _LOW
    ah, al = a >> _HALF, a & _LOW
    ll, lh, hl = al * ml, al * mh, ah * ml
    middle = (ll >> _HALF) + (lh & _LOW) + (hl & _LOW)
    return ah * mh + (lh >> _HALF) + (hl >> _HALF) + (middle >> _HALF), a * m


def philox(c0, c1, c2, c3, k0, k1):
    """Philox4x64-10 of the counter ``(c0, c1, c2, c3)`` with the key
    ``(k0, k1)``: its four outputs. Words are ``np.uint64`` (numbers or
    arrays), whose sums and products wrap."""
    for i in range(10):
        if i:
            k0 = k0 + _W0
            k1 = k1 + _W1
        h0, l0 = _mulhilo(c0, _M0)
        h1, l1 = _mulhilo(c2, _M1)
        c0, c1, c2, c3 = h1 ^ c1 ^ k0, l1, h0 ^ c3 ^ k1, l0
    return c0, c1, c2, c3


def uniforms(key, simulations, first: int, rows: int) -> np.ndarray:
    """Uniforms ``first`` to ``first + rows`` of each of ``simulations``
    (their ``r``, as ``np.uint64``) in the stream with ``key`` (two
    ``np.uint64``): a row per simulation."""
    sims = np.asarray(simulations, dtype=_U)
    if not rows or not sims.size:
        return np.empty((sims.size, rows))
    start, stop = first // 4, (first + rows - 1) // 4 + 1
    counter = np.arange(start + 1, stop + 1, dtype=_U)
    c0 = np.broadcast_to(counter, (sims.size, counter.size)).ravel()
    c1 = np.repeat(sims, counter.size)
    zero = np.zeros_like(c0)
    with np.errstate(over="ignore"):
        words = philox(c0, c1, zero, zero, _U(key[0]), _U(key[1]))
    out = np.stack(words, axis=1).reshape(sims.size, 4 * counter.size)
    skip = first - 4 * start
    return (out[:, skip : skip + rows] >> _TOP) * _SCALE  # noqa: E203
