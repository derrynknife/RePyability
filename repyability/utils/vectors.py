"""Vector arithmetic that keeps off threaded BLAS.

OpenBLAS splits a dot product of more than about 10,000 terms across its
threads, which costs milliseconds a call (thousands of times one thread's
microseconds) while the threads wake, the more so on a busy machine. A loop
of such products, as the renewal sums make, then runs hundreds of times
slower. ``dot`` sums the products in numpy's own loop instead, about as
fast as one thread of BLAS at any length."""

import numpy as np


def dot(a, b) -> float:
    """``a @ b`` for two vectors of one length, off BLAS (see the module
    docstring)."""
    return float(np.einsum("i,i->", a, b))
