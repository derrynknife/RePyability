"""Node models for a component that never fails or has always failed."""

import numpy as np


class PerfectReliability:
    """A node model that never fails.

    Its survival function is 1 at every time, its failure probability 0,
    and a simulated failure time is infinite. A
    [`NonRepairableRBD`][repyability.NonRepairableRBD] uses it for the
    input and output nodes and for nodes forced working (``working_nodes``),
    and it can be given as a node's model, e.g. for a connection that
    cannot fail. Pass the class itself, not an instance: its methods are
    class methods, and the RBD recognises it (as time-invariant, and when
    serialising) by identity.

    Examples
    --------
    A perfectly reliable node in series changes nothing:

    >>> from surpyval import FixedEventProbability
    >>> from repyability import NonRepairableRBD, PerfectReliability
    >>> rbd = NonRepairableRBD(
    ...     [("s", "a"), ("a", "b"), ("b", "t")],
    ...     {
    ...         "a": FixedEventProbability.from_params(0.1),
    ...         "b": PerfectReliability,
    ...     },
    ... )
    >>> round(rbd.sf(), 4)
    0.9
    >>> PerfectReliability.sf([10.0, 20.0]).tolist()
    [1.0, 1.0]
    """

    @classmethod
    def sf(cls, x):
        """Return the survival function (reliability): 1 at every time.

        Parameters
        ----------
        x : array_like
            Time(s); only the shape is used.

        Returns
        -------
        np.ndarray
            Ones (floats) with the shape of ``x``: a 0-d array for a scalar
            ``x``.
        """
        return np.ones_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        """Return the failure probability (unreliability): 0 at every time.

        Parameters
        ----------
        x : array_like
            Time(s); only the shape is used.

        Returns
        -------
        np.ndarray
            Zeros (floats) with the shape of ``x``: a 0-d array for a scalar
            ``x``.
        """
        return np.zeros_like(x).astype(float)

    @classmethod
    def cs(cls, x, X):
        """Return the conditional survival: 1 for any further time.

        The probability of surviving a further ``x`` given survival to age
        ``X``, which is 1 because the node never fails.

        Parameters
        ----------
        x : array_like
            The further time(s); only the shape is used.
        X : array_like
            The age(s) survived to; not used.

        Returns
        -------
        np.ndarray
            Ones (floats) with the shape of ``np.atleast_1d(x)``.
        """
        # Always working -> conditional survival is always 1.
        return np.ones_like(np.atleast_1d(x)).astype(float)

    @classmethod
    def random(cls, size):
        """Draw failure times: all infinite, as the node never fails.

        Parameters
        ----------
        size : int or tuple of int
            The shape of the output.

        Returns
        -------
        np.ndarray
            An array of ``inf`` with shape ``size``.
        """
        return np.ones(size) * np.inf


class PerfectUnreliability:
    """A node model that has always failed.

    Its survival function is 0 at every time, its failure probability 1,
    and a simulated failure time is 0. A
    [`NonRepairableRBD`][repyability.NonRepairableRBD] uses it for nodes
    forced failed (``broken_nodes``) and for failed components in
    condition-based evaluation (a [`NodeState`][repyability.NodeState] with
    ``alive=False``), and it can be given as a node's model, e.g. to study
    the system with a component missing. Pass the class itself, not an
    instance: its methods are class methods, and the RBD recognises it (as
    time-invariant, and when serialising) by identity.

    Examples
    --------
    With one of two parallel nodes failed, the other carries the system:

    >>> from surpyval import FixedEventProbability
    >>> from repyability import NonRepairableRBD, PerfectUnreliability
    >>> rbd = NonRepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {
    ...         "a": FixedEventProbability.from_params(0.1),
    ...         "b": PerfectUnreliability,
    ...     },
    ... )
    >>> round(rbd.sf(), 4)
    0.9
    >>> PerfectUnreliability.sf([10.0, 20.0]).tolist()
    [0.0, 0.0]
    """

    @classmethod
    def sf(cls, x):
        """Return the survival function (reliability): 0 at every time.

        Parameters
        ----------
        x : array_like
            Time(s); only the shape is used.

        Returns
        -------
        np.ndarray
            Zeros (floats) with the shape of ``x``: a 0-d array for a scalar
            ``x``.
        """
        return np.zeros_like(x).astype(float)

    @classmethod
    def ff(cls, x):
        """Return the failure probability (unreliability): 1 at every time.

        Parameters
        ----------
        x : array_like
            Time(s); only the shape is used.

        Returns
        -------
        np.ndarray
            Ones (floats) with the shape of ``x``: a 0-d array for a scalar
            ``x``.
        """
        return np.ones_like(x).astype(float)

    @classmethod
    def cs(cls, x, X):
        """Return the conditional survival: 0 for any further time.

        Parameters
        ----------
        x : array_like
            The further time(s); only the shape is used.
        X : array_like
            The age(s) survived to; not used.

        Returns
        -------
        np.ndarray
            Zeros (floats) with the shape of ``np.atleast_1d(x)``.
        """
        # Never working -> conditional survival is always 0.
        return np.zeros_like(np.atleast_1d(x)).astype(float)

    @classmethod
    def random(cls, size):
        """Draw failure times: all 0, as the node has always failed.

        Parameters
        ----------
        size : int or tuple of int
            The shape of the output.

        Returns
        -------
        np.ndarray
            An array of zeros with shape ``size``.
        """
        return np.zeros(size)
