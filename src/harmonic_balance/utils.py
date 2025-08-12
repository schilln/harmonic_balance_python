from collections import abc

import numpy as np
import scipy

ndarray = np.ndarray
sparray = scipy.sparse.sparray


FFT_NORM = "forward"


def get_block(
    k: int, omega: float, M: ndarray, C: ndarray, K: ndarray
) -> ndarray:
    """Construct block for matrix defining linear dynamics in frequency domain.


    Parameters
    ----------
    k
        Frequency index / integer multiple of the fundamental frequency for
        the block
    omega
        Fundamental frequency
    M
        Mass matrix
        shape (n, n)
    C
        Damping matrix
        shape (n, n)
    K
        Stiffness matrix
        shape (n, n)

    Returns
    -------
    block
        kth diagonal block of frequency-domain linear dynamics matrix A
        shape (n, n)
    """
    return -k * omega**2 * M + 1j * k * omega * C + K


def get_A(NH: int, omega: float, M: ndarray, C: ndarray, K: ndarray) -> sparray:
    """Construct matrix defining linear dynamics in frequency domain.

    Parameters
    ----------
    NH
        Assumed highest harmonic index
    omega
        Fundamental frequency
    M
        Mass matrix
    C
        Damping matrix
    K
        Stiffness matrix

    Returns
    -------
    A
        Frequency-domain linear dynamics matrix
        shape (n * (NH + 1), n * (NH + 1))
    """
    return scipy.sparse.block_diag(
        [get_block(k, omega, M, C, K) for k in range(0, NH + 1)]
    ).tocsr()


def get_b_ext(
    NH: int,
    n: int,
    ks: abc.Iterable[int],
    dofs: abc.Iterable[int],
    is_cosines: abc.Iterable[bool],
    coefficients: abc.Iterable[float],
) -> sparray:
    """Return the exponential Fourier coefficients of the external force given
    cosine and sine coefficients.

    If an index is specified more than once (e.g., for both a cosine and sine
    coefficient), all corresponding coefficients are applied via addition.

    Parameters
    ----------
    NH
        Assumed highest harmonic index
    n
        Number of degrees of freedom
    ks
        The harmonic indices corresponding to each coefficient in `coefficients`
    dofs
        Degree of freedom indices corresponding to each coefficient in
        `coefficients`
    is_cosines
        Whether each corresponding coefficient in `coefficients` is cosine
    coefficients
        Coefficients of cosine and/or sine in the external force (or constant
        for k = 0)

    Returns
    -------
    b_ext
        Frequency coefficients of external force
        shape (n * (NH + 1),)

        b_ext = [c0, c1, ..., cNH]
        ck = [c_k0, c_k1, ..., c_k(n-1)]
        c_ki is the frequency coefficient for the kth harmonic of the ith degree
            of freedom

    Raises
    ------
    ValueError
        If lengths of `ks`, `dofs`, `is_cosines`, and `coefficients` are not
        equal
    ValueError
        If k > NH for any k in `ks`
    ValueError
        If dof >= n for any dof in `dofs`
    """
    length = len(ks)
    if (
        length != len(dofs)
        or length != len(is_cosines)
        or length != len(coefficients)
    ):
        raise ValueError(
            "`ks`, `dofs`, `is_cosines`, `coefficients` do not all have the"
            " same length"
        )

    if any(k > NH for k in ks):
        raise ValueError(
            "At least one specified harmonic index is greater than NH."
        )
    if any(dof >= n for dof in dofs):
        raise ValueError(
            "At least one specified degree of freedom is greater than or equal"
            " to N."
        )

    if not isinstance(ks, ndarray):
        ks = np.array(ks, dtype=int)
    if not isinstance(dofs, ndarray):
        dofs = np.array(dofs, dtype=int)
    if not isinstance(is_cosines, ndarray):
        is_cosines = np.array(is_cosines, dtype=bool)

    k_neq_0_mask = ks != 0
    exp_coefficients = np.array(coefficients, dtype=complex)
    exp_coefficients[k_neq_0_mask & ~is_cosines] *= -1j
    exp_coefficients[k_neq_0_mask] /= 2

    total_length = n * (NH + 1)
    indices = n * ks + dofs

    return scipy.sparse.csc_array(
        (exp_coefficients, (indices, np.zeros(length))),
        shape=(total_length, 1),
        dtype=complex,
    )


def extract_dofs(
    coefficients: ndarray | sparray, NH: int, n: int
) -> ndarray | sparray:
    """Reshape the coefficients by degree of freedom.

    Parameters
    ----------
    coefficients
        Should be of the form (a0, a1, ..., aNH) where ak denotes the
        coefficients of the kth harmonic for the n degrees of freedom.

    Returns
    -------
    reshaped
        Coefficients where each row corresponds to a degree of freedom
        shape (n, NH + 1)
    """
    return coefficients.reshape((n, NH + 1), order="F")


def max_abs(
    a: ndarray | sparray, axis: int | tuple[int] | None = None
) -> ndarray | sparray:
    """Find the maximum absolute value(s) of an array.

    Parameters
    ----------
    a
        Array in which to find maximum absolute value(s)
    axis
        The axis(es) along which to find the max(es)
        A tuple of ints is supported only if `a` is a np.ndarray.
    """
    if isinstance(a, ndarray):
        return _max_abs_np(a, axis)
    elif isinstance(a, sparray):
        return _max_abs_sp(a, axis)
    else:
        raise NotImplementedError(
            "`a` must be one of np.ndarray or scipy.sparse.sparray"
        )


def _max_abs_np(a, axis: int | tuple[int] | None = None) -> ndarray:
    if not isinstance(a, ndarray):
        raise ValueError("`a` is not a np.ndarray")

    if a.dtype != complex:
        a_max = a.max(axis=axis)
        a_min = a.min(axis=axis)
        return np.where(-a_min > a_max, a_min, a_max)
    else:
        a_abs = abs(a)
        return np.argmax(a_abs, axis=axis)


def _max_abs_sp(a, axis: int | None = None) -> sparray:
    if not isinstance(a, sparray):
        raise ValueError("`a` is not a scipy.sparse.sparray")

    return abs(a).max(axis=axis)
