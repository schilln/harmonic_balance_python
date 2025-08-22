from collections import abc

import numpy as np
from scipy import sparse

ndarray = np.ndarray
sparray = sparse.sparray


def extract_dofs_freq(
    coefficients: ndarray | sparray, n: int
) -> ndarray | sparray:
    """Reshape the coefficients by degree of freedom.

    Parameters
    ----------
    coefficients
        Should be of the form [a_{-NH}, a_{-1}, a0, a1, ..., aNH] where ak
        denotes the coefficients of the kth harmonic for the n degrees of
        freedom.
        shape (n * (2 NH + 1),)
    n
        Number of degrees of freedom

    Returns
    -------
    reshaped
        Coefficients where each row corresponds to a degree of freedom
        shape (n, 2 NH + 1)
    """
    result = coefficients.reshape((n, -1), order="F")
    if isinstance(result, sparse.coo_array):
        return result.tocsr()
    else:
        return result


def combine_dofs_freq(coefficients: ndarray | sparray) -> ndarray | sparray:
    """Flatten coefficients from `extract_dofs_freq`.

    Parameters
    ----------
    coefficients
        Coefficients where each row corresponds to a degree of freedom
        shape (n, 2 NH + 1)

    Returns
    -------
    reshaped
        Of the form [a_{-NH}, a_{-1}, a0, a1, ..., aNH] where ak denotes the
        coefficients of the kth harmonic for the n degrees of freedom.
        shape (n * (2 NH + 1),)
    """
    result = coefficients.reshape((-1,), order="F")
    if isinstance(result, sparse.coo_array):
        return result.tocsr()
    else:
        return result


def extract_dofs_time(time: ndarray | sparray, n: int) -> ndarray | sparray:
    """Reshape time signal by degree of freedom.

    Parameters
    ----------
    time
        Should be of the form [x0, x1, ..., x(n-1)] where xi denotes the
        values of the ith degree of freedom for the N time points.
    n
        Number of degrees of freedom

    Returns
    -------
    reshaped
        Time signal where each row corresponds to a degree of freedom
        shape (n, N)
    """
    result = time.reshape((n, -1), order="C")
    if isinstance(result, sparse.coo_array):
        return result.tocsr()
    else:
        return result


def combine_dofs_time(time: ndarray | sparray) -> ndarray | sparray:
    """Flatten time signal from `extract_dofs_time`.

    Parameters
    ----------
    time
        Time signal where each row corresponds to a degree of freedom
        shape (n, N)

    Returns
    -------
    reshaped
        Of the form [x0, x1, ..., x(n-1)] where xi denotes the
        values of the ith degree of freedom for the N time points.
    """
    result = time.reshape((-1), order="C")
    if isinstance(result, sparse.coo_array):
        return result.tocsr()
    else:
        return result


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


def get_f_ext(
    n: int,
    ks: abc.Iterable[int],
    dofs: abc.Iterable[int],
    is_cosines: abc.Iterable[bool],
    coefficients: abc.Iterable[float],
) -> abc.Callable[[float, float], ndarray]:
    """Return a function defining the external force in the time domain.

    See `freq.get_b_ext`.

    If an index is specified more than once (e.g., for both a cosine and sine
    coefficient), all corresponding coefficients are applied via addition.

    Parameters
    ----------
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
    f_ext
        Callable accepting fundamental frequency `omega` and time `t` and
        returning the external force for the n degrees of freedom in an ndarray
        of shape (n,)
    """

    def f_ext(omega, t):
        res = np.zeros(n)
        for k, dof, is_cosine, coefficient in zip(
            ks, dofs, is_cosines, coefficients
        ):
            if k == 0:
                res[dof] += coefficient
            elif is_cosine:
                res[dof] += coefficient * np.cos(k * omega * t)
            else:
                res[dof] += coefficient * np.sin(k * omega * t)
        return res

    return f_ext


def get_time_residual(
    xx: ndarray,
    xxp: ndarray,
    xxpp: ndarray,
    M: ndarray,
    C: ndarray,
    K: ndarray,
    f_ext: abc.Callable[[float, float], ndarray],
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    omega: float,
    tls: ndarray,
    n: int,
):
    """Get the residual in the time domain.

    Parameters
    ----------
    xx
        Time signal
        shape (n, N)
    xxp
        Time derivative of time signal
        shape (n, N)
    xxpp
        Second time derivative of time signal
        shape (n, N)
    M
        Mass matrix
        shape (n, n)
    C
        Damping matrix
        shape (n, n)
    K
        Stiffness matrix
        shape (n, n)
    f_ext
        External force function in time domain
        See `get_f_ext`
    f_nl
        Nonlinear force function in time domain
        [(n * N,), (n * N,), int] -> (n * N,)
    omega
        Fundamental frequency
    tls
        Time values at which to evaluate external force
        shape (N,)
    n
        Number of degrees of freedom

    Returns
    -------
    residual
        Residual R = Mx'' + Cx' + Kx + f_nl(x, x') - f_ext(omega, t)
        shape (n, N)
    """

    return (
        M @ xxpp
        + C @ xxp
        + K @ xx
        + extract_dofs_time(f_nl(xx.ravel(), xxp.ravel(), xx.shape[1]), n)
        - np.stack([f_ext(omega, t) for t in tls]).T
    )


def _max_abs_np(a, axis: int | tuple[int] | None = None) -> ndarray:
    if not isinstance(a, ndarray):
        raise ValueError("`a` is not a np.ndarray")

    if a.dtype != complex:
        a_max = a.max(axis=axis)
        a_min = a.min(axis=axis)
        return np.where(-a_min > a_max, a_min, a_max)
    else:
        return abs(a).max(axis=axis)


def _max_abs_sp(a, axis: int | None = None) -> sparray:
    if not isinstance(a, sparray):
        raise ValueError("`a` is not a scipy.sparse.sparray")

    return abs(a).max(axis=axis)
