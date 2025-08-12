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
        Should be of the form (a0, a1, ..., aNH) where ak denotes the
        coefficients of the kth harmonic for the n degrees of freedom.
    n
        Number of degrees of freedom

    Returns
    -------
    reshaped
        Coefficients where each row corresponds to a degree of freedom
        shape (n, NH + 1)
    """
    return coefficients.reshape((n, -1), order="F")


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
        return abs(a).max(axis=axis)


def _max_abs_sp(a, axis: int | None = None) -> sparray:
    if not isinstance(a, sparray):
        raise ValueError("`a` is not a scipy.sparse.sparray")

    return abs(a).max(axis=axis)
