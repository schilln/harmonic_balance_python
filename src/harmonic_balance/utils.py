from collections import abc

import numpy as np
import scipy

ndarray = np.ndarray


FFT_NORM = "forward"


def get_block(k: int, omega: float, M: ndarray, C: ndarray, K: ndarray):
    return -k * omega**2 * M + 1j * k * omega * C + K


def get_A(NH: int, omega: float, M: ndarray, C: ndarray, K: ndarray):
    return scipy.sparse.block_diag(
        [get_block(k, omega, M, C, K) for k in range(0, NH + 1)]
    ).tocsr()


def get_b_ext(
    NH: int,
    N: int,
    ks: abc.Iterable[int],
    dofs: abc.Iterable[int],
    is_cosines: abc.Iterable[bool],
    coefficients: abc.Iterable[float],
):
    """Return the exponential Fourier coefficients of the external force given
    cosine and sine coefficients.

    If an index is specified more than once (e.g., for both a cosine and sine
    coefficient), all corresponding coefficients are applied via addition.

    Parameters
    ----------
    NH
        The number of assumed harmonics, i.e., 0, 1, ..., N_H
    N
        The number of degrees of freedom in the system
    ks
        The harmonic indices corresponding to each coefficient in `coefficients`
    dofs
        The degree of freedom indices corresponding to each coefficient in
        `coefficients`
    is_cosines
        Whether each corresponding coefficient in `coefficients` is cosine
    coefficients
        The coefficients of cosine and/or sine in the external force
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
    if any(dof >= N for dof in dofs):
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

    total_length = N * (NH + 1)
    indices = N * ks + dofs

    return scipy.sparse.csc_array(
        (exp_coefficients, (indices, np.zeros(length))),
        shape=(total_length, 1),
        dtype=complex,
    )


def extract_dofs(coefficients: ndarray, NH: int, N: int):
    """Reshape the coefficients by degree of freedom.

    Parameters
    ----------
    coefficients
        Should be of the form (a0, a1, ..., aNH) where ak denotes the
        coefficients of the 0th harmonic for all N degrees of freedom.

    Returns
    -------
    reshaped
        Coefficients where each row corresponds to a degree of freedom
        shape (N, NH + 1)
    """
    return np.reshape(coefficients, (N, NH + 1), order="F")
