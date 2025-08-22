from collections import abc

import numpy as np
from scipy import sparse

from harmonic_balance import freq
from harmonic_balance import arclength_continuation as alc

ndarray = np.ndarray
sparray = sparse.sparray
array = ndarray | sparray
# TODO: Fix the use of these type annotations.


def get_rel_error(R: sparray, z: ndarray) -> float:
    """Compute the relative error |R| / |z|.

    Parameters
    ----------
    R
        Frequency domain residual Az + f_nl - b_ext
    z
        Solution

    Returns
    -------
    rel_error
        Relative error |R| / |z|
    """
    return np.linalg.norm(R) / np.linalg.norm(z)


def get_initial_guess(
    A: sparray,
    b_ext: ndarray,
) -> ndarray:
    """Return solution to linear system as initial guess for nonlinear system.

    Parameters
    ----------
    A
        Matrix describing linear dynamics in frequency domain (see `get_A`)
        shape (n * (2 NH + 1), n * (2 NH + 1))
    b_ext
        Exponential Fourier coefficients of external force (see `get_b_ext`)
        shape (n * (2 NH + 1),)

    Returns
    -------
    z
        Solution to linear system Az = b_ext
    """
    return freq.solve_linear_system(A, b_ext)


def solve_nonlinear(
    omega: float,
    z0: ndarray,
    A: sparray,
    b_ext: ndarray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> ndarray:
    """Solve the nonlinear system Az + b = b_ext.

    Parameters
    ----------
    omega
        Fundamental frequency
    z0
        Initial guess for solution
    A
        Matrix describing linear dynamics in frequency domain (see `get_A`)
        shape (n * (2 NH + 1), n * (2 NH + 1))
    b_ext
        Exponential Fourier coefficients of external force (see `get_b_ext`)
        shape (n * (2 NH + 1),)
    f_nl
        Nonlinear force function in time domain
        [(n * N,), (n * N,), int] -> (n * N,)
    f_nl_dx
        Derivative of f_nl with respect to x
        [(n * N,), (n * N,), int] -> (n * N, n * N)
    f_nl_d_xdot
        Derivative of f_nl with respect to x'
        [(n * N,), (n * N,), int] -> (n * N, n * N)
    NH
        Assumed highest harmonic index
    n
        Number of degrees of freedom
    N
        Number of points to sample in time domain
    tol
        Tolerance that relative error |R| / |z| must reach
    max_iter
        Maximum number of allowed iterations

    Returns
    -------
    z
        Solution to nonlinear system Az + b = b_ext
    R
        Frequency domain residual Az + f_nl - b_ext
    converged
        True if relative error is less than tol within max_iter iterations
    i
        Number of iterations
    """
    R = alc.get_R(z0, omega, A, f_nl, b_ext, NH, n, N)
    if get_rel_error(R, z0) < tol:
        return z0, R, True, 0

    z = z0.copy()
    converged = False
    for i in range(max_iter):
        z += _solve_nonlinear_step(
            omega, z, A, b_ext, f_nl, df_nl_dx, df_nl_d_xdot, NH, n, N
        )
        R = alc.get_R(z, omega, A, f_nl, b_ext, NH, n, N)

        if get_rel_error(R, z) < tol:
            converged = True
            break

    return z, R, converged, i + 1 if "i" in locals() else 0


def _solve_nonlinear_step(
    omega: float,
    z: ndarray,
    A: sparray,
    b_ext: ndarray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
) -> ndarray:
    """

    Returns
    -------
    step
        The step to add to z to solve the nonlinear system Az + b = b_ext
    """
    db_dz = alc.get_db_dz(omega, z, df_nl_dx, df_nl_d_xdot, NH, n, N)

    R = alc.get_R(z, omega, A, f_nl, b_ext, NH, n, N)
    dR_dz = alc.get_dR_dz(A, db_dz)

    return np.linalg.solve(dR_dz, -R)
