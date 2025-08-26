from collections import abc
import traceback

import numpy as np
from scipy import sparse

from . import continuation, freq, solve

ndarray = np.ndarray
sparray = sparse.sparray
array = ndarray | sparray
# TODO: Fix the use of these type annotations.


def compute_nlfr_curve(
    num_points: int,
    omega_i0: float,
    b_ext: ndarray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
    s: float,
    M: ndarray,
    C: ndarray,
    K: ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    optimal_num_steps: int = 10,
    min_step_size: float = 1e-3,
    max_step_size: float = 5e-1,
) -> tuple[ndarray[complex], ndarray[float], ndarray[bool], ndarray[int]]:
    """Compute solutions along nonlinear frequency response (NLFR) curve for
    increasing values of fundamental forcing frequency omega.

    Parameters
    ----------
    num_points
        Number of points along solution curve to compute
    omega_i0
        First value of omega to solve
    b_ext
        Exponential Fourier coefficients of external force (see `get_b_ext`)
        shape (n * (NH + 1),)
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
    s
        Goal distance between y_i1 and y_i0
    M
        Mass matrix
        shape (n, n)
    C
        Damping matrix
        shape (n, n)
    K
        Stiffness matrix
        shape (n, n)
    tol
        Tolerance that relative error |rhs| / |y| must reach
    max_iter
        Maximum number of allowed correction iterations
    optimal_num_steps
        Optimal number of correction iterations
    min_step_size, max_step_size
        Minimum and maximum step size `s`

    Returns
    -------
    ys
        Sequence of solutions along NLFR curve
        shape (num_points, n * (NH + 1))
    rel_errors
        Sequence of relative errors
    convergeds
        Whether each iteration converged
    iters
        Number of iterations used to solve for each point on curve
    """
    ys = np.full((num_points, n * (NH + 1) + 1), np.inf, dtype=complex)
    rel_errors = np.full(num_points, np.inf)
    convergeds = np.full(num_points, False)
    iters = np.full(num_points, -1)

    omega = omega_i0
    for i in range(2):
        A = freq.get_A(omega, NH, M, C, K)
        z, rhs, convergeds[i], iters[i] = solve.solve_nonlinear(
            omega,
            freq.solve_linear_system(A, b_ext),
            A,
            b_ext,
            f_nl,
            df_nl_dx,
            df_nl_d_xdot,
            NH,
            n,
            N,
            tol,
            max_iter=max_iter,
        )
        ys[i] = np.concat((z, [omega]))
        rel_errors[i] = get_rel_error(rhs, ys[i])
        if not convergeds[i]:
            print(f"iteration {i:0>3} didn't converge")

        s = update_step_size(
            optimal_num_steps, iters[i], s, min_step_size, max_step_size
        )
        omega += s

    for i in range(2, num_points):
        y_k0 = predict_y(ys[i - 1], ys[i - 2], s)

        try:
            ys[i], rhs, convergeds[i], iters[i] = correct_y(
                y_k0,
                ys[i - 1],
                b_ext,
                f_nl,
                df_nl_dx,
                df_nl_d_xdot,
                NH,
                n,
                N,
                s,
                M,
                C,
                K,
                tol,
                max_iter,
            )
        except np.linalg.LinAlgError:
            print(traceback.format_exc())
            return ys, rel_errors, convergeds, iters

        rel_errors[i] = get_rel_error(rhs, ys[i])
        if not convergeds[i]:
            print(f"iteration {i:0>3} didn't converge")

        s = update_step_size(
            optimal_num_steps, iters[i], s, min_step_size, max_step_size
        )

    return ys, rel_errors, convergeds, iters


def update_step_size(
    optimal_num_steps: int,
    num_steps: int,
    s: float,
    min_step_size: float = 1e-3,
    max_step_size: float = 5e-1,
) -> float:
    """Compute the updated step size.

    Parameters
    ----------
    optimal_num_steps
        Optimal number of correction iterations
    num_steps
        Number of correction iterations used
    s
        Current step size
    min_step_size, max_step_size
        Minimum and maximum step size `s`

    Returns
    -------
    s_new
        New step size
    """
    s_new = s * compute_step_multiplier(optimal_num_steps, num_steps)
    if s_new < min_step_size or max_step_size < s_new:
        return s
    else:
        return s_new


def compute_step_multiplier(optimal_num_steps: int, num_steps: int) -> float:
    """Compute the factor by which to multiply the step size.

    Parameters
    ----------
    optimal_num_steps
        Optimal number of correction iterations
    num_steps
        Number of correction iterations used

    Returns
    -------
    multiplier
        Factor by which to multiply the step size, i.e., s_new = s * update
    """
    return 2 ** ((optimal_num_steps - num_steps) / optimal_num_steps)


def predict_y(
    y_i1: sparray | ndarray, y_i0: sparray | ndarray, s: float
) -> sparray | ndarray:
    """Given the previous two solutions (i = 0, 1), predict the next solution
    (i = 2).

    Parameters
    ----------
    y_i1, y_i0
        Previous solutions y_i = [z_i, omega_i], i = 0, 1
        shape (n(NH + 1) + 1,)
    s
        Step size

    Returns
    -------
    y_i2_k0
        Predicted solution y_ik, i = 2, k = 0
        shape (n(NH + 1) + 1,)
    """
    secant = y_i1 - y_i0
    direction = secant / np.linalg.norm(secant)
    y_i2_k0 = y_i1 + s * direction
    return y_i2_k0


def correct_y(
    y_i1_k0: sparray | ndarray,
    y_i0: sparray | ndarray,
    b_ext: ndarray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
    s: float,
    M: ndarray,
    C: ndarray,
    K: ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> tuple[ndarray, ndarray, bool, int]:
    """Compute the next solution y = [z, omega] to the nonlinear system
    A(omega)z + b = b_ext along the nonlinear frequency response curve.

    Parameters
    ----------
    y_i1_k0
        Predicted solution y_ik, i = 1, k = 0
    y_i0
        Previous solution along nonlinear frequency response curve y_i, i = 0.
    b_ext
        Exponential Fourier coefficients of external force (see `get_b_ext`)
        shape (n * (NH + 1),)
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
    s
        Goal distance between y_i1 and y_i0
    M
        Mass matrix
        shape (n, n)
    C
        Damping matrix
        shape (n, n)
    K
        Stiffness matrix
        shape (n, n)
    tol
        Tolerance that relative error |rhs| / |y| must reach
    max_iter
        Maximum number of allowed iterations

    Returns
    -------
    y
        Solution y = [z, omega] to nonlinear system A(omega)z + b_nl = b_ext
    rhs
        Residual rhs = [R, P]
    converged
        True if relative error is less than tol within max_iter iterations
    i
        Number of iterations
    """
    y = y_i1_k0.copy()

    omega, z = y[-1].real, y[:-1]
    A = freq.get_A(omega, NH, M, C, K)
    b_nl = solve.get_b_nl(z, omega, f_nl, NH, n, N)
    R = solve.get_R(z, A, b_nl, b_ext)
    P = get_P(y, y_i0, s)
    rhs = get_rhs(R, P)
    if get_rel_error(rhs, y) < tol:
        return y, rhs, True, 0

    converged = False
    for i in range(max_iter):
        A = freq.get_A(omega, NH, M, C, K)
        step = _solve_step(
            y,
            y_i0,
            A,
            b_ext,
            f_nl,
            df_nl_dx,
            df_nl_d_xdot,
            NH,
            n,
            N,
            s,
            M,
            C,
        )
        y[-1] += step[-1]
        y[:-1] += step[:-1]

        omega, z = y[-1].real, y[:-1]

        b_nl = solve.get_b_nl(z, omega, f_nl, NH, n, N)
        R = solve.get_R(z, A, b_nl, b_ext)
        P = get_P(y, y_i0, s)
        rhs = get_rhs(R, P)

        if get_rel_error(rhs, y) < tol:
            converged = True
            break

    return y, rhs, converged, i + 1 if "i" in locals() else 0


def get_rel_error(rhs: ndarray, y: ndarray) -> float:
    """Compute the relative error |rhs| / |y|.

    Parameters
    ----------
    rhs
        Residual rhs = [R, P]
    y
        Solution

    Returns
    -------
    rel_error
        Relative error |rhs| / |y|
    """
    return np.linalg.norm(rhs) / np.linalg.norm(y)


def get_P(
    y_i1: sparray | ndarray,
    y_i0: sparray | ndarray,
    s: float,
) -> float:
    """Compute deviation of distance between y_i1 and y_i0 from the step size s.

    Parameters
    ----------
    y_i1, y_i0
        Solutions y_i = [z_i, omega_i], i = 0, 1
        shape (n(NH + 1) + 1,)
    s
        Goal distance between y_i1 and y_i0

    Returns
    -------
    P
        Residual norm(y_i1 - y_i0)^2 - s^2
    """
    if isinstance(y_i1, ndarray) or isinstance(y_i0, ndarray):
        norm = np.linalg.norm
    else:
        norm = sparse.linalg.norm
    return norm(y_i1 - y_i0) ** 2 - s**2


def get_dP_dz(
    z_i1: sparray | ndarray,
    z_i0: sparray | ndarray,
) -> sparray | ndarray:
    return 2 * (z_i1 - z_i0)


def get_dP_d_omega(
    omega_i1: float,
    omega_i0: float,
) -> float:
    return 2 * (omega_i1 - omega_i0)


def get_rhs(R: sparray | ndarray, P: float) -> sparray | ndarray:
    return np.concat((R, [P]))


def _solve_step(
    y_i1: ndarray,
    y_i0: ndarray,
    A: sparray,
    b_ext: ndarray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
    s: float,
    M: ndarray,
    C: ndarray,
) -> ndarray:
    """

    Returns
    -------
    step
        The step to add to y_i1 = [z_i1, omega_i1] to get the (k+1)th correction
    """
    omega_i1, z_i1 = y_i1[-1].real, y_i1[:-1]
    omega_i0, z_i0 = y_i0[-1].real, y_i0[:-1]

    b_nl = solve.get_b_nl(z_i1, omega_i1, f_nl, NH, n, N)
    R = solve.get_R(z_i1, A, b_nl, b_ext)
    P = get_P(y_i1, y_i0, s)
    rhs = get_rhs(R, P)

    db_nl_dz = solve.get_db_nl_dz(
        omega_i1, z_i1, df_nl_dx, df_nl_d_xdot, NH, n, N
    )
    dR_dz = solve.get_dR_dz(A, db_nl_dz)
    dR_d_omega = continuation.get_dR_d_omega(
        z_i1, omega_i1, df_nl_d_xdot, NH, n, N, M, C
    )

    dP_dz = get_dP_dz(z_i1, z_i0)
    dP_d_omega = get_dP_d_omega(omega_i1, omega_i0)

    jacobian = np.block(
        [
            [dR_dz, dR_d_omega.reshape(-1, 1)],
            [dP_dz.reshape(1, -1), dP_d_omega.reshape(-1, 1)],
        ]
    )

    step = np.linalg.solve(jacobian, -rhs)
    step[-1] = step[-1].real
    return step
