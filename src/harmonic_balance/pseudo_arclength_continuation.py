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
    max_iter: int = 50,
    optimal_num_steps: int = 3,
    min_step_size: float = 5e-4,
    max_step_size: float = 5e-1,
    pred_tol: float = 1e-1,
    initial_max_iter: int = 100,
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
            max_iter=initial_max_iter,
        )
        ys[i] = np.concat((z, [omega]))
        b_nl = solve.get_b_nl(z, omega, f_nl, NH, n, N)
        rel_errors[i] = get_rel_error(rhs, b_nl + b_ext)
        if not convergeds[i]:
            print(f"{i: >3} didn't converge")

        s = update_step_size(
            optimal_num_steps, iters[i], s, min_step_size, max_step_size
        )
        omega += s

    # Estimate the first tangent vector with a secant vector.
    V_i0 = ys[1] - ys[0]
    V_i0 /= np.linalg.norm(V_i0)

    for i in range(2, num_points):
        omega, z = ys[i - 1, -1].real, ys[i - 1, :-1]
        A = freq.get_A(omega, NH, M, C, K)
        db_nl_dz = solve.get_db_nl_dz(
            omega, z, df_nl_dx, df_nl_d_xdot, NH, n, N
        )
        dR_dz = solve.get_dR_dz(A, db_nl_dz)
        dR_d_omega = continuation.get_dR_d_omega(
            z, omega, df_nl_d_xdot, NH, n, N, M, C
        )
        V_i1 = compute_tangent(dR_dz, dR_d_omega, V_i0)

        while True:
            y_k0 = predict_y(ys[i - 1], V_i1, s)

            omega, z = y_k0[-1].real, y_k0[:-1]
            A = freq.get_A(omega, NH, M, C, K)
            b_nl = solve.get_b_nl(z, omega, f_nl, NH, n, N)
            R = solve.get_R(z, A, b_nl, b_ext)

            if (pred_rel_error := get_rel_error(R, y_k0)) < pred_tol:
                break
            print(f"{i: >3} rel error (pred): {pred_rel_error:.2e}")

            s /= 2
            s = np.clip(s, min_step_size, max_step_size)
            if s == min_step_size or s == max_step_size:
                break

        try:
            ys[i], rhs, convergeds[i], iters[i] = correct_y(
                y_k0,
                V_i1,
                b_ext,
                f_nl,
                df_nl_dx,
                df_nl_d_xdot,
                NH,
                n,
                N,
                M,
                C,
                K,
                tol,
                max_iter,
            )
        except np.linalg.LinAlgError:
            print(traceback.format_exc())
            return ys, rel_errors, convergeds, iters

        rel_errors[i] = get_rel_error(rhs, b_nl + b_ext)
        if not convergeds[i]:
            print(f"{i: >3} didn't converge")

        s = update_step_size(
            optimal_num_steps, iters[i], s, min_step_size, max_step_size
        )

        V_i0 = V_i1.copy()

    return ys, rel_errors, convergeds, iters


def update_step_size(
    optimal_num_steps: int,
    num_steps: int,
    s: float,
    min_step_size: float = 5e-4,
    max_step_size: float = 5e-1,
):
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
    return np.clip(s_new, min_step_size, max_step_size)


def compute_step_multiplier(
    optimal_num_steps: int,
    num_steps: int,
) -> float:
    """Compute the factor by which to multiply the step size.

    Parameters
    ----------
    optimal_num_steps
        Optimal number of correction iterations
    num_steps
        Number of correction iterations used
    s
        Previous step size

    Returns
    -------
    multiplier
        Factor by which to multiply the step size, i.e., s_new = s * update
    """
    return optimal_num_steps / (num_steps + 1)


def predict_y(
    y_i0: sparray | ndarray,
    V_i1: ndarray,
    s: float,
) -> sparray | ndarray:
    """Predict the next solution.

    Parameters
    ----------
    y_i0
        Previous solution y_i = [z_i, omega_i]
        shape (n(NH + 1) + 1,)
    V_i1
        Current tangent vector
    s
        Current step size

    Returns
    -------
    y_i1_k0
        Predicted solution
    """
    y_i1_k0 = y_i0 + s * V_i1
    return y_i1_k0


def compute_tangent(
    dR_dz: sparray | ndarray,
    dR_d_omega: sparray | ndarray,
    V_i0: ndarray,
) -> sparray | ndarray:
    """Compute the current tangent vector.

    Parameters
    ----------
    dR_dz
        Derivative of residual R with respect to solution z
        See `solve.get_dR_dz`
    dR_d_omega
        Derivative of residual R with respect to omega
        See `continuation.get_dR_d_omega`
    V_i0
        Previous tangent vector

    Returns
    -------
    V_i1
        Current tangent vector
        shape (n(NH + 1) + 1,)
    """
    mat = np.block([[dR_dz, dR_d_omega.reshape(-1, 1)], [V_i0.conj()]])
    rhs = np.zeros_like(V_i0)
    rhs[-1] = 1
    V_i1 = np.linalg.solve(mat, rhs)
    V_i1 /= np.linalg.norm(V_i1)

    return V_i1


def correct_y(
    y_k0: sparray | ndarray,
    V: sparray | ndarray,
    b_ext: ndarray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
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
    y_k0
        Predicted solution
    V
        Current tangent vector
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
    y = y_k0.copy()

    omega, z = y[-1].real, y[:-1]
    A = freq.get_A(omega, NH, M, C, K)
    b_nl = solve.get_b_nl(z, omega, f_nl, NH, n, N)
    R = solve.get_R(z, A, b_nl, b_ext)
    # At the first iteration, the dot product computed by `get_P` is zero since
    # the difference in the dot product is zero.
    if get_rel_error(R, b_nl + b_ext) < tol:
        return y, R, True, 0

    converged = False
    for i in range(max_iter):
        A = freq.get_A(omega, NH, M, C, K)
        step = _solve_step(
            y,
            y_k0,
            V,
            A,
            b_ext,
            f_nl,
            df_nl_dx,
            df_nl_d_xdot,
            NH,
            n,
            N,
            M,
            C,
        )
        y[-1] += step[-1]
        y[:-1] += step[:-1]

        omega, z = y[-1].real, y[:-1]

        b_nl = solve.get_b_nl(z, omega, f_nl, NH, n, N)
        R = solve.get_R(z, A, b_nl, b_ext)
        P = get_P(y, y_k0, V)
        rhs = get_rhs(R, P)

        if get_rel_error(rhs, b_nl + b_ext) < tol:
            converged = True
            break

    return y, rhs, converged, i + 1 if "i" in locals() else 0


def get_rel_error(rhs: ndarray, w: ndarray) -> float:
    """Compute the relative error |rhs| / |w|.

    Parameters
    ----------
    rhs
        Residual rhs = [R, P]
    w
        Vector by which to normalize (e.g., solution y or force b_nl + b_ext)

    Returns
    -------
    rel_error
        Relative error |rhs| / |w|
    """
    return np.linalg.norm(rhs) / np.linalg.norm(w)


def get_P(
    y: sparray | ndarray,
    y_k0: sparray | ndarray,
    V: sparray | ndarray,
) -> float:
    """Compute dot product of tangent vector with difference between current and
    predicted solutions.

    Parameters
    ----------
    y
        Current estimated solution y = [z, omega]
        shape (n(NH + 1) + 1,)
    y_k0
        Predicted solution y
        shape (n(NH + 1) + 1,)
    V
        Current tangent vector
        shape (n(NH + 1) + 1,)

    Returns
    -------
    P
        Dot product V^T (y - y_k0)
    """
    return V.conj() @ (y - y_k0)


def get_rhs(R: sparray | ndarray, P: float) -> sparray | ndarray:
    return np.concat((R, [P]))


def _solve_step(
    y: ndarray,
    y_k0: ndarray,
    V: ndarray,
    A: sparray,
    b_ext: ndarray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
    M: ndarray,
    C: ndarray,
) -> ndarray:
    """

    Returns
    -------
    step
        The step to add to y_i1 = [z_i1, omega_i1] to get the (k+1)th correction
    """
    omega, z = y[-1].real, y[:-1]

    b_nl = solve.get_b_nl(z, omega, f_nl, NH, n, N)
    R = solve.get_R(z, A, b_nl, b_ext)
    P = get_P(y, y_k0, V)
    rhs = get_rhs(R, P)

    db_nl_dz = solve.get_db_nl_dz(omega, z, df_nl_dx, df_nl_d_xdot, NH, n, N)
    dR_dz = solve.get_dR_dz(A, db_nl_dz)
    dR_d_omega = continuation.get_dR_d_omega(
        z, omega, df_nl_d_xdot, NH, n, N, M, C
    )

    jacobian = np.block([[dR_dz, dR_d_omega.reshape(-1, 1)], [V.conj()]])

    step = np.linalg.solve(jacobian, -rhs)
    step[-1] = step[-1].real
    return step
