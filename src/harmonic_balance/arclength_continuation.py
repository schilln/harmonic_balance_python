from collections import abc

import numpy as np
from scipy import sparse

from . import continuation, freq, solve

ndarray = np.ndarray
sparray = sparse.sparray
array = ndarray | sparray
# TODO: Fix the use of these type annotations.


def predict_y(
    y_i1: sparray | ndarray, y_i0: sparray | ndarray, s: float
) -> sparray | ndarray:
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
    y = y_i1_k0.copy()

    omega, z = y[-1].real, y[:-1]
    A = freq.get_A(omega, NH, M, C, K)
    R = solve.get_R(z, omega, A, f_nl, b_ext, NH, n, N)
    P = get_P(y, y_i0, s)
    rhs = get_rhs(R, P)
    if get_rel_error(rhs, y) < tol:
        return y, rhs, True, 0

    converged = False
    for i in range(max_iter):
        omega = y[-1].real
        z = y[:-1]

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
        y[-1] += step[-1].real
        y[:-1] += step[:-1]

        R = solve.get_R(z, omega, A, f_nl, b_ext, NH, n, N)
        P = get_P(y, y_i0, s)
        rhs = get_rhs(R, P)

        if get_rel_error(rhs, y) < tol:
            converged = True
            break

    return y, rhs, converged, i + 1 if "i" in locals() else 0


def get_rel_error(rhs: ndarray, y: ndarray) -> float:
    return np.linalg.norm(rhs) / np.linalg.norm(y)


def get_P(
    y_i1: sparray | ndarray,
    y_i0: sparray | ndarray,
    s: float,
) -> float:
    omega_i1, z_i1 = y_i1[-1].real, y_i1[:-1]
    omega_i0, z_i0 = y_i0[-1].real, y_i0[:-1]

    if isinstance(z_i1, ndarray) or isinstance(z_i0, ndarray):
        norm = np.linalg.norm
    else:
        norm = sparse.linalg.norm
    return norm(z_i1 - z_i0) ** 2 + (omega_i1 - omega_i0) ** 2 - s**2


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

    R = solve.get_R(z_i1, omega_i1, A, f_nl, b_ext, NH, n, N)
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

    return np.linalg.solve(jacobian, -rhs)
