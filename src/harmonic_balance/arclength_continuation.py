from collections import abc

import numpy as np
from scipy import sparse

from . import continuation, freq, solve

ndarray = np.ndarray
sparray = sparse.sparray
array = ndarray | sparray
# TODO: Fix the use of these type annotations.


def predict_y(
    y_current: sparray | ndarray, y_previous: sparray | ndarray, s: float
) -> sparray | ndarray:
    secant = y_current - y_previous
    direction = secant / np.linalg.norm(secant)
    return y_current + s * direction


def correct_y(
    y1: sparray | ndarray,
    y0: sparray | ndarray,
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
    omega1, z1 = y1[-1].real, y1[:-1]
    A = freq.get_A(omega1, NH, M, C, K)
    R = solve.get_R(z1, omega1, A, f_nl, b_ext, NH, n, N)
    P = get_P(y1, y0, s)
    if get_rel_error(get_rhs(R, P), y1) < tol:
        return y1, P, True, 0

    converged = False
    for i in range(max_iter):
        tmp = y1.copy()

        omega1 = y1[-1].real
        z1 = y1[:-1]

        A = freq.get_A(omega1, NH, M, C, K)
        step = _solve_step(
            y1,
            y0,
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
        y1[-1] += step[-1].real
        y1[:-1] += step[:-1]

        y0 = tmp
        R = solve.get_R(z1, omega1, A, f_nl, b_ext, NH, n, N)
        P = get_P(y1, y0, s)
        rhs = get_rhs(R, P)

        print(get_rel_error(rhs, y1))

        if get_rel_error(rhs, y1) < tol:
            converged = True
            break

    return y1, rhs, converged, i + 1 if "i" in locals() else 0


def get_rel_error(rhs: ndarray, y: ndarray) -> float:
    return np.linalg.norm(rhs) / np.linalg.norm(y)


def get_P(
    y1: sparray | ndarray,
    y0: sparray | ndarray,
    s: float,
) -> float:
    omega1, omega0 = y1[-1].real, y0[-1].real
    z1, z0 = y1[:-1], y0[:-1]
    if isinstance(z1, ndarray) or isinstance(z0, ndarray):
        norm = np.linalg.norm
    else:
        norm = sparse.linalg.norm
    return norm(z1 - z0) ** 2 + (omega1 - omega0) ** 2 - s**2


def get_dP_dz(
    z1: sparray | ndarray,
    z0: sparray | ndarray,
) -> sparray | ndarray:
    return 2 * (z1 - z0)


def get_dP_d_omega(
    omega1: float,
    omega0: float,
) -> float:
    return 2 * (omega1 - omega0)


def get_rhs(R: sparray | ndarray, P: float) -> sparray | ndarray:
    return np.concat((R, [P]))


def _solve_step(
    y1: ndarray,
    y0: ndarray,
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
        The step to add to y = [z, omega]
    """
    omega1, z1 = y1[-1].real, y1[:-1]
    omega0, z0 = y0[-1].real, y0[:-1]

    R = solve.get_R(z1, omega1, A, f_nl, b_ext, NH, n, N)
    P = get_P(y1, y0, s)
    rhs = get_rhs(R, P)

    db_nl_dz = solve.get_db_nl_dz(omega1, z1, df_nl_dx, df_nl_d_xdot, NH, n, N)
    dR_dz = solve.get_dR_dz(A, db_nl_dz)
    dR_d_omega = continuation.get_dR_d_omega(
        z1, omega1, df_nl_d_xdot, NH, n, N, M, C
    )

    dP_dz = get_dP_dz(z1, z0)
    dP_d_omega = get_dP_d_omega(omega1, omega0)

    jacobian = np.block(
        [
            [dR_dz, dR_d_omega.reshape(-1, 1)],
            [dP_dz.reshape(1, -1), dP_d_omega.reshape(-1, 1)],
        ]
    )

    return np.linalg.solve(jacobian, -rhs)
