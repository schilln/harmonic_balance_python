from collections import abc

import numpy as np
from scipy import sparse

from . import aft, freq

ndarray = np.ndarray
sparray = sparse.sparray


def get_R(
    z: sparray | ndarray,
    omega: float,
    f_nl: abc.Callable[[ndarray], ndarray],
    b_ext: sparray,
    NH: int,
    n: int,
    N: int,
    M: ndarray,
    C: ndarray,
    K: ndarray,
) -> sparray:
    A = freq.get_A(omega, NH, M, C, K)

    x = aft.time_from_freq(n, aft.get_gamma(omega, NH, n, N), z)
    fx = f_nl(x)
    b = aft.freq_from_time(aft.get_inv_gamma(omega, NH, n, N), fx)

    return A @ z + b - b_ext


def get_dR_dz(
    omega: float,
    NH: int,
    M: ndarray,
    C: ndarray,
    K: ndarray,
    db_dz: sparray | ndarray,
) -> sparray:
    A = freq.get_A(omega, NH, M, C, K)
    return A + db_dz


def get_dR_domega(
    z: sparray | ndarray,
    omega: float,
    NH: int,
    M: ndarray,
    C: ndarray,
) -> sparray:
    return (
        2 * omega * sparse.kron(freq._get_diag_nabla(omega, NH, 2), M)
        + sparse.kron(freq._get_diag_nabla(omega, NH), C)
    ) @ z


def get_db_dz(
    omega: float,
    z: sparray | ndarray,
    df_dx: abc.Callable[[sparray | ndarray], sparray | ndarray],
    df_d_xdot: abc.Callable[[sparray | ndarray], sparray | ndarray],
    NH: int,
    n: int,
    N: int,
) -> sparray:
    inv_gamma = aft.get_inv_gamma(omega, NH, n, N)
    gamma = aft.get_gamma(omega, NH, n, N)

    x = aft.time_from_freq(n, gamma, z)
    zp = freq.get_derivative(omega, z, NH, n)
    xp = aft.time_from_freq(n, gamma, zp)

    db_dx = inv_gamma @ df_dx(x) @ gamma
    db_d_xdot = (
        omega
        * inv_gamma
        @ df_d_xdot(xp)
        @ gamma
        @ sparse.kron(freq._get_diag_nabla(omega, NH), sparse.eye_array(n))
    )

    return db_dx + db_d_xdot


def get_P(
    z1: sparray | ndarray,
    z0: sparray | ndarray,
    omega1: float,
    omega0: float,
    s: float,
):
    return sparse.linalg.norm(z1 - z0) ** 2 + (omega1 - omega0) ** 2 - s**2


def get_dP_dz():
    raise NotImplementedError()


def get_dP_domega():
    raise NotImplementedError()
