from collections import abc

import numpy as np
from scipy import sparse

from . import aft, freq

ndarray = np.ndarray
sparray = sparse.sparray
array = ndarray | sparray
# TODO: Fix the use of these type annotations.


def get_R(
    z: sparray | ndarray,
    omega: float,
    A: sparray,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    b_ext: sparray,
    NH: int,
    n: int,
    N: int,
) -> sparray:
    b_nl = get_b_nl(z, omega, f_nl, NH, n, N)
    return A @ z + b_nl - b_ext


def get_b_nl(
    z: sparray | ndarray,
    omega: float,
    f_nl: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
) -> sparray:
    gamma = aft.get_gamma(omega, NH, n, N)

    x = aft.time_from_freq(n, gamma, z)
    zp = freq.get_derivative(omega, z, NH, n)
    xp = aft.time_from_freq(n, gamma, zp)

    fx = f_nl(x, xp, N)
    return aft.freq_from_time(aft.get_inv_gamma(omega, NH, n, N), fx)


def get_dR_dz(
    A: sparray,
    db_nl_dz: sparray | ndarray,
) -> sparray:
    res = A + db_nl_dz
    if isinstance(res, np.matrix):
        return res.A
    return res


def get_dR_d_omega(
    z: sparray | ndarray,
    omega: float,
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
    M: ndarray,
    C: ndarray,
) -> sparray:
    return (
        2 * sparse.kron(omega * freq.get_nabla(NH, 2), M)
        + sparse.kron(freq.get_nabla(NH), C)
    ) @ z + get_db_nl_d_omega(omega, z, df_nl_d_xdot, NH, n, N)


def get_db_nl_dz(
    omega: float,
    z: sparray | ndarray,
    df_nl_dx: abc.Callable[[ndarray, ndarray, int], ndarray],
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
) -> sparray:
    gamma = aft.get_gamma(omega, NH, n, N)
    inv_gamma = aft.get_inv_gamma(omega, NH, n, N)

    x = aft.time_from_freq(n, gamma, z)
    zp = freq.get_derivative(omega, z, NH, n)
    xp = aft.time_from_freq(n, gamma, zp)

    db_dx = inv_gamma @ df_nl_dx(x, xp, N) @ gamma
    db_d_xdot = (
        inv_gamma
        @ df_nl_d_xdot(x, xp, N)
        @ gamma
        @ sparse.kron(omega * freq.get_nabla(NH), sparse.eye_array(n))
    )

    return db_dx + db_d_xdot


def get_db_nl_d_omega(
    omega: float,
    z: sparray | ndarray,
    df_nl_d_xdot: abc.Callable[[ndarray, ndarray, int], ndarray],
    NH: int,
    n: int,
    N: int,
):
    gamma = aft.get_gamma(omega, NH, n, N)
    inv_gamma = aft.get_inv_gamma(omega, NH, n, N)

    x = aft.time_from_freq(n, gamma, z)
    zp = freq.get_derivative(omega, z, NH, n)
    xp = aft.time_from_freq(n, gamma, zp)

    return (
        inv_gamma
        @ df_nl_d_xdot(x, xp, N)
        @ gamma
        @ freq.get_nabla(omega, NH)
        @ z
    )


def get_P(
    z1: sparray | ndarray,
    z0: sparray | ndarray,
    omega1: float,
    omega0: float,
    s: float,
) -> float:
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
