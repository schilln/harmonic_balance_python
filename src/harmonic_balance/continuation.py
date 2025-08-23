from collections import abc

import numpy as np
from scipy import sparse

from . import aft, freq

ndarray = np.ndarray
sparray = sparse.sparray
array = ndarray | sparray
# TODO: Fix the use of these type annotations.


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
