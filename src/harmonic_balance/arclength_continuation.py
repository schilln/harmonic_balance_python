import numpy as np
from scipy import sparse

ndarray = np.ndarray
sparray = sparse.sparray
array = ndarray | sparray
# TODO: Fix the use of these type annotations.


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
