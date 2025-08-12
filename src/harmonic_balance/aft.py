"""Functions for alternating frequency/time method."""

import numpy as np
from scipy import sparse

ndarray = np.ndarray
sparray = sparse.sparray


def get_gamma(omega: float, n: int, NH: int, N: int) -> sparray:
    return sparse.hstack(
        [_col(n, samples) for samples in _get_gamma_samples(omega, NH, N)],
        format="csr",
    )


def get_inv_gamma(omega: float, n: int, NH: int, N: int) -> sparray:
    return sparse.vstack(
        [_row(n, samples) for samples in _get_inv_gamma_samples(omega, NH, N)],
        format="csr",
    )


def _get_tls(omega: float, N: int) -> ndarray:
    return 2 * np.pi * np.arange(N) / N / omega


def _get_gamma_samples(omega: float, NH: int, N: int) -> list[ndarray]:
    tls = _get_tls(omega, N)
    return [np.exp(1j * k * omega * tls) for k in range(NH + 1)]


def _get_inv_gamma_samples(omega: float, NH: int, N: int) -> list[ndarray]:
    tls = _get_tls(omega, N)
    return [np.exp(-1j * k * omega * tls) / N for k in range(NH + 1)]


def _col(n: int, samples: ndarray) -> sparray:
    return sparse.kron(sparse.eye_array(n), samples.reshape(-1, 1))


def _row(n: int, samples: ndarray) -> sparray:
    return sparse.kron(sparse.eye_array(n), samples.reshape(1, -1))
