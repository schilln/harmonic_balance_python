"""Functions for alternating frequency/time method."""

import numpy as np
from scipy import sparse

ndarray = np.ndarray
sparray = sparse.sparray


def get_gamma(omega: float, NH: int, n: int, N: int) -> sparray:
    """Compute inverse Fourier transform operator.

    Parameters
    ----------
    omega
        Fundamental frequency
    NH
        Assumed highest harmonic index
    n
        Number of degrees of freedom
    N
        Number of points to sample in time domain

    Returns
    -------
    gamma
        Array that computes a time signal via right multiplication with a
        frequency signal
        shape (n * N, n * (NH + 1))
    """
    return sparse.hstack(
        [_col(n, samples) for samples in _get_gamma_samples(omega, NH, N)],
        format="csr",
    )


def get_inv_gamma(omega: float, NH: int, n: int, N: int) -> sparray:
    """Compute Fourier transform operator.

    Parameters
    ----------
    omega
        Fundamental frequency
    NH
        Assumed highest harmonic index
    n
        Number of degrees of freedom
    N
        Number of points to sample in time domain

    Returns
    -------
    inv_gamma
        Array that computes a frequency signal via right multiplication with a
        time signal
        shape (n * (NH + 1), n * N)
    """
    return sparse.vstack(
        [_row(n, samples) for samples in _get_inv_gamma_samples(omega, NH, N)],
        format="csr",
    )


def time_from_freq(n: int, gamma: sparray, freq: sparray) -> sparray:
    """Compute a time signal from a frequency signal.

    Parameters
    ----------
    n
        Number of degrees of freedom
    gamma
        Inverse Fourier transform operator (see `get_gamma`)
        shape (n * N, n * (NH + 1))
    freq
        Frequency signal
        shape (n * (NH + 1),)

        freq = [a0, a1, ..., aNH]
        ak = [a_k0, a_k1, ..., a_k(n-1)]
        a_ki is the frequency coefficient for the kth harmonic of the ith degree
        of freedom

    Returns
    -------
    time
        Time signal
        shape (n * N,)

        time = [x0, x1, ..., x_(n-1))]
        xi = [x_i0, x_i1, ..., x_i(N-1)]
        x_ij is the time signal for the jth sample (in the period) of the ith
        degree of freedom
    """
    return (
        gamma[:, :n].real @ freq[:n].real + 2 * (gamma[:, n:] @ freq[n:]).real
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
