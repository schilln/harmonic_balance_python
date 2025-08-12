"""Functions for alternating frequency/time method."""

import numpy as np


def get_gamma(omega, n, NH, N):
    return np.concat(
        [_col(n, samples) for samples in _get_gamma_samples(omega, NH, N)],
        axis=1,
    )


def get_inv_gamma(omega, n, NH, N):
    return np.concat(
        [_row(n, samples) for samples in _get_inv_gamma_samples(omega, NH, N)],
        axis=0,
    )


def _get_tls(omega, N):
    return 2 * np.pi * np.arange(N) / N / omega


def _get_gamma_samples(omega, NH, N):
    tls = _get_tls(omega, N)
    return [np.exp(1j * k * omega * tls) for k in range(NH + 1)]


def _get_inv_gamma_samples(omega, NH, N):
    tls = _get_tls(omega, N)
    return [np.exp(-1j * k * omega * tls) / N for k in range(NH + 1)]


def _col(n, samples):
    return np.kron(np.eye(n), samples.reshape(-1, 1))


def _row(n, samples):
    return np.kron(np.eye(n), samples)
