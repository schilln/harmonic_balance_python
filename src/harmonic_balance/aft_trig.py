"""Functions for alternating frequency/time method using cosines and sines."""

import itertools

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


def _get_samples(omega, N, k, fn):
    tls = _get_tls(omega, N)
    return fn(k * omega * tls)


def _get_all_samples(omega, NH, N, constant_coefficient):
    cos_sin = [
        (_get_samples(omega, N, k, np.cos), _get_samples(omega, N, k, np.sin))
        for k in range(1, NH + 1)
    ]
    return [np.full(N, constant_coefficient)] + list(itertools.chain(*cos_sin))


def _get_gamma_samples(omega, NH, N):
    return _get_all_samples(omega, NH, N, 1 / 2)


def _get_inv_gamma_samples(omega, NH, N):
    return [2 / N * samples for samples in _get_all_samples(omega, NH, N, 1)]


def _col(n, samples):
    return np.kron(np.eye(n), samples.reshape(-1, 1))


def _row(n, samples):
    return np.kron(np.eye(n), samples)
