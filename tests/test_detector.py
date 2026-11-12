"""Tests for the message-passing and MMSE detectors."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from otfs.channel import dd_channel_matrix, dd_channel_matrix_cp, random_taps
from otfs.detector import QPSK, mmse_detect, mp_detect
from otfs.sim import symbol_errors


def make_channel(N, M, taps, seed=0):
    return dd_channel_matrix_cp(N, M, taps.gain, taps.delay, taps.doppler)


def noisy(H, x, sigma2, rng):
    NM = x.size
    noise = np.sqrt(sigma2 / 2) * (rng.standard_normal(NM) + 1j * rng.standard_normal(NM))
    return H @ x + noise


class TestMessagePassing:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_sparse_matches_dense(self, seed):
        """The sparse factor graph must reproduce the dense iteration exactly."""
        rng = np.random.default_rng(seed)
        N, M = [(4, 8), (8, 8), (4, 16), (8, 16)][seed]
        taps = random_taps(rng, N, M, 3)
        H = make_channel(N, M, taps)
        x = QPSK[rng.integers(0, 4, N * M)]
        for sigma2 in (1e-3, 0.05, 1.0):
            y = noisy(H, x, sigma2, rng)
            dense = mp_detect(y, H, sigma2, method="dense")
            sparse = mp_detect(y, H, sigma2, method="sparse")
            assert np.array_equal(dense, sparse)

    def test_noiseless_recovery(self):
        for N, M in [(4, 8), (8, 8)]:
            taps = random_taps(np.random.default_rng(N), N, M, 3)
            H = make_channel(N, M, taps)
            x = QPSK[np.random.default_rng(N + 1).integers(0, 4, N * M)]
            assert symbol_errors(mp_detect(H @ x, H, 1e-6), x) == 0

    def test_matches_maximum_likelihood_on_a_small_grid(self):
        N, M = 2, 4
        taps = random_taps(np.random.default_rng(7), N, M, 2)
        H = make_channel(N, M, taps)
        rng = np.random.default_rng(8)
        x = QPSK[rng.integers(0, 4, N * M)]
        y = noisy(H, x, 0.02, rng)
        candidates = np.array(list(itertools.product(QPSK, repeat=N * M)))
        distances = np.abs(y[None, :] - candidates @ H.T).sum(axis=1)
        best = candidates[int(np.argmin(distances))]
        assert np.array_equal(mp_detect(y, H, 0.02), best)

    def test_beats_linear_mmse_at_10db(self):
        rng = np.random.default_rng(12)
        errors_mp = errors_mmse = 0
        for _ in range(40):
            taps = random_taps(rng, 4, 8, 3)
            H = make_channel(4, 8, taps)
            x = QPSK[rng.integers(0, 4, 32)]
            y = noisy(H, x, 0.1, rng)
            errors_mp += symbol_errors(mp_detect(y, H, 0.1), x)
            errors_mmse += symbol_errors(mmse_detect(y, H, 0.1), x)
        assert errors_mp < errors_mmse

    def test_beats_zero_forcing(self):
        rng = np.random.default_rng(21)
        taps = random_taps(rng, 4, 8, 3)
        H = make_channel(4, 8, taps)
        x = QPSK[rng.integers(0, 4, 32)]
        y = noisy(H, x, 2.0, rng)
        zf = QPSK[np.argmin(np.abs((np.linalg.solve(H, y))[:, None] - QPSK[None, :]), axis=1)]
        assert symbol_errors(mp_detect(y, H, 2.0), x) <= symbol_errors(zf, x)

    def test_iteration_and_damping_are_honoured(self):
        taps = random_taps(np.random.default_rng(3), 4, 8, 3)
        H = make_channel(4, 8, taps)
        x = QPSK[np.random.default_rng(4).integers(0, 4, 32)]
        y = H @ x
        assert symbol_errors(mp_detect(y, H, 1e-4, iters=1), x) >= 0
        assert symbol_errors(mp_detect(y, H, 1e-4, iters=30), x) == 0

    def test_damping_of_one_equals_undamped_update(self):
        taps = random_taps(np.random.default_rng(6), 4, 8, 3)
        H = make_channel(4, 8, taps)
        x = QPSK[np.random.default_rng(7).integers(0, 4, 32)]
        y = noisy(H, x, 0.2, np.random.default_rng(8))
        assert np.array_equal(mp_detect(y, H, 0.2, damp=1.0),
                              mp_detect(y, H, 0.2, damp=1.0, method="dense"))

    def test_custom_alphabet(self):
        bpsk = np.array([1.0, -1.0])
        taps = random_taps(np.random.default_rng(9), 4, 8, 3)
        H = make_channel(4, 8, taps)
        x = bpsk[np.random.default_rng(10).integers(0, 2, 32)]
        assert symbol_errors(mp_detect(H @ x, H, 1e-6, alphabet=bpsk), x) == 0

    def test_tolerance_removes_negligible_taps(self):
        H = dd_channel_matrix_cp(4, 8, [1.0, 1e-12], [0, 3], [0, 1])
        x = QPSK[np.random.default_rng(11).integers(0, 4, 32)]
        with_default = mp_detect(H @ x, H, 1e-6)
        strict = mp_detect(H @ x, H, 1e-6, tol=0.0)
        assert symbol_errors(with_default, x) == 0
        assert strict.shape == with_default.shape

    def test_works_on_the_dense_block_model(self):
        H = dd_channel_matrix(2, 4, [1, 0.4], [0, 1], [0, -1])
        x = QPSK[np.random.default_rng(13).integers(0, 4, 8)]
        assert symbol_errors(mp_detect(H @ x, H, 1e-6), x) == 0

    def test_constant_channel_is_detected_symbol_by_symbol(self):
        H = np.eye(3, dtype=complex)
        x = np.array([QPSK[0], QPSK[2], QPSK[1]])
        assert np.array_equal(mp_detect(0.5 * x, H, 0.01), x)


class TestDetectorValidation:
    def test_rejects_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            mp_detect(np.zeros(4), np.zeros((4, 5)), 1.0)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="samples"):
            mp_detect(np.zeros(3), np.eye(4), 1.0)

    @pytest.mark.parametrize("sigma2", [0.0, -1.0])
    def test_rejects_non_positive_noise_variance(self, sigma2):
        with pytest.raises(ValueError, match="sigma2"):
            mp_detect(np.zeros(4), np.eye(4), sigma2)

    def test_rejects_bad_method(self):
        with pytest.raises(ValueError, match="method"):
            mp_detect(np.zeros(4), np.eye(4), 1.0, method="turbo")

    def test_rejects_zero_iterations(self):
        with pytest.raises(ValueError, match="iters"):
            mp_detect(np.zeros(4), np.eye(4), 1.0, iters=0)

    def test_rejects_bad_damping(self):
        with pytest.raises(ValueError, match="damp"):
            mp_detect(np.zeros(4), np.eye(4), 1.0, damp=1.5)

    def test_rejects_single_symbol_alphabet(self):
        with pytest.raises(ValueError, match="alphabet"):
            mp_detect(np.zeros(4), np.eye(4), 1.0, alphabet=np.array([1.0]))

    def test_rejects_all_zero_channel(self):
        with pytest.raises(ValueError, match="non-zero"):
            mp_detect(np.zeros(4), np.zeros((4, 4)), 1.0)


class TestMmse:
    def test_returns_alphabet_members(self):
        taps = random_taps(np.random.default_rng(2), 4, 8, 3)
        H = make_channel(4, 8, taps)
        rng = np.random.default_rng(3)
        x = QPSK[rng.integers(0, 4, 32)]
        estimate = mmse_detect(noisy(H, x, 0.5, rng), H, 0.5)
        assert set(np.round(estimate, 12)) <= set(np.round(QPSK, 12))

    def test_high_snr_is_accurate(self):
        taps = random_taps(np.random.default_rng(4), 4, 8, 3)
        H = make_channel(4, 8, taps)
        rng = np.random.default_rng(5)
        x = QPSK[rng.integers(0, 4, 32)]
        assert symbol_errors(mmse_detect(noisy(H, x, 1e-4, rng), H, 1e-4), x) == 0

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            mmse_detect(np.zeros(3), np.eye(4), 1.0)
        with pytest.raises(ValueError):
            mmse_detect(np.zeros(4), np.eye(4), -1.0)
