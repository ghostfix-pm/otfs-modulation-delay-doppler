"""Tests for the delay-Doppler channel models."""

from __future__ import annotations

import numpy as np
import pytest
from otfs.channel import (
    Taps,
    check_grid,
    dd_channel_matrix,
    dd_channel_matrix_cp,
    random_taps,
)


def single_tap_response(N, M, l, k, at=(0, 0)):
    """Apply a one-tap channel to a single delay-Doppler impulse."""
    H = dd_channel_matrix_cp(N, M, [1.0], [l], [k])
    x = np.zeros((N, M), complex)
    x[at] = 1.0
    y = (H @ x.reshape(-1)).reshape(N, M)
    return y


class TestGrid:
    def test_accepts_positive_ints(self):
        assert check_grid(4, 8) == (4, 8)

    @pytest.mark.parametrize("N,M", [(0, 8), (4, 0), (-4, 8)])
    def test_rejects_non_positive(self, N, M):
        with pytest.raises(ValueError, match=">= 1"):
            check_grid(N, M)

    @pytest.mark.parametrize("value", [4.5, "four", None])
    def test_rejects_non_integers(self, value):
        with pytest.raises(ValueError):
            check_grid(value, 8)


class TestTapValidation:
    """A mismatched tap list used to be silently truncated by ``zip``."""

    @pytest.mark.parametrize("g,l,k", [
        ([1.0, 0.4, 0.2], [0, 1], [0, 1]),          # one gain too many
        ([1.0, 0.4], [0, 1, 2], [0, 1]),            # one delay too many
        ([1.0, 0.4], [0, 1], [0]),                  # one doppler too few
        ([], [], []),
    ])
    @pytest.mark.parametrize("constructor", [dd_channel_matrix_cp, dd_channel_matrix])
    def test_mismatched_lengths_raise(self, g, l, k, constructor):
        with pytest.raises(ValueError):
            constructor(4, 8, g, l, k)

    def test_extra_gain_is_not_dropped(self):
        two = dd_channel_matrix_cp(4, 8, [1.0, 0.4], [0, 1], [0, 1])
        with pytest.raises(ValueError, match="same length"):
            dd_channel_matrix_cp(4, 8, [1.0, 0.4, 5.0], [0, 1], [0, 1])
        assert np.allclose(two, dd_channel_matrix_cp(4, 8, [1.0, 0.4], [0, 1], [0, 1]))

    def test_non_integral_delay_has_a_clear_message(self):
        with pytest.raises(ValueError, match="not an integer"):
            dd_channel_matrix_cp(4, 8, [1.0], [1.5], [0])

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError, match="negative"):
            dd_channel_matrix_cp(4, 8, [1.0], [-1], [0])

    def test_delay_beyond_delay_axis_raises(self):
        with pytest.raises(ValueError, match="supports delays"):
            dd_channel_matrix_cp(4, 8, [1.0], [8], [0])

    def test_block_model_rejects_delay_it_cannot_represent(self):
        """``dd_channel_matrix`` indexes delays against N, not M.

        A tap with ``l >= N`` produced a matrix identical to the same channel
        without the tap: the path vanished with no warning.
        """
        kept = dd_channel_matrix(4, 8, [1.0], [0], [0])
        with pytest.raises(ValueError, match="would be dropped"):
            dd_channel_matrix(4, 8, [1.0, 0.4], [0, 5], [0, 1])
        assert np.allclose(kept, dd_channel_matrix(4, 8, [1.0, 0.0], [0, 0], [0, 1]))

    def test_non_finite_gain_raises(self):
        with pytest.raises(ValueError, match="non-finite"):
            dd_channel_matrix_cp(4, 8, [np.nan], [0], [0])

    def test_doppler_beyond_grid_raises(self):
        with pytest.raises(ValueError, match="doppler"):
            dd_channel_matrix_cp(4, 8, [1.0], [0], [32])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            dd_channel_matrix_cp(4, 8, [1.0], [0], [0], method="magic")

    def test_zero_gain_path_is_accepted(self):
        H = dd_channel_matrix_cp(4, 8, [1.0, 0.0], [0, 2], [0, 1])
        assert np.allclose(H, dd_channel_matrix_cp(4, 8, [1.0], [0], [0]))


class TestDelayDopplerStructure:
    def test_single_tap_is_a_pure_permutation(self):
        H = dd_channel_matrix_cp(4, 8, [1.0], [2], [1])
        magnitudes = np.abs(H)
        assert np.allclose(magnitudes.sum(axis=1), 1.0, atol=1e-12)
        assert np.allclose((magnitudes > 1e-9).sum(axis=1), 1)

    def test_tap_shifts_both_grid_axes(self):
        for l, k in [(0, 0), (2, 1), (3, 3), (1, -2), (5, 0)]:
            y = single_tap_response(4, 8, l, k, at=(2, 1))
            peak = np.unravel_index(np.argmax(np.abs(y)), y.shape)
            assert peak == ((2 + k) % 4, (1 + l) % 8)
            assert np.isclose(np.abs(y).max(), 1.0)

    def test_zero_tap_channel_is_identity(self):
        H = dd_channel_matrix_cp(4, 8, [0.7], [0], [0])
        assert np.allclose(H, 0.7 * np.eye(32))

    def test_channel_is_linear_in_the_gains(self):
        a = dd_channel_matrix_cp(4, 8, [1.0], [1], [1])
        b = dd_channel_matrix_cp(4, 8, [0.5], [3], [-2])
        both = dd_channel_matrix_cp(4, 8, [1.0, 0.5], [1, 3], [1, -2])
        assert np.allclose(both, a + b)

    def test_colliding_delays_add_up(self):
        """Two taps on the same delay bin must sum, not overwrite."""
        H = dd_channel_matrix_cp(4, 8, [1.0, 1.0], [2, 2], [0, 0])
        assert np.allclose(H, 2 * dd_channel_matrix_cp(4, 8, [1.0], [2], [0]))

    @pytest.mark.parametrize("grid", [(2, 4), (4, 8), (8, 8), (4, 16), (16, 16)])
    def test_taps_method_matches_reference(self, grid):
        N, M = grid
        rng = np.random.default_rng(N + M)
        for _ in range(3):
            taps = random_taps(rng, N, M, 3)
            fast = dd_channel_matrix_cp(N, M, taps.gain, taps.delay, taps.doppler)
            slow = dd_channel_matrix_cp(N, M, taps.gain, taps.delay, taps.doppler,
                                        method="reference")
            assert np.allclose(fast, slow, atol=1e-12, rtol=1e-12)


class TestBlockModel:
    """Regression pins of the MATLAB port (values from the original implementation)."""

    def test_two_path_toy_magnitude(self):
        H = dd_channel_matrix(2, 4, [1, 0.4], [0, 1], [0, -1])
        assert H.shape == (8, 8)
        assert np.isclose(np.abs(H).sum(), 12.0, atol=1e-9)
        assert np.isclose(np.linalg.norm(H), 3.019933774108, atol=1e-9)
        assert np.isclose(H[0, 0], 1.0)

    def test_single_path_spread(self):
        H = dd_channel_matrix(4, 8, [1.0], [2], [1])
        assert np.isclose(np.abs(H).sum(), 44.0, atol=1e-9)
        assert np.isclose(np.linalg.norm(H), 5.477225575052, atol=1e-9)

    def test_two_path_spread(self):
        H = dd_channel_matrix(4, 8, [1.0, 0.5], [0, 3], [0, -2])
        assert np.isclose(np.abs(H).sum(), 57.0, atol=1e-9)
        assert np.isclose(np.linalg.norm(H), 6.264982043071, atol=1e-9)

    def test_pilot_is_replicated_at_the_second_path(self):
        N, M = 2, 4
        H = dd_channel_matrix(N, M, [1, 0.4], [0, 1], [0, -1])
        pilot = np.zeros((N, M))
        pilot[N // 2 - 1, M // 2 - 1] = 10
        response = np.abs((H @ pilot.reshape(-1)).reshape(N, M))
        assert np.isclose(response[0, 1], 10.0, atol=1e-9)
        assert np.isclose(response[1, 0], 4.0, atol=1e-9)
        assert np.isclose(response.sum(), 14.0, atol=1e-9)


class TestRandomTaps:
    def test_shapes_and_ranges(self):
        taps = random_taps(np.random.default_rng(1), 4, 8, 3)
        assert isinstance(taps, Taps)
        assert len(taps) == 3
        assert taps.gain.shape == taps.delay.shape == taps.doppler.shape == (3,)
        assert taps.delay.min() >= 0 and taps.delay.max() < 4
        assert taps.doppler.min() >= -2 and taps.doppler.max() <= 2
        assert len(set(taps.doppler.tolist())) == 3

    def test_same_seed_same_channel(self):
        a = random_taps(np.random.default_rng(11), 8, 16, 4)
        b = random_taps(np.random.default_rng(11), 8, 16, 4)
        assert np.array_equal(a.gain, b.gain)
        assert np.array_equal(a.delay, b.delay)

    def test_average_gain_power_is_normalised(self):
        rng = np.random.default_rng(5)
        power = np.mean([np.abs(random_taps(rng, 4, 8, 3).gain ** 2).sum() for _ in range(400)])
        assert 0.75 < power < 1.25

    def test_single_path(self):
        taps = random_taps(np.random.default_rng(2), 4, 8, 1)
        assert len(taps) == 1

    def test_too_many_paths_for_the_doppler_grid(self):
        with pytest.raises(ValueError, match="Doppler bins"):
            random_taps(np.random.default_rng(3), 2, 8, 4)

    def test_rejects_zero_paths(self):
        with pytest.raises(ValueError, match=">= 1"):
            random_taps(np.random.default_rng(3), 4, 8, 0)

    def test_channel_from_drawn_taps_is_hermitian_free_but_square(self):
        taps = random_taps(np.random.default_rng(4), 4, 8, 3)
        H = dd_channel_matrix_cp(4, 8, taps.gain, taps.delay, taps.doppler)
        assert H.shape == (32, 32)
        assert np.all(np.isfinite(H))
