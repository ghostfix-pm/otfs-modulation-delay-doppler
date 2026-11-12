"""Tests for the Monte-Carlo simulation and its reporting."""

from __future__ import annotations

import json

import numpy as np
import pytest
from otfs.sim import (
    BerPoint,
    BerResult,
    ber_curve,
    default_snr_db,
    rayleigh_qpsk_ber,
    symbol_errors,
    toy_example,
)

# Frozen output of the original implementation (frames=40, seed=3) so the
# refactor cannot silently change the published curve.
BASELINE_FRAMES = 40
BASELINE_SEED = 3
BASELINE_BER = [
    0.20625, 0.13203125, 0.117578125, 0.0921875, 0.03515625,
    0.028125, 0.008203125, 0.002734375, 0.0015625, 0.000390625,
    0.0,
]


class TestBerPoint:
    def test_rate_and_bound(self):
        point = BerPoint(snr_db=10.0, errors=3, bits=1000)
        assert point.ber == 0.003
        assert point.upper_bound == pytest.approx(3 / 1000)
        assert "3/1000 bits" in point.format()

    def test_zero_error_point_reports_a_bound_not_a_zero(self):
        point = BerPoint(snr_db=18.0, errors=0, bits=5000)
        assert point.ber == 0.0
        assert point.upper_bound == pytest.approx(6e-4)
        assert "no error in 5000 bits" in point.format()
        assert "95%" in point.format()

    def test_empty_point_is_nan(self):
        point = BerPoint(snr_db=0.0, errors=0, bits=0)
        assert np.isnan(point.ber)
        assert np.isnan(point.upper_bound)

    def test_as_dict_is_json_serialisable(self):
        payload = json.dumps(BerPoint(snr_db=4.0, errors=1, bits=64).as_dict())
        assert json.loads(payload)["errors"] == 1


class TestReferenceCurve:
    def test_known_values(self):
        # QPSK over Rayleigh fading: 0.5 * (1 - sqrt(gamma/(1+gamma))), gamma = Eb/N0
        assert rayleigh_qpsk_ber(0.0) == pytest.approx(0.21132486540518713)
        assert rayleigh_qpsk_ber(10.0) == pytest.approx(0.04356453541236155)
        assert rayleigh_qpsk_ber(20.0) == pytest.approx(0.004926228511662856)

    def test_is_monotonically_decreasing(self):
        values = rayleigh_qpsk_ber(default_snr_db())
        assert np.all(np.diff(values) < 0)

    def test_default_grid(self):
        assert np.array_equal(default_snr_db(), np.arange(0, 21, 2))


class TestBerCurve:
    def test_reproduces_the_published_curve(self):
        """Regression: same seed must give the same numbers as the original module."""
        result = ber_curve(frames=BASELINE_FRAMES, seed=BASELINE_SEED)
        assert result.ber.tolist() == pytest.approx(BASELINE_BER, abs=1e-12)

    def test_dense_and_sparse_detectors_agree(self):
        sparse = ber_curve(frames=6, snr_db=[0, 10], seed=5)
        dense = ber_curve(frames=6, snr_db=[0, 10], seed=5, method="dense")
        assert sparse.ber.tolist() == dense.ber.tolist()

    def test_bit_accounting(self):
        result = ber_curve(frames=5, snr_db=[2, 4], N=4, M=8, seed=1)
        assert len(result.points) == 2
        assert all(p.bits == 2 * 4 * 8 * 5 for p in result.points)
        assert result.total_bits == 2 * 2 * 4 * 8 * 5
        assert all(0 <= p.errors <= p.bits for p in result.points)

    def test_error_rate_falls_with_snr(self):
        result = ber_curve(frames=30, snr_db=[0, 12], seed=2)
        assert result.points[0].errors > result.points[1].errors

    def test_seed_is_reproducible_and_used(self):
        a = ber_curve(frames=3, snr_db=[6], seed=9)
        b = ber_curve(frames=3, snr_db=[6], seed=9)
        c = ber_curve(frames=3, snr_db=[6], seed=10)
        assert a.ber.tolist() == b.ber.tolist()
        assert a.params["seed"] == 9
        assert c.params["seed"] == 10

    def test_explicit_rng_takes_precedence(self):
        a = ber_curve(frames=2, snr_db=[6], rng=np.random.default_rng(4), seed=99)
        b = ber_curve(frames=2, snr_db=[6], rng=np.random.default_rng(4), seed=1)
        assert a.ber.tolist() == b.ber.tolist()

    def test_progress_callback_per_point(self):
        seen = []
        ber_curve(frames=1, snr_db=[0, 5, 10], seed=3, progress=seen.append)
        assert [p.snr_db for p in seen] == [0, 5, 10]

    def test_result_is_json_serialisable(self):
        result = ber_curve(frames=1, snr_db=[0], seed=3)
        payload = json.loads(json.dumps(result.as_dict()))
        assert payload["params"]["frames"] == 1
        assert len(payload["points"]) == 1
        assert payload["simulated_bits"] == 64

    def test_no_error_point_is_flagged_in_the_payload(self):
        result = BerResult(points=[BerPoint(snr_db=20.0, errors=0, bits=2560)], params={})
        point = result.as_dict()["points"][0]
        assert point["upper_bound_95"] == pytest.approx(3 / 2560)
        assert point["ber"] == 0.0

    @pytest.mark.parametrize("frames", [0, -1, -20])
    def test_rejects_invalid_frame_counts(self, frames):
        """``--frames 0`` used to raise ZeroDivisionError from the summary."""
        with pytest.raises(ValueError, match="frames"):
            ber_curve(frames=frames, snr_db=[0], seed=0)

    def test_rejects_empty_snr_grid(self):
        with pytest.raises(ValueError, match="snr_db"):
            ber_curve(frames=1, snr_db=[])

    def test_rejects_non_finite_snr(self):
        with pytest.raises(ValueError, match="non-finite"):
            ber_curve(frames=1, snr_db=[0.0, np.nan])

    def test_rejects_zero_paths(self):
        with pytest.raises(ValueError, match="n_paths"):
            ber_curve(frames=1, snr_db=[0], n_paths=0)

    def test_rejects_bad_grid(self):
        with pytest.raises(ValueError):
            ber_curve(frames=1, snr_db=[0], N=0)


class TestSymbolErrors:
    def test_counts_i_and_q_separately(self):
        x = np.array([1 + 1j, 1 - 1j]) / np.sqrt(2)
        same = x.copy()
        flipped_i = np.array([-1 + 1j, 1 - 1j]) / np.sqrt(2)
        assert symbol_errors(same, x) == 0
        assert symbol_errors(flipped_i, x) == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            symbol_errors(np.zeros(3), np.zeros(4))


class TestToyExample:
    def test_noiseless_detection_and_pilot(self):
        result = toy_example(np.random.default_rng(3))
        assert result["symbol_errors"] == 0
        pilot = np.abs(result["received_pilot"])
        assert np.isclose(pilot[0, 1], 10.0)
        assert np.isclose(pilot[1, 0], 4.0)

    def test_writes_both_figures_into_the_given_directory(self, tmp_path):
        out = tmp_path / "figs"
        toy_example(np.random.default_rng(3), out_dir=out)
        for name in ("fig01_dd_channel.png", "fig02_dd_pilot_data.png"):
            assert (out / name).stat().st_size > 1000

    def test_does_not_touch_the_repository_without_an_output_dir(self):
        repository_figures = toy_example(np.random.default_rng(3))["received_pilot"]
        assert repository_figures.shape == (2, 4)
