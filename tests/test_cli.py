"""Tests for the command line interface (run as a subprocess, as a user would)."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parent.parent
WRAPPER = REPOSITORY / "python" / "otfs_sim.py"


def run(*args, cwd=None):
    """Run the module the way the README documents it, from the checkout.

    The checkout is *appended* to PYTHONPATH so inherited entries keep working
    (a replacement breaks interpreters that resolve packages from a shared path).
    """
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    inherited = env.get("PYTHONPATH")
    source = str(REPOSITORY / "python")
    env["PYTHONPATH"] = source if not inherited else source + os.pathsep + inherited
    return subprocess.run([sys.executable, *args], check=False, capture_output=True, text=True,
                          cwd=str(cwd or REPOSITORY), env=env)


def run_module(*args, cwd=None):
    return run("-m", "otfs", *args, cwd=cwd)


class TestBerCommand:
    def test_json_output(self, tmp_path):
        done = run_module("ber", "--frames", "2", "--snr", "4", "--out", str(tmp_path),
                          "--no-figures", "--quiet", "--json")
        assert done.returncode == 0, done.stderr
        payload = json.loads(done.stdout)
        assert payload["points"][0]["snr_db"] == 4.0
        assert payload["params"]["frames"] == 2
        assert payload["points"][0]["bits"] == 2 * 32 * 2

    def test_prints_a_table_with_zero_error_bounds(self, tmp_path):
        done = run_module("ber", "--frames", "1", "--snr", "30", "--out", str(tmp_path))
        assert done.returncode == 0, done.stderr
        assert "no error in" in done.stdout
        assert "95% confidence" in done.stdout

    def test_writes_the_ber_figure(self, tmp_path):
        done = run_module("ber", "--frames", "2", "--snr", "0,10", "--out", str(tmp_path))
        assert done.returncode == 0, done.stderr
        assert (tmp_path / "fig03_ber_mp.png").stat().st_size > 1000

    def test_no_figures_flag(self, tmp_path):
        done = run_module("ber", "--frames", "1", "--snr", "0", "--out", str(tmp_path),
                          "--no-figures")
        assert done.returncode == 0, done.stderr
        assert not (tmp_path / "fig03_ber_mp.png").exists()

    def test_csv_output(self, tmp_path):
        target = tmp_path / "ber.csv"
        done = run_module("ber", "--frames", "2", "--snr", "0,10", "--csv", str(target),
                          "--out", str(tmp_path), "--quiet")
        assert done.returncode == 0, done.stderr
        raw = target.read_bytes()
        assert b"\r\n" not in raw, "CSV must use LF endings"
        with target.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 2
        assert rows[0]["snr_db"] == "0"
        assert rows[0]["bits"] == str(2 * 32 * 2)   # two bits per cell, 2 frames

    def test_quiet_suppresses_the_table(self, tmp_path):
        done = run_module("ber", "--frames", "1", "--snr", "0", "--out", str(tmp_path),
                          "--no-figures", "--quiet")
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == ""

    def test_grid_and_paths_options(self, tmp_path):
        done = run_module("ber", "--frames", "1", "--snr", "5", "--grid", "8x8",
                          "--paths", "2", "--out", str(tmp_path), "--json", "--quiet")
        assert done.returncode == 0, done.stderr
        payload = json.loads(done.stdout)
        assert payload["params"]["N"] == 8 and payload["params"]["M"] == 8
        assert payload["params"]["n_paths"] == 2
        assert payload["points"][0]["bits"] == 2 * 64


class TestArgumentValidation:
    @pytest.mark.parametrize("value", ["0", "-3"])
    def test_zero_frames_is_rejected_not_a_traceback(self, value):
        done = run_module("ber", "--frames", value)
        assert done.returncode == 2
        assert "at least 1" in done.stderr
        assert "ZeroDivisionError" not in done.stderr

    def test_bad_grid_is_rejected(self):
        done = run_module("ber", "--grid", "0x8")
        assert done.returncode == 2
        assert "grid" in done.stderr or "NxM" in done.stderr

    def test_bad_snr_list_is_rejected(self):
        done = run_module("ber", "--snr", "0,abc")
        assert done.returncode == 2
        assert "number" in done.stderr

    def test_too_many_paths_reports_a_clean_error(self, tmp_path):
        done = run_module("ber", "--frames", "1", "--snr", "0", "--grid", "2x8",
                          "--paths", "4", "--out", str(tmp_path))
        assert done.returncode == 1
        assert "otfs: error:" in done.stderr
        assert "Traceback" not in done.stderr


class TestToyAndDefault:
    def test_toy_writes_the_first_two_figures(self, tmp_path):
        done = run_module("toy", "--out", str(tmp_path))
        assert done.returncode == 0, done.stderr
        assert "symbol errors: 0" in done.stdout
        assert (tmp_path / "fig01_dd_channel.png").exists()
        assert (tmp_path / "fig02_dd_pilot_data.png").exists()

    def test_documented_wrapper_still_works(self, tmp_path):
        """The README's ``python python/otfs_sim.py`` invocation must keep working."""
        done = run(str(WRAPPER), "--frames", "1", "--snr", "6", "--out", str(tmp_path),
                   "--no-figures")
        assert done.returncode == 0, done.stderr
        assert "SNR  6 dB" in done.stdout

    def test_help_lists_the_subcommands(self):
        done = run_module("--help")
        assert done.returncode == 0
        assert "toy" in done.stdout and "ber" in done.stdout
