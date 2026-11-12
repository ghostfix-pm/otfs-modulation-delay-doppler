"""Tests for figure helpers and the output path rules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from otfs import figures
from otfs.figures import (
    bar3,
    default_output_dir,
    display_dd_response,
    save_ber_figure,
    save_channel_figure,
    save_pilot_data_figure,
)
from otfs.sim import BerPoint, BerResult


def test_bar3_handles_an_all_zero_grid(tmp_path):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    bar3(fig.add_subplot(111), np.zeros((2, 3)), "empty")   # vmax must not be 0
    target = tmp_path / "empty.png"
    fig.savefig(target)
    plt.close(fig)
    assert target.stat().st_size > 0


def test_display_response_places_taps_like_the_figure():
    # the plotting convention offsets the Doppler column by M//2 - 1
    response = display_dd_response(2, 4, [1, 0.4], [0, 1], [0, -1])
    assert response[0, 1] == 1.0
    assert response[1, 0] == 0.4
    assert np.isclose(response.sum(), 1.4)


def test_default_output_dir_prefers_the_checkout(tmp_path):
    module = tmp_path / "python" / "otfs" / "figures.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    (tmp_path / "docs").mkdir()
    assert default_output_dir(module_path=module) == tmp_path / "docs" / "figures"


def test_default_output_dir_falls_back_to_the_working_directory(tmp_path):
    module = tmp_path / "site-packages" / "otfs" / "figures.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    fallback = tmp_path / "cwd"
    assert default_output_dir(module_path=module, cwd=fallback) == fallback / "figures"


def test_default_output_dir_of_the_checkout_is_inside_the_repository():
    path = Path(figures.__file__).resolve()
    assert default_output_dir(module_path=path).name == "figures"
    assert default_output_dir(module_path=path).parent.name == "docs"


def test_save_channel_and_pilot_figures(tmp_path):
    channel = save_channel_figure(tmp_path / "a" / "fig01.png", 2, 4, [1, 0.4], [0, 1], [0, -1])
    panels = [(np.eye(2), "pilot"), (np.ones((2, 2)), "data")]
    pilot = save_pilot_data_figure(tmp_path / "b" / "fig02.png", panels)
    assert channel.stat().st_size > 1000
    assert pilot.stat().st_size > 1000


def test_save_ber_figure_marks_points_without_errors(tmp_path):
    result = BerResult(params={"N": 4, "M": 8, "n_paths": 3},
                       points=[BerPoint(0.0, errors=10, bits=640),
                               BerPoint(10.0, errors=1, bits=640),
                               BerPoint(20.0, errors=0, bits=640)])
    target = save_ber_figure(tmp_path / "fig03.png", result,
                             reference=[0.1, 0.02, 0.002])
    assert target.stat().st_size > 1000
