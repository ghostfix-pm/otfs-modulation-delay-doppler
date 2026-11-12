"""OTFS modulation over a doubly-dispersive channel with message-passing detection.

Python counterpart of the MATLAB code in ``matlab/``:

* :func:`otfs.channel.dd_channel_matrix` and
  :func:`otfs.channel.dd_channel_matrix_cp` build the effective delay-Doppler
  channel matrix from a set of (gain, delay, Doppler) taps,
* :func:`otfs.sim.toy_example` shows how a pilot and a QAM data frame are spread
  by the channel (fig01, fig02),
* :func:`otfs.sim.ber_curve` evaluates the BER of the message-passing detector
  versus SNR (fig03).
"""

from __future__ import annotations

from .channel import Taps, dd_channel_matrix, dd_channel_matrix_cp, random_taps
from .detector import QPSK, mmse_detect, mp_detect
from .sim import BerPoint, BerResult, ber_curve, rayleigh_qpsk_ber, toy_example

__version__ = "0.2.0"

__all__ = [
    "QPSK",
    "BerPoint",
    "BerResult",
    "Taps",
    "__version__",
    "ber_curve",
    "dd_channel_matrix",
    "dd_channel_matrix_cp",
    "mmse_detect",
    "mp_detect",
    "random_taps",
    "rayleigh_qpsk_ber",
    "toy_example",
]
