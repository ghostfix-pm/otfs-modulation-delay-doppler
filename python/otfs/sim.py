"""Monte-Carlo simulation of OTFS transmission over random delay-Doppler channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .channel import (
    Taps,
    check_grid,
    dd_channel_matrix,
    dd_channel_matrix_cp,
    random_taps,
)
from .detector import QPSK, mp_detect

__all__ = [
    "BerPoint",
    "BerResult",
    "ber_curve",
    "default_snr_db",
    "rayleigh_qpsk_ber",
    "symbol_errors",
    "toy_example",
]


def default_snr_db() -> np.ndarray:
    """SNR grid used by the study (0 to 20 dB in 2 dB steps)."""
    return np.arange(0, 21, 2)


@dataclass(frozen=True)
class BerPoint:
    """One Monte-Carlo point: observed error count over a number of bits."""

    snr_db: float
    errors: int
    bits: int

    @property
    def ber(self) -> float:
        """Bit error rate, ``nan`` when no bit was simulated."""
        return self.errors / self.bits if self.bits else float("nan")

    @property
    def upper_bound(self) -> float:
        """95% confidence upper bound on the BER (rule of three).

        A point with no observed error is not a zero error rate; the honest
        statement is "below ``3/bits``", which is what the figure draws.
        """
        return 3.0 / self.bits if self.bits else float("nan")

    def as_dict(self) -> dict:
        return {
            "snr_db": float(self.snr_db),
            "errors": int(self.errors),
            "bits": int(self.bits),
            "ber": float(self.ber),
            "upper_bound_95": float(self.upper_bound) if not self.errors else None,
        }

    def format(self) -> str:
        if self.errors:
            return f"SNR {self.snr_db:2.0f} dB  BER {self.ber:.2e}  ({self.errors}/{self.bits} bits)"
        return (f"SNR {self.snr_db:2.0f} dB  no error in {self.bits} bits "
                f"(BER < {self.upper_bound:.1e} at 95% confidence)")


@dataclass
class BerResult:
    """Result of :func:`ber_curve`, JSON-serialisable and plot-ready."""

    points: list[BerPoint]
    params: dict = field(default_factory=dict)

    @property
    def snr_db(self) -> np.ndarray:
        return np.array([p.snr_db for p in self.points], float)

    @property
    def ber(self) -> np.ndarray:
        return np.array([p.ber for p in self.points], float)

    @property
    def total_bits(self) -> int:
        return int(sum(p.bits for p in self.points))

    def as_dict(self) -> dict:
        return {
            "params": dict(self.params),
            "simulated_bits": self.total_bits,
            "points": [p.as_dict() for p in self.points],
        }


def rayleigh_qpsk_ber(snr_db) -> np.ndarray:
    """Closed-form BER of QPSK over a single-tap Rayleigh channel.

    ``snr_db`` is ``Es/N0`` in dB, so ``Eb/N0 = Es/N0 / 2`` for Gray QPSK.
    """
    snr = 10 ** (np.asarray(snr_db, float) / 10)
    return 0.5 * (1 - np.sqrt(snr / 2 / (1 + snr / 2)))


def symbol_errors(xhat, x) -> int:
    """Number of wrong QPSK bits between two symbol vectors (I and Q counted)."""
    xhat = np.asarray(xhat, complex).reshape(-1)
    x = np.asarray(x, complex).reshape(-1)
    if xhat.size != x.size:
        raise ValueError(f"symbol vectors differ in length: {xhat.size} vs {x.size}")
    return int(np.sum(xhat.real * x.real < 0) + np.sum(xhat.imag * x.imag < 0))


def _simulate_frame(rng, N, M, n_paths, sigma2, method):
    """Draw one channel and one frame, return (errors, bits)."""
    taps = random_taps(rng, N, M, n_paths)
    H = dd_channel_matrix_cp(N, M, taps.gain, taps.delay, taps.doppler)
    x = QPSK[rng.integers(0, 4, N * M)]
    w = np.sqrt(sigma2 / 2) * (rng.standard_normal(N * M) + 1j * rng.standard_normal(N * M))
    xhat = mp_detect(H @ x + w, H, sigma2, method=method)
    return symbol_errors(xhat, x), 2 * N * M


def ber_curve(frames: int = 400, snr_db=None, N: int = 4, M: int = 8, n_paths: int = 3,
              seed: int | None = None, rng=None, method: str = "sparse",
              progress=None) -> BerResult:
    """Monte-Carlo BER of the OTFS/MP link versus SNR.

    Parameters
    ----------
    frames : int
        Frames (one channel realisation each) per SNR point, must be >= 1.
    snr_db : iterable of float, optional
        SNR grid in dB, default :func:`default_snr_db`.
    N, M : int
        Delay-Doppler grid size (N Doppler bins, M delay bins).
    n_paths : int
        Number of propagation paths per frame.
    seed : int, optional
        Seed for the random generator; ``rng`` takes precedence when given.
    rng : numpy.random.Generator, optional
        Generator to draw from, for reproducing a specific run.
    method : {"sparse", "dense"}
        Detector implementation, see :func:`otfs.detector.mp_detect`.
    progress : callable, optional
        Called with each :class:`BerPoint` as it is produced (used by the CLI).

    Returns
    -------
    BerResult
    """
    N, M = check_grid(N, M)
    frames = int(frames)
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    if rng is None:
        rng = np.random.default_rng(seed)
    snr_values = default_snr_db() if snr_db is None else np.asarray(list(snr_db), float).reshape(-1)
    if snr_values.size == 0:
        raise ValueError("snr_db must contain at least one value")
    if not np.all(np.isfinite(snr_values)):
        raise ValueError("snr_db contains a non-finite value")

    points = []
    for s in snr_values:
        sigma2 = 10 ** (-s / 10)
        errors = bits = 0
        for _ in range(frames):
            e, b = _simulate_frame(rng, N, M, n_paths, sigma2, method)
            errors += e
            bits += b
        point = BerPoint(snr_db=float(s), errors=errors, bits=bits)
        points.append(point)
        if progress is not None:
            progress(point)
    return BerResult(
        points=points,
        params={"frames": frames, "N": N, "M": M, "n_paths": n_paths,
                "method": method, "seed": seed},
    )


def toy_example(rng, out_dir=None) -> dict:
    """Reproduce the 2 x 4 example of matlab/prova.m.

    Returns the number of symbol errors of the noiseless MP detection and the
    received pilot frame; writes fig01/fig02 when ``out_dir`` is given.
    """
    N, M = 2, 4
    gain, delay, doppler = [1, 0.4], [0, 1], [0, -1]
    H = dd_channel_matrix(N, M, gain, delay, doppler)
    X = QPSK[rng.integers(0, 4, N * M)].reshape(N, M)
    GI = np.zeros((N, M))
    GI[N // 2 - 1, M // 2 - 1] = 10
    def flat(grid):
        """Row-major delay-Doppler vectorisation."""
        return grid.reshape(N * M)
    Ygi = (H @ flat(GI)).reshape(N, M)
    Yx = (H @ flat(X)).reshape(N, M)

    if out_dir is not None:
        from . import figures  # imported lazily: `import otfs` must not pull matplotlib

        out_dir = Path(out_dir)
        figures.save_channel_figure(out_dir / "fig01_dd_channel.png", N, M, gain, delay, doppler)
        figures.save_pilot_data_figure(out_dir / "fig02_dd_pilot_data.png",
                                       [(GI, "Input pilot"), (X, "Input data"),
                                        (Ygi, "Output pilot"), (Yx, "Output data")])

    xhat = mp_detect(H @ flat(X), H, 1e-6)
    return {
        "symbol_errors": symbol_errors(xhat, flat(X)),
        "received_pilot": Ygi,
        "taps": Taps(gain=np.asarray(gain, complex), delay=np.asarray(delay),
                     doppler=np.asarray(doppler)),
    }
