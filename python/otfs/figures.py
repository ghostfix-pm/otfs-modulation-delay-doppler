"""Figure generation for the OTFS study.

Every helper takes an explicit output directory: the original module wrote into
``<repo>/docs/figures`` relative to its own file, which points into
``site-packages`` once the package is installed.  :func:`default_output_dir`
picks the source tree when it is there and falls back to the working directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are written to disk, never shown

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "bar3",
    "default_output_dir",
    "display_dd_response",
    "save_ber_figure",
    "save_channel_figure",
    "save_pilot_data_figure",
]


def default_output_dir(module_path=None, cwd=None) -> Path:
    """Directory used when the caller does not pass one.

    Uses ``<checkout>/docs/figures`` when the module runs from the repository
    and ``./figures`` otherwise, so an installed copy never writes into
    ``site-packages``.
    """
    here = Path(module_path).resolve() if module_path is not None else Path(__file__).resolve()
    candidate = here.parents[2]
    if (candidate / "docs").is_dir():
        return candidate / "docs" / "figures"
    return Path(cwd if cwd is not None else Path.cwd()) / "figures"


def display_dd_response(N: int, M: int, gain, delay, doppler) -> np.ndarray:
    """N x M delay-Doppler response map used for the figure.

    The Doppler column is offset by ``M // 2 - 1`` so both paths of the
    two-path example land inside the displayed grid; this is a plotting
    convention of the original study, not the raw bin index.
    """
    response = np.zeros((N, M))
    for gi, li, ki in zip(gain, delay, doppler):
        response[int(li), int(ki) + M // 2 - 1] = gi
    return response


def bar3(ax, Z, title):
    """Annotated magnitude map of a delay-Doppler grid."""
    A = np.abs(Z)
    n, m = A.shape
    ax.imshow(A, cmap="Blues", origin="lower", vmin=0, vmax=A.max() * 1.15 if A.max() else 1)
    for i in range(n):
        for j in range(m):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if A[i, j] > 0.6 * A.max() else "black")
    ax.set_xlabel("Doppler index"); ax.set_ylabel("Delay index"); ax.set_title(title, fontsize=9)
    ax.set_xticks(range(m)); ax.set_yticks(range(n))


def save_channel_figure(path, N, M, gain, delay, doppler) -> Path:
    """Write the delay-Doppler channel response figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4.2, 2.6))
    bar3(fig.add_subplot(111), display_dd_response(N, M, gain, delay, doppler),
         "Delay-Doppler channel response")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def save_pilot_data_figure(path, panels) -> Path:
    """Write the pilot/data input-output figure.

    ``panels`` is a sequence of ``(grid, title)`` pairs, four of them in the
    original study (input pilot, input data, output pilot, output data).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 5.4))
    for i, (grid, title) in enumerate(panels):
        bar3(fig.add_subplot(2, 2, i + 1), grid, title)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path


def save_ber_figure(path, result, reference=None) -> Path:
    """Write the BER-vs-SNR figure for a :class:`otfs.sim.BerResult`.

    Points where no error was observed are not drawn as a rate: they appear as
    open markers at the 95% upper bound ``3/bits``, so a curve that runs out of
    errors at high SNR does not look like an exact zero.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snr_db = np.array([p.snr_db for p in result.points], float)
    ber = np.array([p.ber if p.errors else np.nan for p in result.points], float)
    bounds = np.array([p.upper_bound if not p.errors else np.nan for p in result.points], float)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    label = (f"OTFS (N={result.params['N']}, M={result.params['M']}) + MP detector, "
             f"{result.params['n_paths']}-path channel")
    ax.semilogy(snr_db, ber, "o-", color="tab:blue", label=label)
    if np.any(np.isfinite(bounds)):
        ax.semilogy(snr_db, bounds, "o", mfc="none", color="tab:blue",
                    label="no error observed (95% upper bound)")
    if reference is not None:
        ax.semilogy(snr_db, np.asarray(reference, float), "k--", lw=1,
                    label="QPSK, single-tap Rayleigh (reference)")
    ax.set_xlabel("SNR, $E_s/N_0$ (dB)"); ax.set_ylabel("Bit error rate")
    ax.grid(True, which="both", alpha=0.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return path
