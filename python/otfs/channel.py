"""Delay-Doppler channel models for OTFS.

Two equivalent-by-intent constructions are provided, mirroring the MATLAB
reference in ``matlab/``:

``dd_channel_matrix``
    Block/permutation model ported from ``matlab/channels.m``: the delay-time
    response is built with an explicit sample shift and reordered between
    delay-time and delay-Doppler indexing.  Its delay taps are indexed against
    the *first* grid axis (``l_i < N``), so taps outside that range cannot be
    represented by the model at all and are rejected here instead of being
    dropped silently.

``dd_channel_matrix_cp``
    Reduced-cyclic-prefix model: the effective matrix is the conjugate DFT
    transform of ``sum_i g_i Pi^{l_i} Delta^{k_i}`` (``Pi`` the cyclic shift,
    ``Delta`` the Doppler phase ramp).  Delay taps are indexed against the delay
    axis (``l_i < M``).

Both take the three tap vectors in parallel and validate them: a channel with
mismatched ``g``/``l``/``k`` lengths used to be silently truncated by ``zip``,
so a "three-path" channel could be simulated with two paths and no warning.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Taps",
    "check_grid",
    "dd_channel_matrix",
    "dd_channel_matrix_cp",
    "random_taps",
]


@dataclass(frozen=True)
class Taps:
    """Propagation paths of a doubly-dispersive channel.

    ``gain``, ``delay`` and ``doppler`` are parallel arrays; ``delay`` and
    ``doppler`` are integer bin indices on the delay-Doppler grid.
    """

    gain: np.ndarray
    delay: np.ndarray
    doppler: np.ndarray

    def __len__(self) -> int:
        return int(self.gain.size)


def check_grid(N: int, M: int) -> tuple[int, int]:
    """Validate the frame geometry and return it as ``(N, M)`` ints."""
    out = []
    for name, value in (("N", N), ("M", M)):
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer, got {value!r}") from None
        if ivalue != value:
            raise ValueError(f"{name} must be an integer, got {value!r}")
        if ivalue < 1:
            raise ValueError(f"{name} must be >= 1, got {ivalue}")
        out.append(ivalue)
    return out[0], out[1]


def _as_index(values: Sequence, name: str, count: int) -> np.ndarray:
    """Convert a tap vector to integers, rejecting non-integral and non-finite entries."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != count:
        raise ValueError(f"{name} has {arr.size} entries but {count} paths were declared")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains a non-finite value")
    rounded = np.rint(arr)
    bad = np.flatnonzero(np.abs(arr - rounded) > 1e-9)
    if bad.size:
        raise ValueError(
            f"{name} tap {int(bad[0])} is not an integer ({arr[bad[0]]!r}); "
            "the grid only supports integer delay/Doppler bins"
        )
    return rounded.astype(np.int64)


def _prepare(g, l, k, *, N: int, M: int, max_delay: int) -> Taps:
    """Validate tap vectors against a grid and return them as :class:`Taps`."""
    gains = np.asarray(g, dtype=complex).reshape(-1)
    if gains.size == 0:
        raise ValueError("a channel needs at least one path")
    for name, values in (("g", g), ("l", l), ("k", k)):
        if np.asarray(values).reshape(-1).size != gains.size:
            raise ValueError(
                f"tap vectors must have the same length: g has {gains.size}, "
                f"{name} has {np.asarray(values).reshape(-1).size}"
            )
    if not np.all(np.isfinite(gains)):
        raise ValueError("g contains a non-finite value")
    delay = _as_index(l, "l", gains.size)
    doppler = _as_index(k, "k", gains.size)
    if delay.min() < 0:
        raise ValueError(
            f"delay tap {int(np.argmin(delay))} is negative ({delay.min()}); "
            "delays are bin indices in [0, {max_delay})"
        )
    if delay.max() >= max_delay:
        raise ValueError(
            f"delay tap {int(np.argmax(delay))} is {delay.max()} but this channel "
            f"model supports delays in [0, {max_delay}); the path would be dropped "
            "silently otherwise"
        )
    NM = N * M
    if np.abs(doppler).max() >= NM:
        raise ValueError(
            f"doppler tap {int(np.argmax(np.abs(doppler)))} exceeds the grid "
            f"(|k| < N*M = {NM})"
        )
    return Taps(gain=gains, delay=delay, doppler=doppler)


def random_taps(rng, N: int, M: int, n_paths: int = 3) -> Taps:
    """Draw a random channel realisation used by the Monte-Carlo simulation.

    Delay taps are uniform in ``[0, M/2)`` and Doppler taps are distinct uniform
    draws from ``[-N/2, N/2]``, gains are ``CN(0, 1/n_paths)`` so that the
    average channel power stays independent of the number of paths.
    """
    N, M = check_grid(N, M)
    try:
        n_paths = int(n_paths)
    except (TypeError, ValueError):
        raise ValueError(f"n_paths must be an integer, got {n_paths!r}") from None
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}")
    max_delay = max(1, M // 2)
    doppler_bins = np.arange(-(N // 2), N // 2 + 1)
    if n_paths > doppler_bins.size:
        raise ValueError(
            f"n_paths={n_paths} does not fit the grid: only {doppler_bins.size} "
            f"distinct Doppler bins are available for N={N}"
        )
    l = rng.integers(0, max_delay, n_paths)
    k = rng.choice(doppler_bins, n_paths, replace=False)
    g = (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(2 * n_paths)
    return Taps(gain=g, delay=np.asarray(l, dtype=np.int64), doppler=np.asarray(k, dtype=np.int64))


def dd_channel_matrix(N: int, M: int, g, l, k) -> np.ndarray:
    """Effective NM x NM delay-Doppler channel (same construction as matlab/channels.m).

    Delay taps are limited to ``[0, N)`` by the construction, which represents
    the delay-time response on an ``N``-row grid.
    """
    N, M = check_grid(N, M)
    taps = _prepare(g, l, k, N=N, M=M, max_delay=N)
    NM = N * M
    z = np.exp(2j * np.pi / NM)
    gs = np.zeros((N, NM), complex)                    # delay-time response
    for a in range(N):
        for q in range(NM):
            for gi, li, ki in zip(taps.gain, taps.delay, taps.doppler):
                if a == li:
                    gs[a, q] += gi * z ** (ki * (q - li))
    G = np.zeros((NM, NM), complex)                    # delay-time channel matrix
    for q in range(NM):
        for ell in range(N):
            if q >= ell:
                G[q, q - ell] = gs[ell, q]
    P = np.zeros((NM, NM))                             # permutation matrix
    for a in range(M):
        for b in range(N):
            E = np.zeros((N, M)); E[b, a] = 1
            P[a * N:(a + 1) * N, b * M:(b + 1) * M] = E
    Fm = np.fft.fft(np.eye(M))                         # dftmtx(M)
    K = np.kron(np.eye(N), Fm)
    return (1 / M) * K @ (P.T @ G @ P) @ K.conj().T


def dd_channel_matrix_cp(N: int, M: int, g, l, k, method: str = "taps") -> np.ndarray:
    """Reduced-CP (circular) OTFS model: M delay bins, N Doppler bins.

    ``H_t = sum_i g_i Pi^{l_i} Delta^{k_i}``, ``H_dd = (F_N (x) I_M) H_t (F_N^H (x) I_M)``.

    ``method="taps"`` assembles ``H_t`` in ``O(P*NM)`` and applies the block DFT
    with two ``tensordot`` calls, which is what makes frames beyond a few
    hundred cells practical; ``method="reference"`` builds every matrix densely
    with ``matrix_power`` (the original formulation) and is kept for
    cross-checking.  Both agree to floating-point precision.
    """
    N, M = check_grid(N, M)
    taps = _prepare(g, l, k, N=N, M=M, max_delay=M)
    if method == "reference":
        return _dd_channel_matrix_cp_reference(N, M, taps)
    if method != "taps":
        raise ValueError(f"method must be 'taps' or 'reference', got {method!r}")
    NM = N * M
    Ht = np.zeros((NM, NM), complex)
    samples = np.arange(NM)
    for gi, li, ki in zip(taps.gain, taps.delay, taps.doppler):
        # Pi^{l_i} shifts sample q to q - l_i, Delta^{k_i} multiplies it by the
        # Doppler phase; two taps on the same delay bin add up.
        columns = (samples - int(li)) % NM
        Ht[samples, columns] += gi * np.exp(2j * np.pi * int(ki) * columns / NM)
    F = np.fft.fft(np.eye(N))
    blocks = Ht.reshape(N, M, N, M)
    left = np.tensordot(F / np.sqrt(N), blocks, axes=([1], [0]))        # (n, m, k', m')
    both = np.tensordot(left, F.conj() / np.sqrt(N), axes=([2], [1]))   # (n, m, m', n')
    return both.transpose(0, 1, 3, 2).reshape(NM, NM)


def _dd_channel_matrix_cp_reference(N: int, M: int, taps: Taps) -> np.ndarray:
    """Original formulation: dense shift/phase matrices and ``matrix_power``."""
    NM = N * M
    Pi = np.roll(np.eye(NM), 1, axis=0)                # cyclic delay by one sample
    D = np.diag(np.exp(2j * np.pi * np.arange(NM) / NM))  # Doppler phase ramp
    Ht = sum(gi * np.linalg.matrix_power(Pi, int(li)) @ np.linalg.matrix_power(D, int(ki) % NM)
             for gi, li, ki in zip(taps.gain, taps.delay, taps.doppler))
    A = np.kron(np.fft.fft(np.eye(N)) / np.sqrt(N), np.eye(M))
    return A @ Ht @ A.conj().T
