"""
OTFS modulation over a doubly-dispersive channel with message-passing detection.

Python counterpart of the MATLAB code in ../matlab:
  * builds the effective delay-Doppler channel matrix H from a set of
    (gain, delay, Doppler) taps,
  * illustrates how a pilot and a QAM data frame are spread by the channel
    (fig01, fig02),
  * evaluates the BER of a message-passing (MP) detector versus SNR (fig03).

    python otfs_sim.py [--frames 400] [--seed 3]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"
QPSK = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)


# ---------------------------------------------------------------- channel model
def dd_channel_matrix(N, M, g, l, k):
    """Effective NM x NM delay-Doppler channel (same construction as matlab/channels.m)."""
    NM = N * M
    z = np.exp(2j * np.pi / NM)
    gs = np.zeros((N, NM), complex)                    # delay-time response
    for a in range(N):
        for q in range(NM):
            for gi, li, ki in zip(g, l, k):
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


def dd_channel_matrix_cp(N, M, g, l, k):
    """Reduced-CP (circular) OTFS model: M delay bins, N Doppler bins.

    H_t = sum_i g_i Pi^{l_i} Delta^{k_i},  H_dd = (F_N (x) I_M) H_t (F_N^H (x) I_M)
    """
    NM = N * M
    Pi = np.roll(np.eye(NM), 1, axis=0)                # cyclic delay by one sample
    D = np.diag(np.exp(2j * np.pi * np.arange(NM) / NM))  # Doppler phase ramp
    Ht = sum(gi * np.linalg.matrix_power(Pi, li) @ np.linalg.matrix_power(D, ki % NM)
             for gi, li, ki in zip(g, l, k))
    A = np.kron(np.fft.fft(np.eye(N)) / np.sqrt(N), np.eye(M))
    return A @ Ht @ A.conj().T


# ---------------------------------------------------------------- MP detector
def mp_detect(y, H, sigma2, alphabet=QPSK, iters=15, damp=0.6, tol=1e-9):
    """Message-passing detection over the sparse factor graph of H."""
    NM, Q = H.shape[0], len(alphabet)
    mask = np.abs(H) > tol
    p = np.full((NM, NM, Q), 1 / Q)                    # p[c, d, q]: variable c -> observation d
    a, a2 = alphabet, np.abs(alphabet) ** 2
    absH2 = np.abs(H) ** 2
    for _ in range(iters):
        E = p @ a                                       # E[c, d]
        V = p @ a2 - np.abs(E) ** 2                     # V[c, d]
        Hm = np.where(mask, H, 0)
        tot_mu = np.sum(Hm * E.T, axis=1)               # sum_e H[d,e] E[e,d]
        tot_var = np.sum(np.where(mask, absH2, 0) * V.T, axis=1)
        mu = tot_mu[:, None] - Hm * E.T                 # mu[d, c]
        var = tot_var[:, None] - np.where(mask, absH2, 0) * V.T + sigma2
        # log-likelihoods L[d, c, q]
        L = -np.abs(y[:, None, None] - mu[:, :, None] - Hm[:, :, None] * a[None, None, :]) ** 2 / var[:, :, None]
        L = np.where(mask[:, :, None], L, 0.0)
        Ltot = L.sum(axis=0)                            # Ltot[c, q] = sum_d L[d, c, q]
        msg = Ltot[:, None, :] - np.transpose(L, (1, 0, 2))   # exclude own observation d
        msg -= msg.max(axis=2, keepdims=True)
        new = np.exp(msg); new /= new.sum(axis=2, keepdims=True)
        p = damp * new + (1 - damp) * p
    return alphabet[np.argmax(Ltot, axis=1)]


# ---------------------------------------------------------------- figures
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


def toy_example(rng):
    """Reproduces the 2 x 4 example of matlab/prova.m."""
    N, M = 2, 4
    g, l, k = [1, 0.4], [0, 1], [0, -1]
    Hdd = np.zeros((N, M))
    for gi, li, ki in zip(g, l, k):
        Hdd[li, ki + M // 2 - 1] = gi
    H = dd_channel_matrix(N, M, g, l, k)
    X = QPSK[rng.integers(0, 4, N * M)].reshape(N, M)
    GI = np.zeros((N, M)); GI[N // 2 - 1, M // 2 - 1] = 10
    to_dd = lambda Z: Z.reshape(N * M)                  # row-major DD vectorisation
    Ygi = (H @ to_dd(GI)).reshape(N, M)
    Yx = (H @ to_dd(X)).reshape(N, M)

    fig = plt.figure(figsize=(4.2, 2.6))
    bar3(fig.add_subplot(111), Hdd, "Delay-Doppler channel response")
    fig.tight_layout(); fig.savefig(FIG / "fig01_dd_channel.png", dpi=160); plt.close(fig)

    fig = plt.figure(figsize=(8, 5.4))
    for i, (Z, t) in enumerate([(GI, "Input pilot"), (X, "Input data"),
                                (Ygi, "Output pilot"), (Yx, "Output data")]):
        bar3(fig.add_subplot(2, 2, i + 1), Z, t)
    fig.tight_layout(); fig.savefig(FIG / "fig02_dd_pilot_data.png", dpi=160); plt.close(fig)

    xhat = mp_detect(H @ to_dd(X), H, 1e-6)
    print("toy example – noiseless MP detection, symbol errors:", int(np.sum(np.abs(xhat - to_dd(X)) > 1e-6)))


def ber_curve(rng, frames, N=4, M=8, n_paths=3):
    snr_db = np.arange(0, 21, 2)
    ber = []
    for s in snr_db:
        sigma2 = 10 ** (-s / 10)
        err = tot = 0
        for _ in range(frames):
            l = rng.integers(0, M // 2, n_paths)                                   # delay taps
            k = rng.choice(np.arange(-N // 2, N // 2 + 1), n_paths, replace=False)  # Doppler taps
            g = (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(2 * n_paths)
            H = dd_channel_matrix_cp(N, M, g, l, k)
            x = QPSK[rng.integers(0, 4, N * M)]
            w = np.sqrt(sigma2 / 2) * (rng.standard_normal(N * M) + 1j * rng.standard_normal(N * M))
            xhat = mp_detect(H @ x + w, H, sigma2)
            # Gray-mapped QPSK: bit errors follow the signs of I and Q
            err += np.sum(np.sign(xhat.real) != np.sign(x.real)) + np.sum(np.sign(xhat.imag) != np.sign(x.imag))
            tot += 2 * N * M
        ber.append(err / tot)
        print(f"SNR {s:2d} dB  BER {ber[-1]:.2e}")

    snr = 10 ** (snr_db / 10)
    fig, ax = plt.subplots(figsize=(5.6, 4))
    b = np.array(ber, float); b[b == 0] = np.nan
    ax.semilogy(snr_db, b, "o-", color="tab:blue", label=f"OTFS (N={N}, M={M}) + MP detector, {n_paths}-path channel")
    ax.semilogy(snr_db, 0.5 * (1 - np.sqrt(snr / 2 / (1 + snr / 2))), "k--", lw=1,
                label="QPSK, single-tap Rayleigh (reference)")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Bit error rate")
    ax.grid(True, which="both", alpha=0.3); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig03_ber_mp.png", dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    toy_example(rng)
    ber_curve(rng, a.frames)


if __name__ == "__main__":
    main()
