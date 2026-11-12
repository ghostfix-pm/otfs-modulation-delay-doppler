"""Detectors for the OTFS delay-Doppler input-output relation ``y = H x + w``.

The message-passing (MP) detector exploits the sparsity of ``H``: each received
sample depends on only ``P`` transmitted symbols, so the factor graph has ``P``
edges per node instead of the dense ``NM`` of the original implementation.
``mp_detect`` runs that sparse graph; ``method="dense"`` keeps the original
dense formulation so the two can be cross-checked against each other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["QPSK", "mmse_detect", "mp_detect"]

#: Gray-mapped unit-energy QPSK constellation.
QPSK = np.array([1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]) / np.sqrt(2)


class _FactorGraph:
    """Sparse factor graph of ``H``: variable/observation neighbour lists.

    ``var_of_obs[d, i]`` is the variable connected by the ``i``-th edge of
    observation ``d`` (``-1`` for padding), ``coef[d, i]`` the matching channel
    coefficient.  ``obs_of_var[c, j]`` / ``pos_of_var[c, j]`` give the inverse
    mapping, so a message can be routed from observation to variable and back
    without touching the ``NM x NM`` part of ``H`` beyond the non-zero entries.
    """

    def __init__(self, H, tol: float):
        H = np.asarray(H, dtype=complex)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(f"H must be a square matrix, got shape {H.shape}")
        NM = H.shape[0]
        self.NM = NM
        mask = np.abs(H) > tol
        per_row = mask.sum(axis=1)
        P = int(per_row.max()) if NM else 0
        if P == 0:
            raise ValueError("H has no non-zero entry above the tolerance")
        var_of_obs = np.full((NM, P), -1, dtype=np.int64)
        coef = np.zeros((NM, P), dtype=complex)
        for d in range(NM):
            cols = np.flatnonzero(mask[d])
            var_of_obs[d, :cols.size] = cols
            coef[d, :cols.size] = H[d, cols]

        columns: list[list[tuple[int, int]]] = [[] for _ in range(NM)]
        for d in range(NM):
            for i in range(int(per_row[d])):
                columns[var_of_obs[d, i]].append((d, i))
        Pc = max(len(c) for c in columns)
        obs_of_var = np.full((NM, Pc), -1, dtype=np.int64)
        pos_of_var = np.full((NM, Pc), -1, dtype=np.int64)
        for c, edges in enumerate(columns):
            for j, (d, i) in enumerate(edges):
                obs_of_var[c, j] = d
                pos_of_var[c, j] = i

        self.var_of_obs = var_of_obs
        self.coef = coef
        self.coef2 = np.abs(coef) ** 2
        self.mask = var_of_obs >= 0
        self.obs_of_var = obs_of_var
        self.pos_of_var = pos_of_var
        self.var_mask = obs_of_var >= 0
        # indices clipped to 0 for the padded slots, always combined with a mask
        self._og = np.where(self.var_mask, obs_of_var, 0)
        self._pg = np.where(self.var_mask, pos_of_var, 0)


def mp_detect(y, H, sigma2, alphabet=QPSK, iters=15, damp=0.6, tol=1e-9,
              method="sparse"):
    """Message-passing detection over the sparse factor graph of ``H``.

    Parameters
    ----------
    y : array_like, shape (NM,)
        Received samples.
    H : array_like, shape (NM, NM)
        Effective delay-Doppler channel matrix.
    sigma2 : float
        Noise variance per complex sample.
    alphabet : array_like, optional
        Symbol alphabet (unit average energy assumed for the variance update).
    iters, damp : int, float
        Number of iterations and the damping factor.
    tol : float
        Entries with ``|H| <= tol`` are treated as absent from the factor graph.
    method : {"sparse", "dense"}
        ``"dense"`` reproduces the original implementation, which allocates
        ``NM x NM x Q`` message arrays; it is kept for cross-checking only.

    Returns
    -------
    ndarray, shape (NM,)
        Hard decisions, one symbol per delay-Doppler cell.
    """
    if method not in ("sparse", "dense"):
        raise ValueError(f"method must be 'sparse' or 'dense', got {method!r}")
    y = np.asarray(y, dtype=complex).reshape(-1)
    H = np.asarray(H, dtype=complex)
    sigma2 = float(sigma2)
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {H.shape}")
    if y.size != H.shape[0]:
        raise ValueError(f"y has {y.size} samples but H is {H.shape[0]}x{H.shape[1]}")
    alphabet = np.asarray(alphabet, dtype=complex).reshape(-1)
    if alphabet.size < 2:
        raise ValueError("the alphabet needs at least two symbols")
    iters = int(iters)
    if iters < 1:
        raise ValueError(f"iters must be >= 1, got {iters}")
    if not 0.0 <= float(damp) <= 1.0:
        raise ValueError(f"damp must be in [0, 1], got {damp}")

    if method == "dense":
        return _mp_detect_dense(y, H, sigma2, alphabet, iters, damp, tol)
    return _mp_detect_sparse(y, H, sigma2, alphabet, iters, damp, tol)


def _mp_detect_dense(y, H, sigma2, alphabet, iters, damp, tol):
    """Reference implementation: dense ``NM x NM x Q`` message tensors."""
    NM, Q = H.shape[0], len(alphabet)
    mask = np.abs(H) > tol
    p = np.full((NM, NM, Q), 1 / Q)                    # p[c, d, q]: variable c -> observation d
    a, a2 = alphabet, np.abs(alphabet) ** 2
    absH2 = np.abs(H) ** 2
    Ltot = None
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
    if Ltot is None:                                   # pragma: no cover - iters >= 1
        raise RuntimeError("no message-passing iteration was executed")
    return alphabet[np.argmax(Ltot, axis=1)]


def _mp_detect_sparse(y, H, sigma2, alphabet, iters, damp, tol):
    """Same iteration, routed over the ``P`` non-zero entries per node."""
    graph = _FactorGraph(H, tol)
    Q = len(alphabet)
    a, a2 = alphabet, np.abs(alphabet) ** 2
    p = np.full(graph.var_of_obs.shape + (Q,), 1 / Q)  # p[d, i, q]: variable -> observation d
    Ltot = None
    for _ in range(iters):
        E = p @ a                                       # (NM, P)
        V = p @ a2 - np.abs(E) ** 2
        mu_tot = (graph.coef * E).sum(axis=1)           # sum_e H[d,e] E[e,d]
        var_tot = (graph.coef2 * V).sum(axis=1) + sigma2
        mu = mu_tot[:, None] - graph.coef * E           # mu[d, i]
        var = var_tot[:, None] - graph.coef2 * V
        residual = y[:, None, None] - mu[:, :, None] - graph.coef[:, :, None] * a[None, None, :]
        L = -np.abs(residual) ** 2 / var[:, :, None]
        L = np.where(graph.mask[:, :, None], L, 0.0)
        # route each likelihood to its variable, then sum over all observations
        Lg = L[graph._og, graph._pg]
        Lg = np.where(graph.var_mask[:, :, None], Lg, 0.0)
        Ltot = Lg.sum(axis=1)                           # (NM, Q)
        msg = Ltot[:, None, :] - Lg                     # exclude own observation
        msg -= msg.max(axis=2, keepdims=True)
        new = np.exp(msg); new /= new.sum(axis=2, keepdims=True)
        current = p[graph._og, graph._pg]
        updated = damp * new + (1 - damp) * current
        p[graph._og[graph.var_mask], graph._pg[graph.var_mask]] = updated[graph.var_mask]
    if Ltot is None:                                   # pragma: no cover - iters >= 1
        raise RuntimeError("no message-passing iteration was executed")
    return alphabet[np.argmax(Ltot, axis=1)]


def mmse_detect(y, H, sigma2, alphabet=QPSK) -> np.ndarray:
    """Linear MMSE detector, used as a baseline against the MP detector."""
    y = np.asarray(y, dtype=complex).reshape(-1)
    H = np.asarray(H, dtype=complex)
    sigma2 = float(sigma2)
    if sigma2 <= 0:
        raise ValueError(f"sigma2 must be positive, got {sigma2}")
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be a square matrix, got shape {H.shape}")
    if y.size != H.shape[0]:
        raise ValueError(f"y has {y.size} samples but H is {H.shape[0]}x{H.shape[1]}")
    alphabet = np.asarray(alphabet, dtype=complex).reshape(-1)
    estimate = np.linalg.solve(H.conj().T @ H + sigma2 * np.eye(H.shape[0]), H.conj().T @ y)
    return alphabet[np.argmin(np.abs(estimate[:, None] - alphabet[None, :]), axis=1)]
