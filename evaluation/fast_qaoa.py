"""Exact NumPy simulator for QAOA on a diagonal (Ising) cost Hamiltonian.

Used only for the scaling, depth and cost-accounting experiments in
evaluation/extension_study.py, where Aer's per-evaluation overhead would make
n = 16..20 impractical. It is cross-validated against the Aer EstimatorV2
path used everywhere else (see evaluation/validate_fast_qaoa.py).

Conventions match Qiskit's QAOAAnsatz: initial state |+>^n, cost layer
exp(-i gamma H) with H = sum_{i<j} 2 w_i w_j Z_i Z_j, mixer RX(2 beta) on
every qubit, parameters ordered [beta_1..beta_p, gamma_1..gamma_p].
Bit i of a basis index is task i (little-endian), as in the package.
"""

from __future__ import annotations

import numpy as np


def cost_diagonal(weights: list[float]) -> np.ndarray:
    """c[b] = (sum_i w_i s_i)^2 for every basis state b (s_i = +1 if bit i set)."""
    n = len(weights)
    d = np.zeros(1, dtype=np.float64)
    for i in range(n):  # build the signed sum one bit at a time
        w = weights[i]
        d = np.concatenate([d - w, d + w])  # new top bit: 0 -> -w, 1 -> +w
    # After the loop bit i of the index is task i (the i-th doubling is bit i).
    return d * d


def ising_diagonal(weights: list[float]) -> np.ndarray:
    """H = sum_{i<j} 2 w_i w_j Z_i Z_j = (sum w s)^2 - sum w^2."""
    return cost_diagonal(weights) - float(np.sum(np.square(weights)))


_POP: dict[int, np.ndarray] = {}


def _popcount(n: int) -> np.ndarray:
    if n not in _POP:
        pc = np.zeros(1, dtype=np.int64)
        for _ in range(n):
            pc = np.concatenate([pc, pc + 1])
        _POP[n] = pc
    return _POP[n]


def _wht(psi: np.ndarray, n: int) -> None:
    """In-place unnormalized Walsh-Hadamard transform (butterflies on each bit)."""
    for ax in range(n):
        x = psi.reshape(-1, 2, 2**ax)
        a = x[:, 0, :].copy()
        x[:, 0, :] += x[:, 1, :]
        x[:, 1, :] = a - x[:, 1, :]


def _apply_mixer(psi: np.ndarray, n: int, beta: float) -> np.ndarray:
    """exp(-i beta sum_i X_i) = H^n diag(exp(-i beta (n - 2|b|))) H^n."""
    psi = psi.copy()
    _wht(psi, n)
    psi *= np.exp(-1j * beta * (n - 2 * _popcount(n))) / 2**n
    _wht(psi, n)
    return psi


def qaoa_state(h_diag: np.ndarray, n: int, params: np.ndarray) -> np.ndarray:
    p = len(params) // 2
    betas, gammas = params[:p], params[p:]
    psi = np.full(2**n, 2 ** (-n / 2), dtype=np.complex128)
    for layer in range(p):
        psi = psi * np.exp(-1j * gammas[layer] * h_diag)
        psi = _apply_mixer(psi, n, betas[layer])
    return psi


def expectation(h_diag: np.ndarray, n: int, params: np.ndarray) -> float:
    psi = qaoa_state(h_diag, n, params)
    return float(np.real(np.vdot(psi, h_diag * psi)))


def probabilities(h_diag: np.ndarray, n: int, params: np.ndarray) -> np.ndarray:
    psi = qaoa_state(h_diag, n, params)
    pr = np.abs(psi) ** 2
    return pr / pr.sum()
