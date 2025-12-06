from __future__ import annotations

from typing import Iterable

import numpy as np

from mmlsm.core.interfaces import BasisEvaluation, SpectralBasis


class MuntzLegendreBasis(SpectralBasis):
    """
    Construct Müntz-Legendre polynomials for an arithmetic index set.
    """

    def __init__(self, alpha: float, N: int) -> None:
        if N < 0:
            raise ValueError("N must be non-negative.")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive to define the Müntz sequence.")

        self.alpha: float = float(alpha)
        self.N: int = int(N)
        self._lambdas: np.ndarray = self.alpha * np.arange(self.N + 1, dtype=float)
        gram = 1.0 / (self._lambdas[:, None] + self._lambdas[None, :] + 1.0)
        self._gram = gram
        self._transition: np.ndarray = np.zeros(
            (self.N + 1, self.N + 1), dtype=float, order="C"
        )
        self._construct_transition()

    def _inner(self, coeff_a: np.ndarray, coeff_b: np.ndarray) -> float:
        """L2 inner product between two fractional monomials."""
        return float(coeff_a @ (self._gram @ coeff_b))

    def _construct_transition(self) -> None:
        """Modified Gram-Schmidt that produces the orthonormal polynomials."""
        tol = 1e-14
        for n in range(self.N + 1):
            v = np.zeros(self.N + 1, dtype=float)
            v[n] = 1.0
            for m in range(n):
                projection = self._inner(v, self._transition[m])
                v -= projection * self._transition[m]

            norm_sq = self._inner(v, v)
            if norm_sq < 0.0:
                norm_sq = abs(norm_sq)
            norm = np.sqrt(norm_sq)
            if not np.isfinite(norm) or norm < tol:
                raise RuntimeError("Encountered a degenerate vector during orthonormalisation.")
            self._transition[n] = v / norm

    def evaluate(self, x: Iterable[float] | float) -> BasisEvaluation:
        x_arr = np.array(x, dtype=float, copy=True)
        scalar_input = x_arr.ndim == 0
        if scalar_input:
            x_eval = x_arr.reshape(1)
        else:
            x_eval = x_arr

        powers = np.power(x_eval[np.newaxis, ...], self._lambdas[:, np.newaxis])
        values = self._transition @ powers
        if scalar_input:
            values = values[:, 0]

        return BasisEvaluation(x=x_eval.copy(), values=values)

    def size(self) -> int:
        return self.N + 1

    def transition_matrix(self) -> np.ndarray:
        return self._transition.copy()

    def exponents(self) -> np.ndarray:
        return self._lambdas.copy()
