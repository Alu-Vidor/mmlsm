from __future__ import annotations

from typing import Sequence

import mpmath as mp
import numpy as np

from mmlsm.core.interfaces import LinearOperator, SpectralBasis


class FractionalDerivativeOperator(LinearOperator):
    """Caputo fractional differentiation matrix evaluated on supplied nodes."""

    def __init__(self, order: float, *, mp_dps: int = 80) -> None:
        if order <= 0.0:
            raise ValueError("Fractional order must be positive.")
        if mp_dps <= 0:
            raise ValueError("mp_dps must be positive.")
        self._order = float(order)
        self._mp_dps = int(mp_dps)

    def assemble(self, basis: SpectralBasis, nodes: np.ndarray) -> np.ndarray:
        nodes_arr = np.array(nodes, dtype=float, copy=True).reshape(-1)
        if nodes_arr.size == 0:
            raise ValueError("At least one collocation node is required.")
        if np.any(nodes_arr < 0.0):
            raise ValueError("Caputo derivatives require non-negative nodes.")

        transition = basis.transition_matrix()
        lambdas = basis.exponents()
        num_nodes = nodes_arr.size

        with mp.workdps(self._mp_dps):
            mp_nodes = [mp.mpf(x) for x in nodes_arr]
            mono_samples = self._fractional_monomial_samples(lambdas, mp_nodes)

            mp_columns: list[list[mp.mpf]] = []
            for row in transition:
                column = [mp.mpf("0")] * num_nodes
                for k, weight in enumerate(row):
                    if weight == 0.0:
                        continue
                    samples_k = mono_samples[k]
                    if samples_k is None:
                        continue
                    mp_weight = mp.mpf(weight)
                    for idx in range(num_nodes):
                        column[idx] += mp_weight * samples_k[idx]
                mp_columns.append(column)

        matrix = np.array([[float(value) for value in col] for col in mp_columns])
        return matrix.T

    def _fractional_monomial_samples(
        self, lambdas: Sequence[float], nodes: list[mp.mpf]
    ) -> list[list[mp.mpf] | None]:
        """Evaluate ``D^alpha x^{lambda_k}`` on the supplied nodes."""
        samples: list[list[mp.mpf] | None] = []
        order = mp.mpf(self._order)
        for lam in lambdas:
            if lam == 0.0:
                samples.append(None)
                continue

            lam_mp = mp.mpf(lam)
            coeff = mp.gamma(lam_mp + 1.0) / mp.gamma(lam_mp - order + 1.0)
            exponent = lam_mp - order
            row = [coeff * (x ** exponent) for x in nodes]
            samples.append(row)
        return samples
