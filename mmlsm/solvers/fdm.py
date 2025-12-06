from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .linear import ProblemConfig


class FDMSolver:
    """
    Finite difference solver that applies the classical L1 scheme for Caputo derivatives.

    The solver assumes a uniform grid on ``[0, 1]`` with ``num_basis`` points taken
    from :class:`ProblemConfig`. A Dirichlet condition at ``x = 0`` is enforced
    explicitly, mirroring the Caputo formulation used by the spectral solver.
    """

    def __init__(
        self,
        config: ProblemConfig,
        *,
        a_func: Callable[[float], float],
        f_func: Callable[[float], float],
    ) -> None:
        if config.num_basis < 2:
            raise ValueError("FDMSolver requires at least two grid nodes.")
        self._config = config
        self._a_func = a_func
        self._f_func = f_func
        self._nodes = np.linspace(0.0, 1.0, config.num_basis, dtype=float)

        self._system_matrix: np.ndarray | None = None
        self._rhs: np.ndarray | None = None
        self._solution: np.ndarray | None = None

    def assemble_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the linear system corresponding to the L1 discretisation."""
        derivative = self._assemble_l1_matrix()
        a_vals = np.array([float(self._a_func(x)) for x in self._nodes], dtype=float)
        rhs = np.array([float(self._f_func(x)) for x in self._nodes], dtype=float)

        system = (self._config.epsilon ** self._config.alpha) * derivative
        system += np.diag(a_vals)

        self._system_matrix = system
        self._rhs = rhs
        self._solution = None
        return system.copy(), rhs.copy()

    def solve(self) -> np.ndarray:
        """Apply the boundary condition and solve the lower-triangular system."""
        if self._system_matrix is None or self._rhs is None:
            self.assemble_system()

        matrix = self._system_matrix.copy()
        rhs = self._rhs.copy()

        matrix[0, :] = 0.0
        matrix[0, 0] = 1.0
        rhs[0] = self._config.u0

        solution = np.linalg.solve(matrix, rhs)
        self._solution = solution
        return solution.copy()

    def get_solution(self, x_new: np.ndarray | float) -> np.ndarray | float:
        """Return the piecewise-linear interpolation of the discrete solution."""
        if self._solution is None:
            raise RuntimeError("Call solve() before requesting the solution.")

        scalar_input = np.isscalar(x_new)
        x_arr = np.atleast_1d(np.asarray(x_new, dtype=float))
        if np.any(x_arr < 0.0) or np.any(x_arr > 1.0):
            raise ValueError("Requested points must lie inside [0, 1].")

        values = np.interp(x_arr, self._nodes, self._solution)
        if scalar_input:
            return float(values[0])
        return values

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes.copy()

    @property
    def matrix(self) -> np.ndarray:
        if self._system_matrix is None:
            raise RuntimeError("Call assemble_system() before accessing the matrix.")
        return self._system_matrix.copy()

    @property
    def rhs(self) -> np.ndarray:
        if self._rhs is None:
            raise RuntimeError("Call assemble_system() before accessing the RHS.")
        return self._rhs.copy()

    @property
    def solution(self) -> np.ndarray:
        if self._solution is None:
            raise RuntimeError("Call solve() before accessing the solution vector.")
        return self._solution.copy()

    def _assemble_l1_matrix(self) -> np.ndarray:
        """Assemble the lower-triangular L1 discretisation matrix."""
        num_nodes = self._nodes.size
        if num_nodes < 2:
            raise RuntimeError("Need at least two nodes for L1 discretisation.")

        h = self._nodes[1] - self._nodes[0]
        if not np.allclose(np.diff(self._nodes), h):
            raise RuntimeError("FDMSolver expects a uniform grid.")

        weights = self._l1_weights(num_nodes - 1)
        scale = 1.0 / (math.gamma(2.0 - self._config.alpha) * (h ** self._config.alpha))

        matrix = np.zeros((num_nodes, num_nodes), dtype=float)
        for i in range(1, num_nodes):
            row = np.zeros(num_nodes, dtype=float)
            row[i] = weights[0]
            for j in range(1, i):
                col_idx = i - j
                row[col_idx] = weights[j] - weights[j - 1]
            row[0] -= weights[i - 1]
            matrix[i, :] = scale * row
        return matrix

    def _l1_weights(self, count: int) -> np.ndarray:
        """Return the incremental weights ``b_k = (k+1)^{1-alpha} - k^{1-alpha}``."""
        if count <= 0:
            raise ValueError("Weight count must be positive.")
        k = np.arange(0, count, dtype=float)
        exponent = 1.0 - self._config.alpha
        weights = np.power(k + 1.0, exponent) - np.power(k, exponent)
        return weights
