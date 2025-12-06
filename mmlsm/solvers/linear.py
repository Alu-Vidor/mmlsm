from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from mmlsm.basis.muntz import MuntzLegendreBasis
from mmlsm.core.interfaces import GridGenerator, LinearOperator, SpectralBasis
from mmlsm.grid.mappers import CGLGrid, create_mapper
from mmlsm.operators import FractionalDerivativeOperator

if TYPE_CHECKING:
    from mmlsm.solvers.fdm import FDMSolver


@dataclass(frozen=True)
class ProblemConfig:
    """Container with the fundamental physical and discretisation parameters."""

    epsilon: float
    alpha: float
    u0: float
    N: int

    @property
    def num_basis(self) -> int:
        return self.N + 1


class SpectralSolver:
    """Grid-agnostic solver that operates purely on spectral abstractions."""

    def __init__(
        self,
        config: ProblemConfig,
        basis: SpectralBasis,
        grid: GridGenerator,
        operator: LinearOperator,
        a_func: Callable[[float], float],
        f_func: Callable[[float], float],
    ) -> None:
        if basis.size() != config.num_basis:
            raise ValueError("Basis size must agree with ProblemConfig.N.")

        self._config = config
        self._basis = basis
        self._grid = grid
        self._operator = operator
        self._a_func = a_func
        self._f_func = f_func

        self._nodes: np.ndarray | None = None
        self._matrix: np.ndarray | None = None
        self._rhs: np.ndarray | None = None
        self._coeffs: np.ndarray | None = None

    def assemble_system(self) -> tuple[np.ndarray, np.ndarray]:
        """Construct the linear system before applying constraints."""
        x_nodes, _ = self._grid.generate_nodes(self._config.epsilon)
        order = np.argsort(x_nodes)
        nodes = np.array(x_nodes[order], copy=True)
        if nodes.size != self._basis.size():
            raise ValueError("Grid size must match the number of basis functions.")

        operator_matrix = self._operator.assemble(self._basis, nodes)
        a_vals = np.array([float(self._a_func(x)) for x in nodes], dtype=float)
        f_vals = np.array([float(self._f_func(x)) for x in nodes], dtype=float)
        system_matrix = (self._config.epsilon ** self._config.alpha) * operator_matrix
        system_matrix += np.diag(a_vals)

        self._nodes = nodes
        self._matrix = system_matrix
        self._rhs = f_vals
        self._coeffs = None
        return system_matrix.copy(), f_vals.copy()

    def solve(self) -> np.ndarray:
        """Apply the Dirichlet condition at ``x=0`` and solve for coefficients."""
        if self._matrix is None or self._rhs is None:
            self.assemble_system()

        matrix = self._matrix.copy()
        rhs = self._rhs.copy()

        boundary_row = np.array(self._basis.evaluate(0.0).values, dtype=float, copy=True)
        if boundary_row.ndim != 1 or boundary_row.size != matrix.shape[1]:
            raise RuntimeError("Boundary evaluation is incompatible with the system size.")

        matrix[0, :] = boundary_row
        rhs[0] = self._config.u0

        coeffs = np.linalg.solve(matrix, rhs)
        self._coeffs = coeffs
        return coeffs.copy()

    def get_solution(self, x_new: np.ndarray | float) -> np.ndarray | float:
        """Evaluate ``u(x) = sum_j c_j L_j(x)`` at new points."""
        if self._coeffs is None:
            raise RuntimeError("Call solve() before requesting the solution.")

        evaluation = self._basis.evaluate(x_new)
        u_vals = np.tensordot(self._coeffs, evaluation.values, axes=(0, 0))
        return u_vals

    @property
    def coefficients(self) -> np.ndarray:
        if self._coeffs is None:
            raise RuntimeError("No coefficients are available before solve() is called.")
        return self._coeffs.copy()

    @property
    def nodes(self) -> np.ndarray | None:
        if self._nodes is None:
            return None
        return self._nodes.copy()


class SolverFactory:
    """Utility that wires the standard MMLSM solver stack."""

    def __init__(self, *, mp_dps: int = 80) -> None:
        if mp_dps <= 0:
            raise ValueError("mp_dps must be positive.")
        self._mp_dps = mp_dps

    def create_mmlsm_solver(
        self,
        config: ProblemConfig,
        *,
        mapping: str,
        a_func: Callable[[float], float],
        f_func: Callable[[float], float],
    ) -> SpectralSolver:
        basis = MuntzLegendreBasis(alpha=config.alpha, N=config.N)
        mapper = create_mapper(mapping)
        grid = CGLGrid(num_points=config.num_basis, mapper=mapper)
        operator = FractionalDerivativeOperator(order=config.alpha, mp_dps=self._mp_dps)
        return SpectralSolver(config, basis, grid, operator, a_func, f_func)

    def create_fdm_solver(
        self,
        config: ProblemConfig,
        *,
        a_func: Callable[[float], float],
        f_func: Callable[[float], float],
    ) -> FDMSolver:
        from mmlsm.solvers.fdm import FDMSolver

        return FDMSolver(config, a_func=a_func, f_func=f_func)
