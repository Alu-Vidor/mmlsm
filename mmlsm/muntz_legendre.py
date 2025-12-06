"""
Core numerical routines for the Müntz–Legendre basis used in MMLSM.

The implementation follows the classical construction where the fractional
monomials ``x^{lambda_k}`` (``lambda_k = k * alpha``) are orthonormalised on
``[0, 1]`` with respect to the standard ``L2`` inner product.  This produces the
Müntz–Legendre polynomials ``L_{n, Λ}(x)`` and the triangular transition matrix
between the Müntz and the monomial bases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import mpmath as mp
import numpy as np


class GridMapper:
    """
    Generate Chebyshev-Gauss-Lobatto nodes and map them to ``[0, 1]`` with a
    controllable clustering near ``x = 0``.

    Parameters
    ----------
    num_points : int
        Number of collocation nodes. Must be at least two to include both
        boundaries.
    mapping : str, optional
        Mapping strategy that controls the clustering. Supported options are
        ``"algebraic"`` and ``"logarithmic"``. The former applies a power-law
        stretch, while the latter uses a smooth logarithmic mapping that keeps
        the Jacobian bounded.
    """

    _EPS_FLOOR: float = 1.0e-12
    _MAX_STRENGTH: float = 12.0

    def __init__(self, num_points: int, mapping: str = "algebraic") -> None:
        if num_points < 2:
            raise ValueError("At least two points are required to form a CGL grid.")
        mapping_normalised = mapping.lower().strip()
        if mapping_normalised not in {"algebraic", "logarithmic", "log"}:
            raise ValueError(
                "Unknown mapping strategy. Supported values: 'algebraic', 'logarithmic'."
            )

        self.num_points: int = int(num_points)
        self.mapping: str = (
            "logarithmic"
            if mapping_normalised.startswith("log")
            else "algebraic"
        )
        self._xi: np.ndarray = self._generate_cgl_nodes()

    def _generate_cgl_nodes(self) -> np.ndarray:
        N = self.num_points - 1
        j = np.arange(self.num_points, dtype=float)
        return np.cos(np.pi * j / N)

    def collocation_nodes(self) -> np.ndarray:
        """
        Return a copy of the Chebyshev-Gauss-Lobatto nodes ``xi`` in ``[-1, 1]``.
        """
        return self._xi.copy()

    def map_nodes(self, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Map the CGL nodes to the physical interval ``[0, 1]`` using the
        transformation ``g(xi, epsilon)`` and return both the coordinates and
        the mapping metric ``dx/dxi``.
        """
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive for the mapping to be defined.")

        if self.mapping == "algebraic":
            return self._algebraic_map(epsilon)
        return self._logarithmic_map(epsilon)

    def _boundary_layer_strength(self, epsilon: float) -> float:
        eps_clamped = max(float(epsilon), self._EPS_FLOOR)
        strength = max(0.0, -np.log10(eps_clamped))
        return min(strength, self._MAX_STRENGTH)

    def _algebraic_map(self, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        xi = self._xi
        r = 0.5 * (xi + 1.0)
        exponent = 1.0 + self._boundary_layer_strength(epsilon)
        x = np.power(r, exponent)
        metric = 0.5 * exponent * np.power(r, exponent - 1.0)
        return x, metric

    def _logarithmic_map(self, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        xi = self._xi
        r = 0.5 * (xi + 1.0)
        beta = 1.0 + self._boundary_layer_strength(epsilon)
        expm1_beta = np.expm1(beta)
        numerator = np.log1p(expm1_beta * r)
        x = numerator / beta
        metric = 0.5 * (expm1_beta / (1.0 + expm1_beta * r)) / beta
        return x, metric


@dataclass(frozen=True)
class EvaluationResult:
    """
    Container that stores the sampled values of the basis functions.

    Attributes
    ----------
    x : numpy.ndarray
        Points where the basis was evaluated. The array stores a copy to keep
        the result self-contained.
    values : numpy.ndarray
        Array with shape ``(N + 1, *x.shape)`` that contains the samples
        ``L_{n, Λ}(x)`` for all ``n`` in ``[0, N]``.
    """

    x: np.ndarray
    values: np.ndarray


class MuntzLegendreBasis:
    """
    Construct Müntz–Legendre polynomials for the arithmetic index set.

    Parameters
    ----------
    alpha : float
        Step that defines the exponents ``lambda_k = k * alpha``. Must be
        positive to keep the moments finite.
    N : int
        Number of basis functions ``(N + 1)`` to construct.

    Notes
    -----
    The polynomials are defined through the orthonormality condition

    .. math::
        \\int_0^1 L_{n, \\Lambda}(x) L_{m, \\Lambda}(x) \\, dx = \\delta_{mn},

    which yields a numerically stable basis when the Gram matrix of the
    fractional monomials ``x^{\\lambda_k}`` is explicitly known. The class
    computes the transition matrix ``S`` with entries ``C_{n,k}`` such that

    .. math::
        L_{n, \\Lambda}(x) = \\sum_{k=0}^n C_{n,k} x^{\\lambda_k}.
    """

    def __init__(self, alpha: float, N: int) -> None:
        if N < 0:
            raise ValueError("N must be non-negative.")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive to define the Müntz sequence.")

        self.alpha: float = float(alpha)
        self.N: int = int(N)
        self.lambdas: np.ndarray = self.alpha * np.arange(self.N + 1, dtype=float)
        # Closed-form Gram matrix <x^{lambda_i}, x^{lambda_j}>.
        self._gram: np.ndarray = 1.0 / (
            self.lambdas[:, None] + self.lambdas[None, :] + 1.0
        )
        self._transition: np.ndarray = np.zeros(
            (self.N + 1, self.N + 1), dtype=float, order="C"
        )
        self._construct_transition()

    def _inner(self, coeff_a: np.ndarray, coeff_b: np.ndarray) -> float:
        """
        Compute the L2 inner product between two polynomials represented in the
        fractional monomial basis.
        """
        return float(coeff_a @ (self._gram @ coeff_b))

    def _construct_transition(self) -> None:
        """
        Modified Gram-Schmidt process that yields orthonormal polynomials and
        the associated transition matrix ``S``.
        """
        tol = 1e-14
        for n in range(self.N + 1):
            v = np.zeros(self.N + 1, dtype=float)
            v[n] = 1.0  # Start from x^{lambda_n}.

            for m in range(n):
                projection = self._inner(v, self._transition[m])
                v -= projection * self._transition[m]

            norm_sq = self._inner(v, v)
            if norm_sq < 0.0:
                # Guard against round-off that can produce tiny negative values.
                norm_sq = abs(norm_sq)
            norm = np.sqrt(norm_sq)
            if not np.isfinite(norm) or norm < tol:
                raise RuntimeError(
                    "Encountered a degenerate vector during orthonormalisation."
                )
            self._transition[n] = v / norm

    def evaluate(self, x: Iterable[float]) -> EvaluationResult:
        """
        Evaluate ``L_{n, Λ}(x)`` at the supplied points.

        Parameters
        ----------
        x : Iterable[float]
            Sample locations. The values must be non-negative when ``alpha`` is
            not an integer so that ``x^{lambda_k}`` remains real.

        Returns
        -------
        EvaluationResult
            Stores the sample points and the basis values. The array in
            ``values`` has shape ``(N + 1, len(x))`` for vector inputs and
            ``(N + 1,)`` for scalar inputs.
        """

        x_arr = np.array(x, dtype=float, copy=True)
        scalar_input = x_arr.ndim == 0
        if scalar_input:
            x_eval = x_arr.reshape(1)
        else:
            x_eval = x_arr

        powers = np.power(x_eval[np.newaxis, ...], self.lambdas[:, np.newaxis])
        values = self._transition @ powers
        if scalar_input:
            values = values[:, 0]

        return EvaluationResult(x=x_eval, values=values)

    def transition_matrix(self) -> np.ndarray:
        """
        Return a copy of the transition matrix ``S`` where ``S[n, k] = C_{n,k}``.
        """
        return self._transition.copy()


class FractionalOperator:
    """
    Construct the Caputo fractional differentiation matrix ``D^alpha`` for the
    Mƒ?ntz-Legendre basis.

    Parameters
    ----------
    basis : MuntzLegendreBasis
        Basis that provides the transition matrix ``S`` and the exponents
        ``lambda_k``.
    nodes : Iterable[float]
        Collocation points ``x_i`` where the operator is sampled.
    mp_dps : int, optional
        Decimal precision used by :mod:`mpmath` when forming the intermediate
        monomial derivatives. The default of ``80`` is usually sufficient for
        ill-conditioned transition matrices.
    """

    def __init__(
        self, basis: MuntzLegendreBasis, nodes: Iterable[float], *, mp_dps: int = 80
    ) -> None:
        if mp_dps <= 0:
            raise ValueError("mp_dps must be positive.")

        self._basis = basis
        self._transition = basis.transition_matrix()
        self._lambdas = basis.lambdas.copy()
        self._order = float(basis.alpha)
        self._mp_dps = int(mp_dps)

        nodes_arr = np.array(nodes, dtype=float, copy=True).reshape(-1)
        if nodes_arr.size == 0:
            raise ValueError("At least one collocation node is required.")
        if np.any(nodes_arr < 0.0):
            raise ValueError("Caputo derivatives require non-negative nodes.")
        self._nodes = nodes_arr

    def fractional_differentiation_matrix(self) -> np.ndarray:
        """
        Assemble the matrix ``D^alpha`` whose entries represent
        ``D^alpha L_j(x_i)``.
        """
        num_nodes = self._nodes.size
        num_basis = self._transition.shape[0]

        with mp.workdps(self._mp_dps):
            mp_nodes = [mp.mpf(x) for x in self._nodes]
            mono_samples = self._fractional_monomial_samples(mp_nodes)

            mp_columns: list[list[mp.mpf]] = []
            for row in self._transition:
                column = [mp.mpf("0")] * num_nodes
                for k, weight in enumerate(row):
                    if weight == 0.0:
                        continue
                    mp_weight = mp.mpf(weight)
                    samples_k = mono_samples[k]
                    if samples_k is None:
                        continue
                    for idx in range(num_nodes):
                        column[idx] += mp_weight * samples_k[idx]
                mp_columns.append(column)

        matrix = np.array([[float(value) for value in col] for col in mp_columns])
        return matrix.T

    def _fractional_monomial_samples(
        self, nodes: list[mp.mpf]
    ) -> list[list[mp.mpf] | None]:
        """
        Evaluate ``D^alpha x^{lambda_k}`` at the supplied nodes using mpmath.
        """
        samples: list[list[mp.mpf] | None] = []
        order = mp.mpf(self._order)
        for lam in self._lambdas:
            if lam == 0.0:
                # Caputo derivative of a constant vanishes identically.
                samples.append(None)
                continue

            lam_mp = mp.mpf(lam)
            coeff = mp.gamma(lam_mp + 1.0) / mp.gamma(lam_mp - order + 1.0)
            exponent = lam_mp - order
            row = [coeff * (x ** exponent) for x in nodes]
            samples.append(row)
        return samples


class MMLSMSolver:
    """
    Assemble and solve the collocated MMLSM system with Dirichlet data at ``x = 0``.

    Parameters
    ----------
    basis : MuntzLegendreBasis
        Basis providing the polynomial evaluations.
    grid : GridMapper
        Grid generator used to obtain collocation nodes in ``[0, 1]``.
    a_func, f_func : Callable[[float], float]
        Coefficient and right-hand side functions sampled on the grid.
    epsilon, alpha : float
        Layer width and fractional order parameters that enter the system matrix.
    u0 : float
        Dirichlet value imposed at ``x = 0``.
    mp_dps : int, optional
        Working precision used when building the fractional differentiation matrix.
    """

    def __init__(
        self,
        basis: MuntzLegendreBasis,
        grid: GridMapper,
        a_func: Callable[[float], float],
        f_func: Callable[[float], float],
        epsilon: float,
        alpha: float,
        u0: float,
        *,
        mp_dps: int = 80,
    ) -> None:
        if mp_dps <= 0:
            raise ValueError("mp_dps must be positive.")

        self._basis = basis
        self._grid = grid
        self._a_func = a_func
        self._f_func = f_func
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.u0 = float(u0)
        self._mp_dps = int(mp_dps)

        self._nodes: np.ndarray | None = None
        self._system_matrix: np.ndarray | None = None
        self._rhs: np.ndarray | None = None
        self._coeffs: np.ndarray | None = None

    def assemble_system(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the linear system before boundary conditions are enforced.
        """
        x_nodes, _ = self._grid.map_nodes(self.epsilon)
        order = np.argsort(x_nodes)
        nodes = np.array(x_nodes[order], copy=True)

        expected = self._basis.N + 1
        if nodes.size != expected:
            raise ValueError(
                f"The grid provides {nodes.size} nodes, but the basis requires {expected}."
            )

        frac_op = FractionalOperator(self._basis, nodes, mp_dps=self._mp_dps)
        d_alpha = frac_op.fractional_differentiation_matrix()

        a_vals = np.array([float(self._a_func(x)) for x in nodes], dtype=float)
        f_vals = np.array([float(self._f_func(x)) for x in nodes], dtype=float)
        system_matrix = (self.epsilon ** self.alpha) * d_alpha + np.diag(a_vals)

        self._nodes = nodes
        self._system_matrix = system_matrix
        self._rhs = f_vals
        self._coeffs = None
        return system_matrix.copy(), f_vals.copy()

    def solve(self) -> np.ndarray:
        """
        Apply the Dirichlet constraint at ``x = 0`` and solve for the coefficients.
        """
        if self._system_matrix is None or self._rhs is None:
            self.assemble_system()

        matrix = self._system_matrix.copy()
        rhs = self._rhs.copy()

        boundary_row = np.array(self._basis.evaluate(0.0).values, dtype=float, copy=True)
        if boundary_row.ndim != 1 or boundary_row.size != matrix.shape[1]:
            raise RuntimeError("Boundary row is incompatible with the system size.")

        matrix[0, :] = boundary_row
        rhs[0] = self.u0

        coeffs = np.linalg.solve(matrix, rhs)
        self._coeffs = coeffs
        return coeffs.copy()

    def get_solution(self, x_new: Iterable[float] | float) -> np.ndarray | float:
        """
        Evaluate the reconstructed solution ``u(x) = sum_j c_j L_j(x)``.
        """
        if self._coeffs is None:
            raise RuntimeError("Call solve() before requesting the solution.")

        evaluation = self._basis.evaluate(x_new)
        u_vals = np.tensordot(self._coeffs, evaluation.values, axes=(0, 0))
        return u_vals

    @property
    def coefficients(self) -> np.ndarray:
        """
        Return a copy of the spectral coefficients obtained after solving.
        """
        if self._coeffs is None:
            raise RuntimeError("No coefficients are available before solve() is called.")
        return self._coeffs.copy()

    @property
    def nodes(self) -> np.ndarray | None:
        """
        Collocation nodes used during the assembly.
        """
        if self._nodes is None:
            return None
        return self._nodes.copy()
