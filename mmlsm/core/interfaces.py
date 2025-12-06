from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BasisEvaluation:
    """
    Store sampling points and associated basis values.

    The ``values`` array is organised as ``(num_basis, *x.shape)`` and mirrors
    the result of :meth:`SpectralBasis.evaluate`.
    """

    x: np.ndarray
    values: np.ndarray


class SpectralBasis(ABC):
    """Contract for orthonormal bases used by the spectral solver."""

    @abstractmethod
    def evaluate(self, x: Iterable[float] | float) -> BasisEvaluation:
        """Sample the basis functions at ``x``."""

    @abstractmethod
    def size(self) -> int:
        """Return the number of basis functions."""

    @abstractmethod
    def transition_matrix(self) -> np.ndarray:
        """Return the matrix that maps monomials to the basis."""

    @abstractmethod
    def exponents(self) -> np.ndarray:
        """Return the exponents associated with the fractional monomials."""


class DomainMapper(ABC):
    """Strategy interface for mapping ``[-1, 1]`` to ``[0, 1]``."""

    @abstractmethod
    def map(self, reference_nodes: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """Return mapped nodes and the Jacobian ``dx/dxi``."""


class GridGenerator(ABC):
    """Produce collocation nodes by combining a reference grid and mapper."""

    @abstractmethod
    def reference_nodes(self) -> np.ndarray:
        """Return the nodes on ``[-1, 1]`` before mapping."""

    @abstractmethod
    def generate_nodes(self, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        """Map the reference nodes to ``[0, 1]``."""


class LinearOperator(ABC):
    """Base class for linear operators that act on spectral bases."""

    @abstractmethod
    def assemble(self, basis: SpectralBasis, nodes: np.ndarray) -> np.ndarray:
        """Return the matrix representation of the operator on ``nodes``."""
