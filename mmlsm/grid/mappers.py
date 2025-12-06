from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mmlsm.core.interfaces import DomainMapper, GridGenerator

EPS_FLOOR = 1.0e-12
MAX_STRENGTH = 12.0


def _boundary_layer_strength(epsilon: float) -> float:
    eps_clamped = max(float(epsilon), EPS_FLOOR)
    strength = max(0.0, -np.log10(eps_clamped))
    return min(strength, MAX_STRENGTH)


class AlgebraicMapper(DomainMapper):
    """Power-law stretching that clusters nodes algebraically near ``x=0``."""

    def map(self, reference_nodes: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        xi = np.asarray(reference_nodes, dtype=float)
        r = 0.5 * (xi + 1.0)
        exponent = 1.0 + _boundary_layer_strength(epsilon)
        x = np.power(r, exponent)
        metric = 0.5 * exponent * np.power(r, exponent - 1.0)
        return x, metric


class LogarithmicMapper(DomainMapper):
    """Smooth logarithmic clustering with bounded Jacobian."""

    def map(self, reference_nodes: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        xi = np.asarray(reference_nodes, dtype=float)
        r = 0.5 * (xi + 1.0)
        beta = 1.0 + _boundary_layer_strength(epsilon)
        expm1_beta = np.expm1(beta)
        numerator = np.log1p(expm1_beta * r)
        x = numerator / beta
        metric = 0.5 * (expm1_beta / (1.0 + expm1_beta * r)) / beta
        return x, metric


def create_mapper(mapping: str) -> DomainMapper:
    """Factory function that instantiates a mapper by name."""
    key = mapping.strip().lower()
    if key.startswith("log"):
        return LogarithmicMapper()
    if key == "algebraic":
        return AlgebraicMapper()
    raise ValueError("Unknown mapping strategy. Use 'algebraic' or 'logarithmic'.")


class CGLGrid(GridGenerator):
    """Chebyshev-Gauss-Lobatto grid composed with a domain mapping strategy."""

    def __init__(self, num_points: int, mapper: DomainMapper | str = "algebraic") -> None:
        if num_points < 2:
            raise ValueError("At least two points are required to form a CGL grid.")
        if isinstance(mapper, str):
            mapper_obj = create_mapper(mapper)
        elif isinstance(mapper, DomainMapper):
            mapper_obj = mapper
        else:
            raise TypeError("mapper must be a DomainMapper or a supported string identifier.")

        self.num_points = int(num_points)
        self._mapper = mapper_obj
        self._xi = self._generate_reference_nodes()

    def _generate_reference_nodes(self) -> np.ndarray:
        N = self.num_points - 1
        j = np.arange(self.num_points, dtype=float)
        return np.cos(np.pi * j / N)

    def reference_nodes(self) -> np.ndarray:
        return self._xi.copy()

    def generate_nodes(self, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive for the mapping to be defined.")
        mapped, metric = self._mapper.map(self._xi, epsilon)
        return mapped.copy(), metric.copy()
