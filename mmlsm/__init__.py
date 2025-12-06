"""
Public interface for the modular MMLSM framework.
"""

from .basis.muntz import MuntzLegendreBasis
from .core.interfaces import (
    BasisEvaluation,
    DomainMapper,
    GridGenerator,
    LinearOperator,
    SpectralBasis,
)
from .grid.mappers import AlgebraicMapper, CGLGrid, LogarithmicMapper, create_mapper
from .operators import FractionalDerivativeOperator
from .solvers.linear import ProblemConfig, SolverFactory, SpectralSolver

__all__ = [
    "AlgebraicMapper",
    "BasisEvaluation",
    "CGLGrid",
    "DomainMapper",
    "FractionalDerivativeOperator",
    "GridGenerator",
    "LinearOperator",
    "LogarithmicMapper",
    "MuntzLegendreBasis",
    "ProblemConfig",
    "SolverFactory",
    "SpectralBasis",
    "SpectralSolver",
    "create_mapper",
]
