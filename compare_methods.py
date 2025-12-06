from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from mmlsm import ProblemConfig, SolverFactory

EPSILON = 1e-4
ALPHA = 0.6
MP_DPS = 180
GRID_MAPPING = "logarithmic"
QUAD_ORDER = 512
EPSILON_ALPHA = EPSILON**ALPHA

N_SHARED = (10, 20, 50, 100)
N_FDM_EXTENDED = (10, 20, 50, 100, 200, 500, 1000)
N_SMALL = 30

_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(QUAD_ORDER)
_U = 0.5 * (_LEGENDRE_NODES + 1.0)
_W = 0.5 * _LEGENDRE_WEIGHTS
_TRANSFORM = 1.0 - np.power(_U, 1.0 / (1.0 - ALPHA))
_CAPUTO_PREFAC = 1.0 / (math.gamma(1.0 - ALPHA) * (1.0 - ALPHA))


def exact_solution_scalar(x: float) -> float:
    return float(x ** 2 + np.exp(-x / EPSILON))


def exact_solution_array(x: Iterable[float]) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return x_arr**2 + np.exp(-x_arr / EPSILON)


def caputo_derivative_numeric(x_val: float) -> float:
    x_val = float(x_val)
    if x_val <= 0.0:
        return 0.0
    eval_points = x_val * _TRANSFORM
    deriv_vals = 2.0 * eval_points - (1.0 / EPSILON) * np.exp(-eval_points / EPSILON)
    integral = np.sum(_W * deriv_vals)
    return (x_val ** (1.0 - ALPHA)) * _CAPUTO_PREFAC * integral


def rhs_function(x: float) -> float:
    return EPSILON_ALPHA * caputo_derivative_numeric(x)


def build_config(num_dofs: int) -> ProblemConfig:
    if num_dofs < 2:
        raise ValueError("Need at least two degrees of freedom.")
    return ProblemConfig(
        epsilon=EPSILON,
        alpha=ALPHA,
        u0=exact_solution_scalar(0.0),
        N=num_dofs - 1,
    )


def run_solver_on_grid(
    solver_factory: SolverFactory,
    num_dofs: int,
    dense_x: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    config = build_config(num_dofs)
    a_func = lambda _: 0.0
    if method == "mmlsm":
        solver = solver_factory.create_mmlsm_solver(
            config,
            mapping=GRID_MAPPING,
            a_func=a_func,
            f_func=rhs_function,
        )
    elif method == "fdm":
        solver = solver_factory.create_fdm_solver(
            config,
            a_func=a_func,
            f_func=rhs_function,
        )
    else:
        raise ValueError(f"Unknown method '{method}'.")

    solver.solve()
    dense_values = solver.get_solution(dense_x)
    nodes = solver.nodes
    if nodes is None:
        raise RuntimeError("Solver did not expose its node locations.")
    return np.asarray(dense_values, dtype=float), np.asarray(nodes, dtype=float)


def compute_errors(
    solver_factory: SolverFactory,
    num_dofs_list: Iterable[int],
    dense_x: np.ndarray,
    exact_dense: np.ndarray,
    *,
    method: str,
) -> tuple[list[tuple[int, float]], dict[int, tuple[np.ndarray, np.ndarray]]]:
    errors: list[tuple[int, float]] = []
    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for ndof in num_dofs_list:
        dense_values, nodes = run_solver_on_grid(
            solver_factory,
            ndof,
            dense_x,
            method=method,
        )
        error = float(np.max(np.abs(dense_values - exact_dense)))
        errors.append((ndof, error))
        snapshots[ndof] = (dense_values, nodes)
    return errors, snapshots


def print_error_table(label: str, entries: list[tuple[int, float]]) -> None:
    print(f"\n{label} (L_inf error)")
    for ndof, error in entries:
        print(f"  N = {ndof:4d}: {error:10.3e}")


def plot_results(
    mmlsm_errors: list[tuple[int, float]],
    fdm_errors: list[tuple[int, float]],
    dense_x: np.ndarray,
    exact_dense: np.ndarray,
    mmlsm_snapshot: tuple[np.ndarray, np.ndarray],
    fdm_snapshot: tuple[np.ndarray, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax_err = axes[0]
    mmlsm_n = [val[0] for val in mmlsm_errors]
    mmlsm_vals = [val[1] for val in mmlsm_errors]
    fdm_n = [val[0] for val in fdm_errors]
    fdm_vals = [val[1] for val in fdm_errors]
    ax_err.loglog(mmlsm_n, mmlsm_vals, marker="o", label="MMLSM", color="tab:blue")
    ax_err.loglog(fdm_n, fdm_vals, marker="s", label="FDM (L1)", color="tab:orange")
    ax_err.set_xlabel("Degrees of freedom")
    ax_err.set_ylabel(r"$L_\infty$ error")
    ax_err.set_title("Accuracy vs N")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend()

    ax_sol = axes[1]
    ax_sol.plot(dense_x, exact_dense, label="Exact", color="black", linewidth=2)
    ax_sol.plot(
        dense_x,
        mmlsm_snapshot[0],
        label=f"MMLSM (N={N_SMALL})",
        linestyle="--",
        color="tab:blue",
    )
    ax_sol.plot(
        dense_x,
        fdm_snapshot[0],
        label=f"FDM (N={N_SMALL})",
        linestyle="-.",
        color="tab:orange",
    )
    ax_sol.scatter(
        mmlsm_snapshot[1],
        np.interp(mmlsm_snapshot[1], dense_x, exact_dense),
        marker="|",
        s=80,
        color="tab:blue",
        alpha=0.5,
        label="MMLSM nodes",
    )
    ax_sol.scatter(
        fdm_snapshot[1],
        np.interp(fdm_snapshot[1], dense_x, exact_dense),
        marker="x",
        color="tab:orange",
        alpha=0.7,
        label="FDM nodes",
    )
    ax_sol.set_xlim(0.0, 0.3)
    ax_sol.set_xlabel("x")
    ax_sol.set_ylabel("u(x)")
    ax_sol.set_title(f"Boundary layer capture (N={N_SMALL})")
    ax_sol.grid(True, alpha=0.3)
    ax_sol.legend()

    fig.tight_layout()
    fig.savefig("compare_methods.png", dpi=250)
    backend = plt.get_backend().lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    mp.mp.dps = MP_DPS

    dense_x = np.linspace(0.0, 1.0, 2000)
    exact_dense = exact_solution_array(dense_x)
    factory = SolverFactory(mp_dps=MP_DPS)

    mmlsm_errors, mmlsm_snapshots = compute_errors(
        factory,
        N_SHARED,
        dense_x,
        exact_dense,
        method="mmlsm",
    )
    fdm_errors, fdm_snapshots = compute_errors(
        factory,
        N_FDM_EXTENDED,
        dense_x,
        exact_dense,
        method="fdm",
    )

    print_error_table("MMLSM spectral solver", mmlsm_errors)
    print_error_table("Finite difference L1 solver", fdm_errors)

    if N_SMALL not in mmlsm_snapshots:
        mmlsm_snapshots[N_SMALL] = run_solver_on_grid(
            factory,
            N_SMALL,
            dense_x,
            method="mmlsm",
        )
    if N_SMALL not in fdm_snapshots:
        fdm_snapshots[N_SMALL] = run_solver_on_grid(
            factory,
            N_SMALL,
            dense_x,
            method="fdm",
        )

    plot_results(
        mmlsm_errors,
        fdm_errors,
        dense_x,
        exact_dense,
        mmlsm_snapshots[N_SMALL],
        fdm_snapshots[N_SMALL],
    )


if __name__ == "__main__":
    main()
