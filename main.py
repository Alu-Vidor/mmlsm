from collections import OrderedDict
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

from mmlsm import FractionalOperator, GridMapper, MMLSMSolver, MuntzLegendreBasis


def exact_solution_scalar(x: float, alpha: float, epsilon: float) -> float:
    """Exact solution evaluated with mpmath for improved stability near x=0."""
    if x < 0.0:
        raise ValueError("Exact solution is defined for non-negative x only.")
    x_mp = mp.mpf(x)
    alpha_mp = mp.mpf(alpha)
    epsilon_mp = mp.mpf(epsilon)
    return float(mp.power(x_mp, alpha_mp) + mp.e ** (-x_mp / epsilon_mp))


def exact_solution_array(x: Iterable[float], alpha: float, epsilon: float) -> np.ndarray:
    """Vectorised exact solution on numpy arrays."""
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0.0):
        raise ValueError("Exact solution is defined for x in [0, 1].")
    return np.power(x_arr, alpha) + np.exp(-x_arr / epsilon)


def manufactured_rhs_values(
    basis: MuntzLegendreBasis,
    nodes: np.ndarray,
    alpha: float,
    epsilon: float,
    a_func: Callable[[float], float],
    mp_dps: int,
) -> np.ndarray:
    """
    Evaluate the RHS implied by the manufactured solution on the collocation nodes.
    """
    basis_samples = basis.evaluate(nodes).values.T
    u_nodes = exact_solution_array(nodes, alpha, epsilon)
    coeffs_exact = np.linalg.solve(basis_samples, u_nodes)

    frac_op = FractionalOperator(basis, nodes, mp_dps=mp_dps)
    d_alpha = frac_op.fractional_differentiation_matrix()
    d_vals = d_alpha @ coeffs_exact

    a_vals = np.array([a_func(float(x)) for x in nodes], dtype=float)
    # The current solver formulation applies the algebraic term directly to the
    # spectral coefficients, hence the diagonal contribution multiplies
    # ``coeffs_exact`` rather than the physical solution values.
    rhs_nodes = (epsilon ** alpha) * d_vals + a_vals * coeffs_exact
    return rhs_nodes


def build_rhs_interpolator(
    nodes: np.ndarray,
    rhs_nodes: np.ndarray,
) -> Callable[[float], float]:
    """Interpolate the RHS defined on the collocation nodes."""
    nodes = np.asarray(nodes, dtype=float)
    rhs_nodes = np.asarray(rhs_nodes, dtype=float)
    lookup = {round(float(x), 12): float(val) for x, val in zip(nodes, rhs_nodes)}

    def rhs(x: float) -> float:
        key = round(float(x), 12)
        if key in lookup:
            return lookup[key]
        return float(np.interp(float(x), nodes, rhs_nodes))

    return rhs


def run_case(
    alpha: float,
    epsilon: float,
    n_values: Iterable[int],
    grid_mapping: str,
    mp_dps: int,
) -> tuple[list[tuple[int, float]], dict[str, np.ndarray]]:
    """Solve the manufactured problem for a range of basis sizes."""
    mp.mp.dps = mp_dps
    n_values = list(n_values)
    dense_mapper = GridMapper(num_points=400, mapping=grid_mapping)
    dense_x = np.sort(dense_mapper.map_nodes(epsilon)[0])
    exact_dense = exact_solution_array(dense_x, alpha, epsilon)
    results: list[tuple[int, float]] = []
    plot_payload: dict[str, np.ndarray] = {}
    a_func: Callable[[float], float] = lambda _: 0.0

    for n in n_values:
        if n < 2:
            raise ValueError("N must be at least 2 to form a grid.")
        basis = MuntzLegendreBasis(alpha=alpha, N=n - 1)
        grid = GridMapper(num_points=n, mapping=grid_mapping)
        raw_nodes, _ = grid.map_nodes(epsilon)
        nodes = np.sort(raw_nodes)
        rhs_nodes = manufactured_rhs_values(
            basis,
            nodes,
            alpha,
            epsilon,
            a_func,
            mp_dps,
        )
        rhs_func = build_rhs_interpolator(nodes, rhs_nodes)

        solver = MMLSMSolver(
            basis=basis,
            grid=grid,
            a_func=a_func,
            f_func=rhs_func,
            epsilon=epsilon,
            alpha=alpha,
            u0=exact_solution_scalar(0.0, alpha, epsilon),
            mp_dps=mp_dps,
        )
        solver.solve()
        numeric_dense = solver.get_solution(dense_x)
        linf_error = float(np.max(np.abs(numeric_dense - exact_dense)))
        results.append((n, linf_error))

        if n == max(n_values):
            plot_payload = {
                "x": dense_x,
                "u_numeric": numeric_dense,
                "u_exact": exact_dense,
                "nodes": solver.nodes,
            }

    return results, plot_payload


def print_table(
    alpha: float,
    convergence_table: OrderedDict[float, list[tuple[int, float]]],
) -> None:
    """Pretty-print the convergence study."""
    print(f"\nСходимость для alpha = {alpha}")
    header = "  N    ||u - u_exact||_inf"
    for epsilon, entries in convergence_table.items():
        print(f"\n  epsilon = {epsilon:.0e}")
        print(header)
        for n, error in entries:
            print(f" {n:>3d}        {error:10.3e}")


def plot_results(plot_data: OrderedDict[float, dict[str, np.ndarray]]) -> None:
    """Create comparison plots that highlight the boundary layer resolution."""
    num_cases = len(plot_data)
    fig, axes = plt.subplots(
        1,
        num_cases,
        figsize=(6 * num_cases, 4),
        squeeze=False,
    )

    for ax, (epsilon, payload) in zip(axes.ravel(), plot_data.items()):
        ax.plot(payload["x"], payload["u_exact"], label="Точное решение", color="black")
        ax.plot(
            payload["x"],
            payload["u_numeric"],
            label="MMLSM (N=max)",
            linestyle="--",
            color="tab:blue",
        )

        y_min = min(payload["u_exact"].min(), payload["u_numeric"].min())
        y_max = max(payload["u_exact"].max(), payload["u_numeric"].max())
        span = y_max - y_min
        node_level = y_min - 0.1 * span
        ax.scatter(
            payload["nodes"],
            np.full_like(payload["nodes"], node_level),
            marker="|",
            s=80,
            color="tab:red",
            label="Узлы сетки",
        )
        ax.set_ylim(node_level - 0.05 * span, y_max + 0.05 * span)
        ax.set_title(f"epsilon = {epsilon:.0e}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Решение и распределение узлов MMLSM")
    fig.tight_layout()
    fig.savefig("mmlsm_demo.png", dpi=200)
    backend = plt.get_backend().lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    alpha = 0.5
    epsilon_values = (1e-2, 1e-4)
    n_values = (4, 8, 16, 32, 64)
    mp_dps = 120
    grid_mapping = "logarithmic"

    convergence_table: OrderedDict[float, list[tuple[int, float]]] = OrderedDict()
    plot_payloads: OrderedDict[float, dict[str, np.ndarray]] = OrderedDict()

    for epsilon in epsilon_values:
        results, payload = run_case(alpha, epsilon, n_values, grid_mapping, mp_dps)
        convergence_table[epsilon] = results
        plot_payloads[epsilon] = payload

    print_table(alpha, convergence_table)
    plot_results(plot_payloads)


if __name__ == "__main__":
    main()
