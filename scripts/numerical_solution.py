"""
Compare PINN prediction to numerical solution, using the predictions
u(0, x), u(t, 0), u(t, 1) as the prescribed boundary and initial
conditions.
"""

from pathlib import Path
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from pinn_heat.experiments import EXPERIMENTS
from pinn_heat.model import PINN
import torch
from pinn_heat import analysis
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def exact_vs_approximate(T, X, u_numerical, u_pred, alpha):
    """
    Copy pasted from pinn_heat.visualization to make some changes needed
    for this use.
    """
    error = np.abs(u_pred - u_numerical)
    L2 = np.linalg.norm(u_pred - u_numerical) / np.linalg.norm(u_numerical)
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Predicted solution
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(T, X, u_pred, levels=50, cmap="RdBu_r")
    fig.colorbar(im1, ax=ax1)
    ax1.set_title("PINN prediction")
    ax1.set_xlabel("t")
    ax1.set_ylabel("x")

    # Exact solution
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(T, X, u_numerical, levels=50, cmap="RdBu_r")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Numerical solution")
    ax2.set_xlabel("t")
    ax2.set_ylabel("x")

    # Absolute error
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(T, X, error, levels=50, cmap="hot_r")
    fig.colorbar(im3, ax=ax3)
    ax3.set_title(f"|Error|  (L2={L2:.2e})")
    ax3.set_xlabel("t")
    ax3.set_ylabel("x")

    # Time slices
    ax4 = fig.add_subplot(gs[1, :])
    t_snaps = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t_snap in t_snaps:
        idx = np.argmin(np.abs(T[:, 0] - t_snap))
        x_vals = X[idx]
        color = plt.cm.viridis(t_snap)
        ax4.plot(
            x_vals,
            u_numerical[idx],
            color=color,
            label=f"t={t_snap:.2f} numerical",
            linewidth=2,
            alpha=0.6,
        )
        ax4.plot(
            x_vals,
            u_pred[idx],
            color=color,
            linestyle="--",
            label=f"t={t_snap:.2f} PINN",
            linewidth=1.5,
        )
    ax4.set_xlabel("x")
    ax4.set_ylabel("u(t, x)")
    ax4.set_title("Time slices: numerical (solid) vs PINN (dashed)")
    ax4.legend(loc="upper right", fontsize=7, ncol=2)

    plt.suptitle(f"PINN — Heat equation  (α={alpha})", fontsize=13)
    return fig


def solve_heat_1d(x_array, initial_u, left_boundary, right_boundary, dt, alpha):
    """
    x_array: Numerical array of spatial points
    initial_u: Array of initial temperatures at t=0
    left_boundary: Array of temperatures at x[0] for each time step
    right_boundary: Array of temperatures at x[-1] for each time step
    """
    dx = x_array[1] - x_array[0]
    N = len(x_array)
    nt = len(left_boundary)

    # Fourier number (stability parameter)
    sigma = (alpha * dt) / (2 * dx**2)

    # Create Crank-Nicolson Matrices
    # (1 + 2s) on diagonal, -s on off-diagonals
    main_diag = np.full(N - 2, 1 + 2 * sigma)
    off_diag = np.full(N - 3, -sigma)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsr()

    # (1 - 2s) on diagonal, s on off-diagonals
    main_diag_b = np.full(N - 2, 1 - 2 * sigma)
    off_diag_b = np.full(N - 3, sigma)
    B_mat = diags([off_diag_b, main_diag_b, off_diag_b], [-1, 0, 1]).tocsr()

    u = initial_u.copy()
    results = [u.copy()]

    # Time Stepping
    for t in range(1, nt):
        # Interior points
        b = B_mat.dot(u[1:-1])

        # Add boundary influences to the RHS vector 'b'
        # Current and next step boundary values are averaged in Crank-Nicolson
        b[0] += sigma * (left_boundary[t - 1] + left_boundary[t])
        b[-1] += sigma * (right_boundary[t - 1] + right_boundary[t])

        # Solve the linear system
        u[1:-1] = spsolve(A, b)

        # Explicitly set the boundaries from your arrays
        u[0] = left_boundary[t]
        u[-1] = right_boundary[t]

        results.append(u.copy())

    return np.array(results)


config = EXPERIMENTS["full_no_data_very_fine"]

model = PINN(config.n_neurons, config.n_layers)
model.load_state_dict(torch.load("results/full_no_data_very_fine/model_state_dict.pt"))
T, X = analysis.build_grid()
u_pred = analysis.get_preds(model, T, X)

x_array = X[0]

initial_u = u_pred[0]
left_boundary = u_pred[:, 0]
right_boundary = u_pred[:, -1]

dt = T[1, 0] - T[0, 0]

results = solve_heat_1d(
    x_array, initial_u, left_boundary, right_boundary, dt, config.alpha
)

fig = exact_vs_approximate(T, X, u_pred, results, config.alpha)
fig.tight_layout()
savedir = Path("results") / config.name
fig.savefig(savedir / "numerical.png", dpi=300, bbox_inches="tight")
plt.show()
