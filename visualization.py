import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from analysis import l2_error, fit_snapshots


def exact_vs_approximate(T, X, u_exact, u_pred, alpha, curve_fit=False):
    error = np.abs(u_pred - u_exact)
    L2 = l2_error(u_pred, T, X)
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
    im2 = ax2.contourf(T, X, u_exact, levels=50, cmap="RdBu_r")
    fig.colorbar(im2, ax=ax2)
    ax2.set_title("Exact solution")
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
    if curve_fit:
        fit_data = fit_snapshots(u_pred, T, X, alpha)
    for t_snap in t_snaps:
        idx = np.argmin(np.abs(T[:, 0] - t_snap))
        x_vals = X[idx]
        color = plt.cm.viridis(t_snap)
        if curve_fit:
            fit_fun = fit_data[t_snap]["fit_function"]
            ax4.plot(
                x_vals,
                fit_fun(x_vals, *fit_data[t_snap]["popt"]),
                color=color,
                label=f"t={t_snap:.2f} fit",
                linewidth=2,
                alpha=0.6,
            )
        else:
            ax4.plot(
                x_vals,
                u_exact[idx],
                color=color,
                label=f"t={t_snap:.2f} exact",
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
    if curve_fit:
        ax4.set_title("Time slices: fit (solid) vs PINN (dashed)")
    else:
        ax4.set_title("Time slices: exact (solid) vs PINN (dashed)")
    ax4.legend(loc="upper right", fontsize=7, ncol=2)

    plt.suptitle(f"PINN — Heat equation  (α={alpha})", fontsize=13)
    return fig


def plot_loss(loss):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total loss (log scale)")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
