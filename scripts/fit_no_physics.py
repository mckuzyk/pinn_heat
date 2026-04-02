from pinn_heat.experiments import EXPERIMENTS
from pinn_heat.model import PINN
from pinn_heat import analysis
from pinn_heat import visualization as vis
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import torch


def build_fit_fun(x, t, alpha):
    def inner(x, a, c):
        beta = np.sqrt(c / alpha)
        return a * np.exp(-c * t) * np.sin(beta * x)

    return inner


def fit_snapshots(u_preds, T, X, alpha, t_snaps, p0=None):
    fit_data = {}
    for t_snap in t_snaps:
        idx = np.argmin(np.abs(T[:, 0] - t_snap))
        u_pred = u_preds[idx]
        x_vals = X[idx]
        fit_fun = build_fit_fun(x_vals, t_snap, alpha)
        popt, pcov = curve_fit(fit_fun, x_vals, u_pred, p0=p0)
        fit_data[t_snap] = {"fit_function": fit_fun, "popt": popt, "pcov": pcov}
    return fit_data


def plot_symmetry(u_preds, T, X, alpha, t_snaps):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Solution asymmetry")
    for t_snap in t_snaps:
        idx = np.argmin(np.abs(T[:, 0] - t_snap))
        x_vals = X[idx]
        color = plt.cm.viridis(t_snap)
        ax.plot(
            x_vals,
            u_pred[idx] - u_pred[idx][::-1],
            color=color,
            label=f"t={t_snap:.2f} PINN",
        )
    ax.legend()
    return fig


def plot_fits(T, X, t_snaps, u_pred, fit_data):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Fit to Analytical Solution")
    for t_snap in t_snaps:
        fit_fun = fit_data[t_snap]["fit_function"]

        idx = np.argmin(np.abs(T[:, 0] - t_snap))
        x_vals = X[idx]
        color = plt.cm.viridis(t_snap)
        ax.plot(
            x_vals,
            fit_fun(x_vals, *fit_data[t_snap]["popt"]),
            color=color,
            label=f"t={t_snap:.2f} fit",
            linewidth=2,
            alpha=0.6,
        )
        ax.plot(
            x_vals,
            u_pred[idx],
            color=color,
            linestyle="--",
            label=f"t={t_snap:.2f} PINN",
            linewidth=1.5,
        )
    ax.legend(ncol=2)
    return fig


t_snaps = [0.0, 0.25, 0.5, 0.75, 1.0]
config = EXPERIMENTS["full_no_physics_fine"]
model = PINN(config.n_neurons, config.n_layers)
model.load_state_dict(torch.load("results/full_no_physics_fine/model_state_dict.pt"))
T, X = analysis.build_grid()
u_exact = analysis.exact_solution(T, X, config.alpha)
u_pred = analysis.get_preds(model, T, X)

fit_data = fit_snapshots(
    u_pred, T, X, config.alpha, t_snaps, p0=[1.0, np.pi**2 * config.alpha]
)
for key, val in fit_data.items():
    print(val["popt"])
a_vals = []
for t in t_snaps:
    a_vals.append(fit_data[t]["popt"][0])

fig, ax = plt.subplots()
ax.plot(t_snaps, a_vals, ".")


fig1 = vis.exact_vs_approximate(
    T, X, u_exact, u_pred, config.alpha, curve_fit=config.curve_fit
)
fig2 = plot_fits(T, X, t_snaps, u_pred, fit_data)
fig3 = plot_symmetry(u_pred, T, X, config.alpha, t_snaps)
plt.show()
