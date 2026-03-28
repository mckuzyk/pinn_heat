import numpy as np
import torch
from scipy.optimize import curve_fit


def exact_solution(t, x, alpha=0.5):
    """
    Exact solution of 1D heat equation we seek to solve with PINN:
                        u_t = alpha * u_xx
    """
    return np.exp(-(np.pi**2) * alpha * t) * np.sin(np.pi * x)


# --- Generating Data ----------------------------------------------------------
def build_grid(nt=200, nx=200):
    t_vals = np.linspace(0, 1, nt)
    x_vals = np.linspace(0, 1, nx)
    T, X = np.meshgrid(t_vals, x_vals, indexing="ij")
    return T, X


def get_preds(model, T, X):
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32).unsqueeze(1)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)

    model.eval()
    with torch.no_grad():
        u_pred = model(t_flat, x_flat).numpy().reshape(T.shape)
    return u_pred


# --- Curve Fitting ------------------------------------------------------------
def general_exact_solution(x, t, alpha):
    def inner(x, a0, b0, b1, c):
        beta = np.sqrt(c / alpha)
        return a0 * np.exp(c * t) * (b0 * np.exp(beta * x) + b1 * np.exp(-beta * x))

    return inner


def fit_snapshots(
    u_preds,
    T,
    X,
    alpha,
    t_snaps=[0.0, 0.25, 0.5, 0.75, 1.0],
    p0=[0.2, -0.2, -0.2, 0.01],
):
    fit_data = {}
    for t_snap in t_snaps:
        idx = np.argmin(np.abs(T[:, 0] - t_snap))
        u_pred = u_preds[idx]
        x_vals = X[idx]
        fit_fun = general_exact_solution(x_vals, t_snap, alpha)
        popt, pcov = curve_fit(fit_fun, x_vals, u_pred, p0=p0)
        fit_data[t_snap] = {"fit_function": fit_fun, "popt": popt, "pcov": pcov}
    return fit_data


# --- Error -------------------------------------------------------------------
def l2_error(u_preds, T, X):
    u_exact = exact_solution(T, X)
    return np.linalg.norm(u_preds - u_exact) / np.linalg.norm(u_exact)
