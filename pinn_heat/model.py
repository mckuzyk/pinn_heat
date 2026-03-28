import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


torch.manual_seed(42)
np.random.seed(42)


ALPHA = 0.5


class PINN(nn.Module):
    """
    Defines the data informed neural network, denoted u(t, x) in Raissi et al
    """

    def __init__(self, n_neurons, n_layers):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, t, x):
        input = torch.cat([t, x], dim=1)
        return self.mlp(input)


def sample_boundary_and_ic(n_boundary, n_ic):
    """
    Generate known data points. Namely for our choice of boundary and initial conditions:
    u(t,0) = u(t,1) = 0, t in the range [0,1]
    u(0,x) = sin(pi*x),  x in the range [0,1]
    """

    # Randomly sampled initial condition points
    x_ic = torch.rand(n_ic, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.sin(np.pi * x_ic)

    # Randomly sampled boundary condition points
    t_boundary = torch.rand(n_boundary, 1)
    x_boundary = torch.bernoulli(0.5 * torch.ones_like(t_boundary))
    u_boundary = torch.zeros_like(t_boundary)

    return (
        torch.cat([t_ic, t_boundary], dim=0),
        torch.cat([x_ic, x_boundary], dim=0),
        torch.cat([u_ic, u_boundary], dim=0),
    )


def sample_collocation_points(n_samples):
    """
    A random sampling of pairs (t, x) within the specified domain, in this case
    x in [0,1], t in [0,1], where we don't know the function u(t, x), but we
    do know the physics that dictates the solution!
    """
    t_samples = torch.rand(n_samples, 1, requires_grad=True)
    x_samples = torch.rand(n_samples, 1, requires_grad=True)

    return t_samples, x_samples


def physics_informed_nn(model, t, x, alpha=ALPHA):
    """
    Defines the physics informed neural network, denoted f(t, x) in Raissi et al
    """
    model.eval()
    out = model.forward(t, x)
    u_x = torch.autograd.grad(
        out, x, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    u_t = torch.autograd.grad(
        out, t, grad_outputs=torch.ones_like(out), create_graph=True
    )[0]

    return u_t - alpha * u_xx
