import torch
from pinn_heat.model import (
    PINN,
    sample_collocation_points,
    sample_boundary_and_ic,
    physics_informed_nn,
)


OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
SCHEDULERS = {"StepLR": torch.optim.lr_scheduler.StepLR}


def train(config):

    model = PINN(config.n_neurons, config.n_layers)
    optimizer_class = OPTIMIZERS[config.optimizer]
    optimizer = optimizer_class(model.parameters(), **config.optimizer_params)
    scheduler_class = SCHEDULERS[config.scheduler]
    scheduler = scheduler_class(optimizer, **config.scheduler_params)

    # Fixed boundary points trained each epoch
    t_train, x_train, u_train = sample_boundary_and_ic(
        n_boundary=config.n_bc, n_ic=config.n_ic
    )

    model.train()
    loss_dict = {
        "physics": [],
        "data": [],
        "full": [],
    }
    for epoch in range(config.epochs):
        optimizer.zero_grad()

        u_pred = model(t_train, x_train)
        loss_data = torch.mean((u_pred - u_train) ** 2)

        # Collocation points randomly sampled each epoch
        t_co, x_co = sample_collocation_points(config.n_collocation)
        loss_physics = torch.mean(
            physics_informed_nn(model, t_co, x_co, config.alpha) ** 2
        )

        loss = config.lambda_u * loss_data + config.lambda_f * loss_physics
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_dict["physics"].append(loss_physics.item())
        loss_dict["data"].append(loss_data.item())
        loss_dict["full"].append(loss.item())

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Loss (Total): {loss.item()} | "
                f"Loss (Data): {loss_data.item()} | "
                f"Loss (Phys): {loss_physics.item()} | "
            )

    return model, loss_dict
