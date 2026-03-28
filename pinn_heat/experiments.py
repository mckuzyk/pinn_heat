from config import ExperimentConfig

EXPERIMENTS = {
    "test": ExperimentConfig(
        "test",
        n_neurons=10,
        n_collocation=1000,
        optimizer="SGD",
        lr=0.01,
    ),
    "vanilla": ExperimentConfig(
        "vanilla",
        n_neurons=10,
        n_collocation=2000,
        optimizer="SGD",
    ),
    "scheduled_adam": ExperimentConfig(
        "scheduled_adam",
        n_neurons=10,
        n_collocation=2000,
        optimizer="Adam",
        scheduler_params={"step_size": 2000, "gamma": 0.5},
    ),
    "full": ExperimentConfig(
        "full",
        optimizer="Adam",
        scheduler_params={"step_size": 2000, "gamma": 0.5},
    ),
    "full_no_physics": ExperimentConfig(
        "full_no_physics",
        optimizer="Adam",
        scheduler_params={"step_size": 2000, "gamma": 0.5},
        lambda_f=0.0,
    ),
    "full_no_data": ExperimentConfig(
        "full_no_data",
        optimizer="Adam",
        scheduler_params={"step_size": 2000, "gamma": 0.5},
        lambda_u=0.0,
        curve_fit=True,
    ),
}
