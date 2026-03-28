from dataclasses import dataclass, field, asdict
import json


@dataclass
class ExperimentConfig:
    name: str

    # PDE parameter
    alpha: float = 0.5

    # Model settings
    n_neurons: int = 32
    n_layers: int = 4

    # Balancing data and physics losses
    lambda_u: float = 1.0
    lambda_f: float = 1.0

    # Sample sizes
    n_collocation: int = 5000
    n_ic: int = 100
    n_bc: int = 100

    # Training parameters
    epochs: int = 5000
    lr: float = 1e-3
    optimizer: str = "Adam"
    optimizer_params: dict = field(default_factory=lambda: {"lr": 1e-3})
    scheduler: str = "StepLR"
    scheduler_params: dict = field(
        default_factory=lambda: {"step_size": 2000, "gamma": 1.0, "last_epoch": -1}
    )

    # Other
    curve_fit: bool = False
    output_dir: str = "results"

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
