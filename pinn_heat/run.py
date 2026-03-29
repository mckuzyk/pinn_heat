from pathlib import Path
import argparse
import torch
from pinn_heat.experiments import EXPERIMENTS
from pinn_heat.train import train
from pinn_heat import analysis
from pinn_heat import visualization as vis
import matplotlib.pyplot as plt


def main(config):
    save_dir = Path("results") / config.name
    print(f"Creating output directory {save_dir}")
    if save_dir.exists():
        raise FileExistsError(f"The path {save_dir} already exists")
    save_dir.mkdir(parents=True, exist_ok=False)

    model, loss = train(config)

    T, X = analysis.build_grid()
    u_exact = analysis.exact_solution(T, X, config.alpha)
    u_pred = analysis.get_preds(model, T, X)

    fig1 = vis.exact_vs_approximate(
        T, X, u_exact, u_pred, config.alpha, curve_fit=config.curve_fit
    )
    fig1.tight_layout()
    fig2 = vis.plot_loss(loss)
    fig2.tight_layout()

    fig1.savefig(save_dir / "results.png", dpi=300, bbox_inches="tight")
    fig2.savefig(save_dir / "total_loss.png", dpi=300, bbox_inches="tight")
    config.save(save_dir / "config.json")
    torch.save(model.state_dict(), save_dir / "model_state_dict.pt")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS.keys(), required=True)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        for name, config in EXPERIMENTS.items():
            print(f"Running experiment {name}...")
            main(config)
    else:
        config = EXPERIMENTS[args.experiment]
        main(config)
