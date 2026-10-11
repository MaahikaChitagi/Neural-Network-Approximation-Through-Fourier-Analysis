"""
Experiment 7 - Fourier Features

Compares two ways of feeding x into the network:
  "plain"   : just the raw number x
  "fourier" : phi(x) = [sin(x), cos(x), sin(2x), cos(2x), ..., sin(Bx), cos(Bx)]

Target: f(x) = sin(20x) + 0.5 sin(30x), the "high frequency" target
that Experiment 4 showed a plain network really struggles with.

I use B=32 frequencies in the Fourier feature map, so both k=20 and
k=30 are directly available to the network from the start.

Question: does handing the network sines/cosines up front actually
help it learn a high-frequency target?

Produces:
  results/exp7_results.json
  figures/exp7_mse_comparison.png
  figures/exp7_fit_comparison.png
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fourier_lib.model import MLP, FourierFeatureMLP, set_seed
from fourier_lib.train import train_and_track, learning_time
from fourier_lib.targets import high_freq
from fourier_lib.viz import plot_grouped_bars, plot_function_vs_prediction

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000]
N_EPOCHS = 2000
K_MAX = 32
N_FOURIER_FREQS = 32


def main():
    f, coeffs = high_freq()  # the target: sin(20x) + 0.5 sin(30x)
    results = {"plain": {"mse": [], "T20": [], "T30": []},
               "fourier": {"mse": [], "T20": [], "T30": []}}

    example_pred = {}

    for condition in ["plain", "fourier"]:
        for seed in SEEDS:
            set_seed(seed)
            if condition == "plain":
                model = MLP(width=64, activation="tanh")
            else:
                model = FourierFeatureMLP(n_freqs=N_FOURIER_FREQS, width=64, activation="tanh")
            hist = train_and_track(model, f, k_max=K_MAX, n_epochs=N_EPOCHS, checkpoints=CHECKPOINTS)
            results[condition]["mse"].append(hist["mse"][-1])
            results[condition]["T20"].append(learning_time(hist["epochs"], hist["amp"][:, 20], coeffs[20]))
            results[condition]["T30"].append(learning_time(hist["epochs"], hist["amp"][:, 30], coeffs[30]))
            if seed == 0:
                example_pred[condition] = (hist["eval_x"], hist["eval_pred"][-1])
        print(f"[exp7] {condition}: mean final MSE = {np.mean(results[condition]['mse']):.5f} "
              f"+/- {np.std(results[condition]['mse']):.5f}")
        print(f"        T20={results[condition]['T20']}  T30={results[condition]['T30']}")

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp7_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    categories = ["plain", "fourier"]
    mse_means = [np.mean(results[c]["mse"]) for c in categories]
    mse_stds = [np.std(results[c]["mse"]) for c in categories]
    plot_grouped_bars(
        categories, ["final MSE"],
        {"final MSE": mse_means}, {"final MSE": mse_stds},
        title="Experiment 7: plain input vs. Fourier-feature input\ntarget = sin(20x) + 0.5 sin(30x), mean ± std over 5 seeds",
        ylabel="MSE (log scale)",
        path=os.path.join(FIG_DIR, "exp7_mse_comparison.png"),
        log=True,
    )

    x_plain, y_plain = example_pred["plain"]
    x_four, y_four = example_pred["fourier"]
    plot_function_vs_prediction(
        x_plain, f(x_plain), y_plain,
        title="Experiment 7: plain-input network (seed 0, 2000 epochs)",
        path=os.path.join(FIG_DIR, "exp7_fit_plain.png"),
    )
    plot_function_vs_prediction(
        x_four, f(x_four), y_four,
        title="Experiment 7: Fourier-feature network (seed 0, 2000 epochs)",
        path=os.path.join(FIG_DIR, "exp7_fit_fourier.png"),
    )


if __name__ == "__main__":
    main()
