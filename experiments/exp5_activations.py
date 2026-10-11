"""
Experiment 5 - Does the Activation Function Matter?

Same target the whole time: f(x) = sin(x) + 0.5 sin(10x). Same width,
same seeds, same everything else. The only thing that changes is the
activation function: tanh, ReLU, or sine.

Question: does switching the activation change how fast the network
picks up k=1 vs. k=10?

Produces:
  results/exp5_results.json
  figures/exp5_learning_time_by_activation.png
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fourier_lib.model import MLP, set_seed
from fourier_lib.train import train_and_track, learning_time
from fourier_lib.targets import mixed_freq
from fourier_lib.viz import plot_grouped_bars

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

ACTIVATIONS = ["tanh", "relu", "sine"]
SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000]
N_EPOCHS = 2000
N_EPOCHS_CENSOR = N_EPOCHS  # what to use for "never learned in time"


def main():
    f, coeffs = mixed_freq()
    results = {}
    for act in ACTIVATIONS:
        t1s, t10s, mses = [], [], []
        for seed in SEEDS:
            set_seed(seed)
            model = MLP(width=64, activation=act)
            hist = train_and_track(model, f, k_max=12, n_epochs=N_EPOCHS, checkpoints=CHECKPOINTS)
            t1 = learning_time(hist["epochs"], hist["amp"][:, 1], coeffs[1])
            t10 = learning_time(hist["epochs"], hist["amp"][:, 10], coeffs[10])
            t1s.append(t1)
            t10s.append(t10)
            mses.append(hist["mse"][-1])
        results[act] = {"T1": t1s, "T10": t10s, "mse": mses}
        print(f"[exp5] {act}: T1={t1s}  T10={t10s}  final MSE mean={np.mean(mses):.4f}")

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp5_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    categories = ACTIVATIONS
    t1_means = [np.mean([N_EPOCHS_CENSOR if v is None else v for v in results[a]["T1"]]) for a in categories]
    t1_stds = [np.std([N_EPOCHS_CENSOR if v is None else v for v in results[a]["T1"]]) for a in categories]
    t10_means = [np.mean([N_EPOCHS_CENSOR if v is None else v for v in results[a]["T10"]]) for a in categories]
    t10_stds = [np.std([N_EPOCHS_CENSOR if v is None else v for v in results[a]["T10"]]) for a in categories]

    plot_grouped_bars(
        categories, ["T1 (k=1)", "T10 (k=10)"],
        {"T1 (k=1)": t1_means, "T10 (k=10)": t10_means},
        {"T1 (k=1)": t1_stds, "T10 (k=10)": t10_stds},
        title="Experiment 5: learning time by activation function\n(mean ± std, 5 seeds; capped at 2000 = never reached 90%)",
        ylabel="learning time (epochs)",
        path=os.path.join(FIG_DIR, "exp5_learning_time_by_activation.png"),
    )


if __name__ == "__main__":
    main()
