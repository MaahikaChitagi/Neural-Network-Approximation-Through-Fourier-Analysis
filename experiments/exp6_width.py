"""
Experiment 6 - Does a Bigger Network Help?

Same target the whole time: f(x) = sin(x) + 0.5 sin(10x). Same
everything else. The only thing that changes is the hidden-layer
width: 16, 32, 64, 128.

Question: does making the network wider actually help it learn the
high frequency faster, or does it just get better at the easy low
frequency while the high one stays just as hard?

Produces:
  results/exp6_results.json
  figures/exp6_learning_time_by_width.png
  figures/exp6_mse_by_width.png
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

WIDTHS = [16, 32, 64, 128]
SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000]
N_EPOCHS = 2000


def main():
    f, coeffs = mixed_freq()
    results = {}
    for width in WIDTHS:
        t1s, t10s, mses = [], [], []
        for seed in SEEDS:
            set_seed(seed)
            model = MLP(width=width, activation="tanh")
            hist = train_and_track(model, f, k_max=12, n_epochs=N_EPOCHS, checkpoints=CHECKPOINTS)
            t1 = learning_time(hist["epochs"], hist["amp"][:, 1], coeffs[1])
            t10 = learning_time(hist["epochs"], hist["amp"][:, 10], coeffs[10])
            t1s.append(t1)
            t10s.append(t10)
            mses.append(hist["mse"][-1])
        results[width] = {"T1": t1s, "T10": t10s, "mse": mses}
        print(f"[exp6] width={width}: T1={t1s}  T10={t10s}  final MSE mean={np.mean(mses):.4f}")

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp6_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    categories = [str(w) for w in WIDTHS]

    def censored_stats(key):
        means, stds = [], []
        for w in WIDTHS:
            vals = [N_EPOCHS if v is None else v for v in results[w][key]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return means, stds

    t1_means, t1_stds = censored_stats("T1")
    t10_means, t10_stds = censored_stats("T10")
    plot_grouped_bars(
        categories, ["T1 (k=1)", "T10 (k=10)"],
        {"T1 (k=1)": t1_means, "T10 (k=10)": t10_means},
        {"T1 (k=1)": t1_stds, "T10 (k=10)": t10_stds},
        title="Experiment 6: learning time by network width\n(mean ± std, 5 seeds; capped at 2000 = never reached 90%)",
        ylabel="learning time (epochs)",
        path=os.path.join(FIG_DIR, "exp6_learning_time_by_width.png"),
    )

    mse_means = [np.mean(results[w]["mse"]) for w in WIDTHS]
    mse_stds = [np.std(results[w]["mse"]) for w in WIDTHS]
    plot_grouped_bars(
        categories, ["final MSE"],
        {"final MSE": mse_means}, {"final MSE": mse_stds},
        title="Experiment 6: final MSE by network width\n(mean ± std, 5 seeds)",
        ylabel="MSE (log scale)",
        path=os.path.join(FIG_DIR, "exp6_mse_by_width.png"),
        log=True,
    )


if __name__ == "__main__":
    main()
