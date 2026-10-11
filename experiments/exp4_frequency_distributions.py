"""
Experiment 4 - Different Kinds of Targets

Four target types:
  low   : sin(x) + 0.5 sin(2x)             (all low frequency)
  mixed : sin(x) + 0.5 sin(10x)            (one low, one high)
  high  : sin(20x) + 0.5 sin(30x)          (all high frequency)
  many  : sum_{k=1}^{10} (1/k) sin(kx)     (a bunch of frequencies at once)

For each one, train 5 different seeds and record:
  - the final overall MSE
  - how far off each individual frequency was at the end

The point is to show that "low total error" and "learned every
frequency correctly" are NOT the same thing.

Produces:
  results/exp4_results.json
  figures/exp4_mse_comparison.png
  figures/exp4_relative_error_comparison.png
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fourier_lib.model import MLP, set_seed
from fourier_lib.train import train_and_track
from fourier_lib.targets import low_freq, mixed_freq, high_freq, many_freq
from fourier_lib.viz import plot_grouped_bars

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000]
N_EPOCHS = 2000

FAMILIES = {
    "low": low_freq,
    "mixed": mixed_freq,
    "high": high_freq,
    "many": lambda: many_freq(n=10),
}
K_MAX = 32


def main():
    results = {}
    for name, builder in FAMILIES.items():
        f, coeffs = builder()
        mses = []
        rel_errs_by_k = {k: [] for k in coeffs}
        for seed in SEEDS:
            set_seed(seed)
            model = MLP(width=64, activation="tanh")
            hist = train_and_track(model, f, k_max=K_MAX, n_epochs=N_EPOCHS, checkpoints=CHECKPOINTS)
            mses.append(hist["mse"][-1])
            for k, ak in coeffs.items():
                err = abs(hist["amp"][-1][k] - ak) / ak
                rel_errs_by_k[k].append(err)
        results[name] = {
            "coeffs": coeffs,
            "mse": mses,
            "rel_err_by_k": rel_errs_by_k,
        }
        print(f"[exp4] {name}: mean final MSE = {np.mean(mses):.5f} +/- {np.std(mses):.5f}")
        for k in sorted(coeffs):
            e = rel_errs_by_k[k]
            print(f"        k={k:2d}: mean relative error = {np.mean(e):.3f} +/- {np.std(e):.3f}")

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp4_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    # Chart 1: total MSE for each target type
    categories = list(FAMILIES.keys())
    mse_means = [np.mean(results[name]["mse"]) for name in categories]
    mse_stds = [np.std(results[name]["mse"]) for name in categories]
    plot_grouped_bars(
        categories, ["final MSE"],
        {"final MSE": mse_means}, {"final MSE": mse_stds},
        title="Experiment 4: total MSE by frequency distribution\n(mean ± std, 5 seeds)",
        ylabel="MSE (log scale)",
        path=os.path.join(FIG_DIR, "exp4_mse_comparison.png"),
        log=True,
    )

    # Chart 2: the single worst-learned frequency per target type
    # (this is what shows the gap between "low total error" and
    # "actually got every frequency right")
    worst_err_means = []
    worst_err_stds = []
    for name in categories:
        per_k_means = {k: np.mean(v) for k, v in results[name]["rel_err_by_k"].items()}
        worst_k = max(per_k_means, key=per_k_means.get)
        worst_err_means.append(per_k_means[worst_k])
        worst_err_stds.append(np.std(results[name]["rel_err_by_k"][worst_k]))
    plot_grouped_bars(
        categories, ["worst-component relative error E_k"],
        {"worst-component relative error E_k": worst_err_means},
        {"worst-component relative error E_k": worst_err_stds},
        title="Experiment 4: worst per-frequency relative error by family\n(mean ± std, 5 seeds)",
        ylabel="relative error E_k",
        path=os.path.join(FIG_DIR, "exp4_relative_error_comparison.png"),
        log=False,
    )


if __name__ == "__main__":
    main()
