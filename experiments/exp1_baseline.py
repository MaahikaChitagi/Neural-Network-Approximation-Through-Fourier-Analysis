"""
Experiment 1 - Baseline Check

Just making sure the whole pipeline actually works (train the
network, save checkpoints, pull out Fourier coefficients, make
plots) before trusting it on anything harder. Target is the simplest
thing possible: f(x) = sin(x).

Produces:
  figures/exp1_fit.png - target vs. what the network predicted
  figures/exp1_mse.png - MSE over training
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fourier_lib.model import MLP, set_seed
from fourier_lib.train import train_and_track
from fourier_lib.targets import pure_sine
from fourier_lib.viz import plot_function_vs_prediction, plot_mse_curve

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    set_seed(0)
    f, coeffs = pure_sine()
    model = MLP(width=64, activation="tanh")

    checkpoints = [0, 10, 25, 50, 100, 250, 500, 1000]
    hist = train_and_track(model, f, k_max=3, n_epochs=1000, checkpoints=checkpoints)

    x = hist["eval_x"]
    y_true = f(x)
    y_pred = hist["eval_pred"][-1]

    plot_function_vs_prediction(
        x, y_true, y_pred,
        title="Experiment 1: f(x) = sin(x) after 1000 epochs",
        path=os.path.join(FIG_DIR, "exp1_fit.png"),
    )
    plot_mse_curve(
        hist["epochs"], hist["mse"],
        title="Experiment 1: training MSE vs. epoch",
        path=os.path.join(FIG_DIR, "exp1_mse.png"),
    )

    final_mse = hist["mse"][-1]
    final_amp1 = hist["amp"][-1][1]
    print(f"[exp1] final MSE = {final_mse:.6f}")
    print(f"[exp1] final |a_hat_1| = {final_amp1:.4f} (target = {coeffs[1]})")
    print("[exp1] pipeline check complete. Figures saved to figures/exp1_*.png")


if __name__ == "__main__":
    main()
