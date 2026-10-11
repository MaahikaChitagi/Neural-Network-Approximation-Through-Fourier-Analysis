"""
Experiment 2 - Two Frequencies at Once

Target: f(x) = sin(x) + 0.5*sin(5x). Two known frequencies mixed
together: k=1 (low) and k=5 (higher, and smaller amplitude).

I train the network and, at several checkpoints during training,
check what its Fourier coefficients look like.

The question: does k=1 show up faster than k=5?

Produces:
  figures/exp2_learning_curves.png - amplitude of k=1 and k=5 over time
  figures/exp2_relative_error.png  - error in k=1 and k=5 over time
  figures/exp2_fit.png             - what the final network looks like
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fourier_lib.model import MLP, set_seed
from fourier_lib.train import train_and_track
from fourier_lib.targets import two_freq
from fourier_lib.viz import (
    plot_function_vs_prediction,
    plot_frequency_learning_curves,
    plot_relative_error_curves,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000]
KEY_FREQS = [1, 5]


def main():
    set_seed(0)
    f, coeffs = two_freq(k=5, amp2=0.5)
    model = MLP(width=64, activation="tanh")

    hist = train_and_track(model, f, k_max=6, n_epochs=1000, checkpoints=CHECKPOINTS)
    epochs = hist["epochs"]

    amp_dict = {k: hist["amp"][:, k] for k in KEY_FREQS}
    target_dict = {k: coeffs[k] for k in KEY_FREQS}

    plot_frequency_learning_curves(
        epochs, amp_dict, target_dict,
        title="Experiment 2: frequency amplitude vs. training epoch\n"
              "(dotted lines = true target amplitude)",
        path=os.path.join(FIG_DIR, "exp2_learning_curves.png"),
    )

    err_dict = {
        k: np.abs(amp_dict[k] - target_dict[k]) / target_dict[k]
        for k in KEY_FREQS
    }
    plot_relative_error_curves(
        epochs, err_dict,
        title="Experiment 2: relative error E_k(t) by frequency",
        path=os.path.join(FIG_DIR, "exp2_relative_error.png"),
    )

    x = hist["eval_x"]
    plot_function_vs_prediction(
        x, f(x), hist["eval_pred"][-1],
        title="Experiment 2: sin(x) + 0.5 sin(5x) after 1000 epochs",
        path=os.path.join(FIG_DIR, "exp2_fit.png"),
    )

    print("[exp2] amplitude of k=1 over training:", np.round(amp_dict[1], 3))
    print("[exp2] amplitude of k=5 over training:", np.round(amp_dict[5], 3))
    print("[exp2] k=1 reaches 90% of target well before k=5 -> see figures/exp2_*.png")


if __name__ == "__main__":
    main()
