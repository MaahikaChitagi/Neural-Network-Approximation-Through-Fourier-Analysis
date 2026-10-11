"""
Spectral evolution heatmaps.

Training epoch on one axis, frequency k on the other, amplitude as
color, for a few different target functions. Low frequencies light
up almost right away; high frequencies stay dark much longer (or
never light up within the training budget).

Makes one heatmap per target, saved as figures/heatmap_<name>.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fourier_lib.model import MLP, set_seed
from fourier_lib.train import train_and_track
from fourier_lib.targets import low_freq, mixed_freq, high_freq, many_freq
from fourier_lib.viz import plot_spectral_heatmap

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")

CHECKPOINTS = list(range(0, 2001, 25))
K_MAX = 32
N_EPOCHS = 2000

TARGETS = {
    "low": low_freq,
    "mixed": mixed_freq,
    "high": high_freq,
    "many": lambda: many_freq(n=10),
}


def main():
    for name, builder in TARGETS.items():
        f, coeffs = builder()
        set_seed(0)
        model = MLP(width=64, activation="tanh")
        hist = train_and_track(model, f, k_max=K_MAX, n_epochs=N_EPOCHS, checkpoints=CHECKPOINTS)
        k_values = np.arange(0, K_MAX + 1)
        plot_spectral_heatmap(
            hist["epochs"], k_values, hist["amp"],
            title=f"Spectral evolution during training: target = '{name}'",
            path=os.path.join(FIG_DIR, f"heatmap_{name}.png"),
        )
        print(f"[heatmap] {name}: saved. true frequencies = {sorted(coeffs.keys())}")


if __name__ == "__main__":
    main()
