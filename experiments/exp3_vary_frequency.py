"""
Experiment 3 - What Happens as the Frequency Gets Higher?

Target family: f_k(x) = sin(x) + 0.5*sin(k*x), for k in {2,5,10,20,30}.

For each k, I train the same network from scratch with 5 different
random seeds, and measure:
  - T_1: how long it takes to learn the fixed low frequency (k=1)
  - T_k: how long it takes to learn the changing higher frequency

Question: as k gets bigger, does it just take longer to learn, or
does the network stop being able to learn it at all in the time
given?

Produces:
  results/exp3_results.json
  figures/exp3_learning_time_vs_k.png
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from fourier_lib.model import MLP, set_seed
from fourier_lib.train import train_and_track, learning_time
from fourier_lib.targets import two_freq
from fourier_lib.viz import plot_k_vs_learning_time

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
RES_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

K_VALUES = [2, 5, 10, 20, 30]
SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000, 1500, 2000]
N_EPOCHS = 2000


def run_one(k, seed):
    set_seed(seed)
    f, coeffs = two_freq(k=k, amp2=0.5)
    model = MLP(width=64, activation="tanh")
    hist = train_and_track(model, f, k_max=k + 2, n_epochs=N_EPOCHS, checkpoints=CHECKPOINTS)
    t1 = learning_time(hist["epochs"], hist["amp"][:, 1], coeffs[1])
    tk = learning_time(hist["epochs"], hist["amp"][:, k], coeffs[k])
    return t1, tk, hist["mse"][-1]


def main():
    results = {}
    for k in K_VALUES:
        t1s, tks, final_mses = [], [], []
        for seed in SEEDS:
            t1, tk, mse = run_one(k, seed)
            t1s.append(t1)
            tks.append(tk)
            final_mses.append(mse)
        results[k] = {"T1": t1s, "Tk": tks, "final_mse": final_mses}
        print(f"[exp3] k={k}: T1={t1s}  Tk={tks}")

    os.makedirs(RES_DIR, exist_ok=True)
    with open(os.path.join(RES_DIR, "exp3_results.json"), "w") as fp:
        json.dump(results, fp, indent=2)

    # T_k can be None if the network never got to 90% within N_EPOCHS.
    # For the plot, I just treat "never learned" as N_EPOCHS and make
    # a note of that on the chart itself.
    ks = K_VALUES
    tk_means, tk_stds = [], []
    never_learned = {}
    for k in ks:
        vals = [N_EPOCHS if v is None else v for v in results[k]["Tk"]]
        never_learned[k] = sum(1 for v in results[k]["Tk"] if v is None)
        tk_means.append(np.mean(vals))
        tk_stds.append(np.std(vals))

    plot_k_vs_learning_time(
        ks, tk_means, tk_stds,
        title="Experiment 3: learning time T_k vs. frequency k\n"
              "sin(x) + 0.5 sin(kx), mean ± std over 5 seeds\n"
              "(2000 = capped: network never reached 90% within budget)",
        path=os.path.join(FIG_DIR, "exp3_learning_time_vs_k.png"),
    )
    print("[exp3] never-learned-within-budget counts (out of 5 seeds):", never_learned)


if __name__ == "__main__":
    main()
