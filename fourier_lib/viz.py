"""
Shared plotting helpers, so every experiment script produces figures
that look consistent.

Colors: we use the Okabe-Ito palette, which is a standard colorblind-
safe qualitative palette, assigned in a fixed order so the same
color always means the same thing (e.g. "k=1" is always blue) across
every figure in the project.
"""

import matplotlib.pyplot as plt
import numpy as np

CATEGORICAL = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#999999",  # gray
]

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


def plot_function_vs_prediction(x, y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, y_true, color=CATEGORICAL[6], lw=2.5, label="target f(x)")
    ax.plot(x, y_pred, color=CATEGORICAL[0], lw=2, ls="--", label="network prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_mse_curve(epochs, mse, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, mse, color=CATEGORICAL[0], lw=2, marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_frequency_learning_curves(epochs, amp_dict, target_dict, title, path):
    """amp_dict: {k: array of amplitudes over epochs}
    target_dict: {k: true amplitude}"""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, k in enumerate(sorted(amp_dict)):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(epochs, amp_dict[k], color=color, lw=2, label=f"k={k} (network)")
        ax.axhline(target_dict[k], color=color, lw=1, ls=":", alpha=0.8)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("Fourier coefficient magnitude")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_k_vs_learning_time(ks, means, stds, title, path, ylabel="Learning time T_k (epochs)"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ks, means, yerr=stds, color=CATEGORICAL[3], lw=2, marker="o",
                capsize=4, ms=6)
    ax.set_xlabel("frequency k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_grouped_bars(categories, group_labels, values_by_group, errs_by_group,
                       title, ylabel, path, log=False):
    """categories: x-axis labels. group_labels: legend entries.
    values_by_group[g] is an array aligned with categories."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_groups = len(group_labels)
    width = 0.8 / n_groups
    x = np.arange(len(categories))
    for i, g in enumerate(group_labels):
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, values_by_group[g], width=width, yerr=errs_by_group[g],
               color=CATEGORICAL[i % len(CATEGORICAL)], label=str(g), capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectral_heatmap(epochs, k_values, amp_matrix, title, path, vmax=None):
    """amp_matrix: shape (len(epochs), len(k_values))"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.pcolormesh(epochs, k_values, amp_matrix.T, shading="auto",
                        cmap="viridis", vmax=vmax)
    ax.set_xlabel("training epoch")
    ax.set_ylabel("frequency k")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fourier coefficient magnitude")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_relative_error_curves(epochs, err_dict, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for i, k in enumerate(sorted(err_dict)):
        color = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(epochs, err_dict[k], color=color, lw=2, label=f"k={k}")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("relative error E_k(t)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
