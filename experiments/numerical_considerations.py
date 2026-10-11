"""
A few numerical checks worth doing (not full experiments, more like
sanity-check figures for the paper):

1. Sampling resolution: does the spectrum change if I use more
   sample points?
2. Aliasing: set up a case where a high frequency, sampled too
   coarsely, looks exactly like a low frequency.
3. Frequency resolution: show that recording a shorter stretch of
   the function blurs together frequencies that a longer recording
   can tell apart.

Produces:
  figures/numerics_sampling_resolution.png
  figures/numerics_aliasing.png
  figures/numerics_frequency_resolution.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from fourier_lib.dft import sample_grid, fft_coeffs, direct_coeffs, amplitude, nyquist_frequency
from fourier_lib.viz import CATEGORICAL

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def demo_sampling_resolution():
    """Computes the spectrum of the same function at more and more
    sample points, and shows the answer settles down to the same
    value once you have enough samples."""
    def f(x):
        return np.sin(x) + 0.5 * np.sin(10 * x)

    sample_counts = [32, 64, 128, 256, 512, 1024]
    a1_vals, a10_vals = [], []
    for n in sample_counts:
        x = sample_grid(n)
        y = f(x)
        a, b = fft_coeffs(y, k_max=10)
        amp = amplitude(a, b)
        a1_vals.append(amp[1])
        a10_vals.append(amp[10])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sample_counts, a1_vals, "o-", color=CATEGORICAL[0], label="computed |a_1| (true=1.0)")
    ax.plot(sample_counts, a10_vals, "o-", color=CATEGORICAL[1], label="computed |a_10| (true=0.5)")
    ax.axhline(1.0, color=CATEGORICAL[0], ls=":", lw=1)
    ax.axhline(0.5, color=CATEGORICAL[1], ls=":", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of samples N")
    ax.set_ylabel("computed amplitude")
    ax.set_title("Sampling resolution: coefficients converge once N is large enough")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "numerics_sampling_resolution.png"))
    plt.close(fig)
    print("[numerics] sampling resolution: |a_1| by N =", np.round(a1_vals, 4))
    print("[numerics] sampling resolution: |a_10| by N =", np.round(a10_vals, 4))


def demo_aliasing():
    """Samples sin(30x) with only N=32 points over [-pi, pi]. The
    Nyquist frequency for N=32 is only 16, so k=30 can't be
    represented correctly. It folds back and looks like a lower
    frequency instead.

    The alias of frequency k under sampling rate N is |k mod N|,
    folded into [0, N/2]. For k=30, N=32: 30 mod 32 = 30, and since
    that's above 16, it folds to 32 - 30 = 2. So sin(30x) sampled at
    N=32 should look exactly like sin(2x).
    """
    N = 32
    print(f"[numerics] Nyquist frequency for N={N}:", nyquist_frequency(N))

    x_coarse = sample_grid(N)
    y_coarse = np.sin(30 * x_coarse)

    a, b = fft_coeffs(y_coarse, k_max=15)
    amp = amplitude(a, b)
    print("[numerics] spectrum of sin(30x) sampled at N=32:", np.round(amp, 3))
    print("[numerics] -> all the energy shows up at k=2, not k=30! That's aliasing.")

    x_fine = np.linspace(-np.pi, np.pi, 2000)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(x_fine, np.sin(30 * x_fine), color=CATEGORICAL[6], lw=1, alpha=0.6, label="true sin(30x)")
    axes[0].plot(x_coarse, y_coarse, "o", color=CATEGORICAL[3], ms=5, label=f"N={N} samples")
    axes[0].plot(x_fine, np.sin(2 * x_fine), color=CATEGORICAL[0], lw=2, ls="--", label="sin(2x)")
    axes[0].set_title("The samples of sin(30x) exactly match sin(2x)")
    axes[0].set_xlabel("x")
    axes[0].legend(frameon=False, fontsize=8)

    ks = np.arange(0, 16)
    axes[1].bar(ks, amp, color=CATEGORICAL[3])
    axes[1].set_xlabel("frequency k")
    axes[1].set_ylabel("amplitude")
    axes[1].set_title(f"Computed spectrum (N={N} samples)")
    fig.suptitle("Aliasing: sin(30x) undersampled at N=32 looks like sin(2x)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "numerics_aliasing.png"))
    plt.close(fig)


def demo_frequency_resolution():
    """Frequency resolution depends on how LONG a stretch of the
    function you record, not how many samples you cram into it. My
    `direct_coeffs`/`fft_coeffs` formulas assume the domain covers
    exactly one full period of the k=1 wave (T = 2*pi). If you only
    look at a much shorter stretch instead, sin(10x) and sin(11x)
    stop being exactly orthogonal over that window, so their
    coefficients start leaking into each other — even with plenty of
    samples.

    Here I compare recording a full period [-pi, pi] against a
    quarter of that, [-pi/4, pi/4], using the exact same number of
    samples both times (so sample count isn't the thing changing).
    """
    def f(x):
        return np.sin(10 * x) + np.sin(11 * x)

    domains = [(-np.pi, np.pi), (-np.pi / 4, np.pi / 4)]
    labels = ["full period T=2π (correct resolution assumption)",
              "quarter period T=π/2 (resolution assumption violated)"]
    n = 256

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for domain, label, ax, color in zip(domains, labels, axes, [CATEGORICAL[0], CATEGORICAL[3]]):
        x = sample_grid(n, domain=domain)
        y = f(x)
        a, b = direct_coeffs(x, y, k_max=15, domain=domain)
        amp = amplitude(a, b)
        ax.bar(np.arange(16), amp, color=color)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("frequency k")
        ax.set_ylabel("amplitude")
    fig.suptitle("Frequency resolution depends on record length, not just sample count")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "numerics_frequency_resolution.png"))
    plt.close(fig)
    print("[numerics] frequency resolution demo saved (compares record length, N=256 both cases)")


if __name__ == "__main__":
    demo_sampling_resolution()
    demo_aliasing()
    demo_frequency_resolution()
