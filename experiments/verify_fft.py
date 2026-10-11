"""
Section 6 - verify the Fourier computation.

Compares the FFT-based coefficients (fft_coeffs) against the
from-scratch direct sum (direct_coeffs) on a test function with a
few known frequencies, including cosines, so it checks both a_k
and b_k.

While writing fft_coeffs the two methods disagreed at first (wrong
signs on some coefficients). numpy's FFT assumes samples start at
x=0, but my domain is (-pi, pi), which shifts everything by a
k-dependent phase. See the comment in fourier_lib/dft.py for the fix.
This script is the test that caught the bug.

Produces:
  figures/verify_fft.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from fourier_lib.dft import sample_grid, fft_coeffs, direct_coeffs, amplitude
from fourier_lib.viz import CATEGORICAL

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    x = sample_grid(512)
    y = np.sin(x) + 0.5 * np.sin(5 * x) + 0.3 * np.cos(3 * x) - 0.2 * np.cos(7 * x)
    k_max = 10

    a_fft, b_fft = fft_coeffs(y, k_max)
    a_dir, b_dir = direct_coeffs(x, y, k_max)

    max_diff_a = np.max(np.abs(a_fft - a_dir))
    max_diff_b = np.max(np.abs(b_fft - b_dir))

    print("k    a_fft    a_direct   b_fft    b_direct")
    for k in range(k_max + 1):
        print(f"{k:2d}  {a_fft[k]:8.4f} {a_dir[k]:9.4f}  {b_fft[k]:8.4f} {b_dir[k]:9.4f}")
    print(f"\nmax |a_fft - a_direct| = {max_diff_a:.2e}")
    print(f"max |b_fft - b_direct| = {max_diff_b:.2e}")
    assert max_diff_a < 1e-10 and max_diff_b < 1e-10, "FFT and direct methods disagree!"
    print("PASS: FFT-based and direct-integral coefficients agree to floating-point precision.")

    ks = np.arange(k_max + 1)
    amp_fft = amplitude(a_fft, b_fft)
    amp_dir = amplitude(a_dir, b_dir)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    width = 0.35
    ax.bar(ks - width / 2, amp_fft, width=width, color=CATEGORICAL[0], label="FFT method")
    ax.bar(ks + width / 2, amp_dir, width=width, color=CATEGORICAL[1], label="direct sum method")
    ax.set_xlabel("frequency k")
    ax.set_ylabel("amplitude")
    ax.set_title("Verification: FFT vs. direct-integral Fourier coefficients\n"
                  "f(x) = sin(x) + 0.5 sin(5x) + 0.3 cos(3x) - 0.2 cos(7x)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "verify_fft.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
