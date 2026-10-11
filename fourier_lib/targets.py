"""
Target functions the network tries to learn.

All of these are sums of sine waves, no cosines, so I always know the
correct Fourier coefficients ahead of time: if f(x) = sum_k c_k *
sin(k*x), then b_k = c_k and a_k = 0 exactly. That way I can check if
the network learned each frequency correctly instead of guessing.

Defined on [-pi, pi], one full period of sin(x). A full period
matters for the Fourier math to work, see dft.py.
"""

import numpy as np


def two_freq(k=5, amp2=0.5):
    """sin(x) + amp2 * sin(k*x). The two-frequency target used in
    Experiment 2 and Experiment 3.

    Note: if k == 1, both terms are the same frequency and just merge
    into one coefficient (1 + amp2) instead of two separate
    frequencies. Use pure_sine() for plain sin(x) instead."""
    if k == 1:
        coeffs = {1: 1.0 + amp2}
    else:
        coeffs = {1: 1.0, k: amp2}

    def f(x):
        return np.sin(x) + amp2 * np.sin(k * x)

    return f, coeffs


def pure_sine():
    """f(x) = sin(x). The single-frequency baseline target."""
    coeffs = {1: 1.0}

    def f(x):
        return np.sin(x)

    return f, coeffs


def low_freq():
    """sin(x) + 0.5*sin(2x) -- both frequencies are low."""
    coeffs = {1: 1.0, 2: 0.5}

    def f(x):
        return np.sin(x) + 0.5 * np.sin(2 * x)

    return f, coeffs


def mixed_freq():
    """sin(x) + 0.5*sin(10x) -- one low, one high frequency."""
    coeffs = {1: 1.0, 10: 0.5}

    def f(x):
        return np.sin(x) + 0.5 * np.sin(10 * x)

    return f, coeffs


def high_freq():
    """sin(20x) + 0.5*sin(30x) -- both frequencies are high."""
    coeffs = {20: 1.0, 30: 0.5}

    def f(x):
        return np.sin(20 * x) + 0.5 * np.sin(30 * x)

    return f, coeffs


def many_freq(n=10):
    """sum_{k=1}^{n} (1/k) * sin(k*x) -- a whole staircase of
    frequencies, each weaker than the last."""
    coeffs = {k: 1.0 / k for k in range(1, n + 1)}

    def f(x):
        out = np.zeros_like(x, dtype=float)
        for k in range(1, n + 1):
            out = out + (1.0 / k) * np.sin(k * x)
        return out

    return f, coeffs


TARGETS = {
    "baseline": pure_sine,
    "low": low_freq,
    "mixed": mixed_freq,
    "high": high_freq,
    "many": many_freq,
}
