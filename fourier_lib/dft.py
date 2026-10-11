"""
Two ways to get Fourier coefficients from a function.

1. fft_coeffs -- uses numpy's FFT. Fast, but a black box.
2. direct_coeffs -- computes the same thing straight from the
   integral formula, no FFT. Slower, but I can use it to check that
   fft_coeffs is actually correct.

If they don't agree, something is wrong (bad sampling, aliasing, or a
bug in my code).
"""

import numpy as np


def sample_grid(n_samples=512, domain=(-np.pi, np.pi)):
    """Evenly spaced points across the domain. Skips the right
    endpoint since the function is periodic (it would just repeat
    the start point)."""
    a, b = domain
    x = np.linspace(a, b, n_samples, endpoint=False)
    return x


def direct_coeffs(x, y, k_max, domain=(-np.pi, np.pi)):
    """
    Estimates a_k and b_k using a Riemann sum (rectangles) to
    approximate the integrals:

        a_k ~= (1/pi) * integral of f(x)*cos(k*x) dx
        b_k ~= (1/pi) * integral of f(x)*sin(k*x) dx

    x, y : sample points and the function's values there. x must be
           evenly spaced and cover one full period.
    k_max: highest frequency to compute (k = 0 up to k_max).

    Returns (a, b), arrays of length k_max+1. a[0] is the DC (average)
    term; there's no b[0] since sin(0*x) = 0.
    """
    a_lo, b_hi = domain
    period = b_hi - a_lo
    n = len(x)
    dx = period / n  # width of each rectangle

    a = np.zeros(k_max + 1)
    b = np.zeros(k_max + 1)

    # a_0 is just twice the average value of f
    a[0] = (2.0 / period) * np.sum(y) * dx

    for k in range(1, k_max + 1):
        cos_k = np.cos(k * x)
        sin_k = np.sin(k * x)
        a[k] = (2.0 / period) * np.sum(y * cos_k) * dx
        b[k] = (2.0 / period) * np.sum(y * sin_k) * dx

    return a, b


def fft_coeffs(y, k_max, domain=(-np.pi, np.pi)):
    """
    Computes the same a_k, b_k as direct_coeffs, but using numpy's
    FFT instead of a manual sum.

    numpy's FFT assumes the first sample is at x = 0. My domain is
    (-pi, pi), which starts at x = -pi, so every cos(kx)/sin(kx) is
    shifted by a phase of k*a. Without correcting for that, the FFT
    gives wrong (sign-flipped) coefficients. My first version didn't
    correct for this and it disagreed with direct_coeffs until I
    fixed it.

    Redoing the trig for x_n = a + n*dx gives:

        a_k =  (2/N) * [cos(k*a)*Re(X_k) + sin(k*a)*Im(X_k)]
        b_k =  (2/N) * [sin(k*a)*Re(X_k) - cos(k*a)*Im(X_k)]

    which is the usual a_k = (2/N)Re(X_k), b_k = -(2/N)Im(X_k) only
    when a = 0. For our domain, a = -pi, so it comes out to
    multiplying everything by (-1)^k.

    Needs the samples to be evenly spaced and cover one full period,
    same as direct_coeffs.
    """
    n = len(y)
    a_lo, _ = domain
    X = np.fft.fft(y)

    a = np.zeros(k_max + 1)
    b = np.zeros(k_max + 1)

    a[0] = np.real(X[0]) / n  # DC term (average value), same as the direct method

    for k in range(1, k_max + 1):
        phase = k * a_lo
        c, s = np.cos(phase), np.sin(phase)
        re, im = np.real(X[k]), np.imag(X[k])
        a[k] = 2.0 / n * (c * re + s * im)
        b[k] = 2.0 / n * (s * re - c * im)

    return a, b


def amplitude(a_k, b_k):
    """Amplitude of frequency k: sqrt(a_k^2 + b_k^2). How much of
    frequency k is in the function, regardless of cosine/sine mix."""
    return np.sqrt(a_k ** 2 + b_k ** 2)


def nyquist_frequency(n_samples, domain=(-np.pi, np.pi)):
    """
    Highest frequency k you can trust with n_samples evenly spaced
    points over the domain. Above this, aliasing happens (see
    numerical_considerations.py). For domain length 2*pi, this
    works out to k = N/2.
    """
    a, b = domain
    period = b - a
    return n_samples * np.pi / period  # = N/2 when period = 2*pi
