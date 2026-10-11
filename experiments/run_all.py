"""
Runs every experiment in order and regenerates all the figures and
result files. Takes under a minute on a regular laptop.
"""

import subprocess
import sys
import os

HERE = os.path.dirname(__file__)

SCRIPTS = [
    "verify_fft.py",
    "numerical_considerations.py",
    "exp1_baseline.py",
    "exp2_two_frequencies.py",
    "exp3_vary_frequency.py",
    "exp4_frequency_distributions.py",
    "exp5_activations.py",
    "exp6_width.py",
    "exp7_fourier_features.py",
    "spectral_heatmaps.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n{'=' * 60}\nRunning {script}\n{'=' * 60}")
        subprocess.run([sys.executable, os.path.join(HERE, script)], check=True)


if __name__ == "__main__":
    main()
