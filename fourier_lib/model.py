"""
The neural network used everywhere in this project: a small MLP that
takes in one number x and outputs one number f_hat(x), trying to
match a target function f(x).

  x -> Linear -> activation -> Linear -> activation -> Linear -> f_hat(x)

Keeping the architecture simple on purpose, since the project is
about studying how a normal network learns, not building a new one.
"""

import numpy as np
import torch
import torch.nn as nn


class Sine(nn.Module):
    """A sine activation: sin(x). PyTorch doesn't have this built in
    so I made it myself. Testing this since the target functions are
    also made of sines."""

    def forward(self, x):
        return torch.sin(x)


def make_activation(name):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sine":
        return Sine()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    """A basic MLP with two hidden layers of the same width. The
    "plain input" network used in Experiments 1 through 6."""

    def __init__(self, width=64, activation="tanh"):
        super().__init__()
        act = make_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(1, width),
            act,
            nn.Linear(width, width),
            make_activation(activation),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)


class FourierFeatureMLP(nn.Module):
    """
    Same MLP as before, but instead of feeding the network raw x, I
    turn x into a list of sines and cosines first:

        phi(x) = [sin(x), cos(x), sin(2x), cos(2x), ..., sin(Bx), cos(Bx)]

    and feed phi(x) in instead. This is the idea tested in Experiment 7.
    """

    def __init__(self, n_freqs=10, width=64, activation="tanh"):
        super().__init__()
        self.n_freqs = n_freqs
        act = make_activation(activation)
        in_dim = 2 * n_freqs
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            act,
            nn.Linear(width, width),
            make_activation(activation),
            nn.Linear(width, 1),
        )

    def featurize(self, x):
        # x: (N, 1) tensor. builds [sin(x), cos(x), sin(2x), cos(2x), ...]
        ks = torch.arange(1, self.n_freqs + 1, device=x.device, dtype=x.dtype)
        kx = x * ks.view(1, -1)  # (N, n_freqs)
        return torch.cat([torch.sin(kx), torch.cos(kx)], dim=1)

    def forward(self, x):
        return self.net(self.featurize(x))


def set_seed(seed):
    """Sets the random seed so a run can be repeated exactly, or run
    again with a different seed to check a result isn't just luck."""
    np.random.seed(seed)
    torch.manual_seed(seed)
