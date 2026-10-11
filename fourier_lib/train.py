"""
The training loop. Trains a network on a target function, and at a
few checkpoint epochs, saves:

  - the current MSE loss
  - what the network predicts on a fixed grid of points
  - the Fourier coefficients of those predictions (via FFT), so I can
    see which frequencies the network has learned so far

Every experiment script uses this function.
"""

import numpy as np
import torch

from fourier_lib.dft import fft_coeffs, amplitude, sample_grid


DEFAULT_CHECKPOINTS = [0, 10, 25, 50, 100, 250, 500, 1000]


def train_and_track(
    model,
    target_fn,
    k_max,
    n_train=256,
    n_epochs=1000,
    checkpoints=None,
    lr=1e-3,
    domain=(-np.pi, np.pi),
    device="cpu",
):
    """
    Trains `model` to approximate `target_fn` over `domain`.

    Returns a dict with:
      "epochs"    : the checkpoint epochs that got recorded
      "mse"       : MSE loss at each checkpoint
      "amp"       : shape (n_checkpoints, k_max+1), amplitude of
                    each frequency k at each checkpoint
      "a", "b"    : the raw cosine/sine coefficients, same shape
      "eval_x"    : the fixed grid used to evaluate the network
      "eval_pred" : what the network predicted at each checkpoint,
                    shape (n_checkpoints, n_eval)
    """
    if checkpoints is None:
        checkpoints = DEFAULT_CHECKPOINTS
    checkpoints = sorted(set(checkpoints) | {0})

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Training points: random x-values, picked once and then fixed
    # for the whole run. So one epoch = one pass of gradient descent
    # over the same 256 points.
    x_train = np.random.uniform(domain[0], domain[1], size=(n_train, 1))
    y_train = target_fn(x_train.flatten()).reshape(-1, 1)
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)

    # Evaluation grid: evenly spaced points used only to get the
    # Fourier coefficients of the network's current output. Has to
    # cover a full period or the FFT math breaks (see dft.py).
    eval_x = sample_grid(n_samples=512, domain=domain)
    eval_x_t = torch.tensor(eval_x.reshape(-1, 1), dtype=torch.float32, device=device)

    history = {
        "epochs": [],
        "mse": [],
        "amp": [],
        "a": [],
        "b": [],
        "eval_x": eval_x,
        "eval_pred": [],
    }

    def record(epoch):
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train_t)
            mse = loss_fn(pred_train, y_train_t).item()
            pred_eval = model(eval_x_t).cpu().numpy().flatten()
        a, b = fft_coeffs(pred_eval, k_max, domain=domain)
        amp = amplitude(a, b)
        history["epochs"].append(epoch)
        history["mse"].append(mse)
        history["amp"].append(amp)
        history["a"].append(a)
        history["b"].append(b)
        history["eval_pred"].append(pred_eval)
        model.train()

    checkpoint_set = set(checkpoints)
    record(0)

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if epoch in checkpoint_set:
            record(epoch)

    history["amp"] = np.array(history["amp"])
    history["a"] = np.array(history["a"])
    history["b"] = np.array(history["b"])
    history["eval_pred"] = np.array(history["eval_pred"])
    return history


def learning_time(epochs, amp_k, target_amp_k, threshold=0.9):
    """
    T_k = the first checkpoint where |a_hat_k(t)| / |a_k| >= threshold.
    When did the network get frequency k to 90% correct?

    Returns None if it never hits the threshold within the tracked
    epochs, meaning it never learned that frequency in time.
    """
    if target_amp_k == 0:
        return None
    ratio = np.array(amp_k) / target_amp_k
    hits = np.where(ratio >= threshold)[0]
    if len(hits) == 0:
        return None
    return epochs[hits[0]]
