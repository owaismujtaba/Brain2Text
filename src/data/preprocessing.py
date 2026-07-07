"""Neural-input preprocessing: per-session normalization and augmentation.

Two independent, optional steps are applied to the raw neural ``input_features``
before they reach the model:

* Normalization – per-session, per-channel z-scoring. Corrects the slow drift
  between recording sessions so the model always sees a consistent input scale.
* Augmentation  – light, training-only perturbations (noise, channel dropout,
  amplitude and time jitter) that make the model less likely to memorise the
  exact training trials.
"""
import os

import h5py
import numpy as np
import torch

STATS_DIRNAME = "session_stats"   # sub-folder of features_dir holding the cache


def session_stats(raw_dir, features_dir, session, neural_dim=512):
    """Return per-channel ``(mean, std)`` for one session, cached to disk.

    Statistics are always computed from the session's ``data_train.hdf5`` so the
    same normalization applies to its train and val trials without leaking any
    validation data. Both arrays have shape ``(neural_dim,)`` and dtype float32.
    """
    cache_dir = os.path.join(features_dir, STATS_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{session}.npz")
    if os.path.exists(cache):
        cached = np.load(cache)
        return cached["mean"].astype(np.float32), cached["std"].astype(np.float32)

    train_file = os.path.join(raw_dir, session, "data_train.hdf5")
    count = 0
    total = np.zeros(neural_dim, dtype=np.float64)      # sum of x
    total_sq = np.zeros(neural_dim, dtype=np.float64)   # sum of x^2
    if os.path.exists(train_file):
        with h5py.File(train_file, "r") as f:
            for trial in f.keys():
                if "input_features" not in f[trial]:
                    continue
                x = np.asarray(f[trial]["input_features"][()], dtype=np.float64)
                total += x.sum(axis=0)
                total_sq += (x * x).sum(axis=0)
                count += x.shape[0]

    if count == 0:
        mean = np.zeros(neural_dim, dtype=np.float32)
        std = np.ones(neural_dim, dtype=np.float32)
    else:
        mean = (total / count).astype(np.float32)
        var = np.maximum(total_sq / count - (total / count) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        std[std < 1e-6] = 1.0   # guard silent / constant channels against divide-by-zero

    np.savez(cache, mean=mean, std=std)
    return mean, std


class NeuralAugment:
    """Train-only augmentation applied to a single ``(T, C)`` neural tensor.

    Each perturbation is independent and controlled by its own strength; setting
    a strength to 0 disables that perturbation.
    """

    def __init__(self, noise_std=0.15, channel_drop=0.1, scale=0.1, time_jitter=2):
        self.noise_std = float(noise_std)
        self.channel_drop = float(channel_drop)
        self.scale = float(scale)
        self.time_jitter = int(time_jitter)

    def __call__(self, neural):
        _, channels = neural.shape

        # random temporal shift, zero-filling the exposed edge
        if self.time_jitter > 0:
            shift = int(torch.randint(-self.time_jitter, self.time_jitter + 1, (1,)).item())
            if shift:
                neural = torch.roll(neural, shifts=shift, dims=0)
                if shift > 0:
                    neural[:shift] = 0.0
                else:
                    neural[shift:] = 0.0

        # global amplitude jitter
        if self.scale > 0:
            neural = neural * (1.0 + (torch.rand(1).item() * 2 - 1) * self.scale)

        # additive Gaussian noise
        if self.noise_std > 0:
            neural = neural + torch.randn_like(neural) * self.noise_std

        # randomly zero out a fraction of channels
        if self.channel_drop > 0:
            keep = (torch.rand(channels) >= self.channel_drop).float()
            neural = neural * keep.unsqueeze(0)

        return neural
