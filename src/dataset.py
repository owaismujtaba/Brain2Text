import os
import glob
import math

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

N_CTX = 1500          # Whisper tiny encoder frames
EMB_DIM = 384         # Whisper tiny encoder hidden size
NEURAL_DIM = 512      # neural features per 20 ms bin
FRAME_SAMPLES = 320   # 16 kHz audio samples per encoder frame

STATS_DIRNAME = "session_stats"   # per-session channel mean/std cache


def valid_frames(audio_length):
    return max(1, min(N_CTX, math.ceil(int(audio_length) / FRAME_SAMPLES)))


def _session_stats(raw_dir, features_dir, session):
    """Per-channel (mean, std) of neural input for *session*, cached to disk.

    Stats are always computed from the session's ``data_train.hdf5`` so the
    same normalisation is applied to its train and val trials without leaking
    val statistics. Returns two float32 arrays of shape (NEURAL_DIM,).
    """
    cache_dir = os.path.join(features_dir, STATS_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{session}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["mean"].astype(np.float32), d["std"].astype(np.float32)

    train_file = os.path.join(raw_dir, session, "data_train.hdf5")
    count = 0
    s1 = np.zeros(NEURAL_DIM, dtype=np.float64)
    s2 = np.zeros(NEURAL_DIM, dtype=np.float64)
    if os.path.exists(train_file):
        with h5py.File(train_file, "r") as r:
            for trial in r.keys():
                if "input_features" not in r[trial]:
                    continue
                x = np.asarray(r[trial]["input_features"][()], dtype=np.float64)
                s1 += x.sum(axis=0)
                s2 += (x * x).sum(axis=0)
                count += x.shape[0]

    if count == 0:
        mean = np.zeros(NEURAL_DIM, dtype=np.float32)
        std = np.ones(NEURAL_DIM, dtype=np.float32)
    else:
        mean = (s1 / count).astype(np.float32)
        var = np.maximum(s2 / count - (s1 / count) ** 2, 0.0)
        std = np.sqrt(var).astype(np.float32)
        std[std < 1e-6] = 1.0   # guard silent/constant channels

    np.savez(cache, mean=mean, std=std)
    return mean, std


class NeuralToEmbeddingDataset(Dataset):
    """Pairs neural input_features with Whisper encoder embeddings per (session, trial).

    Returns:
        neural: (T_neural, 512) float32
        target: (valid_frames, 384) float32  -- whisper encoder embedding, content frames only
        transcription: str

    Args:
        normalize: z-score neural input per session/channel (default True).
        augment:   apply train-only augmentation to neural input (default False).
                   Only enable for the training split.
        aug_noise_std / aug_channel_drop / aug_scale / aug_time_jitter:
                   augmentation strengths (see _augment).
    """

    def __init__(self, raw_dir="data/raw/hdf5_data_final",
                 features_dir="data/features", split="train",
                 normalize=True, augment=False,
                 aug_noise_std=0.15, aug_channel_drop=0.1,
                 aug_scale=0.1, aug_time_jitter=2):
        self.raw_dir = raw_dir
        self.features_dir = features_dir
        self.split = split
        self.normalize = normalize
        self.augment = augment
        self.aug_noise_std = float(aug_noise_std)
        self.aug_channel_drop = float(aug_channel_drop)
        self.aug_scale = float(aug_scale)
        self.aug_time_jitter = int(aug_time_jitter)

        self.index = []  # (session, trial, audio_length)
        feat_files = sorted(glob.glob(
            os.path.join(features_dir, split, "*", f"whisper_features_{split}.hdf5")))
        for ff in feat_files:
            session = os.path.basename(os.path.dirname(ff))
            raw_file = os.path.join(raw_dir, session, f"data_{split}.hdf5")
            if not os.path.exists(raw_file):
                continue
            with h5py.File(ff, "r") as f, h5py.File(raw_file, "r") as r:
                raw_keys = set(r.keys())
                for trial in f.keys():
                    if trial not in raw_keys:
                        continue
                    self.index.append((session, trial, int(f[trial]["audio_length"][()])))

        # per-session normalisation stats (mean, std), built once up front
        self._stats = {}
        if self.normalize:
            for session in sorted({s for s, _, _ in self.index}):
                self._stats[session] = _session_stats(raw_dir, features_dir, session)

        # per-worker file-handle cache (built lazily after fork/spawn)
        self._cache = None

    def __len__(self):
        return len(self.index)

    def _file(self, path):
        if self._cache is None:
            self._cache = {}
        h = self._cache.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._cache[path] = h
        return h

    def _augment(self, neural):
        """Train-only augmentation on a (T, 512) tensor. Returns a new tensor."""
        T, C = neural.shape

        if self.aug_time_jitter > 0:
            shift = int(torch.randint(-self.aug_time_jitter,
                                      self.aug_time_jitter + 1, (1,)).item())
            if shift:
                neural = torch.roll(neural, shifts=shift, dims=0)
                if shift > 0:
                    neural[:shift] = 0.0
                else:
                    neural[shift:] = 0.0

        if self.aug_scale > 0:
            neural = neural * (1.0 + (torch.rand(1).item() * 2 - 1) * self.aug_scale)

        if self.aug_noise_std > 0:
            neural = neural + torch.randn_like(neural) * self.aug_noise_std

        if self.aug_channel_drop > 0:
            keep = (torch.rand(C) >= self.aug_channel_drop).float()
            neural = neural * keep.unsqueeze(0)

        return neural

    def __getitem__(self, i):
        session, trial, audio_length = self.index[i]
        raw_file = os.path.join(self.raw_dir, session, f"data_{self.split}.hdf5")
        feat_file = os.path.join(self.features_dir, self.split, session,
                                 f"whisper_features_{self.split}.hdf5")

        r = self._file(raw_file)
        f = self._file(feat_file)

        neural = np.asarray(r[trial]["input_features"][()], dtype=np.float32)
        if self.normalize and session in self._stats:
            mean, std = self._stats[session]
            neural = (neural - mean) / std

        neural = torch.from_numpy(neural)
        if self.augment:
            neural = self._augment(neural)

        vf = valid_frames(audio_length)
        emb = np.asarray(f[trial]["encoder_embedding"][:vf], dtype=np.float32)
        txt = f[trial]["transcription"][()]
        txt = txt.decode() if isinstance(txt, bytes) else str(txt)

        return neural, torch.from_numpy(emb), txt


def collate(batch):
    neurals, embs, txts = zip(*batch)
    B = len(batch)

    n_lengths = torch.tensor([n.shape[0] for n in neurals], dtype=torch.long)
    Tn = int(n_lengths.max())
    neural_pad = torch.zeros(B, Tn, NEURAL_DIM, dtype=torch.float32)
    for i, n in enumerate(neurals):
        neural_pad[i, : n.shape[0]] = n

    target = torch.zeros(B, N_CTX, EMB_DIM, dtype=torch.float32)
    mask = torch.zeros(B, N_CTX, dtype=torch.bool)
    for i, e in enumerate(embs):
        vf = e.shape[0]
        target[i, :vf] = e
        mask[i, :vf] = True

    return neural_pad, n_lengths, target, mask, list(txts)
