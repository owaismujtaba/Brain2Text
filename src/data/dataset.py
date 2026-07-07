"""Pairs each neural recording with its Whisper encoder-embedding target.

The training target for a trial is the Whisper encoder embedding of the audio
that was synthesised from that trial's transcription (see ``features.py``). The
model learns to reproduce that embedding directly from the neural signal.
"""
import glob
import math
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import NeuralAugment, session_stats

__all__ = ["NeuralEmbeddingDataset", "collate", "valid_frames", "NeuralAugment",
           "session_index"]


def valid_frames(audio_length, n_ctx=1500, frame_samples=320):
    """How many Whisper encoder frames actually contain audio for this clip."""
    return max(1, min(n_ctx, math.ceil(int(audio_length) / frame_samples)))


def session_index(features_dir, splits=("train", "val")):
    """Map every session found under ``features_dir`` to a stable integer id.

    The id is the session's position in the lexicographically sorted union of
    sessions across the given splits. Deriving it from both train and val (not
    just the split being loaded) guarantees training and decoding assign the
    same id to the same session, which is what the model's per-session input
    layer indexes into.
    """
    sessions = set()
    for split in splits:
        for path in glob.glob(os.path.join(features_dir, split, "*",
                                           f"whisper_features_{split}.hdf5")):
            sessions.add(os.path.basename(os.path.dirname(path)))
    return {session: i for i, session in enumerate(sorted(sessions))}


class NeuralEmbeddingDataset(Dataset):
    """One item = ``(neural, target, text)`` for a single ``(session, trial)``.

    Shapes returned by ``__getitem__``:
        neural : (T_neural, neural_dim) float32  – model input
        target : (valid_frames, emb_dim) float32 – Whisper encoder embedding
        text   : str                             – ground-truth transcription

    Args:
        normalize: z-score the neural input per session/channel.
        augment:   a ``NeuralAugment`` instance (train split) or ``None``.
    """

    def __init__(self, raw_dir, features_dir, split,
                 normalize=True, augment=None,
                 n_ctx=1500, neural_dim=512, frame_samples=320,
                 session_to_id=None):
        self.raw_dir = raw_dir
        self.features_dir = features_dir
        self.split = split
        self.normalize = normalize
        self.augment = augment
        self.n_ctx = int(n_ctx)
        self.neural_dim = int(neural_dim)
        self.frame_samples = int(frame_samples)
        # session -> integer id for the model's per-session input layer. Empty
        # dict means "no session conditioning" (every trial gets id -1).
        self.session_to_id = session_to_id or {}

        self.index = self._build_index()          # (session, trial, audio_length) per item

        self.stats = {}
        if self.normalize:
            for session in sorted({s for s, _, _ in self.index}):
                self.stats[session] = session_stats(
                    raw_dir, features_dir, session, self.neural_dim)

        self._handles = None    # per-worker HDF5 handle cache, opened lazily after fork

    def _build_index(self):
        """Find every trial that exists in both the feature and the raw files."""
        index = []
        pattern = os.path.join(self.features_dir, self.split, "*",
                               f"whisper_features_{self.split}.hdf5")
        for feat_file in sorted(glob.glob(pattern)):
            session = os.path.basename(os.path.dirname(feat_file))
            raw_file = os.path.join(self.raw_dir, session, f"data_{self.split}.hdf5")
            if not os.path.exists(raw_file):
                continue
            with h5py.File(feat_file, "r") as feats, h5py.File(raw_file, "r") as raw:
                raw_trials = set(raw.keys())
                for trial in feats.keys():
                    if trial in raw_trials:
                        index.append(
                            (session, trial, int(feats[trial]["audio_length"][()])))
        return index

    def __len__(self):
        return len(self.index)

    def _open(self, path):
        if self._handles is None:
            self._handles = {}
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __getitem__(self, i):
        session, trial, audio_length = self.index[i]
        raw_file = os.path.join(self.raw_dir, session, f"data_{self.split}.hdf5")
        feat_file = os.path.join(self.features_dir, self.split, session,
                                 f"whisper_features_{self.split}.hdf5")

        raw = self._open(raw_file)
        feats = self._open(feat_file)

        neural = np.asarray(raw[trial]["input_features"][()], dtype=np.float32)
        if self.normalize and session in self.stats:
            mean, std = self.stats[session]
            neural = (neural - mean) / std
        neural = torch.from_numpy(neural)
        if self.augment is not None:
            neural = self.augment(neural)

        n_frames = valid_frames(audio_length, self.n_ctx, self.frame_samples)
        target = np.asarray(feats[trial]["encoder_embedding"][:n_frames], dtype=np.float32)

        text = feats[trial]["transcription"][()]
        text = text.decode() if isinstance(text, bytes) else str(text)

        session_id = self.session_to_id.get(session, -1)
        return neural, torch.from_numpy(target), text, session_id


def collate(batch, n_ctx=1500):
    """Pad a list of items into batched tensors.

    Returns:
        neural      : (B, T_max, neural_dim)  zero-padded model input
        lengths     : (B,)                    true neural length of each item
        target      : (B, n_ctx, emb_dim)     zero-padded embedding targets
        mask        : (B, n_ctx) bool         True where target frames are real
        texts       : list[str]
        session_ids : (B,) long               per-session input-layer id (-1 = none)
    """
    neurals, targets, texts, session_ids = zip(*batch)
    batch_size = len(batch)
    neural_dim = neurals[0].shape[1]
    emb_dim = targets[0].shape[1]

    lengths = torch.tensor([n.shape[0] for n in neurals], dtype=torch.long)
    t_max = int(lengths.max())
    neural = torch.zeros(batch_size, t_max, neural_dim, dtype=torch.float32)
    for i, n in enumerate(neurals):
        neural[i, : n.shape[0]] = n

    target = torch.zeros(batch_size, n_ctx, emb_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, n_ctx, dtype=torch.bool)
    for i, t in enumerate(targets):
        n_frames = t.shape[0]
        target[i, :n_frames] = t
        mask[i, :n_frames] = True

    session_ids = torch.tensor(session_ids, dtype=torch.long)
    return neural, lengths, target, mask, list(texts), session_ids
