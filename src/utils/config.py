"""Load the YAML config and resolve runtime values.

Every setting for every stage lives in ``config.yaml``. A stage receives its own
section as a small attribute-accessible object (``cfg.epochs``, ``cfg.lr`` ...).
"""
import os
from types import SimpleNamespace

import torch
import yaml

# Whisper encoder hidden size (the embedding dimension we predict) per model.
# n_ctx (1500), neural_dim and frame_samples are the same across these models;
# only this dimension and the model name change.
WHISPER_EMB_DIM = {
    "tiny.en": 384, "tiny": 384,
    "base.en": 512, "base": 512,
    "small.en": 768, "small": 768,
}


def emb_dim_for(model_name: str) -> int:
    """Embedding dimension produced by a given Whisper model's encoder."""
    if model_name not in WHISPER_EMB_DIM:
        raise ValueError(
            f"unknown Whisper model '{model_name}'; known: {sorted(WHISPER_EMB_DIM)}")
    return WHISPER_EMB_DIM[model_name]


# Run-defining knobs encoded into the checkpoint/results folder name, in order.
# Each entry is (config key, label, formatter). Anything not here (num_workers,
# device, epochs, aug sub-params, ...) is deliberately excluded to keep names
# readable; the full config is still saved inside every checkpoint's "args".
def _fnum(x):
    """Compact float: 3e-04, 1e-03, 0.3 -> '3e-04', '1e-03', '0.3'."""
    x = float(x)
    return f"{x:.0e}" if (x != 0 and (x < 1e-2 or x >= 1e4)) else f"{x:g}"


_RUN_FIELDS = (
    ("gru_layers", "gru", lambda v: str(int(v))),
    ("hidden", "h", lambda v: str(int(v))),
    ("conv_channels", "conv", lambda v: str(int(v))),
    ("dropout", "drop", lambda v: f"{float(v):g}"),
    ("lr", "lr", _fnum),
    ("weight_decay", "wd", _fnum),
    ("batch_size", "bs", lambda v: str(int(v))),
    ("session_adapt", "sadapt-", lambda v: str(v)),
    ("dec_loss_weight", "decw", lambda v: f"{float(v):g}"),
    ("normalize", "norm", lambda v: str(int(bool(v)))),
    ("augment", "aug", lambda v: str(int(bool(v)))),
)


def run_name(args) -> str:
    """Folder name encoding the run-defining hyperparameters.

    Accepts a dict or an attribute namespace (e.g. ``vars(cfg)`` saved in a
    checkpoint, or ``cfg`` itself). Fields absent from ``args`` are skipped, so
    train and decode produce the same name from the same settings.
    """
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    model_name = get("model", "tiny.en")
    parts = [str(model_name)]
    _sentinel = object()
    for key, label, fmt in _RUN_FIELDS:
        val = get(key, _sentinel)
        if val is _sentinel:
            continue
        parts.append(f"{label}{fmt(val)}")
    return "_".join(parts)


def model_paths(base_features_dir: str, base_ckpt_dir: str, model_name: str) -> tuple:
    """Per-model feature and checkpoint directories, so the models never collide."""
    return (os.path.join(base_features_dir, model_name),
            os.path.join(base_ckpt_dir, model_name))


def load_config(path: str = "config.yaml") -> dict:
    """Read the whole config file into a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def stage_config(config: dict, stage: str) -> SimpleNamespace:
    """Return one stage's settings as an attribute-accessible namespace."""
    return SimpleNamespace(**dict(config.get(stage, {})))


def resolve_device(value: str) -> str:
    """Turn 'auto' into 'cuda' when a GPU is available, otherwise 'cpu'."""
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value
