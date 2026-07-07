"""Turn predicted embeddings into text via the frozen Whisper decoder, and score.

Two entry points:
    * ``decode_dataset`` – decode trials of a dataset; returns (WER, exact, rows).
      Shared by the training loop's periodic WER check.
    * ``decode_split``   – the ``python main.py decode`` stage: load a checkpoint,
      decode a whole split, and write a per-trial CSV.
"""
import csv
import os

import torch

from ..data.dataset import NeuralEmbeddingDataset
from ..model.model import ConvBiGRU
from ..utils.config import run_name
from ..utils.logging_utils import get_logger
from .metrics import word_error_rate

logger = get_logger("decode")

_normalizer = None


def _english_normalizer():
    """Whisper's English text normalizer, with a plain lower/strip fallback."""
    global _normalizer
    if _normalizer is None:
        try:
            from whisper.normalizers import EnglishTextNormalizer
            _normalizer = EnglishTextNormalizer()
        except Exception:
            _normalizer = lambda s: s.lower().strip()
    return _normalizer


@torch.no_grad()
def decode_dataset(model, dataset, device, whisper_model=None, model_name="tiny.en",
                   beam_size=5, limit=0, n_ctx=1500, emb_dim=384, session_to_id=None):
    """Decode trials and return ``(mean_wer, exact_match_fraction, rows)``.

    ``rows`` is a list of ``{"idx", "wer", "truth", "pred"}`` dicts. ``session_to_id``
    maps each session to its per-session input-layer id; when given, the model's
    day-specific transform is applied. Sessions absent from the map (or a ``None``
    map) fall back to id -1, which the model treats as an identity transform.
    """
    import whisper
    if whisper_model is None:
        whisper_model = whisper.load_model(model_name, device=device)
    options = whisper.DecodingOptions(
        language="en", without_timestamps=True,
        fp16=(str(device) == "cuda"),
        beam_size=(beam_size if beam_size and beam_size > 1 else None),
    )
    normalize_text = _english_normalizer()
    model.eval()

    session_to_id = session_to_id or {}
    count = len(dataset) if not limit else min(limit, len(dataset))
    rows, errors, exact = [], [], 0
    for i in range(count):
        neural, target, truth, _ = dataset[i]
        n_frames = target.shape[0]
        session, trial, _ = dataset.index[i]

        session_ids = torch.tensor([session_to_id.get(session, -1)],
                                   dtype=torch.long, device=device)
        pred = model(neural.unsqueeze(0).to(device),
                     session_ids=session_ids)[0]                # (n_ctx, emb_dim)
        feats = torch.zeros(n_ctx, emb_dim, device=device)
        feats[:n_frames] = pred[:n_frames]                     # keep content frames only
        result = whisper.decode(whisper_model, feats.unsqueeze(0).float(), options)
        text = (result[0] if isinstance(result, list) else result).text.strip()

        ref, hyp = normalize_text(truth), normalize_text(text)
        error = word_error_rate(ref, hyp)
        errors.append(error)
        exact += int(ref == hyp)
        rows.append({"idx": i, "session": session, "trial": trial,
                     "wer": round(error, 4), "truth": truth, "pred": text})

    mean_wer = sum(errors) / max(len(errors), 1)
    return mean_wer, exact / max(len(errors), 1), rows


def decode_split(cfg):
    """Stage: decode a whole split with a saved checkpoint and write a CSV."""
    device = cfg.device
    ckpt = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    saved = ckpt["args"]      # the training config, so we rebuild the exact model

    # The Whisper target the checkpoint was trained for drives everything: which
    # decoder to use, the embedding size, and the model-specific feature folder.
    model_name = saved.get("model", "tiny.en")
    n_ctx = int(saved.get("n_ctx", 1500))
    emb_dim = int(saved.get("emb_dim", 384))
    neural_dim = int(saved.get("neural_dim", 512))
    frame_samples = int(saved.get("frame_samples", 320))
    features_dir = os.path.join(cfg.features_dir, model_name)

    # Per-session input layer: rebuild it exactly as trained. Old checkpoints
    # predate this and have no map -> no session conditioning ("none").
    session_to_id = ckpt.get("session_to_id", {})
    session_adapt = saved.get("session_adapt", "none")

    logger.info("=" * 60)
    logger.info(f"Decoding split='{cfg.split}' with checkpoint {cfg.ckpt}")
    logger.info(f"model={model_name} | emb_dim={emb_dim} | beam_size={cfg.beam_size} "
                f"| features={features_dir} | device={device}")

    model = ConvBiGRU(
        neural_dim=neural_dim, conv_channels=saved.get("conv_channels", 256),
        hidden=saved["hidden"], gru_layers=saved["gru_layers"],
        emb_dim=emb_dim, n_ctx=n_ctx, dropout=saved["dropout"],
        n_sessions=len(session_to_id), session_adapt=session_adapt).to(device)
    model.load_state_dict(ckpt["model"])
    logger.info(f"loaded model (epoch {ckpt.get('epoch')}, gru_layers {saved['gru_layers']}, "
                f"session_adapt={session_adapt}, sessions={len(session_to_id)})")

    normalize = bool(saved.get("normalize", True))   # must match how it was trained
    dataset = NeuralEmbeddingDataset(
        cfg.raw_dir, features_dir, cfg.split,
        normalize=normalize, augment=None,
        n_ctx=n_ctx, neural_dim=neural_dim, frame_samples=frame_samples,
        session_to_id=session_to_id)
    logger.info(f"{cfg.split} trials: {len(dataset)} "
                f"(decoding {cfg.limit or len(dataset)}) | normalize={normalize}")

    mean_wer, exact, rows = decode_dataset(
        model, dataset, device, model_name=model_name,
        beam_size=cfg.beam_size, limit=cfg.limit, n_ctx=n_ctx, emb_dim=emb_dim,
        session_to_id=session_to_id)

    # Namespace the output by the training run (model + hyperparameters) and
    # split, rebuilt from the checkpoint's saved args, so decoding different
    # runs/checkpoints or splits never overwrite each other's results.
    out_dir, out_file = os.path.split(cfg.out)
    root, ext = os.path.splitext(out_file)
    out_path = os.path.join(out_dir, run_name(saved), f"{root}_{cfg.split}{ext}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "session", "trial", "wer", "truth", "pred"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"WER {mean_wer:.4f} | exact {exact * 100:.1f}% "
                f"| wrote {len(rows)} rows -> {out_path}")
    for row in rows[:8]:
        logger.info(f"  [{row['wer']:.2f}] truth: {row['truth']}")
        logger.info(f"          pred : {row['pred']}")
    return mean_wer, exact
