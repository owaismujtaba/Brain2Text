"""Train the ConvBiGRU to map neural activity to Whisper encoder embeddings.

The loop is deliberately plain:

    for each epoch:
        run one pass over the training data (updating weights)
        run one pass over the validation data (no updates)
        step the learning-rate schedule
        save checkpoints (best-so-far, periodic WER, and always the latest)

Two losses can be combined (see ``losses.py``):
    * an embedding regression loss (always on), and
    * an optional decoder-in-the-loop loss that scores decodability directly.
"""
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import NeuralEmbeddingDataset, collate, NeuralAugment, session_index
from ..model.model import ConvBiGRU
from ..model.losses import embedding_loss, WhisperDecoderLoss
from ..utils.config import emb_dim_for, run_name
from ..utils.logging_utils import get_logger
from .decode import decode_dataset

logger = get_logger("train")


def describe_device(device):
    """Human-readable description of the training device."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device() if device == "cuda" else int(str(device).split(":")[-1])
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA (GPU {idx}: {name}, {total:.1f} GB, {torch.cuda.device_count()} visible)"
    return f"CPU ({torch.get_num_threads()} threads)"


def run_epoch(model, loader, device, optimizer=None, decoder_loss=None, dec_weight=0.0):
    """One full pass over ``loader``; returns mean ``[total, l1, cos, dec]`` losses.

    Passing an ``optimizer`` puts the model in train mode and updates weights;
    passing ``None`` runs a no-grad evaluation pass.
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    totals = torch.zeros(4)          # total, l1, cos, dec
    seen = 0

    non_blocking = (str(device) == "cuda")
    with torch.set_grad_enabled(training):
        for neural, lengths, target, mask, texts, session_ids in tqdm(loader, leave=False):
            # lengths stays on CPU (pack_padded_sequence needs it there); the rest
            # copy asynchronously so the H2D transfer overlaps the previous step.
            neural = neural.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)
            session_ids = session_ids.to(device, non_blocking=non_blocking)

            # Only compute over the frames any item in the batch actually uses.
            valid = int(mask.sum(dim=1).max().item())
            pred = model(neural, lengths, n_out=valid, session_ids=session_ids)
            batch_mask = mask[:, :valid]
            loss, l1, cos = embedding_loss(pred, target[:, :valid], batch_mask)

            dec = torch.zeros((), device=device)
            if decoder_loss is not None and dec_weight > 0:
                dec = decoder_loss(pred, batch_mask, texts)
                loss = loss + dec_weight * dec

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            batch_size = neural.size(0)
            totals += torch.tensor([loss.item(), l1.item(), cos.item(), float(dec)]) * batch_size
            seen += batch_size

    return (totals / max(seen, 1)).tolist()


def _build_datasets(cfg, features_dir, dims, session_to_id):
    """Create the train (augmented) and val (clean) datasets."""
    augment = None
    if bool(getattr(cfg, "augment", True)):
        augment = NeuralAugment(
            noise_std=getattr(cfg, "aug_noise_std", 0.15),
            channel_drop=getattr(cfg, "aug_channel_drop", 0.1),
            scale=getattr(cfg, "aug_scale", 0.1),
            time_jitter=getattr(cfg, "aug_time_jitter", 2),
        )
    normalize = bool(getattr(cfg, "normalize", True))

    train_ds = NeuralEmbeddingDataset(
        cfg.raw_dir, features_dir, "train",
        normalize=normalize, augment=augment, session_to_id=session_to_id, **dims)
    val_ds = NeuralEmbeddingDataset(
        cfg.raw_dir, features_dir, "val",
        normalize=normalize, augment=None, session_to_id=session_to_id, **dims)

    if cfg.limit:
        train_ds.index = train_ds.index[: cfg.limit]
        val_ds.index = val_ds.index[: cfg.limit]

    logger.info(f"neural input: normalize={normalize} | augment={augment is not None}")
    logger.info(f"train trials: {len(train_ds)} | val trials: {len(val_ds)}")
    return train_ds, val_ds


def _save(model, optimizer, scheduler, cfg, epoch, best_val, best_wer, name,
          epochs_no_improve=0, session_to_id=None, **extra):
    """Write a checkpoint that records everything needed to resume or rebuild."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(cfg),
        "epoch": epoch,
        "best_val": best_val,
        "best_wer": best_wer,
        "epochs_no_improve": epochs_no_improve,
        "session_to_id": session_to_id or {},
    }
    ckpt.update(extra)
    path = os.path.join(cfg.ckpt_dir, name)
    torch.save(ckpt, path)
    return path


def _resume(model, optimizer, scheduler, cfg, device):
    """Restore the most recent checkpoint if one exists; returns training state."""
    for name in ("last.pt", "best.pt"):
        path = os.path.join(cfg.ckpt_dir, name)
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            # Architecture changed since this checkpoint (e.g. the per-session
            # input layer was added). Don't silently resume an incompatible run.
            logger.warning(f"cannot resume from {path}: {e}")
            logger.warning("checkpoint architecture differs; starting a fresh run "
                           "(move/delete the old checkpoints to silence this)")
            return 1, float("inf"), float("inf"), 0
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scheduler.T_max != cfg.epochs:
            logger.info(f"fixing scheduler T_max: {scheduler.T_max} -> {cfg.epochs}")
            scheduler.T_max = cfg.epochs
        best_val = ckpt.get("best_val", float("inf"))
        best_wer = ckpt.get("best_wer", float("inf"))
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"resumed from {path} (epoch {start_epoch - 1}, "
                    f"best val {best_val:.4f}, best wer {best_wer:.4f})")
        return start_epoch, best_val, best_wer, epochs_no_improve
    return 1, float("inf"), float("inf"), 0


def train(cfg):
    """Full training run driven by the ``train`` config section."""
    device = cfg.device

    # Which Whisper target we're training. The embedding size, the feature
    # folder, and the checkpoint folder all follow from this one choice, so the
    # three model sizes never collide and can't be mismatched.
    model_name = getattr(cfg, "model", "tiny.en")
    emb_dim = emb_dim_for(model_name)
    features_dir = os.path.join(cfg.features_dir, model_name)
    # Features only depend on the Whisper model, so they stay under <model>.
    # Checkpoints depend on the whole run, so they go under <model>/<run_name>.
    cfg.model, cfg.emb_dim = model_name, emb_dim
    # run_name already begins with the model, so the folder is flat:
    # <ckpt_dir>/<model>_<hyperparams>/
    ckpt_dir = os.path.join(cfg.ckpt_dir, run_name(cfg))
    cfg.ckpt_dir = ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Training neural -> Whisper-embedding model | target '{model_name}'")
    logger.info(f"device: {device} -> {describe_device(device)}")
    logger.info(f"emb_dim={emb_dim} | features={features_dir} | ckpts={ckpt_dir}")
    logger.info(f"config: {vars(cfg)}")

    n_ctx = int(getattr(cfg, "n_ctx", 1500))
    neural_dim = int(getattr(cfg, "neural_dim", 512))
    frame_samples = int(getattr(cfg, "frame_samples", 320))
    dims = dict(n_ctx=n_ctx, neural_dim=neural_dim, frame_samples=frame_samples)

    # Stable session -> id map (union of train+val), used by the per-session
    # input layer. Saved in every checkpoint so decoding reproduces it exactly.
    session_to_id = session_index(features_dir)
    session_adapt = getattr(cfg, "session_adapt", "linear")
    cfg.session_adapt = session_adapt
    logger.info(f"session adaptation: {session_adapt} | {len(session_to_id)} sessions")

    train_ds, val_ds = _build_datasets(cfg, features_dir, dims, session_to_id)

    collate_fn = partial(collate, n_ctx=n_ctx)
    pin = (device == "cuda")
    # Keep the GPU fed: workers persist across epochs (no re-fork / HDF5 re-open
    # at every boundary) and each stays several batches ahead via prefetch.
    loader_kwargs = dict(num_workers=cfg.num_workers, collate_fn=collate_fn,
                         pin_memory=pin)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=True,
                             prefetch_factor=getattr(cfg, "prefetch_factor", 4))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            **loader_kwargs)

    model = ConvBiGRU(
        neural_dim=neural_dim, conv_channels=getattr(cfg, "conv_channels", 256),
        hidden=cfg.hidden, gru_layers=cfg.gru_layers,
        emb_dim=emb_dim, n_ctx=n_ctx, dropout=cfg.dropout,
        n_sessions=len(session_to_id), session_adapt=session_adapt).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: ConvBiGRU | {n_params / 1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    dec_weight = float(getattr(cfg, "dec_loss_weight", 0.0))
    decoder_loss = None
    if dec_weight > 0:
        decoder_loss = WhisperDecoderLoss(model_name, device, n_ctx=n_ctx)
        logger.info(f"decoder-in-the-loop loss ENABLED | weight {dec_weight}")

    start_epoch, best_val, best_wer, epochs_no_improve = _resume(
        model, optimizer, scheduler, cfg, device)
    patience = int(getattr(cfg, "patience", 0))

    for epoch in range(start_epoch, cfg.epochs + 1):
        lr_now = scheduler.get_last_lr()[0]
        tr = run_epoch(model, train_loader, device, optimizer, decoder_loss, dec_weight)
        va = run_epoch(model, val_loader, device, None, decoder_loss, dec_weight)
        scheduler.step()

        dec_str = f" | dec {tr[3]:.4f}/{va[3]:.4f}" if dec_weight > 0 else ""
        logger.info(f"epoch {epoch:3d}/{cfg.epochs} | lr {lr_now:.2e} "
                    f"| train {tr[0]:.4f} (l1 {tr[1]:.4f} cos {tr[2]:.4f}) "
                    f"| val {va[0]:.4f} (l1 {va[1]:.4f} cos {va[2]:.4f})" + dec_str)

        if va[0] < best_val:
            best_val = va[0]
            epochs_no_improve = 0
            path = _save(model, optimizer, scheduler, cfg, epoch, best_val, best_wer, "best.pt",
                         epochs_no_improve=epochs_no_improve, session_to_id=session_to_id)
            logger.info(f"           new best val {best_val:.4f} -> {path}")
        else:
            epochs_no_improve += 1

        if cfg.wer_every and epoch % cfg.wer_every == 0:
            mean_wer, exact, rows = decode_dataset(
                model, val_ds, device, model_name=model_name,
                beam_size=getattr(cfg, "beam_size", 5),
                limit=cfg.wer_trials, n_ctx=n_ctx, emb_dim=emb_dim,
                session_to_id=session_to_id)
            logger.info(f"           WER {mean_wer:.4f} | exact {exact * 100:.1f}% "
                        f"(over {cfg.wer_trials} val trials)")
            for row in rows[:3]:
                logger.info(f"             truth: {row['truth']}")
                logger.info(f"             pred : {row['pred']}")
            if mean_wer < best_wer:
                best_wer = mean_wer
                path = _save(model, optimizer, scheduler, cfg, epoch,
                             best_val, best_wer, "best_wer.pt", wer=mean_wer,
                             session_to_id=session_to_id)
                logger.info(f"           new best WER {best_wer:.4f} -> {path}")

        _save(model, optimizer, scheduler, cfg, epoch, best_val, best_wer, "last.pt",
              epochs_no_improve=epochs_no_improve, session_to_id=session_to_id)

        if patience and epochs_no_improve >= patience:
            logger.info(f"early stopping: no val improvement for {epochs_no_improve} epochs "
                        f"(patience={patience})")
            break

    logger.info(f"done. best val {best_val:.4f} | best WER {best_wer:.4f} "
                f"| checkpoints in {cfg.ckpt_dir}/")
    return best_val
