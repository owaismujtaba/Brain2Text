import os
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int):
    """Seed Python, NumPy and Torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

try:
    from dataset import NeuralToEmbeddingDataset, collate, N_CTX, EMB_DIM
    from model import build_model, masked_embedding_loss, load_checkpoint_weights
    from decoder_loss import WhisperDecoderLoss
    from decode import decode_dataset, write_wer_reports
    from utils import create_logger
except ImportError:
    from src.dataset import NeuralToEmbeddingDataset, collate, N_CTX, EMB_DIM
    from src.model import build_model, masked_embedding_loss, load_checkpoint_weights
    from src.decoder_loss import WhisperDecoderLoss
    from src.decode import decode_dataset, write_wer_reports
    from src.utils import create_logger

logger = create_logger("train_neural2emb")


def describe_device(device):
    """Human-readable description of the training device."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device() if device == "cuda" else int(str(device).split(":")[-1])
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory / 1e9
        return f"CUDA (GPU {idx}: {name}, {total:.1f} GB, {torch.cuda.device_count()} visible)"
    return f"CPU ({torch.get_num_threads()} threads)"


def run_epoch(model, loader, device, optimizer=None, dec_loss=None, dec_weight=0.0,
              l1_weight=1.0, cos_weight=1.0):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = tot_l1 = tot_cos = tot_dec = n = 0.0

    with torch.set_grad_enabled(train):
        for neural, lengths, target, mask, txts in tqdm(loader, leave=False):
            neural, target, mask = neural.to(device), target.to(device), mask.to(device)
            out_lengths = mask.sum(dim=1)
            max_vf = int(out_lengths.max().item())
            pred = model(neural, lengths, n_out=max_vf, out_lengths=out_lengths)
            mvf = mask[:, :max_vf]
            loss, l1, cos = masked_embedding_loss(pred, target[:, :max_vf], mvf,
                                                  l1_weight=l1_weight, cos_weight=cos_weight)

            dec = torch.zeros((), device=device)
            if dec_loss is not None and dec_weight > 0:
                dec = dec_loss(pred, mvf, txts, target=target[:, :max_vf])
                loss = loss + dec_weight * dec

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = neural.size(0)
            tot += loss.item() * bs
            tot_l1 += l1.item() * bs
            tot_cos += cos.item() * bs
            tot_dec += float(dec) * bs
            n += bs

    if n == 0:   # e.g. limit < batch_size with drop_last on the train loader
        return 0.0, 0.0, 0.0, 0.0
    return tot / n, tot_l1 / n, tot_cos / n, tot_dec / n


def eval_wer(model, dataset, device, n_trials=1450, model_name="tiny.en",
             beam_size=5, wmodel=None, epoch=None, out_dir=None):
    """Beam-search decode with English normalization; returns (wer, exact, samples).

    Pass ``wmodel`` to reuse an already-loaded frozen Whisper model instead of
    reloading it from disk on every eval. When ``out_dir`` is given, write two
    CSVs tagged by ``epoch``: per-trial (session, trial, actual, predicted, wer)
    and per-session mean WER.
    """
    mw, em, rows = decode_dataset(model, dataset, device, beam_size=beam_size,
                                  limit=n_trials, model_name=model_name, wmodel=wmodel)
    if out_dir:
        tag = f"epoch{epoch:04d}" if epoch is not None else "latest"
        pred_path, session_path = write_wer_reports(
            rows,
            os.path.join(out_dir, f"val_predictions_{tag}.csv"),
            os.path.join(out_dir, f"val_wer_per_session_{tag}.csv"))
        logger.info(f"           wrote per-trial -> {pred_path} | per-session -> {session_path}")
    samples = [(r["truth"], r["pred"]) for r in rows[:5]]
    return mw, em, samples


def train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = args.device

    seed = int(getattr(args, "seed", 42))
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("Training neural -> Whisper-embedding model")
    logger.info(f"seed: {seed}")
    logger.info(f"TRAINING DEVICE: {device}  ->  {describe_device(device)}")
    logger.info(f"config: {vars(args)}")

    norm = bool(getattr(args, "normalize", True))
    aug_kw = dict(
        aug_noise_std=getattr(args, "aug_noise_std", 0.15),
        aug_channel_drop=getattr(args, "aug_channel_drop", 0.1),
        aug_scale=getattr(args, "aug_scale", 0.1),
        aug_time_jitter=getattr(args, "aug_time_jitter", 2),
    )
    stats_dir = getattr(args, "stats_dir", None) or None
    train_ds = NeuralToEmbeddingDataset(
        args.raw_dir, args.features_dir, "train",
        normalize=norm, augment=bool(getattr(args, "augment", True)),
        stats_dir=stats_dir, **aug_kw)
    val_ds = NeuralToEmbeddingDataset(
        args.raw_dir, args.features_dir, "val",
        normalize=norm, augment=False, stats_dir=stats_dir)
    logger.info(f"neural input: normalize={norm} | train augment="
                f"{bool(getattr(args, 'augment', True))} ({aug_kw})")
    if args.limit:
        train_ds.index = train_ds.index[: args.limit]
        val_ds.index = val_ds.index[: args.limit]
    logger.info(f"train trials: {len(train_ds)} | val trials: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate,
                              drop_last=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate,
                            pin_memory=(device == "cuda"))

    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"model: rnn={model.rnn_type} bidir={getattr(args, 'bidirectional', True)} "
        f"layers={getattr(args, 'rnn_layers', getattr(args, 'gru_layers', 2))} "
        f"conv_layers={model.conv_layers} aligner={model.aligner} "
        f"attn_heads={getattr(args, 'attn_heads', 4)} head_layers={getattr(args, 'head_layers', 2)} "
        f"| params: {n_params/1e6:.2f}M | on device: {next(model.parameters()).device}")
    l1_weight = float(getattr(args, "l1_weight", 1.0))
    cos_weight = float(getattr(args, "cos_weight", 1.0))
    logger.info(f"loss weights: l1={l1_weight} cos={cos_weight} dec={getattr(args, 'dec_loss_weight', 0.0)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Decoder-in-the-loop loss (optimises decodability -> WER) ──
    dec_weight = float(getattr(args, "dec_loss_weight", 0.0))
    dec_distill = float(getattr(args, "dec_distill_weight", 0.0))   # 0=pure CE, 1=pure KL
    dec_temp = float(getattr(args, "dec_temperature", 2.0))
    dec_ramp_epochs = int(getattr(args, "dec_ramp_epochs", 0))      # 0 = no ramp (full weight)
    dec_loss = None
    if dec_weight > 0:
        dec_loss = WhisperDecoderLoss(getattr(args, "dec_model", "tiny.en"), device,
                                      distill_weight=dec_distill, temperature=dec_temp)
        logger.info(f"decoder-in-the-loop loss ENABLED | model {getattr(args, 'dec_model', 'tiny.en')} "
                    f"| weight {dec_weight} | distill {dec_distill} (T={dec_temp}) "
                    f"| ramp over {dec_ramp_epochs or 0} epochs")

    def dec_weight_at(epoch):
        """Linearly ramp the decoder-loss weight 0 -> dec_weight over the first
        dec_ramp_epochs (regression warms up first, decoder objective takes over)."""
        if dec_ramp_epochs and dec_ramp_epochs > 0:
            return dec_weight * min(1.0, epoch / dec_ramp_epochs)
        return dec_weight

    best_val = float("inf")
    best_wer = float("inf")
    start_epoch = 1
    epochs_no_improve = 0
    patience = int(getattr(args, "early_stop_patience", 10))   # 0 disables early stopping

    def save_ckpt(name, epoch, **extra):
        d = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "best_val": best_val,
             "best_wer": best_wer, "no_improve": epochs_no_improve,
             "args": vars(args), "epoch": epoch}
        d.update(extra)
        path = os.path.join(args.ckpt_dir, name)
        torch.save(d, path)
        return path

    # ── Resume from checkpoint if available (prefer latest state) ─
    resume_path = None
    for cand in ("last.pt", "best.pt"):
        p = os.path.join(args.ckpt_dir, cand)
        if os.path.exists(p):
            resume_path = p
            break
    if resume_path:
        ckpt_data = torch.load(resume_path, map_location=device, weights_only=False)
        load_checkpoint_weights(model, ckpt_data["model"])
        if "optimizer" in ckpt_data:
            optimizer.load_state_dict(ckpt_data["optimizer"])
        if "scheduler" in ckpt_data:
            scheduler.load_state_dict(ckpt_data["scheduler"])
            if scheduler.T_max != args.epochs:
                logger.info(f"           fixing scheduler T_max: "
                            f"{scheduler.T_max} -> {args.epochs}")
                scheduler.T_max = args.epochs
        best_val = ckpt_data.get("best_val", best_val)
        best_wer = ckpt_data.get("best_wer", best_wer)
        epochs_no_improve = ckpt_data.get("no_improve", 0)
        start_epoch = ckpt_data.get("epoch", 0) + 1
        logger.info(f"Resumed from {resume_path} (epoch {start_epoch - 1}, "
                    f"best val {best_val:.4f}, best wer {best_wer:.4f})")
    # ─────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = scheduler.get_last_lr()[0]
        cur_dec = dec_weight_at(epoch)   # ramped weight for training this epoch
        # Val always uses the full target weight so val loss stays comparable
        # across epochs (early stopping / best-val selection depend on it).
        tr = run_epoch(model, train_loader, device, optimizer, dec_loss, cur_dec,
                       l1_weight=l1_weight, cos_weight=cos_weight)
        va = run_epoch(model, val_loader, device, None, dec_loss, dec_weight,
                       l1_weight=l1_weight, cos_weight=cos_weight)
        scheduler.step()
        dec_str = f" dec {tr[3]:.4f}/{va[3]:.4f}" if dec_weight > 0 else ""
        if dec_weight > 0 and dec_ramp_epochs and cur_dec < dec_weight:
            dec_str += f" (w={cur_dec:.3f})"
        logger.info(f"epoch {epoch:3d}/{args.epochs} | lr {lr_now:.2e} "
                    f"| train loss {tr[0]:.4f} (l1 {tr[1]:.4f} cos {tr[2]:.4f}) "
                    f"| val loss {va[0]:.4f} (l1 {va[1]:.4f} cos {va[2]:.4f})" + dec_str)

        if va[0] < best_val:
            best_val = va[0]
            epochs_no_improve = 0
            p = save_ckpt("best.pt", epoch)
            logger.info(f"           new best val loss {best_val:.4f} -> saved {p}")
        else:
            epochs_no_improve += 1

        if args.wer_every and epoch % args.wer_every == 0:
            # Reuse the frozen Whisper already held by the decoder loss (same
            # model name) to avoid reloading it from disk on every WER eval.
            wmodel = dec_loss.model if dec_loss is not None else None
            mw, em, samples = eval_wer(model, val_ds, device, args.wer_trials,
                                       model_name=getattr(args, "dec_model", "tiny.en"),
                                       beam_size=getattr(args, "beam_size", 5),
                                       wmodel=wmodel, epoch=epoch,
                                       out_dir=getattr(args, "wer_out_dir", "results/wer_eval"))
            logger.info(f"           WER {mw:.4f} | exact {em*100:.1f}% "
                        f"(over {args.wer_trials} val trials)")
            for truth, pred in samples[:3]:
                logger.info(f"             truth: {truth}")
                logger.info(f"             pred : {pred}")
            if mw < best_wer:
                best_wer = mw
                p = save_ckpt("best_wer.pt", epoch, wer=mw)
                logger.info(f"           new best WER {best_wer:.4f} -> saved {p}")

        # always keep the latest weights (survives loss-scale changes)
        save_ckpt("last.pt", epoch)

        # ── early stopping on validation loss ──
        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"early stopping: val loss has not improved for "
                        f"{epochs_no_improve} epochs (patience {patience}); "
                        f"stopping at epoch {epoch}. best val {best_val:.4f}")
            break

    logger.info(f"done. best val {best_val:.4f} | best WER {best_wer:.4f}. "
                f"checkpoints in {args.ckpt_dir}/ (trained on {describe_device(device)})")
    return best_val


def main():
    p = argparse.ArgumentParser(description="Train neural->Whisper-embedding model")
    p.add_argument("--raw_dir", default="data/raw/hdf5_data_final")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--stats_dir", default="", help="dir for the per-session "
                   "normalization cache (default: <features_dir>/session_stats)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    bool_arg = lambda v: str(v).lower() in ("1", "true", "yes")
    # ── model architecture (ablatable) ──
    p.add_argument("--conv_channels", type=int, default=256)
    p.add_argument("--conv_layers", type=int, default=2, help="stride-2 conv blocks")
    p.add_argument("--conv_kernel", type=int, default=5)
    p.add_argument("--rnn_type", default="gru", choices=["gru", "lstm", "none"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--rnn_layers", type=int, default=3, help="GRU/LSTM depth")
    p.add_argument("--gru_layers", type=int, default=None, help="alias for --rnn_layers")
    p.add_argument("--bidirectional", type=bool_arg, default=True)
    p.add_argument("--aligner", default="attn", choices=["attn", "interp"])
    p.add_argument("--attn_heads", type=int, default=4)
    p.add_argument("--head_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    # ── loss term weights (0 = ablate) ──
    p.add_argument("--l1_weight", type=float, default=1.0)
    p.add_argument("--cos_weight", type=float, default=1.0)
    p.add_argument("--normalize", type=lambda v: str(v).lower() in ("1", "true", "yes"), default=True,
                   help="per-session z-score of neural input")
    p.add_argument("--augment", type=lambda v: str(v).lower() in ("1", "true", "yes"), default=True,
                   help="train-only neural-input augmentation")
    p.add_argument("--aug_noise_std", type=float, default=0.15)
    p.add_argument("--aug_channel_drop", type=float, default=0.1)
    p.add_argument("--aug_scale", type=float, default=0.1)
    p.add_argument("--aug_time_jitter", type=int, default=2)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="stop if val loss doesn't improve for N epochs (0=off)")
    p.add_argument("--dec_loss_weight", type=float, default=0.0,
                   help="weight for decoder-in-the-loop loss (0=off)")
    p.add_argument("--dec_distill_weight", type=float, default=0.0,
                   help="blend of soft KL vs hard CE in the decoder loss (0=CE, 1=KL)")
    p.add_argument("--dec_temperature", type=float, default=2.0,
                   help="softmax temperature for distillation KL")
    p.add_argument("--dec_ramp_epochs", type=int, default=0,
                   help="ramp decoder-loss weight 0->dec_loss_weight over N epochs (0=off)")
    p.add_argument("--dec_model", default="tiny.en", help="frozen Whisper model for decoder loss")
    p.add_argument("--wer_every", type=int, default=5, help="run WER eval every N epochs (0=off)")
    p.add_argument("--wer_trials", type=int, default=30)
    p.add_argument("--wer_out_dir", default="results/wer_eval",
                   help="dir for per-validation WER CSVs (per-trial + per-session)")
    p.add_argument("--beam_size", type=int, default=5, help="beam width for WER decoding (1=greedy)")
    p.add_argument("--limit", type=int, default=0, help="cap trials per split (debug)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.gru_layers is not None:      # honor the legacy alias if explicitly set
        args.rnn_layers = args.gru_layers
    train(args)


if __name__ == "__main__":
    main()
