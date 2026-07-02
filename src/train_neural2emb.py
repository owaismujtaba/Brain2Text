import os
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from dataset import NeuralToEmbeddingDataset, collate, N_CTX, EMB_DIM
    from model import ConvBiGRU, masked_embedding_loss
    from decoder_loss import WhisperDecoderLoss
    from decode import decode_dataset
    from utils import create_logger
except ImportError:
    from src.dataset import NeuralToEmbeddingDataset, collate, N_CTX, EMB_DIM
    from src.model import ConvBiGRU, masked_embedding_loss
    from src.decoder_loss import WhisperDecoderLoss
    from src.decode import decode_dataset
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


def run_epoch(model, loader, device, optimizer=None, dec_loss=None, dec_weight=0.0):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot = tot_l1 = tot_cos = tot_dec = n = 0.0

    with torch.set_grad_enabled(train):
        for neural, lengths, target, mask, txts in tqdm(loader, leave=False):
            neural, target, mask = neural.to(device), target.to(device), mask.to(device)
            max_vf = int(mask.sum(dim=1).max().item())
            pred = model(neural, lengths, n_out=max_vf)
            mvf = mask[:, :max_vf]
            loss, l1, cos = masked_embedding_loss(pred, target[:, :max_vf], mvf)

            dec = torch.zeros((), device=device)
            if dec_loss is not None and dec_weight > 0:
                dec = dec_loss(pred, mvf, txts)
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

    return tot / n, tot_l1 / n, tot_cos / n, tot_dec / n


def eval_wer(model, dataset, device, n_trials=1450, model_name="tiny.en", beam_size=5):
    """Beam-search decode with English normalization; returns (wer, exact, samples)."""
    mw, em, rows = decode_dataset(model, dataset, device, beam_size=beam_size,
                                  limit=n_trials, model_name=model_name)
    samples = [(r["truth"], r["pred"]) for r in rows[:5]]
    return mw, em, samples


def train(args):
    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = args.device

    logger.info("=" * 60)
    logger.info("Training neural -> Whisper-embedding model")
    logger.info(f"TRAINING DEVICE: {device}  ->  {describe_device(device)}")
    logger.info(f"config: {vars(args)}")

    norm = bool(getattr(args, "normalize", True))
    aug_kw = dict(
        aug_noise_std=getattr(args, "aug_noise_std", 0.15),
        aug_channel_drop=getattr(args, "aug_channel_drop", 0.1),
        aug_scale=getattr(args, "aug_scale", 0.1),
        aug_time_jitter=getattr(args, "aug_time_jitter", 2),
    )
    train_ds = NeuralToEmbeddingDataset(
        args.raw_dir, args.features_dir, "train",
        normalize=norm, augment=bool(getattr(args, "augment", True)), **aug_kw)
    val_ds = NeuralToEmbeddingDataset(
        args.raw_dir, args.features_dir, "val",
        normalize=norm, augment=False)
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

    model = ConvBiGRU(conv_channels=getattr(args, "conv_channels", 256),
                      hidden=args.hidden, gru_layers=args.gru_layers,
                      dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: ConvBiGRU | params: {n_params/1e6:.2f}M | on device: "
                f"{next(model.parameters()).device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Decoder-in-the-loop loss (optimises decodability -> WER) ──
    dec_weight = float(getattr(args, "dec_loss_weight", 0.0))
    dec_loss = None
    if dec_weight > 0:
        dec_loss = WhisperDecoderLoss(getattr(args, "dec_model", "tiny.en"), device)
        logger.info(f"decoder-in-the-loop loss ENABLED | model {getattr(args, 'dec_model', 'tiny.en')} "
                    f"| weight {dec_weight}")

    best_val = float("inf")
    best_wer = float("inf")
    start_epoch = 1

    def save_ckpt(name, epoch, **extra):
        d = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "best_val": best_val,
             "best_wer": best_wer, "args": vars(args), "epoch": epoch}
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
        model.load_state_dict(ckpt_data["model"])
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
        start_epoch = ckpt_data.get("epoch", 0) + 1
        logger.info(f"Resumed from {resume_path} (epoch {start_epoch - 1}, "
                    f"best val {best_val:.4f}, best wer {best_wer:.4f})")
    # ─────────────────────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs + 1):
        lr_now = scheduler.get_last_lr()[0]
        tr = run_epoch(model, train_loader, device, optimizer, dec_loss, dec_weight)
        va = run_epoch(model, val_loader, device, None, dec_loss, dec_weight)
        scheduler.step()
        dec_str = f" dec {tr[3]:.4f}/{va[3]:.4f}" if dec_weight > 0 else ""
        logger.info(f"epoch {epoch:3d}/{args.epochs} | lr {lr_now:.2e} "
                    f"| train loss {tr[0]:.4f} (l1 {tr[1]:.4f} cos {tr[2]:.4f}) "
                    f"| val loss {va[0]:.4f} (l1 {va[1]:.4f} cos {va[2]:.4f})" + dec_str)

        if va[0] < best_val:
            best_val = va[0]
            p = save_ckpt("best.pt", epoch)
            logger.info(f"           new best val loss {best_val:.4f} -> saved {p}")

        if args.wer_every and epoch % args.wer_every == 0:
            mw, em, samples = eval_wer(model, val_ds, device, args.wer_trials,
                                       beam_size=getattr(args, "beam_size", 5))
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

    logger.info(f"done. best val {best_val:.4f} | best WER {best_wer:.4f}. "
                f"checkpoints in {args.ckpt_dir}/ (trained on {describe_device(device)})")
    return best_val


def main():
    p = argparse.ArgumentParser(description="Train neural->Whisper-embedding model")
    p.add_argument("--raw_dir", default="data/raw/hdf5_data_final")
    p.add_argument("--features_dir", default="data/features")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--conv_channels", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--gru_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--normalize", type=lambda v: str(v).lower() in ("1", "true", "yes"), default=True,
                   help="per-session z-score of neural input")
    p.add_argument("--augment", type=lambda v: str(v).lower() in ("1", "true", "yes"), default=True,
                   help="train-only neural-input augmentation")
    p.add_argument("--aug_noise_std", type=float, default=0.15)
    p.add_argument("--aug_channel_drop", type=float, default=0.1)
    p.add_argument("--aug_scale", type=float, default=0.1)
    p.add_argument("--aug_time_jitter", type=int, default=2)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--dec_loss_weight", type=float, default=0.0,
                   help="weight for decoder-in-the-loop CE loss (0=off)")
    p.add_argument("--dec_model", default="tiny.en", help="frozen Whisper model for decoder loss")
    p.add_argument("--wer_every", type=int, default=5, help="run WER eval every N epochs (0=off)")
    p.add_argument("--wer_trials", type=int, default=30)
    p.add_argument("--beam_size", type=int, default=5, help="beam width for WER decoding (1=greedy)")
    p.add_argument("--limit", type=int, default=0, help="cap trials per split (debug)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    train(p.parse_args())


if __name__ == "__main__":
    main()
