"""Decode predicted neural->embedding outputs into text through the frozen
Whisper decoder, with beam search + English text normalization.

Provides:
  * decode_dataset(...) -> (mean_wer, exact_frac, rows)   shared by training eval
  * decode()  CLI entry used by `python main.py decode`
"""
import os
import csv

import torch

try:
    from dataset import NeuralToEmbeddingDataset, N_CTX, EMB_DIM
    from model import build_model, load_checkpoint_weights
    from utils import create_logger
except ImportError:
    from src.dataset import NeuralToEmbeddingDataset, N_CTX, EMB_DIM
    from src.model import build_model, load_checkpoint_weights
    from src.utils import create_logger

logger = create_logger("decode")

_NORM = None


def _normalizer():
    global _NORM
    if _NORM is None:
        try:
            from whisper.normalizers import EnglishTextNormalizer
            _NORM = EnglishTextNormalizer()
        except Exception:
            _NORM = lambda s: s.lower().strip()
    return _NORM


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d[len(r)][len(h)] / max(len(r), 1)


@torch.no_grad()
def decode_dataset(model, dataset, device, beam_size=5, limit=0,
                   model_name="tiny.en", wmodel=None):
    """Decode (up to `limit`, or all) trials; return (mean_wer, exact_frac, rows)."""
    import whisper
    if wmodel is None:
        wmodel = whisper.load_model(model_name, device=device)
    options = whisper.DecodingOptions(
        language="en", without_timestamps=True,
        fp16=(str(device) == "cuda"),
        beam_size=(beam_size if beam_size and beam_size > 1 else None),
    )
    norm = _normalizer()
    model.eval()

    index = getattr(dataset, "index", None)   # [(session, trial, audio_length), ...]
    n = len(dataset) if not limit else min(limit, len(dataset))
    rows, wers, exact = [], [], 0
    for i in range(n):
        neural, emb, truth = dataset[i]
        vf = emb.shape[0]
        pred = model(neural.unsqueeze(0).to(device))[0]        # (1500, 384)
        feats = torch.zeros(N_CTX, EMB_DIM, device=device)
        feats[:vf] = pred[:vf]                                  # mask to valid frames
        res = whisper.decode(wmodel, feats.unsqueeze(0).float(), options)
        text = (res[0] if isinstance(res, list) else res).text.strip()
        nt, np_ = norm(truth), norm(text)
        w = wer(nt, np_)
        wers.append(w)
        exact += int(nt == np_)
        session, trial = (index[i][0], index[i][1]) if index is not None else ("", i)
        rows.append({"idx": i, "session": session, "trial": trial,
                     "wer": round(w, 4), "truth": truth, "pred": text})
    mean_wer = sum(wers) / max(len(wers), 1)
    return mean_wer, exact / max(len(wers), 1), rows


def write_wer_reports(rows, pred_path, session_path):
    """Write two CSVs from decode_dataset `rows`:

      * pred_path    -> session, trial, actual, predicted, wer   (per trial)
      * session_path -> session, wer, n_trials                   (mean WER per session)
    """
    from collections import defaultdict

    os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
    with open(pred_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["session", "trial", "actual", "predicted", "wer"])
        w.writeheader()
        for r in rows:
            w.writerow({"session": r.get("session", ""), "trial": r.get("trial", ""),
                        "actual": r["truth"], "predicted": r["pred"], "wer": r["wer"]})

    per_session = defaultdict(list)
    for r in rows:
        per_session[r.get("session", "")].append(r["wer"])
    os.makedirs(os.path.dirname(session_path) or ".", exist_ok=True)
    with open(session_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "wer", "n_trials"])
        for session in sorted(per_session):
            vals = per_session[session]
            w.writerow([session, round(sum(vals) / len(vals), 4), len(vals)])
    return pred_path, session_path


def decode(args):
    device = args.device
    logger.info("=" * 60)
    logger.info(f"Decoding split='{args.split}' with checkpoint {args.ckpt}")
    logger.info(f"beam_size={args.beam_size} | model={args.dec_model} | device={device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = build_model(a).to(device)   # rebuilds the exact ablation architecture from stored args
    load_checkpoint_weights(model, ckpt["model"])
    logger.info(f"loaded model (epoch {ckpt.get('epoch')}, rnn={model.rnn_type}, "
                f"aligner={model.aligner}, conv_layers={model.conv_layers})")

    norm = bool(a.get("normalize", True))   # must match how the model was trained
    stats_dir = getattr(args, "stats_dir", None) or None
    ds = NeuralToEmbeddingDataset(args.raw_dir, args.features_dir, args.split,
                                  normalize=norm, augment=False, stats_dir=stats_dir)
    logger.info(f"{args.split} trials: {len(ds)} (decoding {args.limit or len(ds)}) | normalize={norm}")

    mw, em, rows = decode_dataset(model, ds, device, beam_size=args.beam_size,
                                  limit=args.limit, model_name=args.dec_model)

    base, ext = os.path.splitext(args.out)
    session_out = f"{base}_per_session{ext or '.csv'}"
    write_wer_reports(rows, args.out, session_out)

    logger.info(f"WER {mw:.4f} | exact {em*100:.1f}% | wrote {len(rows)} decodings -> "
                f"{args.out} | per-session WER -> {session_out}")
    for r in rows[:8]:
        logger.info(f"  [{r['wer']:.2f}] truth: {r['truth']}")
        logger.info(f"          pred : {r['pred']}")
    return mw, em
