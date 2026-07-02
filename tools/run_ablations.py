#!/usr/bin/env python
"""Ablation study runner for the neural->Whisper-embedding model.

Launches a set of training jobs that each change ONE thing from the config.yaml
baseline, so you can attribute changes in val loss / WER to that one factor.
Everything a run produces is stored under results/ablations/:

    results/ablations/
      logs/<run>.log                     # full stdout+file log of the run
      <run>/checkpoints/{best,best_wer,last}.pt
      <run>/wer/val_predictions_epoch*.csv
      <run>/wer/val_wer_per_session_epoch*.csv

Every non-baseline hyper-parameter comes from config.yaml; each run overrides
only the keys listed below (plus its output dirs). The runner prints the full
"changed from baseline" table at startup. Runs are grouped by the axis they
probe:

    baseline    the config.yaml defaults, unchanged (reference point)
    augment     is augmentation helping, and which transform?
    temporal    the recurrent model (type / direction / depth)
    aligner     variable->fixed length alignment (attention vs interpolation)
    conv        conv front-end depth and receptive field (time subsampling)
    capacity    model width / head depth
    reg         regularisation (dropout, input normalisation)
    loss        which loss terms matter (embedding L1 / cosine / decoder CE)
    optim       optimiser settings (learning rate, weight decay, batch size)
    combo       multi-factor reference configs (change more than one knob)

Examples
--------
    # list everything without running (also prints the config table)
    python tools/run_ablations.py --dry-run

    # quick CPU smoke test of two runs
    python tools/run_ablations.py --python /raid/owais/home/env/bin/python \
        --only baseline no_aug --limit 8 --batch_size 4 --epochs 1 \
        --wer_every 1 --wer_trials 8 --device cpu

    # run a whole axis (e.g. every loss ablation)
    python tools/run_ablations.py --group loss
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = "results/ablations"
LOG_DIR = os.path.join(ROOT, "logs")

# ── the ablation grid ───────────────────────────────────────────────
# name -> {"group": <axis>, "overrides": {train-arg: value, ...}}
# An empty overrides dict means "run the config.yaml baseline unchanged".
# Booleans are passed as "true"/"false" strings (main.py parses them).
ABLATIONS = {}


def _add(name, group, **overrides):
    ABLATIONS[name] = {"group": group, "overrides": overrides}


# reference: the config defaults, nothing changed
_add("baseline", "baseline")

# ── augmentation: on/off and per-transform isolation + strength sweep ──
_add("no_aug",            "augment", augment="false")
_add("aug_noise_only",    "augment", aug_channel_drop=0.0, aug_scale=0.0, aug_time_jitter=0)
_add("aug_chandrop_only", "augment", aug_noise_std=0.0, aug_scale=0.0, aug_time_jitter=0)
_add("aug_scale_only",    "augment", aug_noise_std=0.0, aug_channel_drop=0.0, aug_time_jitter=0)
_add("aug_jitter_only",   "augment", aug_noise_std=0.0, aug_channel_drop=0.0, aug_scale=0.0)
_add("aug_light",         "augment", aug_noise_std=0.075, aug_channel_drop=0.05, aug_scale=0.05, aug_time_jitter=1)
_add("aug_strong",        "augment", aug_noise_std=0.30, aug_channel_drop=0.20, aug_scale=0.20, aug_time_jitter=4)

# ── temporal model ──
_add("rnn_lstm",       "temporal", rnn_type="lstm")
_add("rnn_none",       "temporal", rnn_type="none")
_add("unidirectional", "temporal", bidirectional="false")
_add("rnn_layers_1",   "temporal", rnn_layers=1)
_add("rnn_layers_2",   "temporal", rnn_layers=2)
_add("rnn_layers_5",   "temporal", rnn_layers=5)

# ── length aligner ──
_add("aligner_interp", "aligner", aligner="interp")
_add("attn_heads_2",   "aligner", attn_heads=2)
_add("attn_heads_8",   "aligner", attn_heads=8)

# ── conv front-end (subsampling depth) ──
_add("conv_layers_1", "conv", conv_layers=1)
_add("conv_layers_3", "conv", conv_layers=3)

# ── capacity / width ──
_add("hidden_128",    "capacity", hidden=128, conv_channels=128)
_add("hidden_512",    "capacity", hidden=512, conv_channels=512)
_add("head_layers_1", "capacity", head_layers=1)
_add("head_layers_3", "capacity", head_layers=3)

# ── regularisation ──
_add("dropout_0.1",  "reg", dropout=0.1)
_add("dropout_0.5",  "reg", dropout=0.5)
_add("no_normalize", "reg", normalize="false")

# ── loss terms ──
_add("loss_l1_only",   "loss", cos_weight=0.0)
_add("loss_cos_only",  "loss", l1_weight=0.0)
_add("no_dec_loss",    "loss", dec_loss_weight=0.0)
_add("dec_weight_0.5", "loss", dec_loss_weight=0.5)
_add("dec_weight_1.0", "loss", dec_loss_weight=1.0)
# decoder-loss composition: hard CE only vs pure soft distillation vs no ramp
_add("dec_ce_only",    "loss", dec_distill_weight=0.0)
_add("dec_distill_1.0","loss", dec_distill_weight=1.0)
_add("dec_temp_1",     "loss", dec_temperature=1.0)
_add("dec_temp_4",     "loss", dec_temperature=4.0)
_add("dec_no_ramp",    "loss", dec_ramp_epochs=0)

# ── optimisation (learning rate / weight decay / batch size) ──
_add("lr_1e-4", "optim", lr=1.0e-4)
_add("lr_6e-4", "optim", lr=6.0e-4)
_add("lr_1e-3", "optim", lr=1.0e-3)
_add("wd_0",    "optim", weight_decay=0.0)
_add("wd_1e-4", "optim", weight_decay=1.0e-4)
_add("wd_1e-2", "optim", weight_decay=1.0e-2)
_add("batch_16", "optim", batch_size=16)
_add("batch_64", "optim", batch_size=64)

# ── conv receptive field ──
_add("conv_kernel_3", "conv", conv_kernel=3)
_add("conv_kernel_7", "conv", conv_kernel=7)

# ── minimal-compute reference (multi-factor: no recurrence + no attention) ──
_add("combo_minimal", "combo", rnn_type="none", aligner="interp")

GROUPS = list(dict.fromkeys(v["group"] for v in ABLATIONS.values()))


def _num(v):
    """Format a value: keep strings, ints stay ints, floats drop trailing zeros."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


def config_table(runs):
    """Render each run's overrides vs the config baseline as an aligned table."""
    header = ["group", "run", "changed from baseline"]
    rows = []
    for name in runs:
        spec = ABLATIONS[name]
        ov = spec["overrides"]
        changed = ", ".join(f"{k}={_num(val)}" for k, val in ov.items()) or "(config defaults)"
        rows.append([spec["group"], name, changed])

    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*header), fmt.format(*["-" * w for w in widths])]
    lines += [fmt.format(*r) for r in rows]
    return "\n".join(lines)


def build_command(py, run_name, overrides, passthrough):
    run_dir = os.path.join(ROOT, run_name)
    cmd = [py, "main.py", "train",
           "--ckpt_dir", os.path.join(run_dir, "checkpoints"),
           "--wer_out_dir", os.path.join(run_dir, "wer")]
    for key, val in overrides.items():
        cmd += [f"--{key}", str(val)]
    for key, val in passthrough.items():
        if val is not None:
            cmd += [f"--{key}", str(val)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python", default=sys.executable,
                   help="interpreter to run main.py with (use the env that has openai-whisper)")
    p.add_argument("--only", nargs="+", choices=list(ABLATIONS),
                   help="run only these named ablations")
    p.add_argument("--group", nargs="+", choices=GROUPS,
                   help="run only ablations in these groups")
    p.add_argument("--config", default="config.yaml")
    # passthrough training overrides (None => use config.yaml value)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--wer_every", type=int, default=None)
    p.add_argument("--wer_trials", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print commands, do not run")
    args = p.parse_args()

    # select runs: --only wins; else --group filter; else all (baseline first)
    if args.only:
        runs = args.only
    elif args.group:
        runs = [n for n, s in ABLATIONS.items() if s["group"] in args.group]
    else:
        runs = list(ABLATIONS)

    # main.py accepts --config anywhere (parse_known_args); pass only if non-default.
    passthrough = {"epochs": args.epochs, "batch_size": args.batch_size,
                   "num_workers": args.num_workers, "limit": args.limit,
                   "device": args.device, "wer_every": args.wer_every,
                   "wer_trials": args.wer_trials, "seed": args.seed}
    if args.config != "config.yaml":
        passthrough["config"] = args.config

    os.makedirs(LOG_DIR, exist_ok=True)
    repo_root = Path(__file__).resolve().parent.parent

    print(f"ablation root: {ROOT}  |  logs: {LOG_DIR}  |  runs: {len(runs)}")
    print("\nconfigurations:")
    print(config_table(runs))
    results = {}
    for run_name in runs:
        overrides = ABLATIONS[run_name]["overrides"]
        cmd = build_command(args.python, run_name, overrides, passthrough)
        log_path = os.path.join(LOG_DIR, f"{run_name}.log")

        print("\n" + "=" * 70)
        print(f"[{run_name}] {' '.join(cmd)}")
        print(f"[{run_name}] log -> {log_path}")
        if args.dry_run:
            continue

        env = dict(os.environ, B2T_LOG_FILE=str(repo_root / log_path))
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, cwd=str(repo_root), env=env,
                                  stdout=logf, stderr=subprocess.STDOUT)
        results[run_name] = proc.returncode
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[{run_name}] {status}")

    if results:
        print("\n" + "=" * 70)
        print("summary:")
        for run_name, rc in results.items():
            print(f"  {run_name:20s} {'OK' if rc == 0 else f'FAILED ({rc})'}")
        if any(rc != 0 for rc in results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
