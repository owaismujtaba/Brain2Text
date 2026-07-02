"""
NJEM – Neural Joint Embedding Model
====================================
Unified entry-point driven by config.yaml.

    python main.py workflow                        # run full pipeline
    python main.py generate_audio                  # single stage
    python main.py train                           # single stage
    python main.py train --epochs 50 --device cpu  # override config values
    python main.py --config my_config.yaml train   # use a different config
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch
import yaml


# ── Helpers ─────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def merge_cli_overrides(cfg, overrides):
    """Parse --key value pairs from the remaining CLI args into *cfg*."""
    parser = argparse.ArgumentParser(add_help=False)
    for key, val in cfg.items():
        arg_type = type(val) if val is not None else str
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=lambda v: v.lower() in ("true", "1", "yes"), default=val)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=val)
    parsed, _ = parser.parse_known_args(overrides)
    cfg.update(vars(parsed))
    return cfg


# ── Stage runners ───────────────────────────────────────────────────

def run_generate_audio(cfg: dict):
    from src.audio_generator import AudioGeneratorWorker

    print(f"\n{'=' * 60}")
    print("Stage: generate_audio")
    print(f"{'=' * 60}")

    generator = AudioGeneratorWorker(
        num_workers=cfg["num_workers"],
        threads_per_worker=cfg["threads_per_worker"],
    )
    generator.generate_from_hdf5(
        data_dir=cfg["data_dir"],
        output_dir=cfg["output_dir"],
        split=cfg["split"],
    )


def run_extract_features(cfg: dict):
    from src.whisper_features import WhisperEEGFeatureExtractor

    print(f"\n{'=' * 60}")
    print("Stage: extract_features")
    print(f"{'=' * 60}")

    extractor = WhisperEEGFeatureExtractor(
        audio_dir=cfg["audio_dir"],
        output_dir=cfg["output_dir"],
        num_workers=cfg["num_workers"],
        threads_per_worker=cfg["threads_per_worker"],
    )
    extractor.generate_features()


def run_train(cfg: dict):
    from src.train_neural2emb import train

    print(f"\n{'=' * 60}")
    print("Stage: train")
    print(f"{'=' * 60}")

    cfg["device"] = resolve_device(cfg["device"])
    train(Namespace(**cfg))


def run_decode(cfg: dict):
    from src.decode import decode

    print(f"\n{'=' * 60}")
    print("Stage: decode")
    print(f"{'=' * 60}")

    cfg["device"] = resolve_device(cfg["device"])
    decode(Namespace(**cfg))


STAGES = {
    "generate_audio": run_generate_audio,
    "extract_features": run_extract_features,
    "train": run_train,
    "decode": run_decode,
}


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Neural Joint Embedding Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "command",
        choices=list(STAGES.keys()) + ["workflow"],
        help="Pipeline stage to run, or 'workflow' for the full pipeline",
    )

    args, remaining = parser.parse_known_args()
    config = load_config(args.config)

    if args.command == "workflow":
        steps = config.get("workflow", {}).get("steps", list(STAGES.keys()))
        for step in steps:
            if step not in STAGES:
                print(f"Unknown workflow step '{step}', skipping.")
                continue
            stage_cfg = dict(config.get(step, {}))
            stage_cfg = merge_cli_overrides(stage_cfg, remaining)
            STAGES[step](stage_cfg)
    else:
        stage_cfg = dict(config.get(args.command, {}))
        stage_cfg = merge_cli_overrides(stage_cfg, remaining)
        STAGES[args.command](stage_cfg)


if __name__ == "__main__":
    main()
