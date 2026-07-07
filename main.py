"""NJEM – neural-to-speech decoding pipeline.

Everything is driven by ``config.yaml``. Pick one stage (or run the whole
workflow) from the command line; all settings come from the config file.

    python main.py workflow            # run every step in config's workflow.steps
    python main.py generate_audio      # stage 1: transcription -> speech audio
    python main.py extract_features    # stage 2: audio -> Whisper embeddings
    python main.py train               # stage 3: neural -> embeddings model
    python main.py decode              # stage 4: decode a split, write WER CSV
    python main.py train my.yaml       # use a different config file
"""
import sys

from src.utils.config import load_config, resolve_device, stage_config


def run_generate_audio(cfg):
    from src.pipeline.audio import generate_audio
    generate_audio(cfg)


def run_extract_features(cfg):
    from src.pipeline.features import extract_features
    extract_features(cfg)


def run_train(cfg):
    from src.training.train import train
    cfg.device = resolve_device(cfg.device)
    train(cfg)


def run_decode(cfg):
    from src.training.decode import decode_split
    cfg.device = resolve_device(cfg.device)
    decode_split(cfg)


STAGES = {
    "generate_audio": run_generate_audio,
    "extract_features": run_extract_features,
    "train": run_train,
    "decode": run_decode,
}


def _run(stage, config):
    print(f"\n{'=' * 60}\nStage: {stage}\n{'=' * 60}")
    STAGES[stage](stage_config(config, stage))


def main():
    commands = list(STAGES) + ["workflow"]
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"usage: python main.py <{'|'.join(commands)}> [config.yaml]")
        sys.exit(1)

    command = sys.argv[1]
    config = load_config(sys.argv[2] if len(sys.argv) > 2 else "config.yaml")

    if command == "workflow":
        steps = config.get("workflow", {}).get("steps", list(STAGES))
        for step in steps:
            if step in STAGES:
                _run(step, config)
            else:
                print(f"unknown workflow step '{step}', skipping")
    else:
        _run(command, config)


if __name__ == "__main__":
    main()
