"""Stage 2: turn the synthesised audio into Whisper encoder embeddings.

For each audio clip we compute a log-mel spectrogram and run it through the
frozen Whisper encoder. The resulting embedding is the *training target*: the
model in ``model.py`` learns to reproduce it from neural activity alone.

Each Whisper model (tiny.en, base.en, small.en, ...) produces its own embeddings,
so features are stored per model:

    features/<model>/<split>/<session>/whisper_features_<split>.hdf5

storing, per trial, the ``encoder_embedding``, the ``audio_length`` (samples),
and the ``transcription``. Extraction is resumable and parallelised across
processes, and one run covers every model listed in the config.
"""
import multiprocessing as mp
import os
from pathlib import Path

import h5py
import torch
import whisper
from tqdm import tqdm

from ..utils.logging_utils import get_logger

logger = get_logger("features")

_worker_model = None      # per-worker Whisper model, created in _init_worker


def _init_worker(model_name="tiny.en", threads_per_worker=1):
    global _worker_model
    torch.set_num_threads(threads_per_worker)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"worker {os.getpid()} initialising Whisper '{model_name}' ({device})")
    _worker_model = whisper.load_model(model_name, device=device)


def _extract_session(task):
    """Worker task: extract every missing trial's embedding for one session."""
    session, audio_file, out_file = task
    out_file = Path(out_file)

    written = skipped = 0
    with h5py.File(audio_file, "r") as fin, h5py.File(out_file, "a") as fout:
        for trial in fin.keys():
            if trial in fout:
                continue
            label = fin[trial].attrs.get("sentence_label", "")
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            if not str(label).strip():
                skipped += 1
                continue

            audio = torch.from_numpy(fin[trial][()]).float()
            mel = whisper.log_mel_spectrogram(
                whisper.pad_or_trim(audio)).unsqueeze(0).to(_worker_model.device)
            with torch.no_grad():
                embedding = _worker_model.encoder(mel).squeeze(0).cpu().numpy()

            group = fout.create_group(trial)
            group.create_dataset("encoder_embedding", data=embedding)
            group.create_dataset("audio_length", data=len(audio))
            group.create_dataset("transcription", data=label)
            fout.flush()
            written += 1

    logger.info(f"[{session}] {written} written, {skipped} skipped")
    return session, written, skipped


def _session_tasks(audio_dir, model_output_dir, split):
    """Build one ``(session, audio_file, out_file)`` task per session in a split."""
    split_dir = Path(audio_dir) / split
    if not split_dir.exists():
        logger.info(f"no audio for split '{split}', skipping")
        return []

    tasks = []
    for session_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
        session = session_dir.name
        out_file = Path(model_output_dir) / split / session / f"whisper_features_{split}.hdf5"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tasks.append((session, session_dir / f"audio_{split}.hdf5", out_file))
    return tasks


def _extract_for_model(cfg, model_name, ctx):
    """Extract every split's embeddings for a single Whisper model."""
    model_output_dir = os.path.join(cfg.output_dir, model_name)
    logger.info("-" * 60)
    logger.info(f"model '{model_name}' -> {model_output_dir}")
    for split in ("train", "val"):
        tasks = _session_tasks(cfg.audio_dir, model_output_dir, split)
        if not tasks:
            continue
        with ctx.Pool(processes=cfg.num_workers, initializer=_init_worker,
                      initargs=(model_name, cfg.threads_per_worker)) as pool:
            for _ in tqdm(pool.imap_unordered(_extract_session, tasks),
                          total=len(tasks), desc=f"{model_name} {split}"):
                pass


def extract_features(cfg):
    """Stage entry point: extract Whisper embeddings for every configured model."""
    models = getattr(cfg, "models", ["tiny.en"])
    logger.info(f"feature extraction | models={models} | {cfg.num_workers} workers")
    ctx = mp.get_context("spawn")
    for model_name in models:
        _extract_for_model(cfg, model_name, ctx)
