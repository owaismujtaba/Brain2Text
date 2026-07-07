"""Stage 1: synthesise speech audio from each trial's transcription (StyleTTS2).

For every trial we read its transcription from the raw HDF5, run text-to-speech,
resample to 16 kHz (what Whisper expects), and write the waveform to a mirror
HDF5 under ``generated_audio/<split>/<session>/``.

The work is spread across a pool of worker processes. Each worker loads its own
StyleTTS2 model once (``_init_worker``) and then synthesises trials on demand.
The output is resumable: trials already present in the output file are skipped.
"""
import functools
import multiprocessing
import os
import warnings
from pathlib import Path

if not hasattr(functools, "cache"):
    functools.cache = functools.lru_cache(maxsize=None)

import h5py
import librosa
import numpy as np
import torch

from ..utils.logging_utils import get_logger

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

logger = get_logger("audio")

_worker_tts = None      # per-worker StyleTTS2 model, created in _init_worker


def _init_worker(threads_per_worker=1):
    global _worker_tts
    from styletts2 import tts
    torch.set_num_threads(threads_per_worker)
    logger.info(f"worker {os.getpid()} initialising StyleTTS2 "
                f"(GPU: {torch.cuda.is_available()})")
    _worker_tts = tts.StyleTTS2()


def _synthesize(task):
    """Worker task: turn ``(trial, text)`` into ``(trial, text, audio_16k|None)``."""
    trial, text = task
    try:
        audio = _worker_tts.inference(
            text=text, diffusion_steps=5, embedding_scale=1.0, output_wav_file=None)
        audio_16k = librosa.resample(audio, orig_sr=24000, target_sr=16000)
    except Exception as e:
        logger.warning(f"TTS failed for '{trial}' (text: '{text}'): {e}")
        return trial, text, None
    return trial, text, audio_16k


def _transcription(trial_group):
    """Extract a clean transcription string, or ``None`` if there isn't one."""
    if "transcription" not in trial_group:
        return None
    raw = trial_group["transcription"][()]
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    elif isinstance(raw, np.ndarray):
        text = "".join(chr(c) for c in raw if c != 0)
    else:
        text = str(raw)
    text = text.strip()
    return text or None


def _pending_trials(raw_file, out_file):
    """List ``(trial, text)`` pairs that still need audio for one session."""
    existing = set()
    if os.path.exists(out_file):
        with h5py.File(out_file, "r") as done:
            existing = set(done.keys())

    pending = []
    with h5py.File(raw_file, "r") as raw:
        for trial in raw.keys():
            if trial in existing:
                continue
            text = _transcription(raw[trial])
            if text is None:
                logger.warning(f"trial '{trial}' has no transcription; skipping")
                continue
            pending.append((trial, text))
    return pending


def _process_session(raw_file, output_dir, split, pool):
    """Generate and store any missing audio for a single session file."""
    session = raw_file.parent.name
    session_dir = os.path.join(output_dir, split, session)
    os.makedirs(session_dir, exist_ok=True)
    out_file = os.path.join(session_dir, f"audio_{split}.hdf5")

    pending = _pending_trials(str(raw_file), out_file)
    if not pending:
        logger.info(f"[{session}] nothing to generate")
        return

    total, failed = len(pending), 0
    logger.info(f"[{session}] generating {total} trial(s)")
    with h5py.File(out_file, "a") as out:
        for done, (trial, text, audio_16k) in enumerate(
                pool.imap_unordered(_synthesize, pending), start=1):
            if audio_16k is None:
                failed += 1
                continue
            if trial in out:
                del out[trial]
            dset = out.create_dataset(trial, data=audio_16k, dtype="float32")
            dset.attrs["sentence_label"] = text
            out.flush()
            logger.info(f"[{session}] {done}/{total} '{trial}'")

    if failed:
        logger.warning(f"[{session}] {failed}/{total} trial(s) failed TTS")


def generate_audio(cfg):
    """Stage entry point: synthesise audio for every session in a split."""
    data_dir = Path(cfg.data_dir)
    split = cfg.split
    search = data_dir / split if (data_dir / split).exists() else data_dir
    raw_files = sorted(search.rglob(f"data_{split}.hdf5")) or sorted(search.rglob("*.hdf5"))

    logger.info(f"audio generation | split={split} | {len(raw_files)} session file(s) "
                f"| {cfg.num_workers} workers")

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=cfg.num_workers, initializer=_init_worker,
                    initargs=(cfg.threads_per_worker,))
    try:
        for raw_file in raw_files:
            _process_session(raw_file, cfg.output_dir, split, pool)
    finally:
        pool.close()
        pool.join()
