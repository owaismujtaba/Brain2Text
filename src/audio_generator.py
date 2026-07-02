import os
import sys
import functools

if not hasattr(functools, "cache"):
    functools.cache = functools.lru_cache(maxsize=None)

import numpy as np
import h5py
import librosa
from pathlib import Path
from tqdm import tqdm
import torch
import multiprocessing
from styletts2 import tts
import warnings
import logging

try:
    from utils import create_logger
except ImportError:
    from src.utils import create_logger


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logger = create_logger("audio_generator")

_worker_tts = None

def init_tts_worker(use_gpu, threads_per_worker=1):
    global _worker_tts
    if not hasattr(functools, "cache"):
        functools.cache = functools.lru_cache(maxsize=None)
    torch.set_num_threads(threads_per_worker)
    gpu = use_gpu and torch.cuda.is_available()
    logger.info(f"Worker process {os.getpid()} initializing StyleTTS2 (GPU: {gpu})...")
    _worker_tts = tts.StyleTTS2()

def generate_trial_audio(task_args):
    trial_name, text = task_args
    logger.info(f"Generating audio for trial '{trial_name}' with text: '{text}'")
    try:
        audio_data = _worker_tts.inference(
                text=text,
                diffusion_steps=5,
                embedding_scale=1.0,
                output_wav_file=None
        )
        audio_16k = librosa.resample(audio_data, orig_sr=24000, target_sr=16000)
    except Exception as e:
        logger.warning(f"TTS failed for trial '{trial_name}' (text: '{text}'): {e}. Skipping this trial.")
        return trial_name, text, None
    logger.info(f"Successfully generated audio for trial '{trial_name}' (16kHz samples: {len(audio_16k)})")
    return trial_name, text, audio_16k

class AudioGeneratorWorker:
    def __init__(self, use_gpu=True, num_workers=4, threads_per_worker=1):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.num_workers = num_workers
        self.threads_per_worker = threads_per_worker

    def _get_text_transcription(self, trial):
        if 'transcription' not in trial:
           return None
                        
        transcription = trial['transcription'][()]
                        
        if isinstance(transcription, bytes):
            text = transcription.decode('utf-8')
        elif isinstance(transcription, np.ndarray):
            text = ''.join([chr(c) for c in transcription if c != 0])
        else:
            text = str(transcription)
        
        text = text.strip()
        if not text:
            return None
        
        return text

    def generate_from_hdf5(self, data_dir, output_dir, split="train"):
        search_path = os.path.join(data_dir, split) if os.path.exists(os.path.join(data_dir, split)) else data_dir
        pattern = f"data_{split}.hdf5"
        hdf5_files = sorted(Path(search_path).rglob(pattern))

        if not hdf5_files:
            hdf5_files = sorted(Path(search_path).rglob("*.hdf5"))

        logger.info(f"Starting Audio Generator with {self.num_workers} workers "
                    f"({self.threads_per_worker} thread(s) each)...")

        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(
            processes=self.num_workers,
            initializer=init_tts_worker,
            initargs=(self.use_gpu, self.threads_per_worker),
        )
        try:
            for hdf5_file in hdf5_files:
                self._process_subject(hdf5_file, output_dir, split, pool)
        finally:
            pool.close()
            pool.join()

    def _process_subject(self, hdf5_file, output_dir, split, pool):
        subject_name = hdf5_file.parent.name
        logger.info(f"Processing subject '{subject_name}' from HDF5 file: {hdf5_file}")
        subject_output_dir = os.path.join(output_dir, split, subject_name)
        os.makedirs(subject_output_dir, exist_ok=True)

        output_hdf5_file = os.path.join(subject_output_dir, f"audio_{split}.hdf5")

        if os.path.exists(output_hdf5_file):
            try:
                with h5py.File(output_hdf5_file, 'r') as f_test:
                    list(f_test.keys())
            except Exception as e:
                logger.warning(f"Output file {output_hdf5_file} appears corrupted/truncated ({e}). Deleting and recreating it.")
                os.remove(output_hdf5_file)

        # Collect the trials that still need audio (resumable: skip those already written).
        with h5py.File(hdf5_file, 'r') as f_in:
            existing = set()
            if os.path.exists(output_hdf5_file):
                with h5py.File(output_hdf5_file, 'r') as f_out:
                    existing = set(f_out.keys())

            pending = []
            for trial_name in f_in.keys():
                if trial_name in existing:
                    continue
                text = self._get_text_transcription(f_in[trial_name])
                if text is None:
                    logger.warning(f"Trial '{trial_name}' has no valid transcription. Skipping.")
                    continue
                pending.append((trial_name, text))

        if not pending:
            logger.info(f"No new trials to generate for subject '{subject_name}'.")
            return

        total = len(pending)
        logger.info(f"Dispatching {total} trial(s) for '{subject_name}' across the worker pool.")

        done = 0
        failed = 0
        with h5py.File(output_hdf5_file, 'a') as f_out:
            for trial_name, text, audio_16k in pool.imap_unordered(generate_trial_audio, pending):
                done += 1
                if audio_16k is None:
                    failed += 1
                    logger.warning(f"[{subject_name}] {done}/{total} skipped (TTS failed): '{trial_name}'")
                    continue
                logger.info(f"[{subject_name}] {done}/{total} received: '{trial_name}'")
                if trial_name in f_out:
                    del f_out[trial_name]
                dset = f_out.create_dataset(trial_name, data=audio_16k, dtype='float32')
                dset.attrs['sentence_label'] = text
                f_out.flush()

        if failed:
            logger.warning(f"Subject '{subject_name}': {failed}/{total} trial(s) failed TTS and were skipped.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parallel synthetic audio array generator from HDF5 transcriptions")
    parser.add_argument("--data_dir", type=str, default="data/raw/hdf5_data_final", help="Path to input HDF5 directory")
    parser.add_argument("--output_dir", type=str, default="data/generated_audio", help="Path to output directory")
    parser.add_argument("--split", type=str, default="train", help="Split to process ('train' or 'val')")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--threads_per_worker", type=int, default=1, help="torch intra-op threads per worker")

    args = parser.parse_args()

    generator = AudioGeneratorWorker(
        num_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
    )
    generator.generate_from_hdf5(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split
    )