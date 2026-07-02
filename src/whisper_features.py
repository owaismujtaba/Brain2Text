import os
import argparse
import torch
import h5py
import whisper
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

whisper_model = None

def init_worker(threads_per_worker=1):
    global whisper_model
    torch.set_num_threads(threads_per_worker)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Worker process {os.getpid()} initializing Whisper (device: {device})...")
    whisper_model = whisper.load_model("tiny.en", device=device)


class WhisperEEGFeatureExtractor:
    def __init__(self, audio_dir="data/generated_audio", output_dir="data/features",
                 num_workers=4, threads_per_worker=1):
        self.audio_dir = Path(audio_dir)
        self.output_dir = Path(output_dir)
        self.num_workers = num_workers
        self.threads_per_worker = threads_per_worker
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_features(self):
        self._process_split('train')
        self._process_split('val')

    def _process_split(self, split):
        print(f"Processing split: {split}")
        split_dir = self.audio_dir / split
        if not split_dir.exists():
            print(f"  No audio directory for split '{split}', skipping.")
            return
        subject_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())

        tasks = []
        for subj_dir in subject_dirs:
            subj_id = subj_dir.name
            audio_hdf5 = subj_dir / f"audio_{split}.hdf5"
            out_hdf5 = self.output_dir / split / subj_id / f"whisper_features_{split}.hdf5"
            out_hdf5.parent.mkdir(parents=True, exist_ok=True)
            tasks.append((subj_id, audio_hdf5, out_hdf5))

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=self.num_workers, initializer=init_worker,
                      initargs=(self.threads_per_worker,)) as pool:
            results = [(t[0], pool.apply_async(_worker_extract, args=(t,))) for t in tasks]
            for subj_id, r in tqdm(results, desc=f"Split {split}"):
                r.get()


def _worker_extract(args):
    subj_id, audio_hdf5, out_hdf5 = args
    global whisper_model

    out_hdf5 = Path(out_hdf5)
    if out_hdf5.exists():
        try:
            with h5py.File(out_hdf5, 'r') as f_test:
                list(f_test.keys())
        except Exception as e:
            print(f"[{subj_id}] output {out_hdf5} corrupted ({e}); recreating.")
            out_hdf5.unlink()

    written = skipped = 0
    with h5py.File(audio_hdf5, 'r') as fin, h5py.File(out_hdf5, 'a') as fout:
        for trial_name in fin.keys():
            if trial_name in fout:
                continue

            try:
                audio = fin[trial_name][()]
                sentence_label = fin[trial_name].attrs.get('sentence_label', '')
                if isinstance(sentence_label, bytes):
                    sentence_label = sentence_label.decode('utf-8')

                if not str(sentence_label).strip():
                    print(f"[{subj_id}] trial {trial_name}: missing/empty transcription, skipping.")
                    skipped += 1
                    continue

                audio_tensor = torch.from_numpy(audio).float()
                mel = whisper.log_mel_spectrogram(
                    whisper.pad_or_trim(audio_tensor)
                ).unsqueeze(0).to(whisper_model.device)

                with torch.no_grad():
                    emb = whisper_model.encoder(mel)
                embedding = emb.squeeze(0).cpu().numpy()

                g = fout.create_group(trial_name)
                g.create_dataset('encoder_embedding', data=embedding)
                g.create_dataset('audio_length', data=len(audio))
                g.create_dataset('transcription', data=sentence_label)
                fout.flush()
                written += 1

            except Exception as e:
                print(f"Error {subj_id} trial {trial_name}: {e}")
                continue

    print(f"[{subj_id}] done: {written} written, {skipped} skipped.")
    return subj_id, written, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Whisper encoder features from generated audio")
    parser.add_argument("--audio_dir", type=str, default="data/generated_audio")
    parser.add_argument("--output_dir", type=str, default="data/features")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--threads_per_worker", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    extractor = WhisperEEGFeatureExtractor(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        threads_per_worker=args.threads_per_worker,
    )
    extractor.generate_features()
