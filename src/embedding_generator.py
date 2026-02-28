import os
import torch
import torchaudio
from pathlib import Path
import logging
from typing import Dict, Any
from utils import load_config
import whisper
import pdb
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/embedding_processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("EmbeddingGenerator")


class EmbeddingGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sr = 16000
        self.data_dir = Path(os.getcwd(), "data")
        self._init_whisper_model()

    def _init_whisper_model(self):
        model_name = self.config["embedding"]["model"]
        logger.info(f"Loading Whisper model: {model_name}")

        self.model = whisper.load_model(model_name, device=self.device)
        self.model.eval()

    def _get_audio_files(self, kind: str):
        audio_dir = self.data_dir / "audio" / kind
        return sorted([p for p in audio_dir.glob("*") if p.is_file()])

    def _load_and_resample(self, audio_path: Path):
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.target_sr
            )
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def _get_actual_frame_len(self, waveform):
        n_fft = 400
        hop_length = 160
        n_samples = len(waveform)
        n_frames = 1 + (n_samples - n_fft) // hop_length

        return n_frames
    
    @torch.no_grad()
    def _extract_embedding(self, audio_path: Path):
        flag = True
        waveform = self._load_and_resample(audio_path)
        if len(waveform)/16000>7 :
            flag = False
            embedding, n_actual_frames = [], []
            print('*'*50)
            print(audio_path)
        else:
            n_actual_frames = self._get_actual_frame_len(waveform)
            waveform = whisper.pad_or_trim(waveform.numpy())
            mel = whisper.log_mel_spectrogram(waveform).to(self.device)
            embedding = self.model.encoder(mel.unsqueeze(0))

        return embedding, n_actual_frames, flag

    def _get_text(self, filepath):
        with open(filepath, 'r') as f:
            text = f.readlines()

        return text[0]

    def generate_embeddings(self, kind: str = "train"):
        audio_dir = self.data_dir / "audio" / kind
        eeg_dir = self.data_dir / "eeg" / kind
        text_dir = self.data_dir / "text" / kind
        
        audio_files = self._get_audio_files(kind)
    
        logger.info(f"Processing {len(audio_files)} files in {kind}")
        
        with tqdm(audio_files, unit="file") as pbar:
            for audio_path in pbar:
                pbar.set_description(f"{audio_path.stem}")  # shows file basename (no extension)
    
                eeg_path = eeg_dir / audio_path.with_suffix(".pt").name
                txt_path = text_dir / audio_path.with_suffix(".txt").name
    
                if eeg_path.exists() and audio_path.exists():
                    audio_encoder_emb, n_actual_frames, flag = self._extract_embedding(audio_path)
                    if flag:
                        out_dir = self.data_dir / 'processed'/ kind
                        os.makedirs(out_dir, exist_ok=True)
                        filepath = Path(out_dir, eeg_path.name)
                        if os.path.exists(filepath):
                            print('file exists')
                            continue
                        text = self._get_text(txt_path)
                        eeg = torch.load(eeg_path)
                        self._save_data(
                            audio_encoder_emb.cpu(),  n_actual_frames,
                            text, eeg, eeg_path, kind
                        )

    def _save_data(self, audio_encoder_emb, n_actual_frames, text, eeg, eeg_path, kind):
        filename = eeg_path.name
        out_dir = self.data_dir / 'processed'/ kind
        os.makedirs(out_dir, exist_ok=True)
        filepath = Path(out_dir, filename)

        data = {
            "audio_encoder_emb": audio_encoder_emb.cpu(),
            "n_actual_frames": int(n_actual_frames),
            "text": text,
            "eeg": eeg.cpu()
        }

        torch.save(data, filepath) 
               
               
           


def main():
    try:
        config = load_config("config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml not found.")
        return

    processor = EmbeddingGenerator(config)
    processor.generate_embeddings("train")
    processor.generate_embeddings("val")


if __name__ == "__main__":
    main()