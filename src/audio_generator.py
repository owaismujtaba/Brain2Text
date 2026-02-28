import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from TTS.api import TTS

# Assuming these are defined in your local utils.py
from utils import load_h5py_index_map, get_hdf5_data, load_config, all_filepaths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("logs/eeg_audio_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AudioEEGProcessor")

class AudioEEGProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts = self._init_tts_model()
        
        # Expanded output roots to include text
        self.output_roots = {
            "audio": Path("data/audio"),
            "eeg": Path("data/eeg"),
            "text": Path("data/text")
        }

    def _init_tts_model(self):
        model_name = self.config["audio"]["tts_model"]
        logger.info(f"Initializing {model_name} on {self.device.upper()}...")
        tts_model = TTS(model_name, progress_bar=False, gpu=torch.cuda.is_available())
        return tts_model

    def _get_target_paths(self, source_filepath: str, key: str) -> Dict[str, Path]:
        split = "train" if "train" in source_filepath.lower() else "val"
        folder_name = Path(source_filepath).parent.name
        stem = f"{folder_name}_{key}"
        
        paths = {
            "audio": self.output_roots["audio"] / split / f"{stem}.wav",
            "eeg": self.output_roots["eeg"] / split / f"{stem}.pt",
            "text": self.output_roots["text"] / split / f"{stem}.txt"
        }
        
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
            
        return paths

    def _save_data(self, text: str, eeg: torch.Tensor, paths: Dict[str, Path]):
        try:
            self.tts.tts_to_file(text=text, file_path=str(paths["audio"]))
            torch.save(eeg.cpu(), paths["eeg"])
            with open(paths["text"], "w", encoding="utf-8") as f:
                f.write(text)
                
        except Exception as e:
            logger.error(f"Failed to save data for {paths['audio'].name}: {e}")

    def process_hdf5_file(self, filepath: str):
        index_map = load_h5py_index_map(filepath)
        filename = Path(filepath).name
        
        logger.info(f"Processing file: {filename} ({len(index_map)} entries)")

        for f_path, key in tqdm(index_map, desc=f"→ {filename[:20]}...", leave=False):
            data = get_hdf5_data(f_path, key)
            
            eeg_tensor = data.get('neural_features')
            sentence_text = data.get('sentence_label')
            
            if eeg_tensor is None or sentence_text is None:
                logger.warning(f"Missing data for key {key} in {f_path}")
                continue
                
            target_paths = self._get_target_paths(f_path, key)
            self._save_data(sentence_text, eeg_tensor, target_paths)

def main():
    try:
        config = load_config("config.yaml")
    except FileNotFoundError:
        logger.error("config.yaml not found. Please ensure it exists in the root.")
        return

    processor = AudioEEGProcessor(config)
    data_dir = config["paths"]["data_dir"]
    
    # Process both splits
    for kind in ['train', 'val']:
        target_files = all_filepaths(data_dir, kind=kind)
        logger.info(f"Found {len(target_files)} {kind} files. Starting batch...")
        
        for filepath in tqdm(target_files, desc=f"Overall {kind} Progress"):
            processor.process_hdf5_file(filepath)
            
    logger.info("Batch processing complete. All files synchronized.")

if __name__ == "__main__":
    main()