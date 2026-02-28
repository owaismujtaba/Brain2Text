import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import numpy as np
import pdb

class ProcessedDataset(Dataset):
    # ... (Your existing Dataset code remains the same)
    def __init__(self, config, kind='train', transform=None):
        self.config = config
        self.root_dir = Path(os.getcwd(), self.config['paths']['processed_dir']) / kind
        self.transform = transform
        self.files = sorted([os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith(".pt")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        sample = {
            "audio_encoder_emb": data["audio_encoder_emb"], # Shape: [Seq_Len, Dim]
            "n_actual_frames": data["n_actual_frames"],
            "text": data["text"],
            "eeg": data["eeg"], # Shape: [Seq_Len, Dim]
        }
        if self.transform:
            sample = self.transform(sample)
        return sample

def collate_fn(batch):
    """
    Groups samples into a batch, padding sequences to the max length in the batch.
    """
    # 1. Handle Tensors that need padding (EEG and Audio)
    # We use batch_first=True to get [Batch, Seq_Len, Dim]
    audio_embs = [item['audio_encoder_emb'] for item in batch]
    eeg_signals = [item['eeg'] for item in batch]
    eeg_lengths = [eeg.shape[0] for eeg in eeg_signals]  # Get original lengths for potential masking
    eeg_padded = pad_sequence(eeg_signals, batch_first=True, padding_value=0.0)

    # 2. Handle simple scalars or pre-calculated lengths
    # These can be wrapped in a standard torch.tensor
    n_frames = torch.tensor([item['n_actual_frames'] for item in batch])

    # 3. Handle non-tensor data (like raw text strings)
    texts = [item['text'] for item in batch]
    return {
        "audio_encoder_emb": torch.stack(audio_embs).squeeze(1),
        "eeg": torch.tensor(eeg_padded),
        "eeg_lengths": torch.tensor(eeg_lengths),
        "n_actual_frames": torch.tensor(n_frames),
        "text": texts
    }

def get_dataloader(config, kind='train'):
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    
    dataset = ProcessedDataset(config, kind=kind)
    
    # Pass the collate_fn here
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(kind == 'train'), 
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return dataloader

if __name__ == "__main__":
    from utils import load_config
    config = load_config("config.yaml")
    train_loader = get_dataloader(config, kind='train')
    val_loader = get_dataloader(config, kind='val')

    # Example: Iterate through the training dataloader
    for batch in train_loader:
        print(batch['audio_encoder_emb'].shape)  # Should be [Batch, Seq_Len, Dim]
        print(batch['eeg'].shape)  # Should be [Batch, Seq_Len, Dim]
        print(batch['n_actual_frames'].shape)  # Should be [Batch]
        print(batch['text'])  # List of raw text strings
        break  # Just to check one batch