import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

class BrainEmbed(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        
        eeg_path = os.path.join(data_dir, split, f'{split}_eeg.pt')
        emb_path = os.path.join(data_dir, split, f'{split}_encoder_embeddings.pt')
        text_path = os.path.join(data_dir, split, f'{split}_texts.pt')
        
        print(f"Loading {split} data from {data_dir}...")
        self.eeg_data = torch.load(eeg_path, map_location='cpu')
        self.emb_data = torch.load(emb_path, map_location='cpu')
        self.text_data = torch.load(text_path, map_location='cpu')
        if len(self.eeg_data) != len(self.emb_data):
            print(f"Warning: Length mismatch in {split} set. EEG: {len(self.eeg_data)}, Emb: {len(self.emb_data)}")
            min_len = min(len(self.eeg_data), len(self.emb_data))
            self.eeg_data = self.eeg_data[:min_len]
            self.emb_data = self.emb_data[:min_len]

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_data[idx],
            'embedding': self.emb_data[idx],
            'text': self.text_data[idx]
        }

    @staticmethod
    def collate_fn(batch):
        eeg_list = [item['eeg'] for item in batch]
        emb_list = [item['embedding'] for item in batch]

        lengths = torch.tensor([len(x) for x in eeg_list], dtype=torch.long)
        eeg_padded = pad_sequence(eeg_list, batch_first=True)
        emb_stacked = torch.stack(emb_list)
        
        return {
            'eeg': eeg_padded,
            'embedding': emb_stacked,
            'lengths': lengths
        }


def get_data_loaders(config):
    print("Setting up data loaders...")
    data_dir = config['paths']['data_dir']
    
    train_dataset = BrainEmbed(data_dir, split='train')
    val_dataset = BrainEmbed(data_dir, split='val')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset']['num_workers']
    print(f"Using batch size: {batch_size}, num_workers: {num_workers}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, collate_fn=BrainEmbed.collate_fn, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, 
        collate_fn=BrainEmbed.collate_fn
    )
    
    return train_loader, val_loader