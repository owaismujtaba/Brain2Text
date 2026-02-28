import yaml
import h5py
from pathlib import Path
import torch

def all_filepaths(directory, kind='train'):
    paths =  [str(p.resolve()) for p in Path(directory).rglob("*.hdf5") if p.is_file()]
    if kind == 'train':
        return [p for p in paths if 'train' in p]
    elif kind == 'val':
        return [p for p in paths if 'val' in p]
    elif kind == 'test':
        return [p for p in paths if 'test' in p]
    else:
        return paths

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_h5py_index_map(filepath):
    index_map = []
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        for key in keys:
            index_map.append((filepath, key))

    return index_map


def get_hdf5_data(filepath, key):
    with h5py.File(filepath, 'r') as f:
        g = f[key]
        neural_features = torch.tensor(g['input_features'][:], dtype=torch.float32)
        n_time_steps = g.attrs.get('n_time_steps', neural_features.shape[0])
        seq_class_ids = torch.tensor(g['seq_class_ids'][:], dtype=torch.long) if 'seq_class_ids' in g else None
        seq_len = g.attrs.get('seq_len', None)
        transcription = g['transcription'][:] if 'transcription' in g else None
        sentence_label = g.attrs.get('sentence_label', None)
        session = g.attrs.get('session', None)
        block_num = g.attrs.get('block_num', None)
        trial_num = g.attrs.get('trial_num', None)

        data = {
                'neural_features': neural_features,
                'n_time_steps': n_time_steps,
                'seq_class_ids': seq_class_ids,
                'seq_len': seq_len,
                'transcriptions': transcription,
                'sentence_label': sentence_label,
                'session': session,
                'block_num': block_num,
                'trial_num': trial_num
        }
        return data