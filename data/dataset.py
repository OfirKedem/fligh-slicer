import numpy as np
from torch.utils.data import Dataset

from data.utils import generate_data, resize_data_list


class RoutesDataset(Dataset):
    def __init__(self, size: int,
                 sample_len: int,
                 flatten_channels: bool = False):
        data = generate_data(reps=size)
        self.data = resize_data_list(data, sample_len)
        self.flatten_channels = flatten_channels

    def __getitem__(self, idx):
        sample = self.data[idx]
        signal = np.zeros([2, len(sample['x'])])
        signal[0] = sample['x']
        signal[1] = sample['y']

        if self.flatten_channels:
            signal = signal.flatten()

        gt_idx = np.array([sample['start'], sample['end']])

        return signal.astype(np.float32), gt_idx.astype(np.float32)

    def __len__(self):
        return len(self.data)
