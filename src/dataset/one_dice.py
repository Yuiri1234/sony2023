import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        data = torch.tensor(data, dtype=torch.float32)
        data = data / 255.0
        self.data = data
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
            self.labels = labels - 1
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.data[idx]
        else:
            return self.data[idx], self.labels[idx]
