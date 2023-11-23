import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels=None, adverval=False):
        data = torch.tensor(data, dtype=torch.float32)
        data = data.reshape(-1, 20, 20)
        data = data / 255.0
        self.data = data
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long)
            if not adverval:
                self.labels = labels - 1
            else:
                self.labels = labels
        else:
            self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.data[idx]
        else:
            return self.data[idx], self.labels[idx]


class DetectionDataset(Dataset):
    def __init__(self, data, answers=None, df=None):
        data = torch.tensor(data, dtype=torch.float32)
        data = data.reshape(data.shape[0], 20, 20)
        data = data / 255.0
        self.image = data
        if answers is not None:
            self.answers = torch.tensor(answers, dtype=torch.long)
            self.image_ids = torch.tensor(df["image_id"].values)
            self.boxes = pad_sequence(
                [torch.tensor(box) for box in df["boxes"].values],
                batch_first=True,
                padding_value=-1,
            )
            self.labels = pad_sequence(
                [torch.tensor(label) for label in df["labels"].values],
                batch_first=True,
                padding_value=-1,
            )
            self.area = pad_sequence(
                [torch.tensor(area) for area in df["area"].values],
                batch_first=True,
                padding_value=-1,
            )
            self.iscrowd = pad_sequence(
                [torch.tensor(iscrowd) for iscrowd in df["iscrowd"].values],
                batch_first=True,
                padding_value=-1,
            )
        else:
            self.answers = None
            self.image_ids = None
            self.boxes = None
            self.labels = None
            self.area = None
            self.iscrowd = None

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.image[idx]
        else:
            return (
                self.image[idx],
                self.answers[idx],
                self.image_ids[idx],
                self.boxes[idx],
                self.labels[idx],
                self.area[idx],
                self.iscrowd[idx],
            )
