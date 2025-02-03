from torch.utils.data import Dataset
import torch


class CloudDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.inputs.shape[0] * self.inputs.shape[1]

    def __getitem__(self, idx):
        y_idx = idx // self.inputs.shape[1]
        x_idx = idx % self.inputs.shape[1]
        x = self.inputs[y_idx, x_idx, :]
        y = self.labels[y_idx, x_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


