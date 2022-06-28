import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:

    def __init__(self, capacity, time_window):
        self.capacity = capacity
        self.time_window = time_window

    def initialize(self, dataset):
        self.dataset = dataset

    def update(self, new_data):
        print('Updating reward network...')
        full_dataset = torch.cat([self.dataset.tensors[0], new_data.tensors[0]])
        unique_dataset = torch.unique(full_dataset, dim=0)

        y = [-1 if len(torch.where((x == new_data.tensors[0]).all(dim=1))[0]) > 0 else 0 for x in unique_dataset]
        y = torch.tensor(np.array(y))

        self.dataset = TensorDataset(unique_dataset, y)

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=256, shuffle=True)