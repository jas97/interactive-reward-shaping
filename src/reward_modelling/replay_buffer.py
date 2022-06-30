import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:

    def __init__(self, capacity, time_window):
        self.capacity = capacity
        self.time_window = time_window

    def initialize(self, dataset):
        self.state_diff = dataset
        self.actions = dataset
        self.feature = dataset

    def update(self, new_data, feedback_type='state_diff'):
        print('Updating reward network...')

        if feedback_type == 'state_diff':
            full_dataset = torch.cat([self.state_diff.tensors[0], new_data.tensors[0]])
        elif feedback_type == 'actions':
            full_dataset = torch.cat([self.actions.tensors[0], new_data.tensors[0]])
        elif feedback_type == 'feature':
            full_dataset = torch.cat([self.feature.tensors[0], new_data.tensors[0]])

        unique_dataset = torch.unique(full_dataset, dim=0)

        y = [-1 if len(torch.where((x == new_data.tensors[0]).all(dim=1))[0]) > 0 else 0 for x in unique_dataset]
        y = torch.tensor(np.array(y))

        if feedback_type == 'state_diff':
            self.state_diff = TensorDataset(unique_dataset, y)
        elif feedback_type == 'actions':
            self.actions = TensorDataset(unique_dataset, y)
        elif feedback_type == 'feature':
            self.feature = TensorDataset(unique_dataset, y)

    def get_data_loader(self, feedback_type='state_diff'):
        if feedback_type == 'state_diff':
            return DataLoader(self.state_diff, batch_size=256, shuffle=True)
        elif feedback_type == 'actions':
            return DataLoader(self.actions, batch_size=256, shuffle=True)
        elif feedback_type == 'feature':
            return DataLoader(self.feature, batch_size=256, shuffle=True)

    def get_dataset(self, feedback_type='state_diff'):
        if feedback_type == 'state_diff':
            return self.state_diff
        elif feedback_type == 'actions':
            return self.actions
        elif feedback_type == 'feature':
            return self.feature