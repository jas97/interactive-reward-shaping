import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:

    def __init__(self, capacity, time_window):
        self.capacity = capacity
        self.time_window = time_window

    def initialize(self, dataset):
        self.dataset = dataset

        # measures how many times a trajectory was added
        self.marked = np.zeros((len(self.dataset), ))

    def update(self, new_data, signal, important_features, datatype, actions):
        print('Updating reward buffer...')

        full_dataset = torch.cat([self.dataset.tensors[0], new_data.tensors[0]])
        curr_dataset = self.dataset

        y = torch.cat([curr_dataset.tensors[1], new_data.tensors[1]])
        y = [signal if self.similar_to_data(new_data.tensors[0], full_dataset[i], important_features, datatype, actions) else np.sign(l) for i, l in enumerate(y)]
        y = torch.tensor(y)

        threshold = 0.05
        closest = [self.closest(n, self.dataset.tensors[0], important_features) for n in new_data.tensors[0]]
        new_marked = [max(self.marked[closest[i][0]]) + 1 if closest[i][1] < threshold else 1 for i, n in enumerate(new_data.tensors[0])]
        new_marked = torch.tensor(new_marked)

        self.marked = [m + 1 if self.similar_to_data(new_data.tensors[0], self.dataset.tensors[0][i], important_features, datatype, actions) else m for i, m in enumerate(self.marked)]
        self.marked = torch.tensor(self.marked)
        self.marked = torch.cat([self.marked, new_marked])

        y = self.marked * y

        self.dataset = TensorDataset(full_dataset, y)

    def similar_to_data(self, data, x, important_features, datatype, actions, threshold=0.05):
        state_dtype, action_dtype = datatype
        if state_dtype == 'int' or (action_dtype == 'int' and actions):
            im_feature_vals = x[important_features]
            exists = torch.where((data[:, important_features] == im_feature_vals).all())
            return len(exists[0]) > 0
        elif state_dtype == 'cont' or (action_dtype == 'cont' and actions):
            mean_features = torch.mean(data, axis=0)
            similarity = abs(mean_features[important_features] - x[important_features])

            return (similarity < threshold).all().item()

    def closest(self, x, data, important_features):
        difference = torch.mean(abs(data[:, important_features] - x[important_features]), axis=1)
        min_diff = torch.min(difference)

        min_indices = torch.where(difference == min_diff)[0]

        return min_indices.tolist(), min_diff

    def get_data_loader(self,):
        return DataLoader(self.dataset, batch_size=256, shuffle=True)

    def get_dataset(self):
        print('Unique values in labels = {}'.format(torch.unique(self.dataset.tensors[1], return_counts=True)))
        return self.dataset