import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.feedback.feedback_processing import FeedbackTypes


class ReplayBuffer:

    def __init__(self, capacity, time_window):
        self.capacity = capacity
        self.time_window = time_window

        self.lmbda = 0.1

    def initialize(self, dataset, action_dataset):
        self.state_diff = dataset
        self.actions = action_dataset
        self.feature = dataset

        # measures how many times a trajectory was added
        self.marked = np.zeros((len(self.state_diff), ))

    def update(self, new_data, signal, important_features, datatype, feedback_type=FeedbackTypes.STATE_DIFF):
        print('Updating reward buffer...')

        if feedback_type == FeedbackTypes.STATE_DIFF:
            full_dataset = torch.cat([self.state_diff.tensors[0], new_data.tensors[0]])
            curr_dataset = self.state_diff
        elif feedback_type == FeedbackTypes.ACTIONS:
            full_dataset = torch.cat([self.actions.tensors[0], new_data.tensors[0]])
            curr_dataset = self.actions
        elif feedback_type == FeedbackTypes.FEATURE:
            full_dataset = torch.cat([self.feature.tensors[0], new_data.tensors[0]])
            curr_dataset = self.feature

        y = torch.cat([curr_dataset.tensors[1], new_data.tensors[1]])
        y = [signal if self.similar_to_data(new_data.tensors[0], full_dataset[i], important_features, datatype, feedback_type) else l for i, l in enumerate(y)]
        y = torch.tensor(y)

        threshold = 0.05
        closest = [self.closest(n, self.state_diff.tensors[0], important_features) for n in new_data.tensors[0]]
        new_marked = [max(self.marked[closest[i][0]]) + 1 if closest[i][1] < threshold else 0 for i, n in enumerate(new_data.tensors[0])]
        new_marked = torch.tensor(new_marked)
        self.marked = [m + 1 if self.similar_to_data(new_data.tensors[0], self.state_diff.tensors[0][i], important_features, datatype, feedback_type) else m for i, m in enumerate(self.marked)]
        self.marked = torch.tensor(self.marked)
        self.marked = torch.cat([self.marked, new_marked])

        y = (self.marked * self.lmbda) * y

        if feedback_type == FeedbackTypes.STATE_DIFF:
            self.state_diff = TensorDataset(full_dataset, y)
        elif feedback_type == FeedbackTypes.ACTIONS:
            self.actions = TensorDataset(full_dataset, y)
        elif feedback_type == FeedbackTypes.FEATURE:
            self.feature = TensorDataset(full_dataset, y)

    def similar_to_data(self, data, x, important_features, datatype, feedback_type, threshold=0.05):
        # TODO: different options than mse of important features
        if feedback_type == FeedbackTypes.STATE_DIFF:
            if datatype == 'int':
                im_feature_vals = x[important_features]
                exists = torch.where((data[:, important_features] == im_feature_vals).all())
                return len(exists[0]) > 0
            else:
                mean_features = torch.mean(data, axis=0)
                similarity = abs(mean_features[important_features] - x[important_features])

                return similarity < threshold
        elif feedback_type == FeedbackTypes.ACTIONS:
            return 0

    def closest(self, x, data, important_features):
        difference = torch.mean(abs(data[:, important_features] - x[important_features]), axis=1)
        min_diff = torch.min(difference)

        min_indices = torch.where(difference == min_diff)[0]

        return min_indices.tolist(), min_diff


    def get_data_loader(self, feedback_type=FeedbackTypes.STATE_DIFF):
        if feedback_type == FeedbackTypes.STATE_DIFF:
            return DataLoader(self.state_diff, batch_size=256, shuffle=True)
        elif feedback_type == FeedbackTypes.ACTIONS:
            return DataLoader(self.actions, batch_size=256, shuffle=True)
        elif feedback_type == FeedbackTypes.FEATURE:
            return DataLoader(self.feature, batch_size=256, shuffle=True)

    def get_dataset(self, feedback_type=FeedbackTypes.STATE_DIFF):
        if feedback_type == FeedbackTypes.STATE_DIFF:
            return self.state_diff
        elif feedback_type == FeedbackTypes.ACTIONS:
            return self.actions
        elif feedback_type == FeedbackTypes.ACTIONS:
            return self.feature