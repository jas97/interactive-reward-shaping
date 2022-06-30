import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.feedback.feedback_processing import FeedbackTypes


class ReplayBuffer:

    def __init__(self, capacity, time_window):
        self.capacity = capacity
        self.time_window = time_window

    def initialize(self, dataset):
        self.state_diff = dataset
        self.actions = dataset
        self.feature = dataset

    def update(self, new_data, feedback_type=FeedbackTypes.STATE_DIFF):
        print('Updating reward network...')

        if feedback_type == FeedbackTypes.STATE_DIFF:
            full_dataset = torch.cat([self.state_diff.tensors[0], new_data.tensors[0]])
        elif feedback_type == FeedbackTypes.ACTIONS:
            full_dataset = torch.cat([self.actions.tensors[0], new_data.tensors[0]])
        elif feedback_type == FeedbackTypes.FEATURE:
            full_dataset = torch.cat([self.feature.tensors[0], new_data.tensors[0]])

        unique_dataset = torch.unique(full_dataset, dim=0)

        y = [-1 if len(torch.where((x == new_data.tensors[0]).all(dim=1))[0]) > 0 else 0 for x in unique_dataset]
        y = torch.tensor(np.array(y))

        if feedback_type == FeedbackTypes.STATE_DIFF:
            self.state_diff = TensorDataset(unique_dataset, y)
        elif feedback_type == FeedbackTypes.ACTIONS:
            self.actions = TensorDataset(unique_dataset, y)
        elif feedback_type == FeedbackTypes.FEATURE:
            self.feature = TensorDataset(unique_dataset, y)

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
        elif feedback_type == FeedbackTypes.STATE_DIFF:
            return self.actions
        elif feedback_type == FeedbackTypes.STATE_DIFF:
            return self.feature