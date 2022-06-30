import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn

from src.feedback.feedback_processing import FeedbackTypes
from src.reward_modelling.replay_buffer import ReplayBuffer
from src.reward_modelling.reward_net import RewardNet
import torch.optim as optim

class RewardModel:

    def __init__(self, time_window):
        self.time_window = time_window

        self.buffer = ReplayBuffer(capacity=10000, time_window=self.time_window)

        self.state_diff_predictor = RandomForestRegressor(n_estimators=10, random_state=0)
        self.actions_predictor = RandomForestRegressor(n_estimators=10, random_state=0)
        self.features_predictor = RandomForestRegressor(n_estimators=10, random_state=0)

    def update(self, feedback_type=FeedbackTypes.STATE_DIFF):
        dataset = self.buffer.get_dataset(feedback_type)
        X = np.array(dataset.tensors[0])
        y = np.array(dataset.tensors[1])

        if feedback_type == FeedbackTypes.STATE_DIFF:
            regressor = self.state_diff_predictor
        elif feedback_type == FeedbackTypes.ACTIONS:
            regressor = self.actions_predictor
        elif feedback_type == FeedbackTypes.FEATURE:
            regressor = self.features_predictor

        regressor.fit(X, y)

        pred = regressor.predict(X)
        mse = np.mean((y - pred) ** 2)

        print('Trained with random forest. Mean squared error: {}'.format(mse))

    def update_buffer(self, D, feedback_type):
        self.buffer.update(D, feedback_type)

    def predict(self, encoding, feedback_type=FeedbackTypes.STATE_DIFF):
        if feedback_type == FeedbackTypes.STATE_DIFF:
            predictor = self.state_diff_predictor
        elif feedback_type == FeedbackTypes.ACTIONS:
            predictor = self.actions_predictor
        elif feedback_type == FeedbackTypes.FEATURE:
            predictor = self.features_predictor

        encoding = np.array(encoding).reshape(1, -1)
        return predictor.predict(encoding)




