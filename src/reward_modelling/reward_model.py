import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from torch import nn

from src.reward_modelling.replay_buffer import ReplayBuffer
from src.reward_modelling.reward_net import RewardNet
import torch.optim as optim

class RewardModel:

    def __init__(self, time_window):
        self.time_window = time_window

        self.net = RewardNet()
        self.buffer = ReplayBuffer(capacity=10000, time_window=self.time_window)

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001)
        self.criterion = nn.L1Loss()

    def update(self, D):
        # update the buffer
        self.buffer.update(D)
        dataloader = self.buffer.get_data_loader()

        X = np.array(self.buffer.dataset.tensors[0])
        y = np.array(self.buffer.dataset.tensors[1])
        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(X, y)
        self.predictor = regressor

        pred = regressor.predict(np.array(self.buffer.dataset.tensors[0]))
        mse = np.mean((y - pred) ** 2)

        print('Trained with random forest. Mean squared error: {}'.format(mse))

    def predict(self, state_enc):
        state_enc = np.array(state_enc).reshape(1, -1)
        return self.predictor.predict(state_enc)


