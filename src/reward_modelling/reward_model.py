import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.reward_modelling.replay_buffer import ReplayBuffer


class RewardModel:

    def __init__(self, time_window):
        self.time_window = time_window

        self.buffer = ReplayBuffer(capacity=10000, time_window=self.time_window)
        self.predictor = RandomForestRegressor(n_estimators=1000, random_state=0)

    def update(self):
        dataset = self.buffer.get_dataset()
        X = np.array(dataset.tensors[0])
        y = np.array(dataset.tensors[1])

        regressor = self.predictor

        regressor.fit(X, y)

        pred = regressor.predict(X)
        mse = np.mean((y - pred) ** 2)

        print('Trained with random forest on {} samples. Mean squared error: {}'.format(X.shape[0], mse))

    def update_buffer(self, D, signal, important_features, datatype, actions):
        self.buffer.update(D, signal, important_features, datatype, actions)

    def predict(self, encoding):
        encoding = np.array(encoding).reshape(1, -1)
        return self.predictor.predict(encoding)




