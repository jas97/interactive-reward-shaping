import numpy as np

from src.envs.original.gym_inventory.inventory_env import InventoryEnv

# TODO: create abstract class to define all necessary methods
from src.feedback.feedback_processing import encode_trajectory


class Inventory(InventoryEnv):

    def __init__(self, time_window, shaping=False):
        super().__init__()
        self.shaping = shaping
        self.time_window = time_window

        self.episode = []

        self.lows = np.zeros((1, 0))
        self.highs = np.ones((1, 0))
        self.highs.fill(100)

        self.lmbda = 0.2

    def step(self, action):
        self.episode.append((self.state.flatten(), action))

        self.state, rew, done, info = super().step(action)

        self.state = np.array([self.state]).flatten()

        if self.shaping:
            shaped_rew = self.lmbda * self.augment_reward(action, self.state.flatten())
            rew += shaped_rew

        info['env_rew'] = rew
        info['shaped_rew'] = shaped_rew if self.shaping else 0

        return self.state, rew, done, info

    def reset(self):
        self.episode = []
        self.state = np.array([super().reset()]).flatten()
        return self.state

    def close(self):
        pass

    def render(self):
        print('Obs: {}'.format(self.obs))

    def augment_reward(self, action, state):
        running_rew = 0
        past = self.episode
        curr = 1
        for j in range(len(past)-1, -1, -1):  # go backwards in the past
            state_enc = encode_trajectory(past[j:], curr, self.time_window, self)

            rew = self.reward_model.predict(state_enc)

            running_rew += rew.item()

            if curr >= self.time_window:
                break

            curr += 1

        return running_rew

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean

    def render_state(self, state):
        print('Inventory: {}'.format(state))

    def update_lambda(self, update):
        self.lmbda = self.lmbda * update