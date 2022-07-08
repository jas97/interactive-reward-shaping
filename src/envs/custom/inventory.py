import numpy as np

from src.envs.original.gym_inventory.inventory_env import InventoryEnv

# TODO: create abstract class to define all necessary methods
from src.feedback.feedback_processing import FeedbackTypes


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
            s, a = past[j]
            if curr >= self.time_window:
                break

            state_enc = self.encode_diff(s, state, curr)
            action_enc = self.encode_actions(action, past)

            try:
                state_rew = self.reward_model.predict(state_enc, FeedbackTypes.STATE_DIFF).item()
            except ValueError:
                state_rew = 0.0

            try:
                action_rew = self.reward_model.predict(action_enc, FeedbackTypes.ACTIONS).item()
            except ValueError:
                action_rew = 0.0

            running_rew += state_rew + action_rew

        return running_rew

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean

    def encode_diff(self, start, end, timesteps):
        state_enc = np.array(list(start) + list(start - end) + [timesteps])
        return state_enc

    def encode_actions(self, action, past):
        enc = [self.action_space.n] * self.time_window

        i = 0
        for el in past:
            if i >= self.time_window:
                break
            enc[i] = el[1]
            i += 1

        if i < self.time_window:
            enc[i] = action

        return np.array(enc)

    def render_state(self, state):
        print('Inventory: {}'.format(state))

    def update_lambda(self, update):
        self.lmbda = self.lmbda * update