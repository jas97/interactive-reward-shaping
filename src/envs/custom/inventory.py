import numpy as np

from src.envs.original.gym_inventory.inventory_env import InventoryEnv
from src.feedback.feedback_processing import encode_trajectory


class Inventory(InventoryEnv):

    def __init__(self, time_window, shaping=False):
        super().__init__()
        self.shaping = shaping
        self.time_window = time_window

        self.episode = []

        self.state_len = 1
        self.lows = np.zeros((1, 0))
        self.highs = np.ones((1, 0))
        self.highs.fill(100)

        self.action_dtype = 'cont'

        self.lmbda = 1

        self.immutable_features = []
        self.discrete_features = [0, 1, 2, 3, 4]
        self.cont_features = []

        self.state_dtype = 'int'

        self.action_dtype = 'cont'

        self.lows = [0]
        self.highs = [100]

    def step(self, action):
        self.episode.append((self.state.flatten(), action))

        self.state, rew, done, info = super().step(action)

        self.state = np.array([self.state]).flatten()

        if self.shaping:
            shaped_rew = self.augment_reward(action, self.state.flatten())
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
            state_enc = encode_trajectory(past[j:], state, curr, self.time_window, self, )

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

    def configure(self, rewards):
        super().configure(rewards)

    def set_true_reward(self, rewards):
        super().set_true_reward(rewards)

    def random_state(self):
        return np.random.randint(self.lows, self.highs, (self.state_len,))

    def encode_state(self, state):
        return state