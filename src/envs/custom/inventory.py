import numpy as np

from src.envs.original.gym_inventory.inventory_env import InventoryEnv


class Inventory(InventoryEnv):

    def __init__(self, shaping=False, time_window=5):
        super().__init__()
        self.world_dim = 5
        self.shaping = shaping
        self.time_window = time_window

        self.episode = []

        self.lows = np.zeros((25, 0))
        self.highs = np.ones((25, 0))

        self.lmbda = -0.1

    def step(self, action):
        self.episode.append((self.state.flatten(), action))

        self.state, rew, done, info = super().step(action)

        self.state = np.array([self.state]).flatten()

        if self.shaping:
            print('Shaping')
            rew += self.lmbda * self.augment_reward(action, self.state.flatten())

        info['true_rew'] = rew

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
            state_enc = np.array(list(s) + list(s - state) + [curr])

            rew = self.reward_model.predict(state_enc)
            running_rew += rew.item()

        return running_rew

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean