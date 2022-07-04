import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)


class InventoryEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment

    TO BE EDITED

    This environment corresponds to the version of the inventory control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """

    def __init__(self, n=100, item_cost=-1, item_sale=2, hold_cost=0, loss_cost=-1, delivery_cost=0, lam=30):
        self.n = n
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(0, n + 1, (1, ), dtype=np.int)
        self.max = n
        self.state = n
        self.item_cost = item_cost
        self.item_sale = item_sale
        self.hold_cost = hold_cost
        self.loss_cost = loss_cost
        self.delivery_cost = delivery_cost
        self.lam = lam

        # Set seed
        self.seed()

        # Start the first round
        self.reset()

        self.max_timesteps = 20

    def demand(self):
        return np.random.poisson(self.lam)

    def transition(self, x, a, d):
        m = self.max
        x = x.item()
        return max(min(x + a, m) - d, 0)

    def reward(self, x, a, y):
        x = x.item()
        m = self.max

        # item ordering fee + sale reward + holding fee + loss if not enough to satisfy demand
        new_x = min(x + a, m)
        r = a * self.item_cost + \
            min(y, new_x) * self.item_sale + \
            max(new_x - y, 0) * self.hold_cost +\
            max(y - new_x, 0) * self.loss_cost + \
            (a > 0) * self.delivery_cost

        return r

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        obs = self.state[0]
        demand = self.demand()

        obs2 = self.transition(obs, action, demand)
        self.state = obs2

        reward = self.reward(obs, action, demand)

        done = self.steps >= self.max_timesteps
        self.steps += 1

        return obs2, reward, done, {}

    def reset(self):
        self.steps = 0
        return self.state