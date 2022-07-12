from highway_env import utils

import numpy as np
from highway_env.envs import highway_env
from highway_env.vehicle.controller import ControlledVehicle

from src.feedback.feedback_processing import encode_trajectory


class CustomHighwayEnv(highway_env.HighwayEnvFast):

    def __init__(self, shaping=False, time_window=5):
        super().__init__()
        self.shaping = shaping
        self.time_window = time_window

        self.episode = []

        self.lows = np.zeros((25, ))
        self.highs = np.ones((25, ))
        self.lows.fill(-1)

        self.lmbda = 0.5

    def step(self, action):
        self.episode.append((self.state.flatten(), action))

        self.state, rew, done, info = super().step(action)

        info['true_rew'] = rew

        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]

        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])

        coll_rew = self.config["collision_reward"] * self.vehicle.crashed
        right_lane_rew = self.config["right_lane_reward"] * lane / max(len(neighbours) - 1, 1)
        speed_rew = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)

        if self.shaping:
            rew += self.augment_reward(action, self.state.flatten())

        info['collision_rew'] = coll_rew
        info['right_lane_rew'] = right_lane_rew
        info['speed_rew'] = speed_rew

        return self.state, rew, done, info

    def reset(self):
        self.episode = []
        self.state = super().reset()
        return self.state

    def close(self):
        pass

    def render(self):
        super().render(mode='human')

    def render_state(self, state):
        print('State = {}'.format(state.flatten()[0:5]))

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
