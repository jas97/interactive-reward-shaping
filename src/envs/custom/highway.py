
import numpy as np
from highway_env.envs import highway_env

from src.feedback.feedback_processing import FeedbackTypes


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

        if self.shaping:
            rew += self.augment_reward(action, self.state.flatten())

        return self.state, rew, done, info

    def reset(self):
        self.episode = []
        self.state = super().reset()
        return self.state

    def close(self):
        pass

    def render(self):
        super().render()

    def render_state(self, state):
        print('State = {}'.format(state.flatten()[0:5]))

    def augment_reward(self, action, state):
        running_rew = 0
        past = self.episode
        curr = 1
        for j in range(len(past)-1, -1, -1):  # go backwards in the past
            s, a = past[j]
            if curr >= self.time_window:
                break

            state_enc = self.encode_diff(state, s, curr)
            try:
                rew = self.lmbda * self.reward_model.predict(state_enc, feedback_type=FeedbackTypes.STATE_DIFF).item()
            except ValueError:
                rew = 0.0

            running_rew += rew

            actions = np.array([a for (s, a) in past[0: j]])
            try:
                actions_rew = self.lmbda * self.reward_model.predict(actions, feedback_type=FeedbackTypes.ACTIONS).item()
            except ValueError:
                actions_rew = 0.0

            running_rew += actions_rew

        return running_rew

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean

    def encode_diff(self, start_s, end_s, timesteps):
        enc = np.array(list(start_s.flatten()) + list(end_s.flatten() - start_s.flatten()) + [timesteps])
        return enc

    def encode_actions(self, action, past):
        enc = [self.action_space.n] * self.time_window

        i = 0
        for el in past:
            enc[i] = el[1]
            i += 1

        if i < self.time_window:
            enc[i] = action

        return np.array(enc)