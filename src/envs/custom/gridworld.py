import random

import gym
import numpy as np


class Gridworld(gym.Env):

    def __init__(self, shaping=False, time_window=5):
        self.world_dim = 5
        self.time_window = time_window

        self.lows = np.array([0, 0, 0, 0, 0])
        self.highs = np.array([self.world_dim, self.world_dim, self.world_dim, self.world_dim, 4])
        self.observation_space = gym.spaces.Box(self.lows, self.highs, shape=(5, ))
        self.action_space = gym.spaces.Discrete(2)

        self.state = np.zeros((5, ))

        self.step_pen = -1
        self.turn_pen = 0
        self.goal_rew = 10
        self.shaping = shaping

        self.max_steps = 50
        self.steps = 0

        self.lmbda = 1  # for testing (should be [0,1] otherwise)

        # keep record of the last episode
        self.episode = []

    def step(self, action):
        self.episode.append((self.state, action))

        agent_x, agent_y, goal_x, goal_y, orient = self.state
        if action == 0:
            if orient == 0:
                if agent_x + 1 < self.world_dim:
                    agent_x += 1
            elif orient == 1:
                if agent_y + 1 < self.world_dim:
                    agent_y += 1
            elif orient == 2:
                if agent_x - 1 >= 0:
                    agent_x -= 1
            elif orient == 3:
                if agent_y - 1 >= 0:
                    agent_y -= 1

        if action == 1:
            orient = (orient + 1) % 4

        new_state = np.array([agent_x, agent_y, goal_x, goal_y, orient])

        done = self.check_if_done(new_state)
        rew = self.calculate_reward(action, new_state)

        self.state = new_state
        self.steps += 1

        return new_state.flatten(), rew, done, {}

    def check_if_done(self, state):
        if self.steps >= self.max_steps:
            return True

        agent_x, agent_y, goal_x, goal_y, orient = state

        if (agent_x == goal_x) and (agent_y == goal_y):
            return True

        return False

    def calculate_reward(self, action, state):
        agent_x, agent_y, goal_x, goal_y, orient = state

        rew = 0.0

        if (agent_x == goal_x) and (agent_y == goal_y):
            rew = self.goal_rew
        elif action == 0:
            rew = self.step_pen
        elif action == 1:
            rew = self.turn_pen

        if self.shaping:
            rew += self.lmbda * self.augment_reward(action, state)

        return rew

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

    def reset(self):
        goal_x = random.randint(0, self.world_dim - 1)
        goal_y = random.randint(0, self.world_dim - 1)

        agent_x = random.randint(0, self.world_dim - 1)
        agent_y = random.randint(0, self.world_dim - 1)

        while (agent_x == goal_x) and (agent_y == goal_y):
            agent_x = random.randint(0, self.world_dim - 1)
            agent_y = random.randint(0, self.world_dim - 1)

        orient = random.randint(0, 3)

        self.state = np.array([agent_x, agent_y, goal_x, goal_y, orient])
        self.steps = 0
        self.episode = []

        return self.state.flatten()

    def close(self):
        pass

    def render(self):
        self.render_state(self.state, self.steps)

    def render_state(self, state, steps=0):
        agent_x, agent_y, goal_x, goal_y, orient = state
        rendering = '---------------\n'
        rendering += 'Step = {} State = {}\n'.format(steps, state)

        for j in range(self.world_dim):
            row = ''
            for i in range(self.world_dim):
                if agent_x == i and agent_y == j:
                    if orient == 0:
                        row += ' > '
                    elif orient == 1:
                        row += ' v '
                    elif orient == 2:
                        row += ' < '
                    elif orient == 3:
                        row += ' ^ '
                elif goal_x == i and goal_y == j:
                    row += ' G '
                else:
                    row += ' - '
            rendering += row + '\n'

        rendering += '---------------'
        print(rendering)

    def set_reward_model(self, rm):
        self.reward_model = rm

    def set_shaping(self, boolean):
        self.shaping = boolean

    def encode_diff(self, start_s, end_s, timesteps):
        enc = np.array(list(start_s) + list(end_s - start_s) + [timesteps])
        return enc