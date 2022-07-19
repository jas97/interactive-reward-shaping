import copy

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

from src.feedback.feedback_processing import present_successful_traj
from src.visualization.visualization import visualize_rewards, visualize_feature


class Evaluator:

    def __init__(self, expert_model,  feedback_freq, env):
        self.feedback_freq = feedback_freq
        self.env = env
        self.reward_dict = None
        self.similarities = []

        self.expert_model = expert_model

    def evaluate(self, model, env):
        # Evaluate multiple objectives
        rew_values = self.evaluate_MO(model, env, n_episodes=10)
        if self.reward_dict is None:
            self.reward_dict = rew_values
        else:
            self.reward_dict = {rn: self.reward_dict[rn] + rew_values[rn] for rn in rew_values.keys()}

        # evaluate similarity to the expert
        sim = self.evaluate_similarity(model, self.expert_model, env)
        self.similarities.append(sim)

        print('Rewards: {}'.format(self.reward_dict))
        print('Similarity = {}%'.format(sim*100))

    def visualize(self, iteration):
        # visualize the effect of shaping on objectives
        xs = np.arange(0, self.feedback_freq * iteration, step=self.feedback_freq)
        visualize_rewards(self.reward_dict, title='Average reward objectives with reward shaping', xticks=xs)

        # visualize similarity with the expert model
        plt.plot(self.similarities)
        # plt.xticks(xs)
        plt.xlabel('Time steps')
        plt.ylabel('Percentage of equal actions')
        # plt.title('Policy similarity between expert and trained model')
        plt.show()

    def evaluate_MO(self, model, env, n_episodes=10):
        # estimate number of objectives
        env.reset()
        _, _, _, info = env.step(env.action_space.sample())

        objectives = info['rewards']
        reward_names = [obj_n for obj_n, obj_val in objectives.items()]
        num_objectives = len(info['rewards'])
        ep_average = {rn: 0.0 for rn in reward_names}

        for ep in range(n_episodes):
            rewards = {rn: 0.0 for rn in reward_names}

            done = False
            obs = env.reset()
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, info = env.step(action)

                step_rewards = info['rewards']
                rewards = {rn: rewards[rn] + step_rewards[rn] for rn in rewards.keys()}

            ep_average = {rn: ep_average[rn] + rewards[rn] for rn in ep_average.keys()}

        ep_average = {rn: [ep_average[rn] / n_episodes] for rn in ep_average.keys()}

        return ep_average

    def evaluate_similarity(self, model_A, model_B, env):
        actions_A = []
        actions_B = []

        for i in range(10):
            done = False
            obs = env.reset()
            while not done:
                action_A, _ = model_A.predict(obs, deterministic=True)
                action_B, _ = model_B.predict(obs, deterministic=True)

                obs, rew, done, _ = env.step(action_A)

                actions_A.append(action_A)
                actions_B.append(action_B)

        sim = sum(np.array(actions_A) == np.array(actions_B)) / len(actions_A)

        return sim

