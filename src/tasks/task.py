import numpy as np
import torch
from stable_baselines3 import DQN
from torch.utils.data import TensorDataset

from src.feedback.feedback_processing import present_successful_traj, gather_feedback, augment_feedback
from src.reward_modelling.reward_model import RewardModel
from src.util import evaluate_policy


class Task:

    def __init__(self, env, model_path, time_window=5, feedback_freq=50000, task_name='gridworld'):
        self.model_path = model_path
        self.time_window = time_window
        self.feedback_freq = feedback_freq
        self.task_name = task_name

        self.env = env
        self.reward_model = RewardModel(time_window)

        self.reward_model.buffer.initialize(self.init_replay_buffer())

        self.datatype = self.check_dtype(self.env)

    def init_replay_buffer(self):
        print('Initializing replay buffer with env reward...')
        D = []

        for i in range(10000):
            done = False
            obs = self.env.reset()
            while not done:
                action = self.env.action_space.sample()
                past = self.env.episode
                curr = 1
                for j in range(len(past)-1, -1, -1):
                    s, a = past[j]
                    if curr >= self.time_window:
                        break
                    state_enc = np.array(list(s) + list(s - obs) + [curr])
                    D.append(state_enc)
                    curr += 1

                obs, rew, done, _ = self.env.step(action)

        D = torch.tensor(np.array(D))
        D = torch.unique(D, dim=0)  # remove duplicates

        y = np.zeros((len(D), ))
        y = torch.tensor(y)

        dataset = TensorDataset(D, y)
        print('Generated {} env samples'.format(len(D)))

        return dataset

    def check_dtype(self, env):
        obs = env.reset().flatten()
        is_float = np.issubdtype(obs.flatten()[0], np.floating)
        is_int = np.issubdtype(obs.flatten()[0], np.int)

        if is_int:
            return 'int'
        elif is_float:
            return 'float'
        else:
            raise TypeError('Unknown type of the observation')

    def run(self):
        finished_training = False
        iteration = 1

        while not finished_training:
            print('Iteration = {}'.format(iteration))
            try:
                model = DQN.load(self.model_path, verbose=0, env=self.env)
                print('Loaded saved model')
                self.env.set_shaping(True)
                # if it's not the first iteration reward model should be used
                self.env.set_reward_model(self.reward_model)
            except FileNotFoundError:
                model = DQN('MlpPolicy', self.env, verbose=0, exploration_fraction=0.8, exploration_final_eps=0.1)
                print('First time training the model')

                # initialize the replay buffer
                self.init_replay_buffer()

            print('Training DQN for {} timesteps'.format(self.feedback_freq))
            model.learn(total_timesteps=self.feedback_freq)
            model.save(self.model_path)

            # evaluate partially trained model
            mean_rew = evaluate_policy(model, self.env)
            print('Training timesteps = {}. Mean reward = {}.'.format(self.feedback_freq * iteration, mean_rew))

            # print the best trajectories
            present_successful_traj(model, self.env, n_traj=10)

            if iteration == 2:
                break

            # gather feedback trajectories
            feedback = gather_feedback(self.task_name)
            for f in feedback:
                print('Feedback trajectory = {}. Important features = {}. '.format(f[0], f[1]))

            for feedback_traj, important_features, timesteps in feedback:
                # augment feedback for each trajectory
                D = augment_feedback(feedback_traj,
                                     important_features,
                                     timesteps,
                                     self.env,
                                     feedback_type='diff',
                                     time_window=self.time_window,
                                     datatype=self.datatype,
                                     length=10000)

                self.reward_model.update(D)





