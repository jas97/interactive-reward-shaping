import copy

import numpy as np
import torch
from stable_baselines3 import DQN
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.feedback.feedback_processing import encode_trajectory, present_successful_traj
from src.visualization.visualization import visualize_feature


def check_dtype(env):
    obs = env.reset().flatten()
    is_float = np.issubdtype(obs.flatten()[0], np.floating)
    is_int = np.issubdtype(obs.flatten()[0], np.int)

    action_dtype = env.action_dtype

    if is_int:
        state_dtype = 'int'
    elif is_float:
        state_dtype = 'cont'
    else:
        raise TypeError('Unknown type of the observation')

    return state_dtype, action_dtype


def init_replay_buffer(env, model, time_window):
    print('Initializing replay buffer with env reward...')
    D = []

    for i in tqdm(range(1000)):
        done = False
        obs = env.reset()
        while not done:
            if model is None:
                action = np.random.randint(0, env.action_space.n, size=(1,)).item()
            else:
                action, _ = model.predict(obs, deterministic=True)

            past = env.episode
            curr = 1
            for j in range(len(past)-1, -1, -1):
                enc = encode_trajectory(past[j:], curr, time_window, env)

                D.append(enc)

                if curr >= time_window:
                    break

                curr += 1

            obs, rew, done, _ = env.step(action)

    D = torch.tensor(np.array(D))
    D = torch.unique(D, dim=0)  # remove duplicates

    y_D = np.zeros((len(D), ))
    y_D = torch.tensor(y_D)

    dataset = TensorDataset(D, y_D)
    print('Generated {} env samples for dataset'.format(len(D)))

    return dataset


def train_expert_model(env, env_config, model_config, expert_path, timesteps):
    orig_config = copy.copy(env.config)
    rewards = env_config['true_reward_func']
    env.configure(rewards)
    try:
        model = DQN.load(expert_path, seed=1, env=env)
        print('Loaded expert')
    except FileNotFoundError:
        print('Training expert...')
        model = DQN('MlpPolicy', env, **model_config)
        model.learn(total_timesteps=timesteps)

        model.save(expert_path)

    best_exp_traj = present_successful_traj(model, env)
    visualize_feature(best_exp_traj, 2, plot_actions=False, title='Expert\'s lane change')

    # reset original config
    env.configure(orig_config)
    return model