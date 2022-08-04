import copy

import numpy as np
import torch
from stable_baselines3 import DQN
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.evaluation.evaluator import Evaluator
from src.feedback.feedback_processing import encode_trajectory
from src.util import evaluate_policy


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

            obs, rew, done, _ = env.step(action)

            past = env.episode
            curr = 1
            for j in range(len(past)-1, -1, -1):
                enc = encode_trajectory(past[j:], obs, curr, time_window, env)

                D.append(enc)

                if curr >= time_window:
                    break

                curr += 1

    D = torch.tensor(np.vstack(D))
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

    # evaluate expert on true reward
    true_mean_rew = evaluate_policy(model, env, n_ep=100)
    print('Expert true mean reward = {}'.format(true_mean_rew))

    # evaluate different objectives
    evaluator = Evaluator()
    avg_mo = evaluator.evaluate_MO(model, env, n_episodes=100)
    print('Expert mean reward for objectives = {}'.format(avg_mo))

    # best_exp_traj = present_successful_traj(model, env)
    # visualize_feature(best_exp_traj, 2, plot_actions=False, title='Expert\'s lane change')

    # reset original config
    env.configure(orig_config)
    return model


def check_is_unique(unique_feedback, feedback_traj, timesteps, time_window, env, important_features):
    unique = True
    threshold = 0.05

    for f, imp_f, ts in unique_feedback:
        if (imp_f == important_features) and (len(f) == len(feedback_traj)):

            enc_1 = encode_trajectory(feedback_traj, None, timesteps, time_window, env)
            enc_2 = encode_trajectory(f, None, ts, time_window, env)

            distance = np.mean(abs(enc_1[imp_f] - enc_2[important_features]))
            if distance < threshold:
                unique = False
                break

    return unique

def train_model(env, model_config, path):
    try:
        model = DQN.load(path, seed=1, env=env)
        print('Loaded initial model')
    except FileNotFoundError:
        print('Training initial model...')
        model = DQN('MlpPolicy', env, **model_config)
        model.learn(total_timesteps=20000)

        model.save(path)

    return model