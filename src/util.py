import json
import os
import random

import numpy as np
import torch


def play_episode(model, env, verbose=0):
    done = False
    obs = env.reset()
    if verbose:
        env.render()

    total_rew = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _ = env.step(action)
        if verbose:
            env.render()
        total_rew += rew

    return total_rew


def evaluate_policy(model, env, verbose=False, n_ep=100):
    rews = []
    for i in range(n_ep):
        ep_rew = play_episode(model, env, verbose)
        rews.append(ep_rew)

    return np.mean(rews)


def load_config(config_path):
    with open(config_path) as f:
        data = json.loads(f.read())

    return data


def evaluate_MO(model, env, n_episodes=100):
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
            action, _ = model.predict(obs)
            obs, _, done, info = env.step(action)

            step_rewards = info['rewards']
            rewards = {rn: rewards[rn] + step_rewards[rn] for rn in rewards.keys()}

        ep_average = {rn: ep_average[rn] + rewards[rn] for rn in ep_average}

    ep_average = {rn: ep_average[rn] / n_episodes for rn in ep_average}

    return ep_average


def seed_everything(seed=1):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # tf.random.set_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)