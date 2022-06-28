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
        action, _ = model.predict(obs)
        obs, rew, done, _ = env.step(action)

        total_rew += rew

    return total_rew


def evaluate_policy(model, env, n_ep=100):
    rews = []
    for i in range(n_ep):
        ep_rew = play_episode(model, env)
        rews.append(ep_rew)

    return np.mean(rews)

def seed_everything(seed=1):
    seed_value = seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # tf.random.set_seed(seed_value)
    g = torch.Generator()
    g.manual_seed(seed_value)