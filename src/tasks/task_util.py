import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

from src.feedback.feedback_processing import encode_trajectory


def check_dtype(env):
    obs = env.reset().flatten()
    is_float = np.issubdtype(obs.flatten()[0], np.floating)
    is_int = np.issubdtype(obs.flatten()[0], np.int)

    if is_int:
        return 'int'
    elif is_float:
        return 'float'
    else:
        raise TypeError('Unknown type of the observation')


def init_replay_buffer(env, time_window):
    print('Initializing replay buffer with env reward...')
    D = []

    for i in tqdm(range(50)):
        done = False
        obs = env.reset()
        while not done:
            action = np.random.randint(0, env.action_space.n, size=(1,)).item()
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