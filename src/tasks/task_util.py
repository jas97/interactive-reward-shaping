import numpy as np
import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm


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


def init_replay_buffer(env, time_window):  # TODO: maybe should be initialized also using a trained policy
    print('Initializing replay buffer with env reward...')
    D = []

    for i in tqdm(range(100)):
        done = False
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            past = env.episode
            curr = 1
            for j in range(len(past)-1, -1, -1):
                s, a = past[j]
                if curr >= time_window:
                    break

                state_enc = env.encode_diff(s, obs, curr)

                D.append(state_enc)
                curr += 1

            obs, rew, done, _ = env.step(action)

    D = torch.tensor(np.array(D))
    D = torch.unique(D, dim=0)  # remove duplicates

    y = np.zeros((len(D), ))
    y = torch.tensor(y)

    dataset = TensorDataset(D, y)
    print('Generated {} env samples'.format(len(D)))

    return dataset