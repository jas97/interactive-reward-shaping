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
    A = []

    for i in tqdm(range(100)):
        done = False
        obs = env.reset()
        while not done:
            action = np.random.randint(0, env.action_space.n, size=(1,)).item()
            past = env.episode
            curr = 1
            for j in range(len(past)-1, -1, -1):
                s, a = past[j]
                if curr >= time_window:
                    break

                state_enc = env.encode_diff(s, obs, curr)
                action_enc = env.encode_actions(action, past[j:])

                A.append(action_enc)
                D.append(state_enc)
                curr += 1

            obs, rew, done, _ = env.step(action)

    D = torch.tensor(np.array(D))
    D = torch.unique(D, dim=0)  # remove duplicates

    A = torch.tensor(np.array(A))
    A = torch.unique(A, dim=0)

    y_D = np.zeros((len(D), ))
    y_D = torch.tensor(y_D)

    y_A = np.zeros((len(A),))
    y_A = torch.tensor(y_A)

    state_diff_dataset = TensorDataset(D, y_D)
    print('Generated {} env samples for state_diff'.format(len(D)))

    action_dataset = TensorDataset(A, y_A)
    print('Generated {} env samples for action'.format(len(A)))

    return state_diff_dataset, action_dataset