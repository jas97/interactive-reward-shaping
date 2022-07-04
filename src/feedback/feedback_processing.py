import numpy as np
import torch
from dtw import dtw
from torch.utils.data import TensorDataset
from enum import Enum


class FeedbackTypes(Enum):
    STATE_DIFF = 'state_diff'
    ACTIONS = 'actions'
    FEATURE = 'feature'


def present_successful_traj(model, env, n_traj=10):
    # gather trajectories
    print('Gathering successful trajectories for partially trained model...')
    traj_buffer, rews = gather_trajectories(model, env, 50)

    # filter trajectories
    top_indices = np.argsort(rews)[-n_traj:]
    filtered_traj = [traj_buffer[i] for i in top_indices]

    # play filtered trajectories
    for i, t in enumerate(filtered_traj):
        print('------------------\n Trajectory {} \n------------------\n'.format(i))
        play_trajectory(env, t)

    return filtered_traj


def play_trajectory(env, traj):
    for i, (s, a) in enumerate(traj):
        print('------------------\n Timestep = {}'.format(i))
        env.render_state(s)
        print('Action = {}\n------------------\n '.format(a))


def gather_trajectories(model, env, n_traj):
    traj_buffer = []
    rews = []

    for i in range(n_traj):
        traj, rew = get_ep_traj(model, env)
        traj_buffer.append(traj)
        rews.append(rew)

    return traj_buffer, rews


def get_ep_traj(model, env):
    done = False
    obs = env.reset()
    traj = []

    total_rew = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        traj.append((obs, action))

        obs, rew, done, _ = env.step(action)
        total_rew += rew

    return traj, total_rew


def gather_feedback(best_traj):
    print('Gathering user feedback')
    done = False
    feedback = []

    while not done:
        print('Input feedback type (state_diff, actions or feature)')
        feedback_type = input()
        print('Input trajectory number:')
        traj_id = int(input())
        print('Enter starting timestep:')
        start_timestep = int(input())
        print('Enter ending timestep:')
        end_timestep = int(input())
        print('Enter feedback:')
        feedback_signal = int(input())

        f = best_traj[traj_id][start_timestep:(end_timestep + 1)]  # for inclusivity + 1

        timesteps = end_timestep - start_timestep

        print('Enter ids of important features separated by space:')
        important_features = input()
        important_features = [int(x) for x in important_features.split(' ')]

        feedback.append((feedback_type, f, feedback_signal, important_features, timesteps))

        print('Enter another trajectory (y/n?)')
        cont = input()
        if cont == 'y':
            done = False
        else:
            done = True

    return feedback


def augment_feedback_diff(traj, signal,important_features, timesteps, env, time_window=5, datatype='int', length=100):
    print('Augmenting feedback...')
    start_state = traj[0][0]
    end_state = traj[-1][0]

    # append difference in states to the start state
    state_enc = env.encode_diff(start_state, end_state, timesteps)
    enc_len = state_enc.shape[0]

    # generate mask to preserve important features
    random_mask = np.ones((length, enc_len))
    random_mask[:, important_features] = 0
    inverse_random_mask = 1 - random_mask

    D = np.tile(state_enc, (length, 1))

    if datatype != 'int':
        # adding noise for continuous state features
        D = D + np.random.normal(0, 0.001, (length, enc_len))

    lows = np.zeros((enc_len, ))
    highs = np.zeros((enc_len, ))

    lows[0: int(enc_len/2)] = env.lows
    highs[0:int(enc_len/2)] = env.highs

    lows[int(enc_len/2):-1] = env.lows - env.highs + 1  # because lower bound is inclusive
    highs[int(enc_len/2):-1] = env.highs - env.lows

    # timesteps limits
    lows[-1] = 1
    highs[-1] = time_window

    # generate matrix of random values within allowed ranges
    if datatype == 'int':
        rand_D = np.random.randint(lows, highs, size=(length, enc_len))
    else:
        rand_D = np.random.uniform(lows, highs, size=(length, enc_len))

    # TODO: for continuous data probably should be added small noise
    D = np.multiply(rand_D, random_mask) + np.multiply(inverse_random_mask, D)

    # reward for feedback is always -1
    D = torch.tensor(D)
    D = torch.unique(D, dim=0)

    y = np.zeros((len(D),))
    y.fill(signal)
    y = torch.tensor(y)

    dataset = TensorDataset(D, y)

    print('Generated {} augmented samples'.format(len(dataset)))
    return dataset


def augment_feedback_actions(feedback_traj, signal, env, length=2000, threshold=1):
    actions = [a for (s, a) in feedback_traj]
    traj_len = len(actions)
    max_action = env.action_space.n

    # generate neighbourhood of sequences of actions
    # TODO: different options for generating neighborhood
    random_traj = np.tile(actions, (length*100, 1))

    random_traj = np.round(random_traj + np.random.normal(0, 5, size=(length*100, traj_len)))
    random_traj[random_traj<0] = 0

    # TODO: normalize actions and random_traj instead of increasing the threshold

    # find similarities to the original trajectory actions using dynamic time warping
    sims = [dtw(actions, traj, keep_internals=True).normalizedDistance for traj in random_traj]

    # filter out only the most similar trajectories
    boolean_sim = np.array(sims) < threshold
    filtered_traj = random_traj[boolean_sim]

    D = torch.tensor(filtered_traj)
    D = torch.unique(D, dim=0)

    y = np.zeros((len(D),))
    y.fill(signal)
    y = torch.tensor(y)

    dataset = TensorDataset(D, y)

    return dataset