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

        timesteps = end_timestep - start_timestep + 1

        print('Enter ids of important features separated by space:')
        important_features = input()
        try:
            important_features = [int(x) for x in important_features.split(' ')]
        except ValueError:
            important_features = []

        feedback.append((feedback_type, f, feedback_signal, important_features, timesteps))

        print('Enter another trajectory (y/n?)')
        cont = input()
        if cont == 'y':
            done = False
        else:
            done = True

    return feedback


def augment_feedback_diff(traj, signal, important_features, timesteps, env, actions=False, time_window=5, datatype='int', length=100):
    print('Augmenting feedback...')

    state_len = traj[0][0].flatten().shape[0]
    traj_len = len(traj)
    traj_enc = encode_trajectory(traj, timesteps, time_window, env)
    enc_len = traj_enc.shape[0]

    # generate mask to preserve important features
    random_mask = np.ones((length, enc_len))
    random_mask[:, important_features] = 0
    inverse_random_mask = 1 - random_mask

    D = np.tile(traj_enc, (length, 1))

    # add noise to important features if they are continuous
    if datatype != 'int':
        # adding noise for continuous state features
        D[:, :traj_len*state_len] = D[:, :traj_len*state_len] + np.random.normal(0, 0.001, (length, traj_len*state_len))

    # observation limits
    lows = list(np.tile(env.lows, (traj_len, 1)).flatten())
    highs = list(np.tile(env.highs, (traj_len, 1)).flatten())

    # action limits
    lows += [0] * traj_len
    highs += [env.action_space.n] * traj_len

    # timesteps limits
    lows += [1]
    highs += [time_window]

    # generate matrix of random values within allowed ranges
    if datatype == 'int':
        rand_D = np.random.randint(lows, highs, size=(length, enc_len))
    else:
        rand_D = np.random.uniform(lows, highs, size=(length, enc_len))

    D = np.multiply(rand_D, random_mask) + np.multiply(inverse_random_mask, D)

    # reward for feedback the signal
    D = torch.tensor(D)
    D = torch.unique(D, dim=0)

    y = np.zeros((len(D),))
    y.fill(signal)
    y = torch.tensor(y)

    dataset = TensorDataset(D, y)

    print('Generated {} augmented samples'.format(len(dataset)))
    return dataset


def augment_actions(feedback_traj, env, length=2000):
    actions = [a for (s, a) in feedback_traj]
    traj_len = len(actions)

    # generate neighbourhood of sequences of actions
    # TODO: different options for generating neighborhood
    random_traj = np.tile(actions, (length*100, 1))

    random_traj = np.round(random_traj + np.random.normal(0, 5, size=(length*100, traj_len)))
    random_traj[random_traj<0] = 0

    # find similarities to the original trajectory actions using dynamic time warping
    sims = [dtw(actions, traj, keep_internals=True).normalizedDistance for traj in random_traj]
    top_indices = np.argsort(sims)[-length:]

    # filter out only the most similar trajectories
    filtered_traj = random_traj[top_indices]

    D = torch.tensor(filtered_traj)

    return D


def encode_trajectory(traj, timesteps, time_window, env):
    states = []
    actions = []

    state_len = len(traj[0][0])

    curr = 0
    for s, a in traj:
        if curr >= time_window:
            break

        states += list(s.flatten())
        actions.append(a)
        curr += 1

    # add the last state to fill up the trajectory encoding
    # this will be randomized so it does not matter
    while curr < time_window:
        low = env.lows
        highs = env.highs
        states += list(np.random.randint(low, highs, size=(state_len, )))
        actions.append(env.action_space.sample())
        curr += 1

    enc = states + actions + [timesteps]
    enc = np.array(enc)

    return enc