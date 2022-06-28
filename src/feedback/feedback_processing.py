import json
import numpy as np
import torch
from torch.utils.data import TensorDataset


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


def gather_feedback(task_name):
    json_path = 'feedback/{}.json'.format(task_name)
    with open(json_path, 'r') as json_file:
        data = json.loads(json_file.read())
        f = data['feedback']
        important_features = data['important_features']
        timesteps = data['timesteps']

    return [(f, important_features, timesteps)]


def augment_feedback(traj, important_features, timesteps, env, feedback_type='diff', time_window=5, datatype='int', length=100):
    print('Augmenting feedback...')
    start_state = traj[0][0]
    end_state = traj[-1][0]

    state_diff = list(np.array(start_state) - np.array(end_state))

    if feedback_type == 'diff':
        # append difference in states to the start state
        state_enc = np.array(start_state + state_diff + [timesteps])
        enc_len = state_enc.shape[0]

        # generate mask to preserve important features
        random_mask = np.ones((length, enc_len))
        random_mask[:, important_features] = 0
        inverse_random_mask = 1 - random_mask

        D = np.tile(state_enc, (length, 1))

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

        D = np.multiply(rand_D, random_mask) + np.multiply(inverse_random_mask, D)

        # reward for feedback is always -1
        D = torch.tensor(D)
        D = torch.unique(D, dim=0)

        y = np.zeros((len(D),))
        y.fill(-1)
        y = torch.tensor(y)

        dataset = TensorDataset(D, y)

        print('Generated {} augmented samples'.format(len(dataset)))
        return dataset


def play_trajectory(env, traj):
    for s, a in traj:
        env.render_state(s)
        print('Action = {}'.format(a))


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
        action, _ = model.predict(obs)
        traj.append((obs, action))

        obs, rew, done, _ = env.step(action)
        total_rew += rew

    return traj, total_rew