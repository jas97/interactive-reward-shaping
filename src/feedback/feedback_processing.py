import numpy as np
import torch
from dtw import dtw
from torch.utils.data import TensorDataset


def present_successful_traj(model, env, n_traj=10):
    # gather trajectories
    print('Gathering successful trajectories for partially trained model...')
    traj_buffer, rews = gather_trajectories(model, env, 50)

    # filter trajectories
    top_indices = np.argsort(rews)[-n_traj:]
    filtered_traj = [traj_buffer[i] for i in top_indices]
    top_rews = [rews[i] for i in top_indices]

    # play filtered trajectories
    for j, t in enumerate(filtered_traj):
        print('------------------\n Trajectory {} \n------------------\n'.format(j))
        print('Trajectory reward = {}'.format(top_rews[j]))
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


def gather_feedback(best_traj, time_window, disruptive=False, noisy=False, prob=0):
    print('Gathering user feedback')
    done = False
    feedback_list = []

    while not done:
        print('Input feedback type (s, a, none or done)')
        feedback_type = input()
        if feedback_type == 'done':
            return [], False

        if feedback_type == 'none':
            return [], True

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

        feedback = (feedback_type, f, feedback_signal, important_features, timesteps)

        if disruptive:
            feedback = disrupt(feedback, prob)

        feedback_list.append(feedback)

        print('Enter another trajectory (y/n?)')
        cont = input()
        if cont == 'y':
            done = False
        else:
            done = True

    if noisy:
        feedback_list = noise(feedback_list, best_traj, time_window, prob)

    return feedback_list, True


def noise(feedback_list, best_traj, time_window, prob):
    add_noisy_sample = np.random.randint(0, 1, prob=prob)

    if add_noisy_sample:
        state = best_traj[0][0]
        state_features_len = len(state)

        rand_traj = np.random.randint(0, len(best_traj))
        f_rand_traj = best_traj[rand_traj]

        rand_start = np.random.randint(0, len(f_rand_traj))
        rand_len = np.random.randint(0, time_window)

        f_rand_traj = f_rand_traj[rand_start: (rand_start + rand_len)]

        rand_f_type = np.random.randint(0, len(best_traj))
        rand_f_type = 's' if rand_f_type else 'a'

        rand_f_signal = np.random.choice((-1, +1))

        timesteps = rand_len

        important_features = []
        if rand_f_type == 's':
            important_features = np.random.randint(0, state_features_len)

        feedback_list.append((rand_f_type, f_rand_traj, rand_f_signal, important_features, timesteps))

        return feedback_list

    else: # no noisy samples added
        return feedback_list


def disrupt(feedback, prob):
    feedback_type, f, feedback_signal, important_features, timesteps = feedback

    disrupt_sample = np.random.choice([0, 1], p=prob)
    if disrupt_sample:
        feedback_signal = -feedback_signal
        return (feedback_type, f, feedback_signal, important_features, timesteps)
    else:
        return feedback


def augment_feedback_diff(traj, signal, important_features, timesteps, env, time_window, actions, datatype, length=100):
    print('Augmenting feedback...')

    state_dtype, action_dtype = datatype

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
    if state_dtype != 'int':
        # adding noise for continuous state features
        D[:, :time_window*state_len] = D[:, :time_window*state_len] + np.random.normal(0, 0.001, (length, time_window*state_len))

    # observation limits
    lows = list(np.tile(env.lows, (time_window, 1)).flatten())
    highs = list(np.tile(env.highs, (time_window, 1)).flatten())

    # action limits
    lows += [0] * time_window
    highs += [env.action_space.n] * time_window

    # timesteps limits
    lows += [1]
    highs += [time_window]

    # generate matrix of random values within allowed ranges
    if state_dtype == 'int' or (actions and action_dtype == 'int'):
        rand_D = np.random.randint(lows, highs, size=(length, enc_len))
    else:
        rand_D = np.random.uniform(lows, highs, size=(length, enc_len))
        if actions:
            rand_D[:, time_window*state_len:] = augment_actions(traj, length)

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


def augment_actions(feedback_traj, length=2000):
    actions = [a for (s, a) in feedback_traj]
    traj_len = len(actions)

    # generate neighbourhood of sequences of actions
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

    state_len = len(traj[0][0].flatten())

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


def generate_important_features(important_features, feedback_type, time_window, feedback_traj):
    actions = feedback_type == 'a'
    state_len = feedback_traj[0][0].flatten().shape[0]
    traj_len = len(feedback_traj)
    important_features = [im_f + (state_len * i) for i in range(traj_len) for im_f in important_features]
    important_features += [time_window * state_len + time_window]  # add timesteps as important

    if actions:
        important_features += list(np.arange(time_window * state_len, time_window * state_len + traj_len))

    return important_features, actions
