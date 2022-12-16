# %%
import cvxpy
from aix360.algorithms.rbm import BRCGExplainer, BooleanRuleCG
from aix360.algorithms.rbm import FeatureBinarizer
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.buffers import ReplayBuffer
from src.util import load_config
from src.envs.custom.highway import CustomHighwayEnv
from src import util
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3 import DQN
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.utils import resample
from highway_env.envs.common.graphics import EnvViewer

from highway_env import utils
import gym
from utils import record_videos
import torch
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional, Tuple, Type, Union
import os
import copy
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', UserWarning)

pd.set_option('display.max_columns', 10)
ACTIONS = {'turn': [0, 2],
           'forward': [1, 3, 4],
           'left': 0,
           'right': 2,
           'idle': 1,
           'faster': 3,
           'slower': 4,
           0:'left',
           2:'right',
           1:'idle',
           3:'faster',
           4:'slower'}

FEATURES = {'descriptive': ['speed', 'dist_to_car_in_front', 'car_infront_speed', 'car_infront_dir', 'x_left_car', 'y_left_car',
                            'dir_left_car', 'x_right_car', 'y_right_car', 'dir_right_car'],  # ,'prox_to_right_lane']
            'state': ['ego_y', 'ego_x', 'ego_velocity_y', 'ego_velocity_x',
                      'vehicle_1_y', 'vehicle_1_x', 'vehicle_1_velocity_y', 'vehicle_1_velocity_x',
                      'vehicle_2_y', 'vehicle_2_x', 'vehicle_2_velocity_y', 'vehicle_2_velocity_x']}
                      #'vehicle_3_y', 'vehicle_3_x', 'vehicle_3_velocity_y', 'vehicle_3_velocity_x',
                      #'vehicle_4_y', 'vehicle_4_x', 'vehicle_4_velocity_y', 'vehicle_4_velocity_x']}
CATEG_COL = {'descriptive': ['speed', 'dist_to_car_in_front', 'car_infront_speed', 'car_infront_dir', 'x_left_car', 'y_left_car',
                            'dir_left_car', 'x_right_car', 'y_right_car', 'dir_right_car'],
             'state': None}
#'speed', 'dist_to_car_in_front', 'car_infront_speed', 'car_infront_dir', 'x_left_car',   'dir_left_car', 'x_right_car',  'dir_right_car'

class CustomDQN(DQN):
    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau,
            gamma, train_freq, gradient_steps, replay_buffer_class, replay_buffer_kwargs, optimize_memory_usage,
            target_update_interval, exploration_fraction, exploration_initial_eps, exploration_final_eps, max_grad_norm, tensorboard_log,
            create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)

    def learn(self,
              learn_timesteps=1000,
              total_timesteps: int = 50000,
              callback: MaybeCallback = None,
              log_interval: int = 4,
              eval_env: Optional[GymEnv] = None,
              eval_freq: int = -1,
              n_eval_episodes: int = 5,
              tb_log_name: str = "run",
              eval_log_path: Optional[str] = None,
              reset_num_timesteps: bool = True,
              ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name)
        callback.on_training_start(locals(), globals())
        while self.num_timesteps < learn_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )
            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size,
                               gradient_steps=gradient_steps)
        callback.on_training_end()
        return self


class ImgRenderHighway(CustomHighwayEnv):
    def __init__(self, shaping=False, time_window=5):
        super().__init__(shaping, time_window)

    def render(self, mode: str = 'rgb_arary') -> Optional[np.ndarray]:
        """
        Render the environment as rgb_array only, don't use viewer to display render, just create.

        :param mode: the rendering mode
        """
        self.rendering_mode = mode
        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True
        self.viewer.display()
        assert mode == 'rgb_array', "Only works for rgb_array mode"
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image


def sampler(df, target, lbl, n):
    idx = df[target] == lbl

    return resample(df[idx],
                    replace=sum(idx) < n, n_samples=int(n),
                    random_state=42)


def setup_env():
    env_config = load_config('./config/env/highway.json')
    #env = gym.make('highway-fast-v0')
    env = CustomHighwayEnv(shaping=False, time_window=env_config['time_window'])
    env.config['right_lane_reward'] = env_config['right_lane_reward']
    env.config['lanes_count'] = env_config['lanes_count']
    env.config['offscreen_rendering'] = False
    env.config['simulation_frequency'] = 5
    env.set_true_reward(env_config['true_reward_func'])
    return env

# %%


def get_index_of_car_infront(obs_vehicle_lane, lane_indices):
    in_same_lane = np.where(lane_indices == obs_vehicle_lane)[0]
    if len(in_same_lane) == 0:
        return None
    # return first instance in array and add one to adjust to obs vehicle in state array
    return in_same_lane[0] + 1


def get_index_of_leftmost_car(obs_vehicle_lane, lane_indices):
    left = np.where(lane_indices < obs_vehicle_lane)[0]
    if len(left) == 0:
        return None
    # return first instance in array and add one to adjust to obs vehicle in state array
    return left[0] + 1


def get_index_of_rightmost_car(obs_vehicle_lane, lane_indices):
    right = np.where(lane_indices > obs_vehicle_lane)[0]
    if len(right) == 0:
        return None
    # return first instance in array and add one to adjust to obs vehicle in state array
    return right[0] + 1


def round_to_n(x, n=1):
    if x == 0:
        return 0
    num = np.round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))
    return num


def create_descriptive_based_features(state, env):
    """
    Reward from environment is designed to foster driving at high speed, on rightmost lane and avoid collisions
    Reward seems to capture behavioural decisions that could be useful to summarise in rules.
    Initially just want to capture when a lane change may occur

    Initial Features
    - Lane Changed (Target)
    - Speed
    - Proximity to Car directly infront
    - Car in front slower or faster
    - Proximity to Car left and Right
    - Proximity to right lane
    :param env:
    :return:
    """
    obs_type = env.observation_type
    close_vehicles = obs_type.env.road.close_vehicles_to(obs_type.observer_vehicle,
                                                         obs_type.env.PERCEPTION_DISTANCE,
                                                         count=obs_type.vehicles_count - 1,
                                                         see_behind=obs_type.see_behind,
                                                         sort=obs_type.order == "sorted")
    obs_vehicle_lane = obs_type.observer_vehicle.lane_index[2]
    lane_indices = np.array([v.lane_index[2] for v in close_vehicles])
    idx_of_car_infront = get_index_of_car_infront(
        obs_vehicle_lane, lane_indices)
    car_to_left = get_index_of_leftmost_car(obs_vehicle_lane, lane_indices)
    car_to_right = get_index_of_rightmost_car(obs_vehicle_lane, lane_indices)
    forward_speed = env.vehicle.speed * np.cos(env.vehicle.heading)
    if idx_of_car_infront:
        # 1 is distance column in array
        dist_to_car_in_front = round_to_n(state[idx_of_car_infront, 1], 1)
        car_infront_speed = round_to_n(state[idx_of_car_infront, 3], 1)
        car_infront_dir = round_to_n(state[idx_of_car_infront, 4], 1)
    else:
        dist_to_car_in_front = 99
        car_infront_speed = 99
        car_infront_dir = 99
    if car_to_left:
        x_left_car = round_to_n(state[car_to_left, 1], 1)
        y_left_car = obs_vehicle_lane - lane_indices[car_to_left-1]
        dir_left_car = round_to_n(state[car_to_left, 4], 1)
    else:
        x_left_car = 99
        y_left_car = 99
        dir_left_car = 99
    if car_to_right:
        x_right_car = round_to_n(state[car_to_right, 1], 1)
        y_right_car = lane_indices[car_to_right-1] - obs_vehicle_lane
        dir_right_car = round_to_n(state[car_to_right, 4], 1)
    else:
        x_right_car = 99
        y_right_car = 99
        dir_right_car = 99

    prox_to_right_lane = (
        len(env.road.network.graph['0']['1']) - obs_vehicle_lane) - 1
    features = {'speed': forward_speed, 'dist_to_car_in_front': dist_to_car_in_front,
                'car_infront_speed': car_infront_speed, 'car_infront_dir': car_infront_dir,
                'x_left_car': x_left_car, 'y_left_car': y_left_car, 'dir_left_car': dir_left_car,
                'x_right_car': x_right_car, 'y_right_car': y_right_car, 'dir_right_car': dir_right_car, 'prox_to_right_lane': prox_to_right_lane}
    return features


def create_state_based_features(state):
    ego_y = state[0][2]
    ego_x = state[0][1]
    ego_velocity_y = state[0][4]
    ego_velocity_x = state[0][3]
    vehicle_1_y = state[1][2]
    vehicle_1_x = state[1][1]
    vehicle_1_velocity_y = state[1][4]
    vehicle_1_velocity_x = state[1][3]
    vehicle_2_y = state[2][2]
    vehicle_2_x = state[2][1]
    vehicle_2_velocity_y = state[2][4]
    vehicle_2_velocity_x = state[2][3]
    vehicle_3_y = state[3][2]
    vehicle_3_x = state[3][1]
    vehicle_3_velocity_y = state[3][4]
    vehicle_3_velocity_x = state[3][3]
    vehicle_4_y = state[3][2]
    vehicle_4_x = state[3][1]
    vehicle_4_velocity_y = state[3][4]
    vehicle_4_velocity_x = state[3][3]
    return {'ego_y': ego_y, 'ego_x': ego_x,
            'ego_velocity_y': ego_velocity_y, 'ego_velocity_x': ego_velocity_x,
            'vehicle_1_y': vehicle_1_y, 'vehicle_1_x': vehicle_1_x,
            'vehicle_1_velocity_y': vehicle_1_velocity_y, 'vehicle_1_velocity_x': vehicle_1_velocity_x,
            'vehicle_2_y': vehicle_2_y, 'vehicle_2_x': vehicle_2_x,
            'vehicle_2_velocity_y': vehicle_2_velocity_y, 'vehicle_2_velocity_x': vehicle_2_velocity_x,
            'vehicle_3_y': vehicle_3_y, 'vehicle_3_x': vehicle_3_x,
            'vehicle_3_velocity_y': vehicle_3_velocity_y, 'vehicle_3_velocity_x': vehicle_3_velocity_x,
            'vehicle_4_y': vehicle_4_y, 'vehicle_4_x': vehicle_4_x,
            'vehicle_4_velocity_y': vehicle_4_velocity_y, 'vehicle_4_velocity_x': vehicle_4_velocity_x
            }


def collect_features(feature_dict: dict = {}, new_feature_dict: dict = None):
    if len(feature_dict) == 0:
        feature_dict = {key: [value] if type(value) != list else value
                        for key, value in new_feature_dict.items()}
        feature_dict['index'] = [0] if type(feature_dict['baseline_action']) != list else [
            i for i in range(len(feature_dict['baseline_action']))]
        return feature_dict
    else:
        for key in new_feature_dict.keys():
            if type(new_feature_dict[key]) != list and key != 'index':
                feature_dict[key].append(new_feature_dict[key])
                increase = 1
            elif type(new_feature_dict[key]) == list and key != 'index':
                feature_dict[key].extend(new_feature_dict[key])
                increase = len(new_feature_dict[key])
        feature_dict['index'].extend(
            np.arange(feature_dict['index'][-1] + 1, (feature_dict['index'][-1]+1)+increase))
    return feature_dict


def collect_episode_features(env, baseline_model, alternate_model, feature_dict={}, save_img=False):
    obs,info = env.reset()
    done = False
    rewards = 0
    max_steps = 50
    step = 0
    while not done and step <= max_steps:
        state = obs
        obs = obs_as_tensor(obs, device=baseline_model.device).reshape(1, -1)
        baseline_q_values = baseline_model.q_net(obs)
        action = torch.argmax(baseline_q_values).item()
        alternate_q_values = alternate_model.q_net(obs)
        alternate_action = torch.argmax(alternate_q_values).item()
        if save_img:
            img = env.render(mode='rgb_array')
        desc_features = create_descriptive_based_features(state, env)
        features = create_state_based_features(state)
        features = {**features, ** desc_features}
        new_obs, reward, done, info = env.step(action)
        if save_img:
            features['img'] = img

        features['done'] = done
        features['baseline_q_values'] = [
            baseline_q_values.cpu().detach().numpy()[0]]
        features['alternate_q_values'] = [
            alternate_q_values.cpu().detach().numpy()[0]]
        features['baseline_action'] = action
        features['alternate_action'] = alternate_action
        features['reward'] = reward
        features['lane_changed'] = info['rewards']['lane_changed']
        features['true_reward'] = info['true_rew']
        features['rewards_dict'] = np.array(list(info['rewards'].items()))
        features['info'] = np.array(list(info.items()))
        rewards += reward
        features['state'] = state
        feature_dict = collect_features(
            feature_dict=feature_dict, new_feature_dict=features)
        obs = new_obs
        step += 1
    return feature_dict, rewards



def construct_rules(explainer_model,dtype='object'):
    """
    Parses the rules generated by the explanation model and returns a form that is suitable for
    dataframe "querying"
    """
    constructed_rules = []
    for rule in explainer_model['rules']:
        split_rule = [part.lower() if part ==
                      'AND' else part for part in rule.split(' ')]
        constructed_rule = []
        if len(split_rule) == 0:
            break
        for i in range(len(split_rule)):
            if i != 0:
                if split_rule[i-1] == '==':
                    if dtype == 'object':
                        value  = str(int(float(split_rule[i])))
                        constructed_rule.append(f"{value}")
                    else:
                        value = split_rule[i]
                        constructed_rule.append(f"'{value}'")
                    continue
            constructed_rule.append(split_rule[i])
        constructed_rules.append(" ".join(constructed_rule))
    return constructed_rules


def create_summary_df(df, constructed_rules, features, target_feature, target):
    """
    Creates summary dataframe for agreements and disagreements of each rule
    """
    summary_df = pd.DataFrame(
        columns=['rule', 'total_support', 'agree', 'disagree', 'unique_states'])
    data_cols = features + [target_feature]
    for rule in constructed_rules:
        row = {}
        d = df[data_cols].query(rule)
        row['rule'] = rule
        row['total_support'] = d[target_feature].count()
        row['agree'] = len(d[d[target_feature] == target])
        row['agree_percentage'] = row['agree']/row['total_support']
        row['disagree'] = len(d[d[target_feature] != target])
        row['disagree_percentage'] = row['disagree']/row['total_support']
        row['unique_states'] = len(
            d.drop(target_feature, axis=1).value_counts())
        summary_df = summary_df.append(row, ignore_index=True)
    return summary_df


def create_stability_features(data, stability_threshold=0.1):
    """
    Creates the features marking each action from the baseline model and the alternative model (new model) as stable or not
    :param data: pd.DataFrame
    :return: data: pd.DataFrame
    """
    data['stable_baseline'] = data['baseline_q_values'].apply(
        lambda x: 1 if np.abs(x[0]-x[1]) > stability_threshold else 0)
    data['stable_alt'] = data['alternate_q_values'].apply(
        lambda x: 1 if np.abs(x[0]-x[1]) > stability_threshold else 0)
    data['stable'] = data[['stable_baseline', 'stable_alt']].apply(
        lambda x: 1 if (x[0] == 1 and x[0] == x[1]) else 0, axis=1)
    return data


def create_agreement_targets(data):
    turn_actions = [0, 2]
    not_turn_actions = [1, 3, 4]
    # agents disagree when to turn
    data['disagreement_turn'] = data[['baseline_action', 'alternate_action']].apply(
        lambda x: 1 if x[0] in turn_actions and x[1] not in turn_actions else 0, axis=1)
    # agents disagree when to go straight
    data['disagreement_forward'] = data[['baseline_action', 'alternate_action']].apply(
        lambda x: 1 if x[0] in not_turn_actions and x[1] not in not_turn_actions else 0, axis=1)
    # agents agree when to turn
    data['agreement_turn'] = data[['baseline_action', 'alternate_action']].apply(
        lambda x: 1 if x[0] in turn_actions and x[1] in turn_actions else 0, axis=1)
    # agents agree when to go straight
    data['agreement_forward'] = data[['baseline_action', 'alternate_action']].apply(
        lambda x: 1 if x[0] in not_turn_actions and x[1] in not_turn_actions else 0, axis=1)
    return data


def create_train_test_datasets(data, features, target):
    splitter = ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(splitter.split(data[features], data[target]))
    x_train, y_train = data[features].iloc[train_idx], data[target].iloc[train_idx]
    x_test, y_test = data[features].iloc[test_idx], data[target].iloc[test_idx]
    data['train_test'] = 'train'
    data['train_test'].iloc[test_idx] = 'test'
    return x_train, x_test, y_train, y_test


def create_binary_features(data_dict, colCateg):
    x_train, x_test, y_train, y_test = data_dict['x_train'], data_dict[
        'x_test'], data_dict['y_train'], data_dict['y_test']
    if colCateg:
        fb = FeatureBinarizer(colCateg=colCateg, numThresh=4, negations=False)
    else:
        fb = FeatureBinarizer(numThresh=4, negations=False)
    fb = fb.fit(X=x_train)
    x_train_fb = fb.transform(x_train)
    x_test_fb = fb.transform(X=x_test)
    tmp_cat = y_train.astype('category')
    y_train_code = tmp_cat.cat.codes
    y_test_code = y_test.astype('category').cat.codes
    assert x_train_fb.shape[-1] == x_test_fb.shape[-1]
    return x_train_fb, x_test_fb, y_train_code, y_test_code


def print_rules(explanations, target,action):
    if 'agreement' in target:
        if 'disagreement' not in target:
            isCNF = 'Agents does NOT disagree if ANY of the following rules are satisfied, otherwise Agents disagree:'
            notCNF = 'Agent agree if ANY of the following rules are satisfied, otherwise Agent disagree:'
        else:
            isCNF = 'Agents does NOT agree if ANY of the following rules are satisfied, otherwise Agents agree:'
            notCNF = 'Agent disagree if ANY of the following rules are satisfied, otherwise Agent agree:'
    elif 'agreement' not in target:
        action_ = action.split('_')[1]
        isCNF = f'Agents does NOT choose {action_} if ANY of the following rules are satisfied, otherwise Agents chooses {action_}:'
        notCNF = f'Agent chooses {action_} if ANY of the following rules are satisfied, otherwise Agent does NOT choose {action_}:'

    print(isCNF if explanations['isCNF'] else notCNF)
    for rule in explanations['rules']:
        print(f'  - {rule}')


def summarise_rules(data, features, target, explanations):
    target_value = 1  # always positive value for target
    constructed_rules = construct_rules(explainer_model=explanations)
    if not constructed_rules or len(constructed_rules[0]) == 0:
        print('no rules')
        return
    summary_df = create_summary_df(df=data, constructed_rules=constructed_rules,
                                   features=features, target_feature=target, target=target_value)
    print(summary_df)
    return summary_df


def create_training_dataframes(df, feature_type='state', target='baseline_action', action='turn_left',
                               stability_filter=False,features=FEATURES,categ_col=CATEG_COL,method = 'single_type'):

    # filter dataframe for selected action type i.e. turn or forward
    if action:
        action_split = action.split('_')
        action_ = action_split[0]
        # specific action specified 
        if method == 'single_type':
            print(action_)
            target_values = ACTIONS[action_]
            df = df[df['baseline_action'].isin(target_values)]
        
    # Filter for stable actions only, i.e. actions baseline model is confident about
    if stability_filter:
        df = create_stability_features(df, stability_threshold=0.1)
        stability_check = 'stable_baseline' if 'agreement' not in action else 'stable'
        df = df[df[stability_check] == 1]
        if len(df) < 100:
            print('Data is too unstable, skipping rules model')
            return

    #df = create_lagged_features(df)

    # if target is agreement/disagreement between action of two models drop out action
    # that the agreement/disagreement is not about
    df = create_agreement_targets(df)

    # make action positive case in target if not agreement target
    if 'agreement' not in target and method != 'dual_type':
        df[target] = df[target].apply(
            lambda x: 1 if x == ACTIONS[action.split('_')[1]] else 0)
    if 'agreement' not in target and method == 'dual_type':
            df[target] = df[target].apply(
            lambda x: 1 if x in ACTIONS[action.split('_')[0]] else 0)

    features = features[feature_type]
    categ_col = categ_col[feature_type]

    # recheck size of data
    if len(df) < 100:
        print('Data is too unstable, skipping rules model')
        return

    x_train, x_test, y_train, y_test = create_train_test_datasets(
        df, features, target)

    # check if rules can even be made by checking if more than one unique class in target
    if len(y_train.unique()) == 1:
        return
    classes = list(df[target].unique())
    
    sample_size_per_class = y_train.value_counts()
    d = {k:sample_size_per_class[k] for k in classes}
    max_val = np.max(list(d.values()))
    sample_size_per_class = {k: np.min([int(np.floor(v+(v*0.30))),max_val]) for k,v in d.items()}
    x_train[target] = y_train
    x_train = pd.concat([sampler(x_train, target, lbl, sample_size_per_class[lbl])
                         for lbl in classes], ignore_index=True)
    y_train = x_train[target]
    x_train.drop(columns=target, inplace=True)
    # binarise features into classes within features
    data_dict = {'x_train': x_train, 'x_test': x_test,
                 'y_train': y_train, 'y_test': y_test}
    x_train_fb, x_test_fb, y_train_code, y_test_code = create_binary_features(
        data_dict, colCateg=categ_col)
    return df,data_dict, x_train_fb, x_test_fb, y_train_code, y_test_code


def train_rules_model(df, feature_type='state', target='baseline_action', action='turn_left',stability_filter=False,features=FEATURES,categ_col=CATEG_COL,method='single_type'):

    df,_, x_train_fb,x_test_fb, y_train_code, y_test_code = create_training_dataframes(df,feature_type=feature_type,target=target,action=action,
    stability_filter=stability_filter,features=features,categ_col=categ_col,method=method)
    boolean_model = BooleanRuleCG(
        silent=False, lambda0=0.001, lambda1=0.001, CNF=False, iterMax=1000)
    explainer = BRCGExplainer(boolean_model)
    explainer.fit(x_train_fb, y_train_code)
    y_pred = explainer.predict(x_test_fb)
    print(f'Accuracy {metrics.accuracy_score(y_test_code, y_pred)}')
    print(metrics.f1_score(y_test_code, y_pred))
    e = explainer.explain()
    print(str(e))
    # the labels are flipped between Conjunctive/Disjunctive normal forms
    print_rules(e, target,action)
    summary_df = summarise_rules(df,features[feature_type], target,explanations=e)
    return explainer, df, summary_df, (x_train_fb, x_test_fb, y_train_code, y_test_code)

# %%


def run_rules_models(df, action='turn_right', feature_type='state',method='single_type',df_out=None):
    print('|--------------------------------------------------|')
    print('|-------------- Baseline Rules:  ------------------|')
    print('|---- Turning Action ----|')
    explainer,_,summary_df, _ = train_rules_model(df[df['agent'] == 'baseline'], feature_type=feature_type, target='baseline_action', action=action,method=method)
    if df_out:
        summary_df.to_csv(f'{df_out}/baseline_{action}.csv')
    else:
        summary_df.to_csv(f'data/summary/baseline_{action}.csv')
    print('|--------------------------------------------------|')

    print('|-------------- Expert Rules:  ------------------|')
    print('|---- Turning Action ----|')
    explainer,_,summary_df, _ = train_rules_model(df[df['agent'] == 'expert'], feature_type=feature_type, target='baseline_action', action=action,method=method)
    if df_out:
        summary_df.to_csv(f'{df_out}/expert_{action}.csv')
    else:
        summary_df.to_csv(f'data/summary/expert_{action}.csv')
    """
    feat_dict = baseline_feat.copy()
    feat_dict = collect_features(feat_dict, expert_feat)
    print('|--------------     Agreement    ------------------|')
    print('|---- Turning Action ----|')
    train_rules_model(feat_dict, target='agreement_turn', action=action,
                      run_number=1, experiment='highway_agreement')
    print('|--------------   Disagreement   ------------------|')
    print('|---- Turning Action ----|')
    train_rules_model(feat_dict, target='disagreement_turn', action=action,
                      run_number=1, experiment='baseline_comparison')
    """


def main(train=True, eval_eps=1000, record_vid=False,feature_type='descriptive'):
    env = setup_env()
    model_config = util.load_config('./config/model/highway.json')
    baseline_model = DQN.load('./trained_models/highway_init.zip')
    expert_model = DQN.load('./trained_models/highway_expert.zip')
    baseline_feat = {}
    expert_feat = {}
    baseline_rewards = []
    expert_rewards = []
    if train:
        if record_vid:
            env = record_videos(env)
        for eval_ep in range(eval_eps):
            baseline_feat, ep_reward = collect_episode_features(env,
                                                                baseline_model=baseline_model, alternate_model=expert_model, feature_dict=baseline_feat)
            baseline_rewards.append(ep_reward)
            expert_feat, ep_reward = collect_episode_features(env,
                                                            baseline_model=expert_model, alternate_model=baseline_model, feature_dict=expert_feat)
            expert_rewards.append(ep_reward)
        data_dict = {'baseline_features': baseline_feat, 'baseline_rewards': baseline_rewards,
                    'expert_features': expert_feat, 'expert_rewards': expert_rewards}
        data = data_dict['baseline_features']
        baseline_df = pd.DataFrame.from_dict(data)
        model_name = 'baseline'
        df_name = f'{model_name}_model'
        baseline_df.to_csv(f'data/dataframes/{df_name}_2.csv')
        data = data_dict['expert_features']
        expert_df = pd.DataFrame.from_dict(data)
        model_name = 'expert'
        df_name = f'{model_name}_model'
        expert_df.to_csv(f'data/dataframes/{df_name}_2.csv')
        baseline_df['agent'] = 'baseline'
        expert_df['agent'] = 'expert'
        combined_df = baseline_df.append(expert_df)
    else:
        baseline_df = pd.read_csv('data/dataframes/baseline_model.csv')
        expert_df = pd.read_csv('data/dataframes/expert_model.csv')
        baseline_df['agent'] = 'baseline'
        expert_df['agent'] = 'expert'
        combined_df = baseline_df.append(expert_df).reset_index().drop('level_0',axis=1)


    combined_df['car_infront_dir'] = combined_df[['car_infront_speed','car_infront_dir']].apply(lambda x: 99 if (x.car_infront_speed == 99 and x.car_infront_dir == 0) else x.car_infront_dir,axis=1)
    combined_df['dir_left_car'] = combined_df[['x_left_car','dir_left_car']].apply(lambda x: 99 if (x.x_left_car == 99 and x.dir_left_car ==0) else x.dir_left_car,axis=1)
    combined_df['dir_right_car'] = combined_df[['x_right_car','dir_right_car']].apply(lambda x: 99 if (x.x_right_car == 99 and x.dir_right_car == 0) else x.dir_right_car,axis=1)
    combined_df.replace(99,np.nan,inplace=True)

    combined_df['speed_labels'] = pd.cut(combined_df['speed'],bins=4,include_lowest=True)
    combined_df['speed'] = pd.cut(combined_df['speed'],bins=4,labels=False,include_lowest=True)
    combined_df['dist_to_car_in_front_labels'] = pd.cut(combined_df['dist_to_car_in_front'],bins=5,include_lowest=True)
    combined_df['dist_to_car_in_front'] = pd.cut(combined_df['dist_to_car_in_front'],bins=5,labels=False,include_lowest=True)
    combined_df['car_infront_dir_labels'] = pd.cut(combined_df['car_infront_dir'],bins=4,include_lowest=True)
    combined_df['car_infront_dir'] = pd.cut(combined_df['car_infront_dir'],bins=4,labels=False,include_lowest=True)
    combined_df['car_infront_speed_labels'] = pd.cut(combined_df['car_infront_speed'],bins=4,include_lowest=True)
    combined_df['car_infront_speed'] = pd.cut(combined_df['car_infront_speed'],bins=4,labels=False,include_lowest=True)
    combined_df['x_left_car_labels'] = pd.cut(combined_df['x_left_car'],bins=6,include_lowest=True)
    combined_df['x_left_car'] = pd.cut(combined_df['x_left_car'],bins=6,labels=False,include_lowest=True)
    combined_df['y_left_car_labels'] = pd.cut(combined_df['y_left_car'],bins=6,include_lowest=True)
    combined_df['y_left_car'] = pd.cut(combined_df['y_left_car'],bins=6,labels=False,include_lowest=True)
    combined_df['dir_left_car_labels'] = pd.cut(combined_df['dir_left_car'],bins=4,include_lowest=True)
    combined_df['dir_left_car'] = pd.cut(combined_df['dir_left_car'],bins=4,labels=False,include_lowest=True)
    combined_df['x_right_car_labels'] = pd.cut(combined_df['x_right_car'],bins=6,include_lowest=True)
    combined_df['x_right_car'] = pd.cut(combined_df['x_right_car'],bins=6,labels=False,include_lowest=True)
    combined_df['y_right_car_labels'] = pd.cut(combined_df['y_right_car'],bins=6,include_lowest=True)
    combined_df['y_right_car'] = pd.cut(combined_df['y_right_car'],bins=6,labels=False,include_lowest=True)
    combined_df['dir_right_car_labels'] = pd.cut(combined_df['dir_right_car'],bins=4,include_lowest=True)
    combined_df['dir_right_car'] = pd.cut(combined_df['dir_right_car'],bins=4,labels=False,include_lowest=True)
    combined_df[FEATURES['descriptive']] = combined_df[FEATURES['descriptive']].fillna(99)
    for col in FEATURES['descriptive']:
        combined_df[col] = combined_df[col].astype('object')
    dones = np.where(combined_df.done == True)[0]
    start = 0 
    combined_df['episode'] = 0
    episode = 1
    for i in dones:
        combined_df['episode'].iloc[start:i+1] = episode
        start = i + 1
        episode += 1
    summary_stats = combined_df.groupby(['agent','episode'])['reward'].sum().reset_index()
    summary_stats['ep_len'] = combined_df.groupby(['agent','episode'])['reward'].count().reset_index()['reward']
    summary_stats
    over_20s = summary_stats[summary_stats['reward'] >=20]
    combined_df =  combined_df[combined_df['episode'].isin(over_20s['episode'])]
    method = 'single_type'
    print('Action is Turn right')
    run_rules_models(combined_df, feature_type=feature_type,action='turn_right',method=method)
    print('Action is Turn Left')
    run_rules_models(combined_df, feature_type=feature_type, action='turn_left',method=method)
    print('Action is Forward Idle')
    run_rules_models(combined_df, feature_type=feature_type,
                     action='forward_idle',method=method)
    print('Action is Forward Faster')
    run_rules_models(combined_df, feature_type=feature_type,
                     action='forward_faster',method=method)
    print('Action is Forward Slower')
    run_rules_models(combined_df, feature_type=feature_type,
                     action='forward_slower',method=method)
    method = 'dual_type'
    run_rules_models(combined_df, feature_type=feature_type,action='turn_right',method=method,df_out='data/summary/action_type')
    run_rules_models(combined_df, feature_type=feature_type,
                     action='forward_faster',method=method,df_out='data/summary/action_type')

if __name__ == "__main__":
    main(train=False)

# %%
