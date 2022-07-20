import gym
from stable_baselines3 import DQN

from src.envs.custom.highway import CustomHighwayEnv
from src.util import play_episode, load_config, evaluate_policy


def main():
    task_name = 'highway'
    env_config_path = 'config/env/{}.json'.format(task_name)
    model_config_path = 'config/model/{}.json'.format(task_name)

    # Load configs
    env_config = load_config(env_config_path)
    model_config = load_config(model_config_path)

    env = CustomHighwayEnv(shaping=False, time_window=env_config['time_window'])
    env.config['right_lane_reward'] = env_config['right_lane_reward']
    env.config['lanes_count'] = env_config['lanes_count']
    env.reset()

    env.set_true_reward(env_config['true_reward_func'])

    model = DQN('MlpPolicy', env, **model_config)

    model.learn(total_timesteps=20000)

    print('Mean reward = {}'.format(evaluate_policy(model, env)))

    env.configure(env_config['true_reward_func'])

    print('True mean reward = {}'.format(evaluate_policy(model, env)))


if __name__ == '__main__':
    main()