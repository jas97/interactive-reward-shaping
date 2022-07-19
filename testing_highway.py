import gym
from stable_baselines3 import DQN

from src.envs.custom.highway import CustomHighwayEnv
from src.feedback.feedback_processing import present_successful_traj
from src.util import play_episode, load_config
from src.visualization.visualization import visualize_feature


def main():
    task_name = 'highway'
    env_config_path = 'config/env/{}.json'.format(task_name)

    # Load configs
    env_config = load_config(env_config_path)
    env = CustomHighwayEnv(shaping=False, time_window=env_config['time_window'])
    env.config['right_lane_reward'] = env_config['right_lane_reward']
    env.config['lanes_count'] = env_config['lanes_count']
    env.reset()

    env.reset()

    model_path = "trained_models/highway_expert.zip"

    model = DQN.load(model_path, env=env)

    for i in range(10):
        play_episode(model, env, verbose=1)

    best_traj = present_successful_traj(model, env)
    visualize_feature(best_traj, 2, plot_actions=False, title='Only environment reward function')


if __name__ == '__main__':
    main()