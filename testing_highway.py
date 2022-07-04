import gym
from stable_baselines3 import DQN

from src.feedback.feedback_processing import present_successful_traj
from src.visualization.visualization import visualize_feature


def main():
    env = gym.make("highway-fast-v0")
    env.config['lanes_count'] = 4

    env.reset()

    model_path = "trained_models/highway"

    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                seed=1,
                verbose=1)

    model.learn(int(2e4))
    model.save(model_path)

    model = DQN.load(model_path, env=env)

    best_traj = present_successful_traj(model, env)
    visualize_feature(best_traj, 2, plot_actions=False, title='Only environment reward function')


if __name__ == '__main__':
    main()