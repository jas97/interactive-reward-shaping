from stable_baselines3 import DQN

from src.envs.custom.gridworld import Gridworld
from src.feedback.feedback_processing import present_successful_traj
from src.util import play_episode
from src.visualization.visualization import visualize_feature


def main():
    env = Gridworld(time_window=5)

    env.reset()

    model_path = "trained_models/gridworld_18"

    # model = DQN('MlpPolicy', env,
    #             policy_kwargs=dict(net_arch=[256, 256]),
    #             learning_rate=5e-4,
    #             buffer_size=15000,
    #             learning_starts=200,
    #             batch_size=32,
    #             gamma=0.8,
    #             train_freq=1,
    #             gradient_steps=1,
    #             exploration_fraction=0.8,
    #             seed=1,
    #             verbose=1)
    #
    # model.learn(int(1e5))
    # model.save(model_path)

    model = DQN.load(model_path, env=env)
    model.learn(int(1e5))

    for i in range(10):
        print('EPISODE {}'.format(i))
        play_episode(model, env, verbose=1)

    # best_traj = present_successful_traj(model, env)
    # visualize_feature(best_traj, 4, plot_actions=True, title='Only environment reward function')


if __name__ == '__main__':
    main()