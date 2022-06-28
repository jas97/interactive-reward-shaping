import gym
from stable_baselines3 import DQN
import highway_env


def main():
    env = gym.make("highway-v0")
    change_pen = 0
    env.config['lane_change_reward'] = change_pen
    env.reset()

    model_path = "trained_models/highway_{}".format(change_pen)

    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1)

    #model.learn(int(2e4))
    #model.save(model_path)

    model = DQN.load(model_path, env=env)

    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        print('Obs: {}'.format(obs))
        print('Action: {}'.format(action))
        obs, reward, done, info = env.step(action)
        env.render()


if __name__ == '__main__':
    main()