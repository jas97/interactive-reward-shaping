import gym
from stable_baselines3 import DQN, DDPG
import highway_env


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
                verbose=1)

    #model.learn(int(2e4))
    #model.save(model_path)

    model = DQN.load(model_path, env=env)

    for i in range(10):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            print('Obs: {}'.format(obs[0]))
            print('Action: {}'.format(action))
            obs, reward, done, info = env.step(action)
            env.render()


if __name__ == '__main__':
    main()