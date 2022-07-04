from stable_baselines3 import DQN

from src.envs.custom.inventory import Inventory


def main():
    env = Inventory(time_window=5)
    model_path = "trained_models/inventory_1.zip"

    model = DQN.load(model_path, env=env)

    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        print('Obs: {}'.format(obs))
        print('Action: {}'.format(action))
        obs, reward, done, info = env.step(action)
        print('Reward: {}'.format(reward))


if __name__ == '__main__':
    main()