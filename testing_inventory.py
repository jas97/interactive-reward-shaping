import json
from csv import writer

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src.envs.custom.inventory import Inventory
from src.util import evaluate_MO
from src.visualization.visualization import visualize_rewards

class MOCallback(EvalCallback):

    def __init__(self, feedback_freq, eval_env, verbose=1):
        super().__init__(eval_env, eval_freq=feedback_freq)

        self.eval_env = eval_env
        self.feedback_freq = feedback_freq
        self.iter = 0
        self.n_calls = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.feedback_freq == 0:
            rew_vals = evaluate_MO(self.model, self.eval_env, n_episodes=100).items()

            names = [rv[0] for rv in rew_vals]
            values = [rv[1] for rv in rew_vals]

            print('Trained for {} timesteps. Rewards = {}'.format(self.n_calls, rew_vals))

            # write reward_dict to csv file
            if self.iter == 0:
                with open('rewards.csv', 'a', newline='') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(names)
                    writer_object.writerow(values)
                    f_object.close()
            else:
                with open('rewards.csv', 'a', newline='') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(values)
                    f_object.close()

            self.iter += 1

        return True


def main():
    env = Inventory(time_window=5)
    model_path = "trained_models/inventory"
    total_timesteps = int(1e5)
    feedback_freq = 1000

    callback = MOCallback(feedback_freq, env)

    model = DQN('MlpPolicy',
                env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                exploration_fraction=0.5,
                seed=1,
                verbose=0)

    model.learn(total_timesteps, callback=callback)
    model.save(model_path)

    xticks = np.arange(0, total_timesteps, step=feedback_freq)

    reward_df = pd.read_csv('rewards.csv')
    reward_dict = reward_df.to_dict('list')
    visualize_rewards(reward_dict, title='Average reward objectives without reward shaping', xticks=xticks)


if __name__ == '__main__':
    main()