from src.envs.custom.gridworld import Gridworld
from src.envs.custom.highway import CustomHighwayEnv
from src.envs.custom.inventory import Inventory
from src.tasks.task import Task
import numpy as np

from src.tasks.task_util import train_expert_model, train_model
from src.util import seed_everything, load_config
import argparse

from src.visualization.visualization import visualize_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    args = parser.parse_args()

    task_name = args.task

    print('Task = {}'.format(task_name))

    # Define paths
    model_path = 'trained_models/{}'.format(task_name)
    env_config_path = 'config/env/{}.json'.format(task_name)
    model_config_path = 'config/model/{}.json'.format(task_name)
    task_config_path = 'config/task/{}.json'.format(task_name)

    # Load configs
    env_config = load_config(env_config_path)
    model_config = load_config(model_config_path)
    task_config = load_config(task_config_path)

    if task_name == 'gridworld':
        env = Gridworld(env_config['time_window'], shaping=False)
    elif task_name == 'highway':
        env = CustomHighwayEnv(shaping=False, time_window=env_config['time_window'])
        env.config['right_lane_reward'] = env_config['right_lane_reward']
        env.config['lanes_count'] = env_config['lanes_count']
        env.reset()
    elif task_name == 'inventory':
        env = Inventory(time_window=env_config['time_window'], shaping=False)

    # set true reward function
    env.set_true_reward(env_config['true_reward_func'])

    eval_path = 'eval/{}/'.format(task_name)
    max_iter = 100

    # initialize starting and expert model
    init_model_path = 'trained_models/{}_init'.format(task_name)
    expert_path = 'trained_models/{}_expert'.format(task_name)
    eval_path = 'eval/{}/'.format(task_name)

    model_env = train_model(env, model_config, init_model_path, eval_path, task_config['feedback_freq'], max_iter)
    expert_model = train_expert_model(env, env_config, model_config, expert_path, eval_path, task_config['feedback_freq'], max_iter)

    probs = [0.1, 0.2, 0.5]
    seeds = [0, 1, 2, 3, 4]

    print('Running regular experiments')

    for s in seeds:
        seed_everything(s)
        task = Task(env, model_path, model_env, expert_model, task_name, max_iter, env_config, model_config, eval_path, **task_config, auto=True, seed=s)
        task.run(experiment_type='regular')

    # print('Running noisy experiments')
    # for p in probs:
    #     for s in seeds:
    #         seed_everything(s)
    #         task = Task(env, model_path, task_name, env_config, model_config, **task_config, auto=True, seed=s)
    #         task.run(noisy=True, disruptive=False,  experiment_type='noisy', prob=p)


    visualize_experiments(task_name, eval_path)


if __name__ == '__main__':
    main()