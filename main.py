from src.envs.custom.gridworld import Gridworld
from src.envs.custom.highway import CustomHighwayEnv
from src.envs.custom.inventory import Inventory
from src.tasks.task import Task
from src.util import seed_everything, load_config
import argparse

from src.visualization.visualization import visualize_experiments


def main():
    seed_everything()

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
        env = Inventory(**env_config)

    print('Running regular experiments')
    task = Task(env, model_path, task_name, env_config, model_config, **task_config, auto=True)
    # task.run( experiment_type='regular')

    probs = [0.2, 0.5]

    print('Running noisy experiments')
    # for p in probs:
    #     task = Task(env, model_path, task_name, env_config, model_config, **task_config, auto=True)
    #     task.run(noisy=True, disruptive=False,  experiment_type='noisy', prob=p)

    print('Running disruptive experiments')
    for p in probs:
        task = Task(env, model_path, task_name, env_config, model_config, **task_config, auto=True)
        task.run(noisy=False, disruptive=True,  experiment_type='disruptive', prob=p)

    visualize_experiments(task_name)


if __name__ == '__main__':
    main()