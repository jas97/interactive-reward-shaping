from src.envs.custom.gridworld import Gridworld
from src.envs.custom.highway import CustomHighwayEnv
from src.tasks.task import Task
from src.util import seed_everything
import argparse


def main():
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    args = parser.parse_args()

    task_name = args.task

    print('Task = {}'.format(task_name))

    model_path = 'trained_models/{}'.format(task_name)

    if task_name == 'gridworld':
        feedback_freq = int(15000)
        time_window = 5
        env = Gridworld(time_window=time_window)
    if task_name == 'highway':
        feedback_freq = int(2e4)
        time_window = 5
        env = CustomHighwayEnv(time_window=5)
        env.config['lanes_count'] = 4
        env.config['right_lane_reward'] = 0
        env.reset()

    task = Task(env, model_path, time_window, feedback_freq=feedback_freq, task_name=task_name)
    task.run()


if __name__ == '__main__':
    main()