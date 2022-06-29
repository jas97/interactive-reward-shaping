import gym

from src.envs.custom.gridworld import Gridworld
from src.tasks.task import Task
from src.util import seed_everything
import argparse


def main():
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    args = parser.parse_args()

    task_name = args.task

    model_path = 'trained_models/{}'.format(task_name)
    feedback_freq = 50000  # training stops every 50k steps and feedback is gathered
    time_window = 5

    if task_name == 'gridworld':
        env = Gridworld(time_window=time_window)
    if task_name == 'highway':
        env = gym.make('highway-v0')
        # TODO: might need to flatten the state

    task = Task(env, model_path, time_window, feedback_freq=feedback_freq, task_name=task_name)
    task.run()


if __name__ == '__main__':
    main()