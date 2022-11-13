from stable_baselines3 import DQN

from src.envs.custom.gridworld import Gridworld
from src.envs.custom.highway import CustomHighwayEnv
from src.envs.custom.inventory import Inventory
from src.evaluation.evaluator import Evaluator
from src.tasks.task import Task

from src.tasks.task_util import train_expert_model, train_model
from src.util import seed_everything, load_config
import argparse

from src.visualization.visualization import visualize_experiments, visualize_best_experiment, \
    visualize_best_vs_rand_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    args = parser.parse_args()

    task_name = args.task

    print('Task = {}'.format(task_name))

    # Define paths
    model_path = 'trained_models_test/{}'.format(task_name)
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
    max_iter = 30

    # initialize starting and expert.csv model
    init_model_path = 'trained_models/{}_init'.format(task_name)
    expert_path = 'trained_models/{}_expert'.format(task_name)
    eval_path = 'eval/{}/'.format(task_name)

    model_env = train_model(env, model_config, init_model_path, eval_path, task_config['feedback_freq'], max_iter)
    expert_model = train_expert_model(env, env_config, model_config, expert_path, eval_path, task_config['feedback_freq'], max_iter)

    seeds = [0, 1, 2]
    lmbdas = [0.5, 1]

    # evaluate experiments
    experiments = [('best_summary', 'expl'), ('best_summary', 'no_exp'), ('rand_summary', 'expl')]

    for sum, expl in experiments:
        for l in lmbdas:
            for s in seeds:
                print('Running experiment with summary = {}, expl = {}, lambda = {}, seed = {}'.format(sum, expl, l, s))
                seed_everything(s)

                eval_path = 'eval/{}/{}_{}/'.format(task_name, sum, expl)

                task = Task(env, model_path, model_env, expert_model, task_name, max_iter, env_config, model_config,
                            eval_path, **task_config, expl_type=expl, auto=True, seed=s)
                task.run(experiment_type='regular', lmbda=l, summary_type=sum, expl_type=expl)

    # visualizing true reward for different values of lambda
    eval_path = 'eval/{}/best_summary_expl/IRS.csv'.format(task_name)
    best_summary_path = eval_path
    rand_summary_path = 'eval/{}/rand_summary_expl/IRS.csv'.format(task_name)
    expert_path = 'eval/{}/expert.csv'.format(task_name)
    model_env_path = 'eval/{}/model_env.csv'.format(task_name)

    # visualize_best_experiment(eval_path, expert_path, model_env_path, task_name, 'IRS for different values of \u03BB for Inventory Management task')
    visualize_best_vs_rand_summary(best_summary_path, rand_summary_path, lmbdas, task_name, 'IRS for different summary types in Inventory Management task')



if __name__ == '__main__':
    main()