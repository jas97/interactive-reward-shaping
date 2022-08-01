import copy
import os
import random

from stable_baselines3 import DQN

from src.evaluation.evaluator import Evaluator
from src.feedback.feedback_processing import present_successful_traj, gather_feedback, augment_feedback_diff, \
    generate_important_features
from src.reward_modelling.reward_model import RewardModel
from src.tasks.task_util import init_replay_buffer, check_dtype, train_expert_model, check_is_unique, train_model
from src.visualization.visualization import visualize_feature


class Task:

    def __init__(self, env, model_path, task_name, env_config, model_config, feedback_freq, auto):
        self.model_path = model_path
        self.time_window = env_config['time_window']
        self.feedback_freq = feedback_freq
        self.task_name = task_name
        self.model_config = model_config
        self.env = env
        self.auto = auto

        self.init_model_path = 'trained_models/{}_init'.format(task_name)
        self.eval_path = 'eval/{}/'.format(task_name)

        # set true reward function
        self.env.set_true_reward(env_config['true_reward_func'])

        self.expert_path = 'trained_models/{}_expert'.format(task_name)
        self.expert_model = train_expert_model(env, env_config, model_config, self.expert_path, env_config['expert_timesteps'])
        # self.init_model = train_model(env, model_config, self.init_model_path)
        expert_data = init_replay_buffer(self.env, None, self.time_window)

        self.reward_model = RewardModel(self.time_window, env_config['input_size'])

        # initialize buffer of the reward model
        self.reward_model.buffer.initialize(expert_data)

        # evaluator object
        self.evaluator = Evaluator(self.expert_model, self.feedback_freq, copy.copy(env))

        # check the dtype of env state space
        self.state_dtype, self.action_dtype = check_dtype(self.env)

        self.max_iter = 50

        self.seed = random.seed(10)

    def run(self, noisy=False, disruptive=False, experiment_type='regular', prob=0):
        finished_training = False
        iteration = 1
        reward_dict = {}
        self.evaluator.reset_reward_dict()

        while not finished_training:
            print('Iteration = {}'.format(iteration))
            try:
                model_path = self.model_path + '/{}_{}/iter_{}'.format(experiment_type, prob, iteration-1)
                model = DQN.load(model_path, verbose=1, seed=random.randint(0, 100), exploration_fraction=max(0.1, 1-(0.05*iteration)), env=self.env)
                print('Loaded saved model')

                # if it's not the first iteration reward model should be used
                self.env.set_shaping(True)
                self.env.set_reward_model(self.reward_model)

            except FileNotFoundError:
                model = DQN('MlpPolicy',
                            self.env,
                            **self.model_config)
                print('First time training the model')

            print('Training DQN for {} timesteps'.format(self.feedback_freq))

            model.learn(total_timesteps=self.feedback_freq)
            model.save(self.model_path + '/{}_{}/iter_{}'.format(experiment_type, prob, iteration))

            # print the best trajectories
            best_traj = present_successful_traj(model, self.env, n_traj=10)

            # visualize features and/or actions
            title = 'Iteration = {}'.format(iteration)
            visualize_feature(best_traj, 2, plot_actions=False, title=title)

            # gather feedback trajectories
            feedback, cont = gather_feedback(best_traj, self.time_window, self.env, disruptive, noisy, prob, auto=self.auto)

            if iteration >= self.max_iter:
                cont = False

            if not cont:
                self.reward_model.update()
                if not noisy and not disruptive:
                    title = 'regular.csv'
                else:
                    title = 'noisy_{}.csv'.format(prob) if noisy else 'disruptive_{}.csv'.format(prob)

                self.evaluator.evaluate(model, self.env, os.path.join(self.eval_path, title), write=True)
                break

            unique_feedback = []
            for feedback_type, feedback_traj, signal, important_features, timesteps in feedback:
                important_features, actions = generate_important_features(important_features, self.env.state_len, feedback_type, self.time_window, feedback_traj)
                unique = check_is_unique(unique_feedback, feedback_traj, timesteps, self.time_window, self.env, important_features)

                print('Feedback = {} Important features = {} Signal = {}'.format(feedback_traj, important_features, signal))

                if not unique:
                    continue
                else:
                    unique_feedback.append((feedback_traj, important_features, timesteps))

                # augment feedback for each trajectory
                D = augment_feedback_diff(feedback_traj,
                                          signal,
                                          copy.copy(important_features),
                                          timesteps,
                                          self.env,
                                          self.time_window,
                                          actions,
                                          datatype=(self.state_dtype, self.action_dtype),
                                          length=10000)

                # Update reward buffer with augmented data
                self.reward_model.update_buffer(D,
                                                signal,
                                                important_features,
                                                (self.state_dtype, self.action_dtype),
                                                actions)

            # Update reward model with augmented data
            self.reward_model.update()

            # evaluate different rewards
            self.evaluator.evaluate(model, self.env)

            iteration += 1

        # # visualize different rewards
        self.evaluator.visualize(iteration)








