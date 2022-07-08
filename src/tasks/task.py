import numpy as np
from stable_baselines3 import DQN

from src.feedback.feedback_processing import present_successful_traj, gather_feedback, \
    augment_feedback_actions, augment_feedback_diff, FeedbackTypes
from src.reward_modelling.reward_model import RewardModel
from src.tasks.task_util import init_replay_buffer, check_dtype
from src.util import evaluate_policy, evaluate_MO
from src.visualization.visualization import visualize_feature, visualize_rewards


class Task:

    def __init__(self, env, model_path, task_name, model_config, time_window, feedback_freq):
        self.model_path = model_path
        self.time_window = time_window
        self.feedback_freq = feedback_freq
        self.task_name = task_name
        self.model_config = model_config

        self.env = env
        self.reward_model = RewardModel(time_window)

        # initialize buffer of the reward model
        self.reward_model.buffer.initialize(*init_replay_buffer(self.env, self.time_window))

        # check the dtype of env state space
        self.datatype = check_dtype(self.env)

    def run(self):
        finished_training = False
        iteration = 1
        reward_dict = {}

        while not finished_training:
            print('Iteration = {}'.format(iteration))
            try:
                model_path = self.model_path + '_{}'.format(iteration-1)
                model = DQN.load(model_path, verbose=1, seed=1, env=self.env)
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
            model.save(self.model_path + '_{}'.format(iteration))

            # print the best trajectories
            best_traj = present_successful_traj(model, self.env, n_traj=10)

            # visualize features and/or actions
            title = 'Before reward shaping' if iteration == 1 else 'After reward shaping'
            visualize_feature(best_traj, 0, plot_actions=True, title=title)  # TODO: remove hardcoding

            if iteration == 20:  # so far for initial experiments
                break

            # gather feedback trajectories
            feedback = gather_feedback(best_traj)
            for f in feedback:
                print('Feedback trajectory = {}. Important features = {}. '.format(f[0], f[1]))

            state, action, feature = False, False, False
            for feedback_type, feedback_traj, signal, important_features, timesteps in feedback:
                # augment feedback for each trajectory
                if feedback_type == FeedbackTypes.STATE_DIFF.value:
                    state = True
                    feedback_type = FeedbackTypes.STATE_DIFF

                    D = augment_feedback_diff(feedback_traj,
                                         signal,
                                         important_features,
                                         timesteps,
                                         self.env,
                                         time_window=self.time_window,
                                         datatype=self.datatype,
                                         length=1000)
                elif feedback_type == FeedbackTypes.ACTIONS.value:
                    feedback_type = FeedbackTypes.ACTIONS
                    action = True
                    D = augment_feedback_actions(feedback_traj, signal, self.env, length=1000)

                # Update reward buffer with augmented data
                self.reward_model.update_buffer(D,
                                                signal,
                                                important_features,
                                                self.datatype,
                                                feedback_type)

            # TODO: update other feedback rewards separately
            # Update reward model with augmented data
            if state:
                self.reward_model.update(FeedbackTypes.STATE_DIFF)
            if action:
                self.reward_model.update(FeedbackTypes.ACTIONS)

            # evaluate different rewards
        #     rew_values = evaluate_MO(model, self.env, n_episodes=100)
        #     if iteration == 1:
        #         reward_dict = rew_values
        #     else:
        #         reward_dict = {rn: reward_dict[rn] + rew_values[rn] for rn in rew_values.keys()}
        #
            iteration += 1
        #
        # # visualize different rewards
        # xticks = np.arange(0, self.feedback_freq*iteration, step=self.feedback_freq)
        # visualize_rewards(reward_dict, title='Average reward objectives without reward shaping', xticks=xticks)








