from stable_baselines3 import DQN

from src.feedback.feedback_processing import present_successful_traj, gather_feedback, \
    augment_feedback_actions, augment_feedback_diff
from src.reward_modelling.reward_model import RewardModel
from src.tasks.task_util import init_replay_buffer, check_dtype
from src.util import evaluate_policy
from src.visualization.visualization import visualize_feature


class Task:

    def __init__(self, env, model_path, time_window=5, feedback_freq=50000, task_name='gridworld'):
        self.model_path = model_path
        self.time_window = time_window
        self.feedback_freq = feedback_freq
        self.task_name = task_name

        self.env = env
        self.reward_model = RewardModel(time_window)

        # initialize buffer of the reward model
        self.reward_model.buffer.initialize(init_replay_buffer(self.env, self.time_window))

        # check the dtype of env state space
        self.datatype = check_dtype(self.env)

    def run(self):
        finished_training = False
        iteration = 1

        while not finished_training:
            print('Iteration = {}'.format(iteration))
            try:
                model = DQN.load(self.model_path, verbose=1, env=self.env)
                print('Loaded saved model')
                self.env.set_shaping(True)
                # if it's not the first iteration reward model should be used
                self.env.set_reward_model(self.reward_model)
            except FileNotFoundError:
                model = DQN('MlpPolicy',  # TODO: load config from file
                            self.env,
                            policy_kwargs=dict(net_arch=[256, 256]),
                            learning_rate=5e-4,
                            buffer_size=15000,
                            learning_starts=200,
                            batch_size=32,
                            gamma=0.8,
                            train_freq=1,
                            gradient_steps=1,
                            target_update_interval=50,
                            verbose=1)
                print('First time training the model')

            print('Training DQN for {} timesteps'.format(self.feedback_freq))
            model.learn(total_timesteps=self.feedback_freq)
            model.save(self.model_path)

            # evaluate partially trained model
            mean_rew = evaluate_policy(model, self.env)
            print('Training timesteps = {}. Mean reward = {}.'.format(self.feedback_freq * iteration, mean_rew))

            # print the best trajectories
            best_traj = present_successful_traj(model, self.env, n_traj=10)

            # visualize y feature in highway env
            title = 'Before reward shaping' if iteration == 1 else 'After reward shaping'
            visualize_feature(best_traj, 2, title)

            if iteration == 2:
                break

            # gather feedback trajectories
            feedback = gather_feedback(best_traj)
            for f in feedback:
                print('Feedback trajectory = {}. Important features = {}. '.format(f[0], f[1]))

            for feedback_type, feedback_traj, important_features, timesteps in feedback:
                # augment feedback for each trajectory
                if feedback_type == 'state_diff':
                    D = augment_feedback_diff(feedback_traj,
                                         important_features,
                                         timesteps,
                                         self.env,
                                         time_window=self.time_window,
                                         datatype=self.datatype,
                                         length=500)
                elif feedback_type == 'actions':
                    D = augment_feedback_actions(feedback_traj,
                                                 self.env,
                                                 length=2000)

                self.reward_model.update_buffer(D, feedback_type)

            # TODO: update other feedback rewards separately
            self.reward_model.update('state_diff')






