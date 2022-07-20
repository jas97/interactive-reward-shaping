from matplotlib import pyplot as plt


class TaskManager:

    def __init__(self, task):
        self.task = task

        self.probs = [0.01, 0.05, 0.1, 0.2]

    def run(self):
        noisy_rewards = []
        for p in self.probs:
            self.task.run(noisy=True, prob=p)
            rewards = self.task.evaluator.get_rewards_dict()
            rew_names = rewards.keys()
            noisy_rewards.append(rewards)

        for rn in rew_names:
            for i, p in enumerate(noisy_rewards):
                plt.plot(noisy_rewards[i][rn], label='p = {}'.format(p))

            plt.title('Noisy feedback, objective = {}'.format(rn))
            plt.show()

        disruptive_rewards = []
        for p in self.probs:
            self.task.run(disruptive=True, prob=p)
            rewards = self.task.evaluator.get_rewards_dict()
            disruptive_rewards.append(rewards)

        for rn in rew_names:
            for i, p in enumerate(disruptive_rewards):
                plt.plot(disruptive_rewards[i][rn], label='p = {}'.format(p))

            plt.title('Disruptive feedback, objective = {}'.format(rn))
            plt.show()