import os
import seaborn as sns
import pandas
import pandas as pd
from matplotlib import pyplot as plt


def visualize_experiments(task):
    path = 'eval/{}/'.format(task)
    dfs = []
    experiment_names = []

    for file_name in os.listdir(path):
        file = os.path.join(path, file_name)
        df = pd.read_csv(file)

        dfs.append(df)
        experiment_names.append(file_name.split('.csv')[0])

        col_names = df.columns

    for metric in col_names:
        for i, df in enumerate(dfs):
            sns.lineplot(data=df, x="iter", y=metric, label=experiment_names[i])

    plt.legend()
    plt.show()

def visualize_feature(traj, feature_id, plot_actions=False, title=''):
    feature_vals = []
    for t in traj:
        ep_vals = [p[0].flatten()[feature_id] for p in t]
        feature_vals.append(ep_vals)

    for f_vals in feature_vals:
        plt.plot(f_vals)

    plt.title(title)
    plt.xlabel('Time step')
    plt.ylabel('Agent\'s lane')

    plt.show()

    if plot_actions:
        actions = []
        for t in traj:
            ep_actions = [p[1] for p in t]
            actions.append(ep_actions)

        for a_vals in actions:
            plt.plot(a_vals)

        plt.title('Action distribution through an episode across successful trajectories')
        plt.xlabel('Time step')
        plt.ylabel('Action')
        plt.show()


def visualize_rewards(rew_dict, title='', xticks=None):
    for rew_name, rew_values in rew_dict.items():
        plt.plot(rew_values)

        plt.title(rew_name)

        if xticks is not None:
            plt.xticks = xticks

        plt.xlabel('Time steps')
        plt.ylabel('Average reward')

        plt.show()