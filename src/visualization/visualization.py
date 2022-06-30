from matplotlib import pyplot as plt

def visualize_feature(traj, feature_id, title=''):
    feature_vals = []
    for t in traj:
        ep_vals = [p[0].flatten()[feature_id] for p in t]
        feature_vals.append(ep_vals)

    for f_vals in feature_vals:
        plt.plot(f_vals)

    plt.title(title)

    plt.show()
