import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from src.feedback.feedback_processing import encode_trajectory


def give_rule_feedback(model_A, model_B, env):
    dataset = []
    n_ep = 1000

    for i in range(n_ep):
        obs = env.reset()

        done = False
        while not done:
            action_A, _ = model_A.predict(obs, deterministic=True)
            action_B, _ = model_B.predict(obs, deterministic=True)

            t = encode_trajectory(env.episode[-(env.time_window):], obs, env.time_window, env.time_window, env)
            record = list(t[-(env.time_window * (env.state_len+1) + 1):-2]) + [int(action_A != action_B)]

            dataset.append(record)

            obs, rew, done, _ = env.step(action_A)

    df = pd.DataFrame(dataset, columns=env.feature_names + ['diff'])

    print('Training decision tree on {} samples'.format(len(df)))

    clf = DecisionTreeClassifier(max_depth=10, min_samples_leaf=1000)

    # Train Decision Tree Classifer
    train_cols = [c for c in df.columns if c != 'diff']
    clf = clf.fit(df[train_cols], df['diff'])

    text_representation = tree.export_text(clf, feature_names=train_cols)
    print(text_representation)