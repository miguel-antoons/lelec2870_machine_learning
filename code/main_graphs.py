import random

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor

import utils.utils as utils
import scoring.scoring as scoring
from code.feature_selection.embedded import decision_tree_feature_importance
from code.feature_selection.filter import max_relevance_min_redundancy_filter
from code.feature_selection.wrapper import forward_search, backward_search


if __name__ == '__main__':
    # Constants
    TEST_RATIO = 0.1

    # load data and set seed in order to have reproducible results
    np.random.seed(1234)
    random.seed(1234)
    staging_path = "../data/staging/"
    data_file = "Xtab1_Y1_cleaned.csv"
    image_file = "MyXimg1.csv"
    cleaned_set = pd.read_csv(staging_path + data_file, sep=",", header=0)

    # split data into training, validation and test sets
    (training_set,
     test_set,
     training_target,
     test_target) = utils.split_train_validation_test(cleaned_set, TEST_RATIO, image_path=staging_path + image_file)

    train_corr = training_set.corr()
    # position the heatmp in the center of the graph
    _, axis = plt.subplots(figsize=(14, 12))
    heatmap = sns.heatmap(train_corr, annot=True, fmt=".2f", linewidths=.5, ax=axis)
    figure = heatmap.get_figure()
    figure.savefig("../graphs/heatmap.png", bbox_inches='tight', dpi=300)

    # selected_features = utils.return_best_features(training_set, training_target, test_set, test_target)

    n_features = training_set.shape[1] - 1
    # n_features = 10

    # 2.2 Set Features
    # model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=256)
    model = linear_model.LinearRegression()
    features = [
        max_relevance_min_redundancy_filter(training_set.copy(), training_target.copy(), n_features),
        forward_search(training_set.copy(), training_target.copy(), n_features, model=model),
        backward_search(training_set.copy(), training_target.copy(), n_features, model=model),
        decision_tree_feature_importance(
            training_set.copy(), training_target.copy(), test_set.copy(), test_target.copy(), n_features, decision_tree_depth=20
        )
    ]

    # 3.2 Cross validation
    # 3.2.1 Test Features sets
    # scores = []
    # # print('Cross validation Score')
    # for feature_set in features:
    #     scores.append(
    #         cross_val_score(model, training_set[feature_set], training_target, cv=8, scoring=scoring.rmse_score).mean()
    #     )
    #     # print(scores[-1])
    #
    # best_score = min(scores)
    # best_index = scores.index(best_score)
    # print(best_index)

    # 3.2.2 Test Features of best set
    selected_features = []
    # transform above code into loop
    model = linear_model.LinearRegression()
    for feature_set in features:
        selected_features.append([(cross_val_score(linear_model.LinearRegression(), training_set[feature_set].copy(), training_target.copy(), cv=8, scoring=scoring.rmse_score).mean(), len(feature_set))])
        for i in range(1, len(feature_set)):
            selected_features[-1].append((
                cross_val_score(linear_model.LinearRegression(), training_set[feature_set[0:-i]].copy(), training_target.copy(), cv=8, scoring=scoring.rmse_score).mean(),
                len(feature_set) - i
             ))

    print(selected_features)

    # plot values stored in selected_features
    _, axis = plt.subplots(figsize=(14, 12))
    for feature_set in selected_features:
        axis.plot([i[1] for i in feature_set], [i[0] for i in feature_set])
    axis.legend(["Max relevance min redundancy", "Forward search", "Backward search", "Decision tree feature importance"])
    axis.set_xlabel("Number of features")
    axis.set_ylabel("RMSE")
    figure = axis.get_figure()
    figure.savefig("../graphs/feature_selection.png", bbox_inches='tight', dpi=300)
