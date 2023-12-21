import random

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import mutual_info_regression as mutual_info


import utils.utils as utils
import scoring.scoring as scoring
from code.feature_selection.embedded import decision_tree_feature_importance
from code.feature_selection.filter import max_relevance_min_redundancy_filter
from code.feature_selection.wrapper import forward_search, backward_search


if __name__ == '__main__':
    # Constants
    TEST_RATIO = 0.1

    y1_data = [0.046, 0.044, 0.053, 0.046]
    y2_data = [0.056, 0.054, 0.057, 0.054]
    y3_data = [0.053, 0.049, 0.055, 0.050]

    bar_width = 0.2
    br1 = np.arange(len(y1_data))
    br2 = [x + bar_width + 0.02 for x in br1]
    br3 = [x + bar_width + 0.02 for x in br2]

    # Create labels for the data points
    x_labels = ["K Nearest Neigbors", "MLP", "Linear", "Kernel Ridge"]
    _, axis = plt.subplots(figsize=(14, 12))
    axis.bar(br1, y1_data, width=0.2, label="Test set ratio = 0.10")
    axis.bar(br2, y2_data, width=0.2, label="Test set ratio = 0.15")
    axis.bar(br3, y3_data, width=0.2, label="Test set ratio = 0.20")
    axis.set_xticks([r + bar_width + 0.02 for r in range(len(y1_data))], x_labels)
    figure = axis.get_figure()
    axis.legend()
    # increase font size
    for item in ([axis.title, axis.xaxis.label, axis.yaxis.label] +
                 axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(20)
    # increase legend font size
    legend = axis.legend()
    for label in legend.get_texts():
        label.set_fontsize(15)
    plt.tight_layout()
    plt.show()
    figure.savefig("../graphs/rmse.png", bbox_inches='tight', dpi=300)
    exit(0)


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

    # print correlation with target
    # print(training_set.corrwith(training_target))

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
    model = MLPRegressor(hidden_layer_sizes=(8, 16, 8), max_iter=256)
    # model = linear_model.LinearRegression()
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
    # model = MLPRegressor(hidden_layer_sizes=(8, 16, 8), max_iter=256)
    model = linear_model.LinearRegression()
    for feature_set in features:
        selected_features.append([(cross_val_score(model, training_set[feature_set].copy(), training_target.copy(), cv=8, scoring=scoring.rmse_score).mean(), len(feature_set))])
        for i in range(1, len(feature_set)):
            selected_features[-1].append((
                cross_val_score(model, training_set[feature_set[0:-i]].copy(), training_target.copy(), cv=8, scoring=scoring.rmse_score).mean(),
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
