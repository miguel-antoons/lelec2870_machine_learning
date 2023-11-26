"""
NOTE : a lot of the code in this file comes from the 5th TP of the ML course, which was about feature selection.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression as mutual_info


def correlation_filter(x_train, y_train, n_features):
    """
    Selects the n_features features with the highest correlation with the target variable
    :param x_train: the training data
    :param y_train: the training target results
    :param n_features: the number of features to select
    :return: the list of the n_features features with the highest correlation with the target variable
    """
    corr = x_train.corrwith(y_train)
    return list(corr.abs().sort_values(ascending=False)[:n_features].index)


def mutual_info_filter(x_train, y_train, n_features):
    """
    Selects the n_features features with the highest mutual information with the target variable
    :param x_train: the training data
    :param y_train: the training target results
    :param n_features: the number of features to select
    :return: the list of the n_features features with the highest mutual information with the target variable
    """
    mi = pd.Series(mutual_info(x_train.values, y_train.values.ravel()), index=x_train.columns)
    return list(mi.sort_values(ascending=False)[:n_features].index)


def max_relevance_min_redundancy_filter(x_train, y_train, n_features, corr_threshold=0.7):
    """
    Selects the n_features features with the highest mutual information with the target variable, while keeping the
    redundancy between the selected features below a given threshold.
    The redundancy between two features is defined as the absolute value of their correlation.
    :param x_train: the training data
    :param y_train: the training target results
    :param n_features: the number of features to select
    :param corr_threshold: the maximum correlation allowed between two selected features
    :return: a list of the n_features features with the highest mutual information with the target variable, while
             keeping the redundancy between the selected features below a given threshold
    """
    ranked_features = mutual_info_filter(x_train, y_train, x_train.columns.size)
    selected_features = []
    corr = x_train.corr()

    # iterate through the features, starting with the most relevant ones first
    for feature in ranked_features:
        n_selected = len(selected_features)
        if n_selected == n_features:
            break  # stop if we reach the desired number of features
        elif n_selected == 0:
            selected_features.append(feature)
        else:
            feature_is_redundant = False
            # iterate through the already selected features and check for redundancy
            for selected_feature in selected_features:
                if np.abs(corr[feature][selected_feature]) > corr_threshold:
                    feature_is_redundant = True
                    break
            if not feature_is_redundant:
                selected_features.append(feature)

    return selected_features
