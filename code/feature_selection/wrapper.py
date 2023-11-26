"""
NOTE : a lot of the code in this file comes from the 5th TP of the ML course, which was about feature selection.
"""
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPRegressor


def forward_search(x_train, y_train, n_features, cross_validation=2, model=None):
    """
    Find the n_features features that minimize the mean squared error of the model
    using forward search.
    :param x_train: training data
    :param y_train: training target
    :param n_features: number of features to select
    :param cross_validation: number of folds to use for cross validation
    :param model: the model to use for the feature selection
    :return: the n_features features that minimize the mean squared error of the model according to forward search
    """
    if model is None:
        model = MLPRegressor(hidden_layer_sizes=16, max_iter=25)

    selector = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, direction="forward",
                                         scoring="neg_mean_squared_error", cv=cross_validation)
    selector.fit(x_train, y_train.values.ravel())
    return x_train.columns[selector.get_support()]


def backward_search(x_train, y_train, n_features, cross_validation=2, model=None):
    """
    Find the n_features features that minimize the mean squared error of the model
    using backward search.
    :param x_train: training data
    :param y_train: training target
    :param n_features: number of features to select
    :param cross_validation: number of folds to use for cross validation
    :param model: the model to use for the feature selection
    :return: the n_features features that minimize the mean squared error of the model according to backward search
    """
    if model is None:
        model = MLPRegressor(hidden_layer_sizes=16, max_iter=25)

    selector = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, direction="backward",
                                         scoring="neg_mean_squared_error", cv=cross_validation)
    selector.fit(x_train, y_train.values.ravel())
    return x_train.columns[selector.get_support()]
