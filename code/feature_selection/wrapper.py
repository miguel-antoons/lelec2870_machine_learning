from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neural_network import MLPRegressor


def forward_search(x_train, y_train, n_features, cross_validation=2, model=None):
    if model is None:
        model = MLPRegressor(hidden_layer_sizes=16, max_iter=25)

    selector = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, direction="forward",
                                         scoring="neg_mean_squared_error", cv=cross_validation)
    selector.fit(x_train, y_train.values.ravel())
    return x_train.columns[selector.get_support()]


def backward_search(x_train, y_train, n_features, cross_validation=2, model=None):
    if model is None:
        model = MLPRegressor(hidden_layer_sizes=16, max_iter=25)

    selector = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, direction="backward",
                                         scoring="neg_mean_squared_error", cv=cross_validation)
    selector.fit(x_train, y_train.values.ravel())
    return x_train.columns[selector.get_support()]
