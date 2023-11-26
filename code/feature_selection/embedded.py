"""
NOTE : a lot of the code in this file comes from the 5th TP of the ML course, which was about feature selection.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def decision_tree_feature_importance(x_train, y_train, x_test, y_test, n_features, decision_tree_depth=10):
    """
    Selects the n_features features with the highest importance according to a decision tree regressor
    :param x_train: the training data
    :param y_train: the training target results
    :param x_test: the test data
    :param y_test: the test target results
    :param n_features: the number of features to select
    :param decision_tree_depth: the maximum depth of the decision tree
    :return: a list of the n_features features with the highest importance according to a decision tree regressor
    """
    # Train decision tree
    dt = DecisionTreeRegressor(max_depth=decision_tree_depth)
    dt.fit(x_train, y_train)

    # Evaluate model
    y_pred = dt.predict(x_test)
    rmse = np.sqrt(np.mean((y_pred - y_test.values.ravel()) ** 2))

    # Retrieve feature importances
    feature_importances = pd.Series(dt.feature_importances_, index=x_train.columns)

    print("Feature importances:")
    for feature, feature_importance in feature_importances.items():
        print(f"- {feature:20} {feature_importance:5.3f}")
    print("--------------------------------------")
    print(f"Decision tree RMSE: {rmse:5.3f}")

    # feature selection
    return list(feature_importances.sort_values(ascending=False)[:n_features].index)
