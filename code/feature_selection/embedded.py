import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def decision_tree_feature_importance(x_train, y_train, x_test, y_test, n_features, decision_tree_depth=10):
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
