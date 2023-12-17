import numpy as np


def rmse(predict, target):
    return np.sqrt(np.mean((predict - target) ** 2))


def rmse_score(model, x, y):
    y_pred = model.predict(x)
    return rmse(y_pred, y)


def evaluate_feature_selection(x_train, x_test, y_train, y_test, mlp_model, selected_features):
    """
    Train and evaluate a model using the given features.
    The evaluation is done using the RMSE metric.
    :param x_train: training data
    :param x_test: test data
    :param y_train: training target
    :param y_test: test target
    :param mlp_model: model to train and evaluate
    :param selected_features: features to use for the training
    :return: None
    """
    # train model
    mlp_model.fit(x_train[selected_features].values, y_train.values.ravel())

    # evaluate model
    y_pred = mlp_model.predict(x_test[selected_features].values)
    error = rmse(y_pred, y_test.values.ravel())

    print(f"selected features: ")
    for i, feature in enumerate(selected_features):
        print(f"- {feature}")
    print("----------------------------------")
    print(f"RMSE basic MLP: {error:5.3f}")
