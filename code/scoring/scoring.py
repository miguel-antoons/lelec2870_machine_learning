import numpy as np


def rmse(predict, target):
    return np.sqrt(np.mean((predict - target) ** 2))


def rmse_score(model, x, y):
    y_pred = model.predict(x)
    return rmse(y_pred, y)
