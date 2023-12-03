import numpy as np


def rmse(predict, target):
    return np.sqrt(np.mean((predict - target) ** 2))
