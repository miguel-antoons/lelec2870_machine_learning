# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:19:48 2023

@author: skida
"""
import random

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

import utils.utils as utils
import scoring.scoring as scoring
import pandas as pd
import numpy as np
import model_selection.model_selection as m_select


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

    selected_features = utils.return_best_features(training_set, training_target, test_set, test_target)
    model = KernelRidge()
    k_fold = KFold(n_splits=8)
    param_grid = {
        "alpha": [10, 20, 50, 100, 150],
        "kernel": ["poly"],
        "gamma": [0.001, 0.05, 0.1, 0.5, 1, 2, 5],
        "degree": [2, 3, 4, 5],
        "coef0": [10, 20, 50, 100, 150, 200, 250]
    }
    grid = m_select.perform_grid_search(model, param_grid, scoring.rmse, k_fold, training_set[selected_features], training_target, n_jobs=-1)
    print(grid.best_params_)
    model = KernelRidge(**grid.best_params_)
    # model = KernelRidge(**{'alpha': 20, 'coef0': 100, 'degree': 4, 'gamma': 0.001, 'kernel': 'poly'})

    scoring.evaluate_feature_selection(training_set, test_set, training_target, test_target, model, selected_features)

    predict = True
    # predict = False
    if predict:
        prediction_file = "Xtab2_cleaned.csv"
        cleaned_train = pd.read_csv(staging_path + data_file, sep=",", header=0)
        cleaned_predict = pd.read_csv(staging_path + prediction_file, sep=",", header=0)

        x_train, y_train, x_prediction = utils.prepare_data(cleaned_train, cleaned_predict)

        selected_features = ['BloodPr_Std', 'Weight_Std', 'h5_std', 'Cholesterol_Std', 'h7_std',
                             'physicalActivity_num']
        model = KernelRidge(**{'alpha': 20, 'coef0': 100, 'degree': 4, 'gamma': 0.001, 'kernel': 'poly'})

        model.fit(x_train[selected_features], y_train.values.ravel())

        y_prediction = model.predict(x_prediction[selected_features])

        print(y_prediction)
        # write y_prediction to csv file
        y_prediction = pd.DataFrame(y_prediction)
        y_prediction.to_csv("../data/staging/Y2_pred.csv", index=False, header=False)
        print(len(y_prediction))
