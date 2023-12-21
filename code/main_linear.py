# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:19:48 2023

@author: skida
"""
from sklearn import linear_model

import utils.utils as utils
import scoring.scoring as scoring
from feature_selection.embedded import *


if __name__ == '__main__':
    # Constants
    TEST_RATIO = 0.1

    # load data and set seed in order to have reproducible results
    np.random.seed(1234)
    staging_path = "../data/staging/"
    data_file = "Xtab1_Y1_cleaned.csv"
    image_file = "MyXimg1.csv"
    cleaned_set = pd.read_csv(staging_path + data_file, sep=",", header=0)

    # split data into training, validation and test sets
    (training_set,
     test_set,
     training_target,
     test_target) = utils.split_train_validation_test(cleaned_set, TEST_RATIO, image_path=staging_path + image_file)

    # get all the features that are important for prediction
    model = linear_model.LinearRegression()
    selected_features = utils.return_best_features(training_set, training_target, test_set, test_target, model)

    scoring.evaluate_feature_selection(training_set, test_set, training_target, test_target, model, selected_features)
