from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import random

import code.utils.utils as utils
import scoring.scoring as scoring
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

    # get all the features that are important for prediction
    selected_features = utils.return_best_features(training_set, training_target, test_set, test_target)
    # perform an grid search to find the best hyperparameters
    model = MLPRegressor(random_state=1234)
    k_fold = KFold(n_splits=8)
    param_grid = {
        "hidden_layer_sizes": [(4, 4), (8, 8), (4, 8, 4), (16, 16), (8, 16, 8), (8, 8, 8), (16, 16, 16), (4, 4, 4)],
        "learning_rate": ["invscaling", "adaptive"],
        "solver": ["lbfgs", "sgd", "adam"],
        "learning_rate_init": [0.01, 0.014, 0.012, 0.008],
        "max_iter": [256, 512, 750, 1024]
    }
    grid = m_select.perform_grid_search(model, param_grid, scoring.rmse, k_fold, training_set[selected_features], training_target, n_jobs=-1)
    print(grid.best_params_)
    model = MLPRegressor(**grid.best_params_, random_state=1234)
    # model = MLPRegressor(**{'hidden_layer_sizes': (4, 8, 4), 'learning_rate': 'invscaling', 'learning_rate_init': 0.01, 'max_iter': 256, 'solver': 'lbfgs'}, random_state=1234)

    scoring.evaluate_feature_selection(training_set, test_set, training_target, test_target, model, selected_features)
