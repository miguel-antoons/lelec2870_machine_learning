import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

import utils.utils as utils
import scoring.scoring as scoring
import model_selection.model_selection as m_select


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

    selected_features = utils.return_best_features(training_set, training_target, test_set, test_target, model=LinearRegression())
    # model = KNeighborsRegressor()
    # k_fold = KFold(n_splits=8)
    # param_grid = {
    #     "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19],
    #     "weights": ["uniform", "distance"],
    #     "algorithm": ["auto"],
    #     "leaf_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #     "p": [1, 2],
    #     "metric": ["minkowski", "euclidean", "manhattan"]
    # }
    # grid = m_select.perform_grid_search(
    #     model, param_grid, scoring.rmse, k_fold, training_set[selected_features], training_target, n_jobs=-1
    # )
    # print(grid.best_params_)
    # model = KNeighborsRegressor(**grid.best_params_)
    model = KNeighborsRegressor(**{
        'algorithm': 'auto', 'leaf_size': 10, 'metric': 'minkowski', 'n_neighbors': 11, 'p': 1, 'weights': 'distance'
    })
    # selected_features = utils.return_best_features(training_set, training_target, test_set, test_target, model)

    scoring.evaluate_feature_selection(training_set, test_set, training_target, test_target, model, selected_features)
