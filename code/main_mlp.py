from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import pandas as pd

import code.utils.utils as utils
import scoring.scoring as scoring
import model_selection.model_selection as m_select


if __name__ == '__main__':
    # Constants
    TEST_RATIO = 0.1

    # load data and set seed in order to have reproducible results
    # np.random.seed(123)
    staging_path = "../data/staging/"
    data_file = "Xtab1_Y1_cleaned.csv"
    image_file = "MyXimg1.csv"
    cleaned_set = pd.read_csv(staging_path + data_file, sep=",", header=0)

    # split data into training, validation and test sets
    (training_set,
     test_set,
     training_target,
     test_target) = utils.split_train_validation_test(cleaned_set, TEST_RATIO, image_path=staging_path + image_file)

    # selected_features = utils.return_best_features(training_set, training_target, test_set, test_target)
    model = MLPRegressor()
    k_fold = KFold(n_splits=8)
    param_grid = {
        "hidden_layer_sizes": [(128, 256, 128), (256, 512, 256), (32, 64, 128, 64, 32), (16, 32, 64, 128, 64, 32, 16)],
        "learning_rate": ["invscaling", "adaptive"],
        "learning_rate_init": [0.01, 0.014, 0.012],
        "max_iter": [128, 256, 512]
    }
    grid = m_select.perform_grid_search(model, param_grid, scoring.rmse, k_fold, training_set, training_target, n_jobs=-1)
    print(grid.best_params_)
    model = MLPRegressor(**grid.best_params_)
    selected_features = utils.return_best_features(training_set, training_target, test_set, test_target, model)

    scoring.evaluate_feature_selection(training_set, test_set, training_target, test_target, model, selected_features)
