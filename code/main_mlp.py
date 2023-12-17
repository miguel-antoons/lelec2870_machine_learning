from sklearn.model_selection import KFold

import code.utils.utils as utils
from feature_selection.filter import *
from feature_selection.wrapper import *
from feature_selection.embedded import *
import scoring.scoring as scoring
import model_selection.model_selection as m_select


if __name__ == '__main__':
    # Constants
    TEST_RATIO = 0.1

    # load data and set seed in order to have reproducible results
    np.random.seed(123)
    staging_path = "../data/staging/"
    data_file = "Xtab1_Y1_cleaned.csv"
    cleaned_set = pd.read_csv(staging_path + data_file, sep=",", header=0)

    # split data into training, validation and test sets
    (training_set,
     test_set,
     training_target,
     test_target) = utils.split_train_validation_test(cleaned_set, TEST_RATIO)

    # FILTER methods
    # selected_features = correlation_filter(training_set, training_target, 10)
    # selected_features = mutual_info_filter(training_set, training_target, 10)
    # selected_features = max_relevance_min_redundancy_filter(training_set, training_target, 10)

    # WRAPPER methods
    # model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=128)
    # selected_features = forward_search(training_set, training_target, 10, model=model)
    # selected_features = backward_search(training_set, training_target, 10, model=model)

    # EMBEDDED methods
    selected_features = decision_tree_feature_importance(
        training_set, training_target, test_set, test_target, 8, decision_tree_depth=20
    )

    model = MLPRegressor()
    k_fold = KFold(n_splits=10, shuffle=True, random_state=123)
    param_grid = {
        "hidden_layer_sizes": [(128, 256, 128), (128, 128), (32, 32, 32), (32, 64, 128, 64, 32), (16, 32, 64, 128, 64, 32, 16), (32, 64, 128, 256, 128, 64, 32)],
        "learning_rate": ["constant", "invscaling", "adaptive"],
        "learning_rate_init": [0.01, 0.014, 0.008, 0.012],
        "max_iter": [64, 128, 256, 512]
    }
    grid = m_select.perform_grid_search(model, param_grid, scoring.rmse, k_fold, training_set[selected_features], training_target, n_jobs=-1)
    print(grid.best_params_)
    model = MLPRegressor(**grid.best_params_)

    print(selected_features)
    scoring.evaluate_feature_selection(training_set, test_set, training_target, test_target, model, selected_features)
