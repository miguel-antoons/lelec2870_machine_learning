import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import utils.utils as utils


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

    train_corr = training_set.corr()
    # position the heatmp in the center of the graph
    _, axis = plt.subplots(figsize=(14, 12))
    heatmap = sns.heatmap(train_corr, annot=True, fmt=".2f", linewidths=.5, ax=axis)
    figure = heatmap.get_figure()
    figure.savefig("../graphs/heatmap.png", bbox_inches='tight', dpi=300)

    # selected_features = utils.return_best_features(training_set, training_target, test_set, test_target)

