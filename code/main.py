from cleaning.cleaning import mean_norm
from sklearn.model_selection import train_test_split
from feature_selection.filter import *
from feature_selection.wrapper import *
from feature_selection.embedded import *


def split_train_validation_test(data, test_ratio, validation_ratio):
    """
    Split the data into a training set, a validation set and a test set.
    While doing all this, also normalize the data.
    :param data: data to split, note that the data can only contain numerical values
    :param test_ratio: test set size as a ratio of the total data size
    :param validation_ratio: validation set size as a ratio of the total data size
    :return: 3 X sets (training, validation, test) and 3 targets (training, validation, test)
    """
    X = data.drop("target", axis=1)
    y = data["target"].copy()
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=test_ratio)

    x_train_val['Age_Std'] = mean_norm(x_train_val['age'])
    x_train_val['BloodPr_Std'] = mean_norm(x_train_val['blood pressure'])
    x_train_val['Cholesterol_Std'] = mean_norm(x_train_val['cholesterol'])
    x_train_val['Hemoglobin_Std'] = mean_norm(x_train_val['hemoglobin'])
    x_train_val['Temperature_Std'] = mean_norm(x_train_val['temperature'])
    x_train_val['Testosterone_Std'] = mean_norm(x_train_val['testosterone'])
    x_train_val['Weight_Std'] = mean_norm(x_train_val['weight'])

    x_test['Age_Std'] = mean_norm(x_test['age'], x_train_val['age'])
    x_test['BloodPr_Std'] = mean_norm(x_test['blood pressure'], x_train_val['blood pressure'])
    x_test['Cholesterol_Std'] = mean_norm(x_test['cholesterol'], x_train_val['cholesterol'])
    x_test['Hemoglobin_Std'] = mean_norm(x_test['hemoglobin'], x_train_val['hemoglobin'])
    x_test['Temperature_Std'] = mean_norm(x_test['temperature'], x_train_val['temperature'])
    x_test['Testosterone_Std'] = mean_norm(x_test['testosterone'], x_train_val['testosterone'])
    x_test['Weight_Std'] = mean_norm(x_test['weight'], x_train_val['weight'])

    x_train, x_validation, y_train, y_validation = train_test_split(x_train_val, y_train_val, test_size=validation_ratio)
    x_train = remove_unused_fields(x_train)
    x_validation = remove_unused_fields(x_validation)
    x_test = remove_unused_fields(x_test)

    return x_train, x_validation, x_test, y_train, y_validation, y_test


def remove_unused_fields(data):
    """
    Remove the fields that are not used in the model
    :param data: data to clean
    :return: cleaned data
    """
    field_to_keep = ['Age_Std', 'BloodPr_Std', 'Cholesterol_Std',
                   'Hemoglobin_Std', 'Temperature_Std', 'Testosterone_Std', 'Weight_Std',
                   'sarsaparilla_num', 'smurfberryLiquor_num', 'smurfinDonuts_num',
                   'physicalActivity_num', 'IsRhesusPositive',
                   'IsBlGrp_A', 'IsBlGrp_B', 'IsBlGrp_O', 'IsBlGrp_AB'
                   ]

    return data[field_to_keep].copy()


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
    rmse = np.sqrt(np.mean((y_pred - y_test.values.ravel()) ** 2))

    print(f"selected features: ")
    for i, feature in enumerate(selected_features):
        print(f"- {feature}")
    print("----------------------------------")
    print(f"RMSE basic MLP: {rmse:5.3f}")


if __name__ == '__main__':
    # Constants
    TEST_RATIO = 0.1
    VALIDATION_RATIO = 0.11

    # load data and set seed in order to have reproducible results
    np.random.seed(123)
    staging_path = "../data/staging/"
    data_file = "Xtab1_Y1_cleaned.csv"
    cleaned_set = pd.read_csv(staging_path + data_file, sep=",", header=0)

    # split data into training, validation and test sets
    (training_set,
     validation_set,
     test_set,
     training_target,
     validation_target,
     test_target) = split_train_validation_test(cleaned_set, TEST_RATIO, VALIDATION_RATIO)
    print(training_target)

    # print(training_set.shape)
    # print(validation_set.shape)
    # print(test_set.shape)
    # print(training_target.shape)
    # print(validation_target.shape)
    # print(test_target.shape)

    model = MLPRegressor(hidden_layer_sizes=32, max_iter=160)

    # FILTER methods
    # selected_features = correlation_filter(training_set, training_target, 10)
    # selected_features = mutual_info_filter(training_set, training_target, 10)
    # selected_features = max_relevance_min_redundancy_filter(training_set, training_target, 10)

    # WRAPPER methods
    # selected_features = forward_search(training_set, training_target, 10)
    selected_features = backward_search(training_set, training_target, 10)

    # EMBEDDED methods
    # selected_features = decision_tree_feature_importance(
    #     training_set, training_target, validation_set, validation_target, 8, decision_tree_depth=20
    # )

    print(selected_features)
    evaluate_feature_selection(training_set, test_set, training_target, test_target, model, selected_features)
