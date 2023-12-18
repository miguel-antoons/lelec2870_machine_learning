from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor

from cleaning.cleaning import mean_norm
import pandas as pd

from feature_selection.embedded import decision_tree_feature_importance
from feature_selection.filter import correlation_filter, mutual_info_filter, max_relevance_min_redundancy_filter
from feature_selection.wrapper import forward_search, backward_search
import scoring.scoring as scoring


def split_train_validation_test(data, test_ratio, image_path=None):
    """
    Split the data into a training set, a validation set and a test set.
    While doing all this, also normalize the data.
    :param data: data to split, note that the data can only contain numerical values
    :param test_ratio: test set size as a ratio of the total data size
    :param image_path: path to the image features file
    :return: 3 X sets (training, validation, test) and 3 targets (training, validation, test)
    """
    X = data.drop("target", axis=1)
    y = data["target"].copy()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)

    x_train['Age_Std'] = mean_norm(x_train['age'])
    x_train['BloodPr_Std'] = mean_norm(x_train['blood pressure'])
    x_train['Cholesterol_Std'] = mean_norm(x_train['cholesterol'])
    x_train['Hemoglobin_Std'] = mean_norm(x_train['hemoglobin'])
    x_train['Temperature_Std'] = mean_norm(x_train['temperature'])
    x_train['Testosterone_Std'] = mean_norm(x_train['testosterone'])
    x_train['Weight_Std'] = mean_norm(x_train['weight'])

    x_test['Age_Std'] = mean_norm(x_test['age'], x_train['age'])
    x_test['BloodPr_Std'] = mean_norm(x_test['blood pressure'], x_train['blood pressure'])
    x_test['Cholesterol_Std'] = mean_norm(x_test['cholesterol'], x_train['cholesterol'])
    x_test['Hemoglobin_Std'] = mean_norm(x_test['hemoglobin'], x_train['hemoglobin'])
    x_test['Temperature_Std'] = mean_norm(x_test['temperature'], x_train['temperature'])
    x_test['Testosterone_Std'] = mean_norm(x_test['testosterone'], x_train['testosterone'])
    x_test['Weight_Std'] = mean_norm(x_test['weight'], x_train['weight'])

    if image_path is not None:
        x_train, x_test = add_image_features(image_path, x_train, x_test)
    # x_train, x_validation, y_train, y_validation = train_test_split(x_train_val, y_train, test_size=validation_ratio)
    x_train = remove_unused_fields(x_train)
    # x_validation = remove_unused_fields(x_validation)
    x_test = remove_unused_fields(x_test)

    return x_train, x_test, y_train, y_test


def remove_unused_fields(data):
    """
    Remove the fields that are not used in the model
    :param data: data to clean
    :return: cleaned data
    """
    field_to_keep = [
        'Age_Std', 'BloodPr_Std', 'Cholesterol_Std', 'Hemoglobin_Std', 'Temperature_Std', 'Testosterone_Std',
        'Weight_Std', 'smurfberryLiquor_num', 'physicalActivity_num', 'h1', 'h2', 'h3', 'h4', 'h5', 'h7', 'h8'
    ]

    return data[field_to_keep].copy()


def add_image_features(path, training_data, test_data):
    """
    Add image features to the training and test data
    :param path: path to the image features file
    :param training_data: training data
    :param test_data: test data
    :return: training and test data with the image features
    """
    image_features = pd.read_csv(path, sep=",", header=0)

    # add image features having the same image_filename as the training data
    training_data = training_data.merge(image_features, on="img_filename", how="left")
    test_data = test_data.merge(image_features, on="img_filename", how="left")

    return training_data, test_data


def return_best_features(training_set, training_target, test_set, test_target, model=None):
    """
    Return the best features set
    :param training_set: training data
    :param training_target: training target
    :param test_set: test data
    :param test_target: test target
    :param model: model to use
    :return: best features set
    """
    n_features = training_set.shape[1]
    print(f"Number of features: {n_features}")
    # 2.2 Set Features
    if model is None:
        model = MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=200)
    features = [
        correlation_filter(training_set, training_target, 10),
        max_relevance_min_redundancy_filter(training_set, training_target, 10),
        forward_search(training_set, training_target, 10, model=model),
        backward_search(training_set, training_target, 10, model=model),
        decision_tree_feature_importance(
            training_set, training_target, test_set, test_target, 10, decision_tree_depth=20
        )
    ]

    # 3.2 Cross validation
    # 3.2.1 Test Features sets
    scores = []
    lm0 = linear_model.LinearRegression()
    # print('Cross validation Score')
    for feature_set in features:
        scores.append(
            cross_val_score(lm0, training_set[feature_set], training_target, cv=8, scoring=scoring.rmse_score).mean())
        # print(scores[-1])

    best_score = min(scores)
    best_index = scores.index(best_score)

    # 3.2.2 Test Features of best set
    selected_features = []
    # transform above code into loop
    for i in range(1, len(features[best_index])):
        selected_features.append(features[best_index][0:-i])

    ScoresList5 = []
    for feature_set in selected_features:
        # print(X_TrainVal['Target'])
        ScoresList5.append(
            cross_val_score(lm0, training_set[feature_set], training_target, cv=8, scoring=scoring.rmse_score).mean()
        )

    best_score = min(ScoresList5)
    best_index = ScoresList5.index(best_score)

    return selected_features[best_index]
