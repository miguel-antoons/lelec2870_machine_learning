from sklearn import linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor

from code.cleaning.cleaning import mean_norm
import pandas as pd

from code.feature_selection.embedded import decision_tree_feature_importance
from code.feature_selection.filter import max_relevance_min_redundancy_filter
from code.feature_selection.wrapper import forward_search, backward_search
import code.scoring.scoring as scoring


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
    if image_path is not None:
        x_train, x_test = add_image_features(image_path, x_train, x_test)

    # normalize training data
    x_train['Age_Std'] = mean_norm(x_train['age'])
    x_train['BloodPr_Std'] = mean_norm(x_train['blood pressure'])
    x_train['Cholesterol_Std'] = mean_norm(x_train['cholesterol'])
    x_train['Hemoglobin_Std'] = mean_norm(x_train['hemoglobin'])
    x_train['Temperature_Std'] = mean_norm(x_train['temperature'])
    x_train['Testosterone_Std'] = mean_norm(x_train['testosterone'])
    x_train['Weight_Std'] = mean_norm(x_train['weight'])
    x_train['h1_std'] = mean_norm(x_train['h1'])
    x_train['h2_std'] = mean_norm(x_train['h2'])
    x_train['h3_std'] = mean_norm(x_train['h3'])
    x_train['h4_std'] = mean_norm(x_train['h4'])
    x_train['h5_std'] = mean_norm(x_train['h5'])
    x_train['h6_std'] = mean_norm(x_train['h6'])
    x_train['h7_std'] = mean_norm(x_train['h7'])
    x_train['h8_std'] = mean_norm(x_train['h8'])

    # normalize test data with mean and std of training data
    x_test['Age_Std'] = mean_norm(x_test['age'], x_train['age'])
    x_test['BloodPr_Std'] = mean_norm(x_test['blood pressure'], x_train['blood pressure'])
    x_test['Cholesterol_Std'] = mean_norm(x_test['cholesterol'], x_train['cholesterol'])
    x_test['Hemoglobin_Std'] = mean_norm(x_test['hemoglobin'], x_train['hemoglobin'])
    x_test['Temperature_Std'] = mean_norm(x_test['temperature'], x_train['temperature'])
    x_test['Testosterone_Std'] = mean_norm(x_test['testosterone'], x_train['testosterone'])
    x_test['Weight_Std'] = mean_norm(x_test['weight'], x_train['weight'])
    x_test['h1_std'] = mean_norm(x_test['h1'], x_train['h1'])
    x_test['h2_std'] = mean_norm(x_test['h2'], x_train['h2'])
    x_test['h3_std'] = mean_norm(x_test['h3'], x_train['h3'])
    x_test['h4_std'] = mean_norm(x_test['h4'], x_train['h4'])
    x_test['h5_std'] = mean_norm(x_test['h5'], x_train['h5'])
    x_test['h6_std'] = mean_norm(x_test['h6'], x_train['h6'])
    x_test['h7_std'] = mean_norm(x_test['h7'], x_train['h7'])
    x_test['h8_std'] = mean_norm(x_test['h8'], x_train['h8'])

    # x_train, x_validation, y_train, y_validation = train_test_split(x_train_val, y_train, test_size=validation_ratio)
    x_train = remove_unused_fields(x_train)
    # x_validation = remove_unused_fields(x_validation)
    x_test = remove_unused_fields(x_test)

    return x_train, x_test, y_train, y_test


def prepare_data(train_data, prediction_data):
    train_img_path = "../data/staging/MyXimg1.csv"
    prediction_img_path = "../data/staging/MyXimg2.csv"

    x_train = train_data.drop("target", axis=1)
    y_train = train_data["target"].copy()

    x_prediction = prediction_data

    image_features = pd.read_csv(train_img_path, sep=",", header=0)
    # add image features having the same image_filename as the training data
    x_train = x_train.merge(image_features, on="img_filename", how="left")

    image_features = pd.read_csv(prediction_img_path, sep=",", header=0)
    # add image features having the same image_filename as the training data
    x_prediction = x_prediction.merge(image_features, on="img_filename", how="left")

    # normalize all the training data
    x_train['Age_Std'] = mean_norm(x_train['age'])
    x_train['BloodPr_Std'] = mean_norm(x_train['blood pressure'])
    x_train['Cholesterol_Std'] = mean_norm(x_train['cholesterol'])
    x_train['Hemoglobin_Std'] = mean_norm(x_train['hemoglobin'])
    x_train['Temperature_Std'] = mean_norm(x_train['temperature'])
    x_train['Testosterone_Std'] = mean_norm(x_train['testosterone'])
    x_train['Weight_Std'] = mean_norm(x_train['weight'])
    x_train['h1_std'] = mean_norm(x_train['h1'])
    x_train['h2_std'] = mean_norm(x_train['h2'])
    x_train['h3_std'] = mean_norm(x_train['h3'])
    x_train['h4_std'] = mean_norm(x_train['h4'])
    x_train['h5_std'] = mean_norm(x_train['h5'])
    x_train['h6_std'] = mean_norm(x_train['h6'])
    x_train['h7_std'] = mean_norm(x_train['h7'])
    x_train['h8_std'] = mean_norm(x_train['h8'])

    # normalize the data to predict with mean and std of training data
    x_prediction['Age_Std'] = mean_norm(x_prediction['age'], x_train['age'])
    x_prediction['BloodPr_Std'] = mean_norm(x_prediction['blood pressure'], x_train['blood pressure'])
    x_prediction['Cholesterol_Std'] = mean_norm(x_prediction['cholesterol'], x_train['cholesterol'])
    x_prediction['Hemoglobin_Std'] = mean_norm(x_prediction['hemoglobin'], x_train['hemoglobin'])
    x_prediction['Temperature_Std'] = mean_norm(x_prediction['temperature'], x_train['temperature'])
    x_prediction['Testosterone_Std'] = mean_norm(x_prediction['testosterone'], x_train['testosterone'])
    x_prediction['Weight_Std'] = mean_norm(x_prediction['weight'], x_train['weight'])
    x_prediction['h1_std'] = mean_norm(x_prediction['h1'], x_train['h1'])
    x_prediction['h2_std'] = mean_norm(x_prediction['h2'], x_train['h2'])
    x_prediction['h3_std'] = mean_norm(x_prediction['h3'], x_train['h3'])
    x_prediction['h4_std'] = mean_norm(x_prediction['h4'], x_train['h4'])
    x_prediction['h5_std'] = mean_norm(x_prediction['h5'], x_train['h5'])
    x_prediction['h6_std'] = mean_norm(x_prediction['h6'], x_train['h6'])
    x_prediction['h7_std'] = mean_norm(x_prediction['h7'], x_train['h7'])
    x_prediction['h8_std'] = mean_norm(x_prediction['h8'], x_train['h8'])

    x_train = remove_unused_fields(x_train)
    x_prediction = remove_unused_fields(x_prediction)

    return x_train, y_train, x_prediction


def remove_unused_fields(data):
    """
    Remove the fields that are not used in the model
    :param data: data to clean
    :return: cleaned data
    """
    field_to_keep = [
        'Age_Std', 'BloodPr_Std', 'Cholesterol_Std', 'Hemoglobin_Std', 'Temperature_Std', 'Testosterone_Std', 'Weight_Std',
        'smurfberryLiquor_num', 'physicalActivity_num', 'h3_std', 'h5_std', 'h7_std', 'h8_std'
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
    # 2.2 Set Features
    if model is None:
        model = MLPRegressor(hidden_layer_sizes=(100, 100, 100), max_iter=256)
    features = [
        # max_relevance_min_redundancy_filter(training_set, training_target, 10),
        # forward_search(training_set, training_target, 10, model=model),
        # backward_search(training_set, training_target, 10, model=model),
        decision_tree_feature_importance(
            training_set, training_target, test_set, test_target, 10, decision_tree_depth=20
        )
    ]

    # 3.2 Cross validation
    # 3.2.1 Test Features sets
    scores = []
    model = linear_model.LinearRegression()
    # print('Cross validation Score')
    for feature_set in features:
        scores.append(
            cross_val_score(model, training_set[feature_set], training_target, cv=8, scoring=scoring.rmse_score).mean()
        )
        # print(scores[-1])

    best_score = min(scores)
    best_index = scores.index(best_score)

    # 3.2.2 Test Features of best set
    selected_features = []
    # select different number of features
    for i in range(1, len(features[best_index])):
        selected_features.append(features[best_index][0:-i])

    ScoresList5 = []
    # calculate the score for each set of features
    for feature_set in selected_features:
        # print(X_TrainVal['Target'])
        ScoresList5.append(
            cross_val_score(model, training_set[feature_set], training_target, cv=8, scoring=scoring.rmse_score).mean()
        )

    best_score = min(ScoresList5)
    best_index = ScoresList5.index(best_score)
    # print(ScoresList5)

    # return the best set of features
    return selected_features[best_index]
