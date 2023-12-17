from sklearn.model_selection import train_test_split
import code.scoring.scoring as scoring
from code.cleaning.cleaning import mean_norm


def split_train_validation_test(data, test_ratio):
    """
    Split the data into a training set, a validation set and a test set.
    While doing all this, also normalize the data.
    :param data: data to split, note that the data can only contain numerical values
    :param test_ratio: test set size as a ratio of the total data size
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
        'Age_Std', 'BloodPr_Std', 'Cholesterol_Std',
        'Hemoglobin_Std', 'Temperature_Std', 'Testosterone_Std', 'Weight_Std',
        'sarsaparilla_num', 'smurfberryLiquor_num', 'smurfinDonuts_num',
        'physicalActivity_num', 'IsRhesusPositive',
        'IsBlGrp_A', 'IsBlGrp_B', 'IsBlGrp_O', 'IsBlGrp_AB'
    ]

    return data[field_to_keep].copy()
