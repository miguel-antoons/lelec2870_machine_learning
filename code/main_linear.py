# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:19:48 2023

@author: skida
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import utils.utils as utils
import scoring.scoring as scoring


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

    # 2.2 Set Features
    Feat1 = ['Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std',
             'smurfinDonuts_num', 'Age_Std', 'sarsaparilla_num', 'smurfberryLiquor_num', 'Temperature_Std']
    Feat2 = ['physicalActivity_num', 'Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'sarsaparilla_num',
             'smurfinDonuts_num', 'Age_Std', 'Testosterone_Std', 'IsBlGrp_O', 'smurfberryLiquor_num']
    Feat3 = ['physicalActivity_num', 'Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'smurfberryLiquor_num', 'Age_Std',
             'Testosterone_Std', 'sarsaparilla_num', 'IsRhesusPositive', 'IsBlGrp_B']
    Feat4 = ['Age_Std', 'BloodPr_Std', 'Hemoglobin_Std', 'Testosterone_Std', 'physicalActivity_num', 'IsRhesusPositive',
             'IsBlGrp_A', 'IsBlGrp_B', 'IsBlGrp_O', 'IsBlGrp_AB']
    Feat5 = ['Age_Std', 'Cholesterol_Std', 'Temperature_Std', 'Testosterone_Std', 'Weight_Std', 'smurfberryLiquor_num',
             'IsBlGrp_A', 'IsBlGrp_B', 'IsBlGrp_O', 'IsBlGrp_AB']
    Feat6 = ['Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'Age_Std', 'physicalActivity_num', 'Testosterone_Std',
             'Temperature_Std', 'Hemoglobin_Std']

    # 3.2 Cross validation
    # 3.2.1 Test Features sets
    lm0 = linear_model.LinearRegression()
    Scores1 = cross_val_score(lm0, training_set[Feat1], training_target, cv=10, scoring=scoring.rmse_score).mean()
    Scores2 = cross_val_score(lm0, training_set[Feat2], training_target, cv=10, scoring=scoring.rmse_score).mean()
    Scores3 = cross_val_score(lm0, training_set[Feat3], training_target, cv=10, scoring=scoring.rmse_score).mean()
    Scores4 = cross_val_score(lm0, training_set[Feat4], training_target, cv=10, scoring=scoring.rmse_score).mean()
    Scores5 = cross_val_score(lm0, training_set[Feat5], training_target, cv=10, scoring=scoring.rmse_score).mean()
    Scores6 = cross_val_score(lm0, training_set[Feat6], training_target, cv=10, scoring=scoring.rmse_score).mean()

    print('Cross validation Score')
    print(Scores1)
    print(Scores2)
    print(Scores3)
    print(Scores4)
    print(Scores5)
    print(Scores6)

    # --> Best Score for Feat1
    # 3.2.2 Test Features of best set
    MyFeat = []
    MyFeat.append([
        'Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std',
        'smurfinDonuts_num', 'Age_Std', 'sarsaparilla_num', 'smurfberryLiquor_num', 'Temperature_Std'
    ])
    MyFeat.append([
        'Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std',
        'smurfinDonuts_num', 'Age_Std', 'sarsaparilla_num', 'smurfberryLiquor_num'
    ])
    MyFeat.append([
        'Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std',
        'smurfinDonuts_num', 'Age_Std', 'sarsaparilla_num'
    ])
    MyFeat.append([
        'Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std',
        'smurfinDonuts_num', 'Age_Std'
    ])
    MyFeat.append([
        'Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std',
        'smurfinDonuts_num'
    ])
    MyFeat.append(['Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std'])
    MyFeat.append(['Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num'])
    MyFeat.append(['Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', ])
    MyFeat.append(['Weight_Std', 'BloodPr_Std'])
    MyFeat.append(['Weight_Std'])

    ScoresList5 = []
    for feature_set in MyFeat:
        # print(X_TrainVal['Target'])
        ScoresList5.append(cross_val_score(lm0, training_set[feature_set], training_target, cv=5, scoring=scoring.rmse_score).mean())

    print('Cross validation Score by Features')
    print(ScoresList5)

    model = linear_model.LinearRegression()
    utils.evaluate_feature_selection(training_set, test_set, training_target, test_target, model, MyFeat[1])

    # --> Best Features selection : MyFeat[1] = ['Weight_Std', 'BloodPr_Std', 'Cholesterol_Std', 'physicalActivity_num', 'Testosterone_Std', 'smurfinDonuts_num', 'Age_Std', 'sarsaparilla_num', 'smurfberryLiquor_num']