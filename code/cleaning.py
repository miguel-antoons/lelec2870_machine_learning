# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:08:04 2023

@author: Alexis C. SPYROU
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 0. Define Inputs and outputs
# 0.1 Inputs
IpPath = "../data/"
Fl1_nm = "Xtab1.csv"
Fl2_nm = "Y1.csv"
Fl3_nm = "Ximg1.csv"

# 0.1 Output
OpPath = "../data/staging/"
OpFl1_nm = "Xt1_train_Cl.csv"
OpFl2_nm = "Xt1_Val_Cl.csv"
OpFl3_nm = "Xt1_Test_Cl.csv"

# create opPath if not exists
if not os.path.exists(OpPath):
    os.makedirs(OpPath)

# 0.2 Parameters
MySeed = 123
TestPartion = 0.10
ValPartition = 0.11

# Import Files
Data_Set = pd.read_csv(IpPath + Fl1_nm, sep=",", header=0)
Data_Target = pd.read_csv(IpPath + Fl2_nm, sep=",", header=0)
Data_Img = pd.read_csv(IpPath + Fl3_nm, sep=",", header=0)


# 1. Cleaning
# 1.1 Transform Txt fields
# 1.1.1 Case Very low --> Very High
# 1.1.1.1 Function definition
def TurnOrderToNum(MyField):
    match MyField:
        case "Very low":
            return -2
        case "Low":
            return -1
        case "Moderate":
            return 0
        case "High":
            return 1
        case "Very high":
            return 2
        case _:
            return -99


# 1.1.1.2 Function Application
Data_Set['sarsaparilla_num'] = Data_Set['sarsaparilla'].apply(TurnOrderToNum)
Data_Set['smurfberryLiquor_num'] = Data_Set['smurfberry liquor'].apply(TurnOrderToNum)
Data_Set['smurfinDonuts_num'] = Data_Set['smurfin donuts'].apply(TurnOrderToNum)
# 1.1.1.3 Check Error Value
if Data_Set['sarsaparilla_num'].min() == -99:
    print('Error sarsaparilla_num')
if Data_Set['smurfberryLiquor_num'].min == -99:
    print('Error smurfberryLiquor_num')
if Data_Set['smurfinDonuts_num'].min == -99:
    print('Error smurfinDonuts_num')


# 1.1.2 Case Yes/No
# 1.1.2.1 Function definition
def YesNoToNum(MyField):
    if MyField == "Yes":
        return 1
    elif MyField == "No":
        return 0
    else:
        return -99


# 1.1.2.2 Function Application
Data_Set['physicalActivity_num'] = Data_Set['physical activity'].apply(YesNoToNum)

# 1.1.2.3 Check Error Value
if Data_Set['physicalActivity_num'].min == -99:
    print('Error physicalActivity_num')


# 1.1.3 Case Blood Type
# 1.1.3.1 Rhesus
# 1.1.3.1.1 Function definition
def IsResPositive(BloodTp):
    Rhesus = BloodTp[-1:]
    if Rhesus == "+":
        return 1
    elif Rhesus == "-":
        return 0
    else:
        return -99


# 1.1.3.1.2 Function Application
Data_Set['IsRhesusPositive'] = Data_Set['blood type'].apply(IsResPositive)

# 1.1.3.1.3 Check Error Value
if Data_Set['IsRhesusPositive'].min == -99:
    print('Error IsRhesusPositive')


# 1.1.3.2 Blood Group
# 1.1.3.2.1 Function definition
def IsInBloodGrp(MyValue, BloodTp):
    MyValue2 = MyValue[:-1]
    if MyValue2 == BloodTp:
        return 1
    else:
        return 0


# 1.1.3.2.2 Function Application
Data_Set['IsBlGrp_A'] = Data_Set['blood type'].apply(IsInBloodGrp, BloodTp="A")
Data_Set['IsBlGrp_B'] = Data_Set['blood type'].apply(IsInBloodGrp, BloodTp="B")
Data_Set['IsBlGrp_O'] = Data_Set['blood type'].apply(IsInBloodGrp, BloodTp="O")
Data_Set['IsBlGrp_AB'] = Data_Set['blood type'].apply(IsInBloodGrp, BloodTp="AB")
Data_Set['Check_BldTp'] = Data_Set['IsBlGrp_A'] + Data_Set['IsBlGrp_B'] + Data_Set['IsBlGrp_O'] + Data_Set['IsBlGrp_AB']

# 1.1.3.2.3 Check Error Value
if Data_Set['Check_BldTp'].min() < 1:
    print('Blood Group error I : Some observations without Blood Group')
if Data_Set['Check_BldTp'].max() > 1:
    print('Blood Group error II : Some observations with more than 1 Blood Group')

# 1.2 Standardize fields
# 1.2.0 Split Train and Test
np.random.seed(MySeed)
X_trainVal, X_test = train_test_split(Data_Set, test_size=TestPartion)


# 1.2.1 Function definition
# def mean_norm(df_input):
#    return df_input.apply(lambda x: (x - x.mean()) / x.std()) #, axis=0)
def mean_norm(df_input):
    def std(x, m, s):
        return (x - m) / s

    Mu = df_input.mean()
    sigma = df_input.std()
    return df_input.apply(std, m=Mu, s=sigma)


def mean_norm2(df_input, df_ref):
    def std(x, m, s):
        return (x - m) / s

    Mu = df_ref.mean()
    sigma = df_ref.std()
    return df_input.apply(std, m=Mu, s=sigma)


# 1.2.2 Function Application
X_trainVal['Age_Std'] = mean_norm(X_trainVal['age'])
X_trainVal['BloodPr_Std'] = mean_norm(X_trainVal['blood pressure'])
X_trainVal['Cholesterol_Std'] = mean_norm(X_trainVal['cholesterol'])
X_trainVal['Hemoglobin_Std'] = mean_norm(X_trainVal['hemoglobin'])
X_trainVal['Temperature_Std'] = mean_norm(X_trainVal['temperature'])
X_trainVal['Testosterone_Std'] = mean_norm(X_trainVal['testosterone'])
X_trainVal['Weight_Std'] = mean_norm(X_trainVal['weight'])

X_test['Age_Std'] = mean_norm2(X_test['age'], X_trainVal['age'])
X_test['BloodPr_Std'] = mean_norm2(X_test['blood pressure'], X_trainVal['blood pressure'])
X_test['Cholesterol_Std'] = mean_norm2(X_test['cholesterol'], X_trainVal['cholesterol'])
X_test['Hemoglobin_Std'] = mean_norm2(X_test['hemoglobin'], X_trainVal['hemoglobin'])
X_test['Temperature_Std'] = mean_norm2(X_test['temperature'], X_trainVal['temperature'])
X_test['Testosterone_Std'] = mean_norm2(X_test['testosterone'], X_trainVal['testosterone'])
X_test['Weight_Std'] = mean_norm2(X_test['weight'], X_trainVal['weight'])

# 1.3 Output
# 1.3.1 Creation
FieldToKeep = ['Age_Std', 'BloodPr_Std', 'Cholesterol_Std',
               'Hemoglobin_Std', 'Temperature_Std', 'Testosterone_Std', 'Weight_Std',
               'sarsaparilla_num', 'smurfberryLiquor_num', 'smurfinDonuts_num',
               'physicalActivity_num', 'IsRhesusPositive',
               'IsBlGrp_A', 'IsBlGrp_B', 'IsBlGrp_O', 'IsBlGrp_AB'
               ]

X_trainVal_Cl = X_trainVal[FieldToKeep].copy()

X_test_Cl = X_test[FieldToKeep].copy()
X_train_Cl, X_Val_Cl = train_test_split(X_trainVal_Cl, test_size=ValPartition)

# 1.3.2 Export
X_train_Cl.to_csv(OpPath + OpFl1_nm, index=False)
X_Val_Cl.to_csv(OpPath + OpFl2_nm, index=False)
X_test_Cl.to_csv(OpPath + OpFl3_nm, index=False)