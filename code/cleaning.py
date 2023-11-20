# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 12:08:04 2023

@author: Alexis C. SPYROU
"""
import pandas as pd
import os

# import sklearn as sk
# import numpy as np

# 0. Define Inputs and outputs
# 0.1 Inputs
IpPath = "../data/"
Fl1_nm = "Xtab1.csv"
Fl2_nm = "Y1.csv"
Fl3_nm = "Ximg1.csv"

# 0.1 Output
OpPath = "../data/staging/"
OpFl_nm = "Xtab1_Cl.csv"

# checking if the directory demo_folder
# exist or not.
if not os.path.exists(OpPath):
    # if the demo_folder directory is not present
    # then create it.
    os.makedirs(OpPath)

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
# ?1.1.1.3 Check Error Value --> juste pour vérifier?
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

# ?1.1.2.3 Check Error Value
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

# ?1.1.3.1.3 Check Error Value
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
# 1.2.1 Function definition
# def mean_norm(df_input):
#    return df_input.apply(lambda x: (x - x.mean()) / x.std()) #, axis=0)
def mean_norm(df_input):
    def std(x, m, s):
        return (x - m) / s

    Mu = df_input.mean()
    sigma = df_input.std()
    return df_input.apply(std, m=Mu, s=sigma)


# 1.2.2 Function Application
Data_Set['Age_Std'] = mean_norm(Data_Set['age'])
Data_Set['BloodPr_Std'] = mean_norm(Data_Set['blood pressure'])
Data_Set['Cholesterol_Std'] = mean_norm(Data_Set['cholesterol'])
Data_Set['Hemoglobin_Std'] = mean_norm(Data_Set['hemoglobin'])
Data_Set['Temperature_Std'] = mean_norm(Data_Set['temperature'])
Data_Set['Testosterone_Std'] = mean_norm(Data_Set['testosterone'])
Data_Set['Weight_Std'] = mean_norm(Data_Set['weight'])

# df_mean_norm = mean_norm(df)

# 1.3 Output
FieldToKeep = ['Age_Std', 'BloodPr_Std', 'Cholesterol_Std',
               'Hemoglobin_Std', 'Temperature_Std', 'Testosterone_Std', 'Weight_Std',
               'sarsaparilla_num', 'smurfberryLiquor_num', 'smurfinDonuts_num',
               'physicalActivity_num', 'IsRhesusPositive',
               'IsBlGrp_A', 'IsBlGrp_B', 'IsBlGrp_O', 'IsBlGrp_AB'
               ]
Data_Set_Cl = Data_Set[FieldToKeep].copy()

Data_Set_Cl.to_csv(OpPath + OpFl_nm, index=False)
