from data_analysis import *
from get_data import TRAIN_DATA_A
from get_data import TRAIN_DATA_B
from sklearn.impute import SimpleImputer

import pandas as pd
import random as r
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

#THIS FILE IS FILLED WITH DIFFERENT TYPES OF IMPUTATION TECHNIQUE FUCNTIONS
# THE MAIN IMPUTING FUNCTION IS THE COMPLEX IMPUTER

# impute the origninal data set to fill in some of the blank datapoints WITH SEPSIS
def imputeData(DATA):

    columns = list(keyColumns(DATA).index.values)
    # columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'DBP']
    columns.append('SepsisLabel')
    df = DATA[columns]

    print("Imputing data...\n")

    # impute the temperature with a range

    df['Temp'] = df['Temp'].fillna(0.0)
    df = df.sort_values(by=['Temp']).reset_index(drop=True)
    index_of_nan = len(df.index[df['Temp'] == 0.0].tolist())
    df_temp = df.iloc[:index_of_nan-1,:]
    length_of_df = df_temp['SepsisLabel'].size

    tempFrame = pd.DataFrame(np.random.rand(length_of_df, 1) + 36.5)
    df_temp = df_temp.assign(Temp=tempFrame[0])
    df = df.iloc[index_of_nan-1:,:]
    df = pd.concat([df, df_temp], ignore_index=True)

    # impute the blood pressure with a range

    df['DBP'] = df['DBP'].fillna(0.0)
    df = df.sort_values(by=['DBP']).reset_index(drop=True)
    index_of_nan = len(df.index[df['DBP'] == 0.0].tolist())
    df_dbp = df.iloc[:index_of_nan-1,:]
    length_of_df = df_dbp['SepsisLabel'].size

    dbpFrame = pd.DataFrame(np.random.rand(length_of_df, 1) * 10 + 90)
    df_dbp = df_dbp.assign(DBP=dbpFrame[0])
    df = df.iloc[index_of_nan-1:,:]
    df = pd.concat([df, df_dbp], ignore_index=True)

    df = df.dropna().reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

# using scikit learns imputing functions
def imputeSimpleImputer(DATA):
    columns = list(keyColumns(DATA).index.values)
    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP',
    'Resp', 'Glucose', 'Potassium', 'WBC', 'Hct', 'Hgb', 'HCO3', 'Age',
    'Creatinine', 'Platelets',
    'SepsisLabel']


    print(len(columns))
    df = DATA[columns]

    imputer = SimpleImputer(strategy='median')
    imputer.fit(df)
    df_trans = pd.DataFrame(imputer.transform(df))
    df_trans.columns = columns
    df = cleanDataset(df_trans)
    # df = df.loc[(df['Age'] > 80) & (df['Age'] < 90)]
    return df

# MAIN IMPUTER USED
def complexImputer(DATA):
    # get feature column names
    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP',
    'Resp', 'Glucose', 'Potassium', 'WBC', 'Hct', 'Hgb', 'HCO3', 'Age',
    'Creatinine', 'Platelets', 'ICULOS', 'Gender',
    'SepsisLabel']
    df = DATA[columns]
    print(df)


    # if my file doesn't exist containing the patients id, create it
    # this is just saving time in future runs

    my_file = Path("../medical data/imputed_dataset.pkl")
    if not my_file.is_file():

        df = addPatientID(df)

        df.to_pickle("../medical data/imputed_dataset.pkl")


    df = pd.read_pickle("../medical data/imputed_dataset.pkl")


    #impute the last observation carried forward upto 24 hours

    df = df.groupby("PatientID").fillna(method='ffill', inplace=False, limit=24)
    df = df.fillna(df.mean())


    # again saving file to help speed up runtime
    my_file = Path("../medical data/final_imputed_dataset.pkl")
    if not my_file.is_file():
        df = addPatientID(df)
        df.to_pickle("../medical data/final_imputed_dataset.pkl")
    # df = df.drop(columns=['ICULOS'])

    df = pd.read_pickle("../medical data/final_imputed_dataset.pkl")


    #
    pat_df = df.loc[df['SepsisLabel'] == 1]
    list_of_indexs = list(pat_df.index.values)
    num_of_patients_with_sep = len(list(set(pat_df['PatientID'].tolist())))

    # get the indecies where the patient is about to septic
    key_indices = []
    start = True
    for index, item in enumerate(list_of_indexs):
        try:
            if start == True:
                key_indices.append(item)
                start = False
            if item != list_of_indexs[index+1] - 1:
                start = True

        except:
            pass

    # fill a dataframe with septic data from the key indices used.
    time_before_sepsis = 50 # change to amount of hours before sepsis onset for early prediction
    sep_df = pd.DataFrame([])
    for index in key_indices:
        subdf = df.loc[:index-time_before_sepsis].tail(1000)
        subdf = subdf.loc[df['PatientID'] == int(df.loc[:index-1].tail(1)['PatientID'])]
        sep_df = pd.concat([sep_df, subdf], ignore_index=True)

    sep_df['SepsisLabel'] = 1

    #double sepsis values for oversampling
    sep_df = pd.concat([sep_df, sep_df], ignore_index=True)


    # get non septic patients that havent been used in the sepsis dataset already
    no_sep_df = df[~df['PatientID'].isin(list(set(pat_df['PatientID'].tolist())))]
    df = pd.concat([sep_df, no_sep_df], ignore_index=True) # put them together

    # drop non important columns for training
    df = df.drop(columns=['ICULOS'])
    df = df.drop(columns=['PatientID'])

    # clean the dataset with values
    df = cleanDataset(df)

    # FILTER THE DATASET FOR GROUPINGS

    df = df.loc[(df['Age'] > 20) & (df['Age'] < 40)]
    # male 1 female 0
    df = df.loc[(df['Gender'] == 0)]
    # blood pressure values unhealthy values
    df = df.loc[(df['DBP'] < 60) | (df['DBP'] > 80)]
    df = df.loc[(df['SBP'] < 90) | (df['SBP'] > 120)]
    # healthy values
    # df = df.loc[(df['DBP'] > 60) & (df['DBP'] < 80)]
    # df = df.loc[(df['SBP'] > 90) & (df['SBP'] < 120)]
    # heart rate values unhealthy
    df = df.loc[(df['HR'] > 90)]
    #healthy
    # df = df.loc[(df['HR'] < 90)]



    # df = df.loc[(df['Temp'] < 36.0) | (df['Temp'] > 38.0)]
    df = df.drop(columns=['Gender'])


    return df

# clean dataset from clinical ranges given
def cleanDataset(df):

    columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP',
    'Resp', 'Glucose', 'Potassium', 'WBC', 'Hct', 'Hgb', 'HCO3', 'Age',
    'Creatinine', 'Platelets']

    ranges = [(1,320), (1,100), (25,42), (1,400), (1,400),
    (1,300), (1,150), (1, 600), (1,20), (0.1,100), (1, 100),
    (1,50), (1,50), (15, 90), (1,20), (1,600)]

    indexs = []

    for x in zip(columns, ranges):

        indexs.append(df[(df[x[0]] < x[1][0]) | (df[x[0]] > x[1][1])].index.tolist())

    indexs = sum(indexs, [])

    df.drop(indexs, inplace=True)
    df = df.reset_index(drop=True)

    print(df)

    return df

def addPatientID(df):
    df['PatientID'] = np.nan
    count = 0

    for x in range(len(df)):
        if x % 100_000 == 0:
            print(x)
        try:
            if df.loc[x, :]['ICULOS'] > df.loc[x+1, :]['ICULOS']:
                df.at[x, 'PatientID'] = count
                count += 1
            else:
                df.at[x, 'PatientID'] = count

        except:
            df.at[x, 'PatientID'] = count
    print(df)

    df = df[df['HR'].notna()].reset_index(drop=True)
    return df

def undersampling(DATA): # doubles the size of the sepsis samples with duplicates
    with_sep_df = DATA.loc[DATA['SepsisLabel'] == 1].reset_index(drop=True)
    dup_with_sep_df = with_sep_df

    # df = pd.concat([with_sep_df, dup_with_sep_df], ignore_index=True)
    df = pd.concat([with_sep_df, DATA], ignore_index=True)

    return df

# if get raw is true and impute is false, it will return the complete raw dataset
def createDataset(get_raw=False, impute=True):

    # concat the two datafarmes to make one big one
    df = pd.concat([TRAIN_DATA_A, TRAIN_DATA_B], ignore_index=True)


    if not get_raw:
        if impute:
            # impute the data
            # df = imputeData(df)
            # df = imputeSimpleImputer(df)
            df = complexImputer(df)

        # get sepsis values
        withSepDF = df.loc[df['SepsisLabel'] == 1].reset_index(drop=True)
        # withSepDF = pd.concat([withSepDF, withSepDF], ignore_index=True)

        # get non sepsis values
        withoutSepDF = df.loc[df['SepsisLabel'] == 0].reset_index(drop=True)

        # get length of sepsis dataframe so can use a 50/50 split
        lengthOfSepDF = withSepDF['SepsisLabel'].size
        # sample withoutSepsis dataframe to get same size as other dataframe
        withoutSepDFsample = withoutSepDF.sample(n=lengthOfSepDF).reset_index(drop=True)

        # concat the two dataframes and shuffle
        df = pd.concat([withSepDF, withoutSepDFsample], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)



        lengthOfdf = df['SepsisLabel'].size
        print(lengthOfdf)

        print("Done")
        return df[:int(lengthOfdf*0.7)], df[int(lengthOfdf*0.7):] #70/30 training split
    else:
        return df

# df = complexImputer(pd.concat([TRAIN_DATA_A, TRAIN_DATA_B], ignore_index=True))

