# =====================================================
# =           data analysis for sepsis data           =
# =====================================================

from get_data import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


"""
gets percentage of positive sepsis patients in the dataset
parameter (dataframe)
returns (int, int, int)
"""
def percentageOfSepsis(DATA):
    df = DATA['SepsisLabel'].sort_values(ascending=True, na_position='first')
    sepPos = df.sum().sum()
    sepNeg = len(df.index) - sepPos
    percentage = int(df.sum().sum() / len(df.index) * 100)
    return sepPos, sepNeg, percentage


"""
saves a graph of how temperature correlates to sepsis
"""
def tempWithSepsis(DATA):
    df = DATA[["Temp", "SepsisLabel"]].sort_values(by='SepsisLabel', ascending=True, na_position='first').dropna()
    df.plot(kind='scatter', x='Temp', y='SepsisLabel', color='red')
    plt.savefig('../Images/temp with sepsis')

def heartrateAnal(DATA): # looking at heartrate ranges
    df = DATA.loc[DATA['SepsisLabel'] == 1].reset_index(drop=True)
    df = DATA['DBP'].sort_values(ascending=True)
    return df.dropna()


# looks at ages in the dataset
def agePercentage(DATA):
    df = DATA[['Age', 'SepsisLabel']]
    print('average age of patients', sum(list(df['Age'])) / len(list(df["Age"])))
    df = df.loc[df['SepsisLabel'] == 1].reset_index(drop=True)
    print('average age with sepsis', sum(list(df['Age'])) / len(list(df["Age"])))

#looking at gender splits
def gender(df):
    male = df.loc[df['Gender'] == 1].reset_index(drop=True)
    female = df.loc[df['Gender'] == 0].reset_index(drop=True)
    print("male", len(male['SepsisLabel'].tolist()) / 1552210  * 100)
    print("female", len(female['SepsisLabel'].tolist()) / 1552210  * 100)

    df = df.loc[df['SepsisLabel'] == 1].reset_index(drop=True)
    male = df.loc[df['Gender'] == 1].reset_index(drop=True)
    female = df.loc[df['Gender'] == 0].reset_index(drop=True)
    print("male with sepsis", len(male['SepsisLabel'].tolist()) / 27916  * 100)
    print("female with sepsis", len(female['SepsisLabel'].tolist()) / 27916  * 100)

# return the most important columns based on how many times they are taken
def keyColumns(DATA):
    df = DATA.loc[DATA['SepsisLabel'] == 1].reset_index(drop=True)
    df = df.isna().sum().sort_values(ascending=True)
    df = df[(df != 0)].head(9)
    df = df.drop(["Unit1", "Unit2"]) # these values are meaningless
    return df

# returns the percentage of times that the key column values are taken if the
# patient has sepsis
def keyColumnsAnal(DATA):
    length = DATA.loc[DATA['SepsisLabel'] == 1].reset_index(drop=True)
    length = length['SepsisLabel'].size

    df = keyColumns(DATA)
    df = 100 - (df / length) * 100
    return df

# graph the missing data
def missingValues(DATA):
    df = DATA
    n_miss = df.isnull().sum(axis=0).tolist()
    total = df['SepsisLabel'].size
    percentage = []
    heading = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2', 'HST', 'ICULOS', 'SepsisLabel']

    for x in n_miss:
        percentage.append(int(x / total * 100))

    l = []
    for x in zip(heading, percentage):
        print(x)
        l.append(x)

    print("\nNumber datapoints per patient: ",len(l), "\n")

    fig1 = plt.figure(2)
    plt.bar(heading[34:], percentage[34:])
    plt.title('Percentage of missing data')
    plt.xlabel('Demographics - Co-variates')
    plt.ylabel('Percentage')
    plt.savefig('../Images/Missing_data_dataset_Dem')

# creating the recall and auroc graphs for analysis
def createOversampleGraph():
    fig1 = plt.figure(3)
    l = [
    0.626   ,0.550,
    0.752   ,0.626,
    0.692   ,0.583,
    0.801   ,0.708,
    0.652   ,0.578,
    0.667   ,0.649,
    0.641   ,0.513,
    0.720   ,0.661,
    0.659   ,0.534,
    0.665   ,0.642,
    0.690   ,0.524,
    0.686   ,0.607,
    0.678   ,0.679,
    0.700   ,0.730,
    0.759   ,0.638,
    0.705   ,0.627,
    ]
    aucroc_no_oversample = [ l[x] for x in range(0, len(l), 2)]
    recall_no_oversample = [ l[x] for x in range(1, len(l), 2)]
    aucroc = [0.719, 0.783, 0.783, 0.779, 0.655, 0.656, 0.669, 0.712, 0.632, 0.660, 0.688, 0.663, 0.713, 0.712, 0.738, 0.728]
    recall = [0.596, 0.721, 0.694, 0.731, 0.524, 0.596, 0.603, 0.682, 0.546, 0.652, 0.607, 0.604, 0.674, 0.707, 0.677, 0.649]

    axes = plt.gca()
    axes.set_ylim([0.5, 0.85])

    plt.bar(np.asarray([x for x in range(1,len(recall) +1)]) - 0.15,
        aucroc_no_oversample, color="Blue", width=0.3)
    plt.bar(np.asarray([x for x in range(1,len(recall) +1)]) +0.15,
        aucroc, color="Red", width=0.3)


    plt.xlabel("Index of health and age grouping")
    plt.ylabel("AUROC score")
    plt.title("Effect of Oversampling on AUROC score")

    colors = {'After Oversampling':'red', 'Before Oversampling':'blue'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.savefig('../Images/OversampleBarAUROCscore')

    fig1 = plt.figure(4)

    axes = plt.gca()
    axes.set_ylim([0.4, 0.8])

    plt.bar(np.asarray([x for x in range(1,len(recall) +1)]) - 0.15,
        recall_no_oversample, color="Blue", width=0.3)
    plt.bar(np.asarray([x for x in range(1,len(recall) +1)]) +0.15,
        recall, color="Red", width=0.3)


    plt.xlabel("Index of health and age grouping")
    plt.ylabel("Recall value")
    plt.title("Effect of Oversampling on Recall")

    colors = {'After Oversampling':'red', 'Before Oversampling':'blue'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.savefig('../Images/OversampleBarRecall')

# plotting data from experiments with auroc score over time
def EarlyAUROCscore():
    fig1 = plt.figure(5)

    auroc_good = [0.782, 0.821, 0.871, 0.890, 0.893, 0.901]
    auroc_bad = [0.634, 0.660, 0.68, 0.716, 0.719, 0.715]
    time = [x for x in range(0,51,10)]

    plt.plot(time, auroc_good, marker="|", label="Best grouping")
    plt.plot(time, auroc_bad, marker="|", label="Worst grouping")

    axes = plt.gca()
    axes.set_ylim([0.5, 1])
    # Title
    plt.title('AUROC score at different hours before sepsis onset')
    # Axis labels
    plt.xlabel('Time (hours before onset)')
    plt.ylabel('AUROC score')
    # Show legend
    plt.legend() #
    # Show plot
    plt.savefig('../Images/AUROC_before_sepsis_onset')



if __name__ == '__main__':

    df = pd.concat([TRAIN_DATA_A, TRAIN_DATA_B], ignore_index=True)


    print(df)
    print("Number of missing values:", sum(df.isna().sum().tolist()))

    sepPos, sepNeg, percentage = percentageOfSepsis(df)
    print(sepPos + sepNeg)
    print(" number of positive sepsis:", sepPos, "\n", "number of negative sepsis:",
        sepNeg, "\n", "percentage of positive sepsis:", str(percentage) + "%")

    gender(df)



    tempWithSepsis(df)

    agePercentage(df)

    keyColumns(df)

    missingValues(df)

    print(keyColumnsAnal(df))

    createOversampleGraph()

    EarlyAUROCscore()


