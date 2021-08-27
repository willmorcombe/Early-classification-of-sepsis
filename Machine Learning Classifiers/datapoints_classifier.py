import numpy as np
import pandas as pd
import seaborn as sn
import random as r
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import sys
import matplotlib as plt

sys.path.append('C:/python/python code/Sepsis-classification-main/Data Preparation/')
from data_imputation import *
from get_data import *

def createGraph(data):
    sepsis_df = sorted(data.loc[data[1] == 1][0].tolist())
    no_sepsis_df = sorted(data.loc[data[1] == 0][0].tolist())

    sep_occ = []
    for x in range(sepsis_df[len(sepsis_df) -1]):
        sep_occ.append(sepsis_df.count(x))

    no_sep_occ = []
    for x in range(sepsis_df[len(sepsis_df) -1]):
        no_sep_occ.append(no_sepsis_df.count(x))


    x = [i for i in range(sepsis_df[len(sepsis_df) -1])]

    y1 = sep_occ
    y2 = no_sep_occ

    # print(len(y1), len(y2))

    plt.bar(np.asarray(x) + 0.0, y1, color='red', width = 0.5)
    plt.bar(np.asarray(x) + 0.2, y2, color='blue', width = 0.5)
    plt.xlabel("Length of stay (Hours)")
    plt.ylabel("Sepsis occurences")
    plt.title("Patient outcome at end of stay")

    colors = {'Septic patients':'red', 'Non-septic patients':'blue'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    plt.savefig('../Images/number of hours spent')

def accuracy(predicted, target):
    acc = np.equal(predicted, target).sum() / float(target.shape[0])
    return acc

def createConfusionMatrix(preds, labels):
    cm = confusion_matrix(labels.squeeze().tolist(), preds.squeeze().tolist())

    df_cm = pd.DataFrame(cm, index = [i for i in ['Non-Sepsis', 'Sepsis']],
                  columns = [i for i in ['Non-Sepsis', 'Sepsis']])
    plt.figure(figsize = (3,3))
    sn.heatmap(df_cm, annot=True, cmap='Purples', fmt='g')
    plt.savefig('../Images/confustion_matrix_ICOLUS')



if __name__ == '__main__':

    _, df = createDataset(impute=False)
    df = df[['ICULOS', 'SepsisLabel']]
    df_list = list(zip(df['ICULOS'].tolist(), df['SepsisLabel'].tolist()))
    l = []

    # gets the spesis at end of each patients stay at the hospital
    for x in range(len(df_list)):
        try:
            if df_list[x][0] > df_list[x+1][0]:
                l.append(df_list[x])
        except:
            l.append(df_list[x])


    createGraph(pd.DataFrame(l))

    _, test_df = createDataset(impute=False)

    target = test_df['SepsisLabel'].tolist()
    data = test_df['ICULOS'].tolist()

    predicted_class = []

    for x in range(len(data)):
        if data[x] < 55:
            predicted_class.append(0)
        else:
            predicted_class.append(1)

    target = np.asarray([target]).T
    predicted_class = np.asarray([predicted_class]).T

    # print(target.T.tolist())
    # print(predicted_class.T.tolist())

    print(accuracy(predicted_class, target))
    createConfusionMatrix(predicted_class, target)
