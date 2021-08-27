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
    data_sep = sorted(df.loc[df['Sepsis Label'] == 1]['Feature List'].tolist())
    data_nosep = sorted(df.loc[df['Sepsis Label'] == 0]['Feature List'].tolist())

    occurences_sep = []
    for x in range(len(set(data_sep))):
        occurences_sep.append(data_sep.count(x + 3))

    occurences_nosep = []
    for x in range(len(set(data_sep))):
        occurences_nosep.append(data_nosep.count(x + 3))


    x = list(set(data_sep))
    y1 = occurences_sep
    y2 = occurences_nosep

    plt.bar(np.asarray(x) + 0.0, y1, color='red', width = 0.4)
    plt.bar(np.asarray(x) + 0.5, y2, color='blue', width = 0.4)
    plt.xlabel("Number of missing features")
    plt.ylabel("Sepsis occurences")
    plt.title("Sepsis compared with missing features")

    colors = {'Has Sepsis':'red', 'No Sepsis':'blue'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)

    plt.savefig('../Images/number of feature in sepsis patients')

def accuracy(predicted, target):
    acc = np.equal(predicted, target).sum() / float(target.shape[0])
    return acc

def createConfusionMatrix(preds, labels):
    cm = confusion_matrix(labels.squeeze().tolist(), preds.squeeze().tolist())

    df_cm = pd.DataFrame(cm, index = [i for i in ['Non-Sepsis', 'Sepsis']],
                  columns = [i for i in ['Non-Sepsis', 'Sepsis']])
    plt.figure(figsize = (3,3))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.savefig('../Images/confustion_matrix_featurs')

if __name__ == '__main__':
    # df = createDataset(get_raw=True)
    _, df = createDataset(impute=False)

    sepsis_list = df['SepsisLabel'].tolist()
    feature_list = df.isnull().sum(axis=1).tolist()

    d = {'Feature List':feature_list, 'Sepsis Label':sepsis_list}
    df = pd.DataFrame(d)
    # this graph shows that the main sepsis spread is between 26 and 31 featuers so this
    # will be the crude classifier
    createGraph(df)


    # MODEL

    _, test_df = createDataset(impute=False)

    target = test_df['SepsisLabel'].tolist()
    feature_list = test_df.isnull().sum(axis=1).tolist()

    predicted_class = []


    for x in range(len(feature_list)):
        if feature_list[x] < 22: # has sepsis if the datapoint has this many features
            predicted_class.append(1)
        else:
            predicted_class.append(0)

    target = np.asarray([target]).T
    predicted_class = np.asarray([predicted_class]).T

    # print(target.T.tolist())
    # print(predicted_class.T.tolist())

    print(accuracy(predicted_class, target))
    createConfusionMatrix(predicted_class, target)







