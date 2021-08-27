import numpy as np
import pandas as pd
import seaborn as sn
import random as r
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, precision_score
from matplotlib import pyplot
import sys

sys.path.append('C:/python/python code/Sepsis-classification-main/Data Preparation/')
from data_imputation import *
from get_data import *

# model from scratch
class LogisticRegression:

    def __init__(self, lr, features, length_of_dataset):
        self.lr = lr
        self.features = features
        self.len_data = length_of_dataset
        self.weights = np.zeros((self.features, 1))
        self.bias = 0

    def sigmiod(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, data):
        data = data.T
        Z = np.dot(self.weights.T, data) + self.bias
        A = self.sigmiod(Z)

        return A


    def back(self, A, data, target):

        data = data.T
        target = target.T

        cost = -(1/self.len_data) * np.sum( target * np.log(A) + (1-target) * np.log(1-A))
        dw = (1/self.len_data) * np.dot(A - target, data.T)
        db = (1/self.len_data) * np.sum(A - target)
        self.weights = self.weights - self.lr * dw.T
        self.bias = self.bias - self.lr * db

        return cost

    def train(self, data, target):
        output = self.forward(data)
        return self.back(output, data, target)

# create AUROC curve from testing results and labels

def createAUROC(preds, labels):
    lg_score = roc_auc_score(labels, preds)
    r_score = roc_auc_score(labels, [0 for x in range(len(labels))])


    r_fpr, r_tpr, _ = roc_curve(labels, [0 for x in range(len(labels))])
    lg_fpr, lg_tpr, _ = roc_curve(labels, preds)

    fig2 = plt.figure(2)

    plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_score)
    plt.plot(lg_fpr, lg_tpr, marker='.', label='Logistic regression (AUROC = %0.3f)' % lg_score)

    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # Show legend
    plt.legend() #
    # Show plot
    plt.savefig('../Images/ROC_curve')


# create confusion matrix from testing resutls and labels
def createConfusionMatrix(preds, labels):
    cm = confusion_matrix(labels.squeeze().tolist(), preds.squeeze().tolist())
    catagories = ['Non-Sepsis', 'Sepsis']
    fig1 = plt.figure(1)
    df_cm = pd.DataFrame(cm, index = [i for i in ['Non-Sepsis', 'Sepsis']],
                  columns = [i for i in ['Non-Sepsis', 'Sepsis']])
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (4,4))
    sn.heatmap(df_cm, annot=True, cmap='Reds', fmt='g')
    plt.savefig('../Images/confusion_matrix_scratch')

#get accuracy of model
def accuracy(predicted, target):
    predicted = predicted.T
    predicted_class = predicted.round()
    acc = np.equal(predicted_class, target).sum() / float(target.shape[0])
    return acc

if __name__ == '__main__':

    train_set_df, test_set_df = createDataset() # get training and testing datasets

    #list the gaussian featuers and the non gaussian features
    gaussian_features = ['HR', 'Temp', 'SBP', 'MAP', 'DBP',
    'Glucose', 'Potassium', 'Hct', 'Hgb', 'HCO3', 'Resp', 'Platelets', 'Age', 'WBC']

    non_gaussian_features = ['Creatinine', 'O2Sat']

    #plot histogram
    test_set_df.hist(alpha=0.5, figsize=(14, 14))
    pyplot.savefig('../Images/Histogram_of_features')

    #scale and fit the data
    sc = StandardScaler()
    ms = MinMaxScaler()

    train_set_gaus = pd.DataFrame(sc.fit_transform(train_set_df[gaussian_features]))
    train_set_non_gaus = pd.DataFrame(sc.fit_transform(train_set_df[non_gaussian_features]))
    train_set_df= pd.concat([train_set_gaus, train_set_non_gaus, train_set_df['SepsisLabel']],
        axis=1,ignore_index=True).rename(columns={16 : "SepsisLabel"})

    test_set_gaus = pd.DataFrame(sc.fit_transform(test_set_df[gaussian_features]))
    test_set_non_gaus = pd.DataFrame(sc.fit_transform(test_set_df[non_gaussian_features]))
    test_set_df = pd.concat([test_set_gaus, test_set_non_gaus, test_set_df['SepsisLabel'].reset_index(drop=True)],
        axis=1,ignore_index=True).rename(columns={16 : "SepsisLabel"})





    # split test and train dataset separating their labels and making them a numpy array
    train_data = train_set_df.drop(columns=['SepsisLabel']).to_numpy()
    train_target = train_set_df['SepsisLabel'].to_numpy()

    test_data = test_set_df.drop(columns=['SepsisLabel']).to_numpy()
    test_target = test_set_df['SepsisLabel'].to_numpy()


    #make targets columns
    train_target = train_target.reshape(-1,1)
    test_target = test_target.reshape(-1,1)

    learning_rate = 0.1
    epochs = 100
    features = train_data.shape[1]
    length_of_dataset = train_data.shape[0]



    model = LogisticRegression(learning_rate, features, length_of_dataset)

    for epoch in range(epochs):

        cost = model.train(train_data, train_target)

        if epoch % 10 == 0:
            predicted = model.forward(train_data)
            print("Epoch:", epoch, "\nLoss:", cost, "\nAcc:", accuracy(predicted, train_target),"\n")


    # test
    predicted = model.forward(test_data)
    predicted_class = np.where(predicted > 0.5, 1, 0).T


    print("Testing acc:", '%.3f' % accuracy(predicted, test_target))
    print("Precision:", '%.3f' % precision_score(test_target.T.tolist()[0],predicted_class.T.tolist()[0]))
    print("ROC:", '%.3f' % roc_auc_score(test_target.T.tolist()[0],predicted_class.T.tolist()[0]))
    print("Recall:", '%.3f' % recall_score(test_target.T.tolist()[0],predicted_class.T.tolist()[0]))

    # confustion matrix



    # puts values in 2dp
    l = [ '%.2f' % elem for elem in predicted[0] ]
    l = [float(x) for x in l]
    l = np.array([l]).T

    createAUROC(l, test_target)
    createConfusionMatrix(predicted_class, test_target)












