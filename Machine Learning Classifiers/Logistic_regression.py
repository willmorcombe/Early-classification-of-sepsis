import numpy as np
import pandas as pd
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import seaborn as sn
import random as r
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import sys

sys.path.append('C:/python/python code/Sepsis-classification-main/Data Preparation/')
from data_imputation import *
from get_data import *

class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(16, 1)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


def createConfusionMatrix(preds, labels):
    cm = confusion_matrix(labels.squeeze().tolist(), preds.squeeze().tolist())

    df_cm = pd.DataFrame(cm, index = [i for i in ['Non-Sepsis', 'Sepsis']],
                  columns = [i for i in ['Non-Sepsis', 'Sepsis']])
    plt.figure(figsize = (3,3))
    sn.heatmap(df_cm, annot=True, cmap='Greens', fmt='g')
    plt.savefig('../Images/confusion_matrix_pytorch_implementation)


if __name__ == '__main__':

    train_set_df, test_set_df = createDataset()

    # split test and train dataset separating their labels and making them a numpy array
    train_data = train_set_df.drop(columns=['SepsisLabel']).to_numpy()
    train_target = train_set_df['SepsisLabel'].to_numpy()

    test_data = test_set_df.drop(columns=['SepsisLabel']).to_numpy()
    test_target = test_set_df['SepsisLabel'].to_numpy()


    # scale and fit the data
    sc = StandardScaler()

    train_data = sc.fit_transform(train_data)
    test_data = sc.fit_transform(test_data)

    # make tensor

    train_data = torch.from_numpy(train_data.astype(np.float32))
    test_data = torch.from_numpy(test_data.astype(np.float32))
    train_target = torch.from_numpy(train_target.astype(np.float32))
    test_target = torch.from_numpy(test_target.astype(np.float32))

    # make targets columns

    train_target = train_target.view(train_target.shape[0], 1)
    test_target = test_target.view(test_target.shape[0], 1)

    model = LogisticRegression()

    learning_rate = 0.1
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100
    for epoch in range(num_epochs):
        predicted = model(train_data)
        loss = criterion(predicted, train_target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



        if (epoch+1) % 10 == 0:
            predicted_class = predicted.round()
            acc = predicted_class.eq(train_target).sum() / float(train_target.shape[0])
            print('epoch:', epoch+1,'\n loss:', loss.item(),
                '\n training accuracy:', acc)


    with torch.no_grad():
        predicted = model(test_data)
        predicted_class = predicted.round()
        acc = predicted_class.eq(test_target).sum() / float(test_target.shape[0])

        print('accuracy =', acc)

        createConfusionMatrix(predicted_class, test_target)





