import numpy as np
import pandas as pd
import torch
from torch import autograd, nn, optim
import torch.nn.functional as F
import seaborn as sn
import random as r
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append('C:/python/python code/Sepsis-classification-main/Data Preparation/')
from data_imputation import *
from get_data import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(7,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,1)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# gets prediction from network and returns the prediction and labels
def getPrediction(net, loader):
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

    for data in loader:
        values, labels = data

        preds = net(values)
        all_preds = torch.cat((all_preds, preds), dim=0)
        all_labels = torch.cat((all_labels, labels), dim=0)

    return all_preds, all_labels

# creats a confusion matrix with test predictions and labels
def createConfusionMatrix(preds, labels):
    stacked = torch.stack((labels, preds.argmax(dim=1)), dim=1)
    cmt = torch.zeros(2,2, dtype=torch.int32)
    classes = ['Sepsis', 'No Sepsis']

    for p in stacked:
        x, y = p.tolist()
        cmt[int(x), int(y)] = cmt[int(x), int(y)] + 1

    df_cm = pd.DataFrame(cmt.tolist(), index = [i for i in classes], columns = [i for i in classes])
    plt.figure(figsize = (3,3))
    sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    plt.savefig('../Images/confusion_matrix')


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

    # add targets back to data to put into dataloader
    train_data_final = torch.cat((train_data, train_target), dim=1)
    test_data_final = torch.cat((test_data, test_target), dim=1)

    # parameters
    batchSize = 32
    learningRate = 0.001
    nEpochs = 700

    # data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data_final,
        batch_size=batchSize,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data_final,
        batch_size=batchSize,
        shuffle=False
    )

    model = Model()
    optimizer = optim.SGD(params=model.parameters(), lr = learningRate)
    criterion = nn.BCELoss()

    losses = []

    # training loop
    for epoch in range(nEpochs):
        for index, data in enumerate(train_loader):

            inputs = data[:, :7]
            targets = data[:, -1::]

            output = model(inputs)


            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if epoch % 2 == 0:
            losses.append(loss)
            print("epoch {}\tloss : {}".format(epoch,loss))
