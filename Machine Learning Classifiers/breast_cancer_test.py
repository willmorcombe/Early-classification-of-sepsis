from sklearn import datasets
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, roc_auc_score

from Lg_scratch import LogisticRegression

bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target
n_samples, n_features = x.shape


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        return x

torchmodel = False

if torchmodel:

    model = Model(n_features)

    learning_rate = 0.4
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_epochs = 100
    for epoch in range(num_epochs):
        y_predicted = model(x_train)
        loss = criterion(y_predicted, y_train)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            print('epoch:', epoch+1,'\n loss:', loss.item(), "\n")

    with torch.no_grad():
        y_predicted = model(x_test)
        y_predicted_cls = y_predicted.round()

        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])

else:
    model = LogisticRegression(0.3, n_features, n_samples)
    epoch = 100

    x_train = np.array(x_train.tolist())
    y_train = np.array(y_train.tolist())

    for e in range(epoch):

        cost = model.train(x_train, y_train)

    predicted = model.forward(x_test)

    predicted = predicted.T
    y_predicted_cls = predicted.round()
    acc = np.equal(y_predicted_cls, y_test).sum() / float(y_test.shape[0])


pred, targ = [y_predicted_cls.T.tolist()[0], y_test.T.tolist()[0]]

print(targ)

print('recall:', '%.3f' % recall_score(targ, pred))
print('precision', '%.3f' % precision_score(targ, pred))
print('ROC', '%.3f' % roc_auc_score(pred, targ))


print('accuracy', '%.3f' % float(acc))




