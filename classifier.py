import torch
from sklearn import metrics
from torch import nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# hyper parameters
lr = 0.01
epochs = 10
batchSize = 500
criterion = nn.BCELoss()


# Dataset of one label, given original data with 3 labels
class myDataset(Dataset):
    def __init__(self, dataEncoded):
        super(myDataset, self).__init__()
        self.IDs = [sample[0] for sample in dataEncoded]
        self.tensors = [sample[1][0] for sample in dataEncoded]  # this is already tensor at the right device
        self.labels = torch.tensor([sample[1][1] for sample in dataEncoded], device=device, dtype=torch.float)

    def __getitem__(self, ind):
        return self.IDs[ind], self.tensors[ind], self.labels[ind]  # We could have returned only the desired label for each model, but then we would be storing all data in 3 different loaders

    def __len__(self):
        return len(self.IDs)


# MLP for binary classification
class Classifier(nn.Module):
    def __init__(self, inputSize):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(inputSize, 500),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.BatchNorm1d(300),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(500, 200),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.BatchNorm1d(40),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(200, 40),
            nn.ReLU(),
            # nn.Dropout(),
            # nn.BatchNorm1d(40),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(40, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# do train
def train(model, optimizer, trainLoader):
    model.train()
    totalLoss, totalLength = 0, 0
    for epoch in range(epochs):
        for IDsBatch, xBatch, yBatch in trainLoader:
            optimizer.zero_grad()
            output = model(xBatch)
            loss = criterion(output, yBatch.view_as(output))
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            totalLength += len(xBatch)
    return totalLoss / totalLength


# to validation
def validation(model, valLoader):
    model.eval()
    totalLoss, totalLength = 0, 0
    with torch.no_grad():
        for IDsBatch, xBatch, yBatch in valLoader:
            output = model(xBatch)
            loss = criterion(output, yBatch.view_as(output))
            totalLoss += loss.item()
            totalLength += len(xBatch)
    return totalLoss / totalLength


# do test, write to csv file, each row is: ID, pred1, pred2, pred3
def test(model, testData):
    model.eval()
    testLoader = DataLoader(myDataset(testData))
    f = open('prediction_test.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["IDs", "2 years", "5 years", "10 years"])
    with torch.no_grad():
        for ID, x, y in testLoader:
            ID = ID.tolist()[0]
            output = model(x)
            pred = torch.round(output)[0].tolist()
            writer.writerow([ID, int(pred[0]), int(pred[1]), int(pred[2])])
    f.close()


# plot curves for train and validation losses
def plotLoss(trainLoss, valLoss):
    epochsList = range(epochs)
    plt.figure()
    plt.title(f"Model losses")
    plt.plot(epochsList, trainLoss, label="Train")
    plt.plot(epochsList, valLoss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    plt.legend()
    plt.savefig(f"Model losses")


# return list of length epochs, each value is mean for that epoch
def foldMean(allLosses):
    meaned = []
    for i in range(epochs):
        meaned.append(np.mean([allLosses[j] for j in range(i, len(allLosses), epochs)]))
    return meaned


# make models, train them, plot their losses, and retur them
def makeClassifier(dataEncoded):
    inputSize = dataEncoded[0][1][0].size()[0]
    # results lists
    trainLoss, valLoss = [], []

    # run k-fold
    kfold = KFold(n_splits=3, random_state=777, shuffle=True)
    for foldInd, (trainInds, valInds) in enumerate(kfold.split(dataEncoded)):
        print("fold", foldInd)
        # prepare data
        trainData, valData = dataEncoded[trainInds], dataEncoded[valInds]
        trainLoader, valLoader = DataLoader(myDataset(trainData), batch_size=batchSize), DataLoader(myDataset(valData), batch_size=batchSize)

        # models
        model = Classifier(inputSize).to(device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # run MLP
        for epoch in range(epochs):
            print("\tMLPs epoch", epoch)
            # do train and validation
            curTrainLoss = train(model, optimizer, trainLoader)
            curValLoss = validation(model, valLoader)
            # save values in lists
            trainLoss.append(curTrainLoss)
            valLoss.append(curValLoss)

    # make loss_list[i] mean of epoch[i] from all folds
    trainLoss, valLoss = foldMean(trainLoss), foldMean(valLoss)

    # plot graphs
    plotLoss(trainLoss, valLoss)

    return model
