import torch
from sklearn import metrics
from torch import nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import numpy as np
use_cuda = torch.cuda.is_available()
# use_cuda = False


# hyper parameters
lr = 0.01
epochs = 4
batchSize = 500
criterion = nn.BCELoss()

# Dataset of one label, given original data with 3 labels
class myDataset(Dataset):
    def __init__(self, dataEncoded):
        super(myDataset, self).__init__()
        self.IDs = torch.tensor([sample[0] for sample in dataEncoded], device=torch.device('cuda') if use_cuda else torch.device('cpu'), dtype=torch.float)
        self.tensors = [sample[1][0] for sample in dataEncoded]     # this is already tensor at the right device
        self.labels = torch.tensor([sample[1][1] for sample in dataEncoded], device=torch.device('cuda') if use_cuda else torch.device('cpu'), dtype=torch.float)

    def __getitem__(self, ind):
        return self.IDs[ind], self.tensors[ind], (self.labels[ind][0], self.labels[ind][1], self.labels[ind][2])    # We could have returned only the desired label for each model, but then we would be storing all data in 3 different loaders

    def __len__(self):
        return len(self.IDs)


# MLP for binary classification
class Classifier(nn.Module):
    def __init__(self, inputSize):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(inputSize, 300),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(300),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(300, 40),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(40),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(40, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# do train
def train(model, optimizer, trainLoader, labelNum):
    model.train()
    totalLoss = 0
    for epoch in range(epochs):
        for IDsBatch, xBatch, yBatchList in trainLoader:
            yBatch = yBatchList[labelNum-1]                 # -1 because we expect 1,2,3 and want 0,1,2
            optimizer.zero_grad()
            output = model(xBatch)
            loss = criterion(output, yBatch.view_as(output))
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
    return totalLoss / len(trainLoader)


# to validation
def validation(modely, valLoader, labelNum):
    modely.eval()
    totalLoss = 0
    with torch.no_grad():
        for IDsBatch, xBatch, yBatchList in valLoader:
            yBatch = yBatchList[labelNum-1]                 # -1 because we expect 1,2,3 and want 0,1,2
            output = modely(xBatch)
            loss = criterion(output, yBatch.view_as(output))
            totalLoss += loss.item()
    return totalLoss / len(valLoader)


# do test
def test(model, testLoader):
    pass


# plot curves for train and validation losses
def plotLoss(trainLoss, valLoss, modelNum):
    epochsList = range(epochs)
    plt.figure()
    plt.title(f"Model{modelNum} loss")
    plt.plot(epochsList, trainLoss, label="Train")
    plt.plot(epochsList, valLoss, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.locator_params(axis="x", integer=True, tight=True)  # make x axis to display only whole number (iterations)
    plt.legend()
    plt.savefig(f"Model{modelNum} loss")


# return list of length epochs, each value is mean for that epoch
def foldMean(allLosses):
    meaned = []
    for i in range(epochs):
        meaned.append(np.mean([allLosses[j] for j in range(i, len(allLosses), epochs)]))
    return meaned




# make models, train them, plot their losses, and retur them
def makeClassifier(dataEncoded):
    # data
    inputSize = dataEncoded[0][1][0].size()[0]

    # models
    model1, model2, model3 = Classifier(inputSize), Classifier(inputSize), Classifier(inputSize)
    if use_cuda:
        model1, model2, model3 = model1.to(torch.device('cuda')), model2.to(torch.device('cuda')), model3.to(torch.device('cuda'))
    optimizer1, optimizer2, optimizer3 = torch.optim.Adam(model1.parameters(), lr=lr), torch.optim.Adam(model2.parameters(), lr=lr), torch.optim.Adam(model3.parameters(), lr=lr)

    # results lists
    trainLoss1, valLoss1 = [], []
    trainLoss2, valLoss2 = [], []
    trainLoss3, valLoss3 = [], []

    # run k-fold
    kfoldy = KFold(n_splits=3, random_state=777, shuffle=True)
    for foldInd, (trainInds, valInds) in enumerate(kfoldy.split(dataEncoded)):
        print("fold", foldInd)
        # prepare data
        trainData, valData = dataEncoded[trainInds], dataEncoded[valInds]
        trainLoader, valLoader = DataLoader(myDataset(trainData), batch_size=batchSize), DataLoader(myDataset(valData), batch_size=batchSize)
        # run MLPs
        for epoch in range(epochs):
            print("\tMLPs epoch", epoch)
            # do train and validation
            curTrainLoss1 = train(model1, optimizer1, trainLoader, 1)
            curTrainLoss2 = train(model2, optimizer2, trainLoader, 2)
            curTrainLoss3 = train(model3, optimizer3, trainLoader, 3)
            curValLoss1 = validation(model1, valLoader, 1)
            curValLoss2 = validation(model2, valLoader, 2)
            curValLoss3 = validation(model3, valLoader, 3)
            # save values in lists
            trainLoss1.append(curTrainLoss1)
            valLoss1.append(curValLoss1)
            trainLoss2.append(curTrainLoss2)
            valLoss2.append(curValLoss2)
            trainLoss3.append(curTrainLoss3)
            valLoss3.append(curValLoss3)


    # make loss_list[i] mean of epoch[i] from all folds
    trainLoss1, valLoss1 = foldMean(trainLoss1), foldMean(valLoss1)
    trainLoss2, valLoss2 = foldMean(trainLoss2), foldMean(valLoss2)
    trainLoss3, valLoss3 = foldMean(trainLoss3), foldMean(valLoss3)

    # plot graphs
    plotLoss(trainLoss1, valLoss1, 1)
    plotLoss(trainLoss2, valLoss2, 2)
    plotLoss(trainLoss3, valLoss3, 3)

    return model1, model2, model3