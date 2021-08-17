import torch
from sklearn import metrics
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# hyper parameters
lr = 0.01
epochs = 10
batchSize = 200
criterion = nn.BCELoss()


# Dataset of one label, given original data with 3 labels
class myDataset(Dataset):
    def __init__(self, dataEncoded, labelNum):
        super(myDataset, self).__init__()
        self.IDs = [sample[0] for sample in dataEncoded]
        self.tensors = [sample[1][0] for sample in dataEncoded]
        self.labels = [sample[1][1][labelNum - 1] for sample in dataEncoded]  # -1 because we expect 1,2,3 and want 0,1,2

    def __getitem__(self, ind):
        return self.IDs[ind], self.tensors[ind], self.labels[ind]

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
            nn.Linear(40, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# do train
def train(model, optimizer, trainLoader):
    model.train()
    totalLoss = 0
    for epoch in range(epochs):
        for IDsBatch, xBatch, yBatch in trainLoader:
            optimizer.zero_grad()
            output = model(xBatch)
            loss = criterion(output, yBatch.view_as(output))
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
    return totalLoss / len(trainLoader)


# to validation
def validation(modely, valLoader):
    # do predictions and save them. also save real labels
    modely.eval()
    totalLoss = 0
    with torch.no_grad():
        for IDsBatch, xBatch, yBatch in valLoader:
            output = modely(xBatch)
            loss = criterion(output, yBatch.view_as(output))
            totalLoss += loss.item()
    return totalLoss / len(valLoader)


# do test
def test(model, testLoader):
    pass


# tain models and return them
def makeClassifier(dataEncoded):
    # data
    inputSize = dataEncoded[0][1][0].size()[0]
    trainData, valData = train_test_split(dataEncoded)
    trainLoader1, valLoader1 = DataLoader(myDataset(trainData, 1), batch_size=batchSize), DataLoader(myDataset(valData, 1), batch_size=batchSize)
    trainLoader2, valLoader2 = DataLoader(myDataset(trainData, 2), batch_size=batchSize), DataLoader(myDataset(valData, 2), batch_size=batchSize)
    trainLoader3, valLoader3 = DataLoader(myDataset(trainData, 3), batch_size=batchSize), DataLoader(myDataset(valData, 3), batch_size=batchSize)

    # models
    model1, model2, model3 = Classifier(inputSize), Classifier(inputSize), Classifier(inputSize)
    optimizer1, optimizer2, optimizer3 = torch.optim.Adam(model1.parameters(), lr=lr), torch.optim.Adam(model2.parameters(), lr=lr), torch.optim.Adam(model3.parameters(), lr=lr)

    # results lists
    trainLoss1, valLoss1 = [], []
    trainLoss2, valLoss2 = [], []
    trainLoss3, valLoss3 = [], []

    ##### k-fold
    # run
    for epoch in range(epochs):
        print("MLPs epoch", epoch)
        # do train and validation
        curTrainLoss1 = train(model1, optimizer1, trainLoader1)
        curTrainLoss2 = train(model2, optimizer2, trainLoader2)
        curTrainLoss3 = train(model3, optimizer3, trainLoader3)
        curValLoss1 = validation(model1, valLoader1)
        curValLoss2 = validation(model2, valLoader2)
        curValLoss3 = validation(model3, valLoader3)
        # save values in lists
        trainLoss1.append(curTrainLoss1)
        valLoss1.append(curValLoss1)
        trainLoss2.append(curTrainLoss2)
        valLoss2.append(curValLoss2)
        trainLoss3.append(curTrainLoss3)
        valLoss3.append(curValLoss3)

    # plot graphs
