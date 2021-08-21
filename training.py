import torch
from torch import nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# hyper parameters
lr = 0.01
epochs = 10
batchSize = 500

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


# training loop - one epoch
def train(model, optimizer, train_loader, criterion):
    model.train()
    total_loss, total_length = 0, 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch.view_as(output))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_length += x_batch.shape[0]
    return total_loss/total_length


# validation
def validation(model, val_loader, criterion):
    model.eval()
    total_loss, total_length = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch.view_as(output))
            total_loss += loss.item()
            total_length += len(x_batch)
            total_length += x_batch.shape[0]
    return total_loss / total_length


def fit(model, x, y):
    # results lists
    train_loss, val_loss = [], []

    # run k-fold
    k_fold = KFold(n_splits=3, random_state=777, shuffle=True)
    for train_idx, val_idx in k_fold.split(x, y):
        # split data to train and validation
        x_train, x_val, y_train, y_val = x.iloc[train_idx].values, x.iloc[val_idx].values, y.iloc[train_idx].values, y.iloc[val_idx].values

        # create tensors from Dataframes
        train_dataset = data_utils.TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        val_dataset = data_utils.TensorDataset(torch.FloatTensor(x_val), torch.FloatTensor(y_val))

        train_loader, val_loader = DataLoader(train_dataset, batch_size=batchSize), DataLoader(val_dataset, batch_size=batchSize)

        model.to(device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        model.train()
        criterion = nn.BCELoss()

        train_loss, val_loss = [], []

        # training loop
        for epoch in range(epochs):
            print("\tMLPs epoch", epoch)
            # train
            cur_train_loss = train(model, optimizer, train_loader, criterion)

            # evaluate on validation set
            cur_val_loss = validation(model, val_loader, criterion)

            # save values in lists
            train_loss.append(cur_train_loss)
            val_loss.append(cur_val_loss)

    # make loss_list[i] mean of epoch[i] from all folds, and plot graphs
    mean_trainloss, mean_valloss = foldMean(train_loss), foldMean(val_loss)

    plotLoss(mean_trainloss, mean_valloss)

    return model
