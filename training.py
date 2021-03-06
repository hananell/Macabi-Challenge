import torch
from torch import nn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# hyper parameters
lr = 0.02
epochs = 10
batchSize = 500
n_splits = 4
oversampling = False


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
def train_epoch(model, optimizer, train_loader, criterion):
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
    return total_loss / total_length


# validation
def valid_epoch(model, val_loader, criterion):
    model.eval()
    total_loss, total_length = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            output = model(x_batch)
            loss = criterion(output, y_batch.view_as(output))
            total_loss += loss.item()
            total_length += x_batch.shape[0]
    return total_loss / total_length


# duplicate samples that contain a 1 label
def duplicate_ills(x_train, y_train):
    duplicated_x, duplicated_y = np.copy(x_train), np.copy(y_train)

    # for each (vector, labels) if the labels contain 1, add the (vector, label) to duplicated arrays
    for i, (x, y) in enumerate(zip(x_train, y_train)):
        if 1 in y:
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            duplicated_x = np.append(duplicated_x, x, axis=0)
            duplicated_y = np.append(duplicated_y, y, axis=0)

    return duplicated_x, duplicated_y


def cross_validate(model, x, y):
    # results lists
    train_loss, val_loss = [], []

    # run k-fold
    k_fold = KFold(n_splits=n_splits, random_state=777, shuffle=True)
    for foldNum, (train_idx, val_idx) in enumerate(k_fold.split(x, y)):
        print("fold", foldNum + 1)
        # split data to train and validation
        x_train, x_val, y_train, y_val = x.iloc[train_idx].values, x.iloc[val_idx].values, y.iloc[train_idx].values, \
                                         y.iloc[val_idx].values
        # if oversampling:
        # x_train, y_train = duplicate_ills(x_train, y_train)

        # create tensors from Dataframes, and DataLoaders
        train_dataset = data_utils.TensorDataset(torch.tensor(x_train, dtype=torch.float, device=device),
                                                 torch.tensor(y_train, dtype=torch.float, device=device))
        val_dataset = data_utils.TensorDataset(torch.tensor(x_val, dtype=torch.float, device=device),
                                               torch.tensor(y_val, dtype=torch.float, device=device))
        train_loader, val_loader = DataLoader(train_dataset, batch_size=batchSize), DataLoader(val_dataset,
                                                                                               batch_size=batchSize)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        train_loss, val_loss = [], []

        model.weight_reset()

        # training loop
        for epoch in range(epochs):
            # train
            cur_train_loss = train_epoch(model, optimizer, train_loader, criterion)

            # evaluate on validation set
            cur_val_loss = valid_epoch(model, val_loader, criterion)

            # save values in lists
            train_loss.append(cur_train_loss)
            val_loss.append(cur_val_loss)

    # make loss_list[i] mean of epoch[i] from all folds, and plot graphs
    mean_trainloss, mean_valloss = foldMean(train_loss), foldMean(val_loss)

    return mean_trainloss, mean_valloss


def fit(model, x, y):
    train_dataset = data_utils.TensorDataset(torch.tensor(x.values, dtype=torch.float, device=device),
                                             torch.tensor(y.values, dtype=torch.float, device=device))
    train_loader = DataLoader(train_dataset, batch_size=batchSize)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    train_loss = []

    for epoch in range(epochs):
        cur_train_loss = train_epoch(model, optimizer, train_loader, criterion)
        # save values in lists
        train_loss.append(cur_train_loss)

    print(train_loss)

    return model


