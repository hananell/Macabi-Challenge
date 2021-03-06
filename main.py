from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from data_preprocessing import encode_data
import training
import time
import model
import eval

startTime = time.time()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
data_path = "diab_ckd_data.csv"

if __name__ == '__main__':
    # read and encode data
    data = pd.read_csv(data_path)

    IDs = data.pop("IDS").tolist()  # save IDs for test. and we don't need them till then
    encoded_data, encoded_target = encode_data(data)

    # split the data to train and test
    x_train, x_test, y_train, y_test = train_test_split(encoded_data, encoded_target, test_size=0.25)

    mpl_classifier = model.Classifier(input_size=len(encoded_data.columns)).to(device)

    # k-fold cross validation for parameter tuning
    # mean_trainloss, mean_valloss = training.cross_validate(mpl_classifier, x_train, y_train)
    # training.plotLoss(mean_trainloss, mean_valloss)

    training.fit(mpl_classifier, x_train, y_train)
    eval.predict(mpl_classifier, x_test, y_test, IDs)

    # print run time
    minutes = (time.time() - startTime) / 60
    print(f"\n--- {minutes:.1f} minutes ---")
    print("cuda: ", torch.cuda.is_available())
