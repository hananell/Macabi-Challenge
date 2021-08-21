from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from data_preprocessing import encode_data
from model import Classifier
import training
import time
import eval

startTime = time.time()
use_cuda = torch.cuda.is_available()
# use_cuda = False
data_path = "diab_ckd_data.csv"


if __name__ == '__main__':
    # read and encode data
    data = pd.read_csv(data_path)
    encoded_data, encoded_target = encode_data(data)

    # split the data to train and test
    X_train, X_test, y_train, y_test = train_test_split(encoded_data, encoded_target, test_size=0.25)

    # make models, then use them to predict labels of all the data
    mpl_classifier = Classifier(input_size=len(encoded_data.columns))

    model = training.fit(mpl_classifier, X_train, y_train)
    eval.predict(mpl_classifier, X_test, y_test)

    # print run time
    minutes = (time.time() - startTime) / 60
    print(f"\n--- {minutes:.1f} minutes ---")
    if use_cuda:
        print("with cuda")
    else:
        print("without cuda")
