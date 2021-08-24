import torch.utils.data as data_utils
import csv
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def predict(model, x, y, IDs):
    model.eval()

    # prepare data and result list
    test_dataset = data_utils.TensorDataset(torch.tensor(x.values, dtype=torch.float, device=device), torch.tensor(y.values, dtype=torch.float, device=device))
    test_loader = data_utils.DataLoader(test_dataset, shuffle=True)
    preds2, preds5, preds10 = [], [], []
    labels2, labels5, labels10 = [], [], []

    # open file, write titles
    f = open('prediction_test.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["IDs", "2 years", "5 years", "10 years"])

    # predict
    with torch.no_grad():
        for ID, (x, y) in zip(IDs, test_loader):
            # make prediction and write to file
            output = model(x)
            pred = torch.round(output)[0].tolist()
            pred = [int(item) for item in pred]
            writer.writerow([ID] + pred)
            # save predictions and real labels
            preds2.append(pred[0])
            preds5.append(pred[1])
            preds10.append(pred[2])
            y = torch.squeeze(y).tolist()
            y = [int(item) for item in y]
            labels2.append(y[0])
            labels5.append(y[1])
            labels10.append(y[2])
    f.close()

    # print measurements
    f1_2, f1_5, f1_10 = f1_score(labels2, preds2), f1_score(labels5, preds5), f1_score(labels10, preds10)
    print(f"f1 scores:\n\t2 years: {f1_2:.3f}\t\t5 years: {f1_5:.3f}\t\t10 years: {f1_10:.3f}")

    accuracy_2, accuracy_5, accuracy_10 = accuracy_score(labels2, preds2), accuracy_score(labels5, preds5), accuracy_score(labels10, preds10)
    print(f"accuracies:\n\t2 years: {accuracy_2:.3f}\t\t5 years: {accuracy_5:.3f}\t\t10 years: {accuracy_10:.3f}")

    prec1, prec2, prec3 = precision_score(labels2, preds2), precision_score(labels5, preds5), precision_score(labels10, preds10)
    print(f"precision:\n\t2 years: {prec1:.3f}\t\t5 years: {prec2:.3f}\t\t10 years: {prec3:.3f}")

    rec1, rec2, rec3 = recall_score(labels2, preds2), recall_score(labels5, preds5), recall_score(labels10, preds10)
    print(f"recall:\n\t2 years: {rec1:.3f}\t\t5 years: {rec2:.3f}\t\t10 years: {rec3:.3f}")


