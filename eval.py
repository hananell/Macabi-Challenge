import torch.utils.data as data_utils
import csv
import torch


def predict(model, x, y):
    model.eval()
    test_dataset = data_utils.TensorDataset(torch.FloatTensor(x.values), torch.FloatTensor(y.values))
    test_loader = data_utils.DataLoader(test_dataset, batch_size=10, shuffle=True)
    f = open('prediction_test.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(["IDs", "2 years", "5 years", "10 years"])
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            pred = torch.round(output)[0].tolist()
            writer.writerow([int(pred[0]), int(pred[1]), int(pred[2])])
    f.close()
