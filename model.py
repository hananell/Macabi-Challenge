from torch import nn


# MLP for binary classification
class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 500),
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
            nn.Sigmoid() ,
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x