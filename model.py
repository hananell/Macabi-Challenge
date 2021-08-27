from torch import nn


# MLP for binary classification of 3 the time horizons
class Classifier(nn.Module):
    def __init__(self, input_size):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 4000),
            nn.Tanh(),
            nn.BatchNorm1d(4000),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(4000, 2000),
            nn.Tanh(),
            nn.BatchNorm1d(2000),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(2000, 1000),
            nn.Tanh(),
            nn.BatchNorm1d(1000),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1000, 300),
            nn.Tanh(),
            nn.BatchNorm1d(300),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(300, 40),
            nn.Tanh(),
            nn.BatchNorm1d(40),
        )
        self.layer6 = nn.Sequential(
            nn.Linear(40, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

    def weight_reset(self):
        for layers in self.children():
            for layer in layers:
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()


