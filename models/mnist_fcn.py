import torch.nn as nn

class MnistFCN(nn.Module):
    def __init__(self):
        super(MnistFCN, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),             # 28x28 -> 784
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)        # 10 classes
        )

    def forward(self, x):
        return self.layers(x)