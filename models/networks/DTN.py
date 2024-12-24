import torch
import torch.nn as nn


class DTN(nn.Module):
    "Classifier used for SVHN->MNIST Experiment"
    def __init__(self, input_shape, num_classes=10):
        super(DTN, self).__init__()
        self.num_channels = input_shape[0]
        self.num_cls = num_classes
        self.conv_params = nn.Sequential (
                nn.Conv2d(self.num_channels, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )

        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.LayerNorm(512),
                )

        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, self.num_cls)
                )
    
    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y


# a = DTN((3, 32, 32))
# b = torch.randn((1, 3, 32, 32))
# x, y = a(b)
# print(x.shape)
# print(y.shape)