import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, input_shape):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        y = self.fc3(x)
        return x, y



# class LeNet(nn.Module):
#     "Network used for MNIST or USPS experiments."    
#     def __init__(self, input_shape, num_classes=10):
#         super(LeNet, self).__init__()
#         self.num_channels = input_shape[0]
#         self.num_cls = num_classes
#         self.conv_params = nn.Sequential(
#                 nn.Conv2d(self.num_channels, 20, kernel_size=5),
#                 nn.MaxPool2d(2),
#                 nn.ReLU(),
#                 nn.Conv2d(20, 50, kernel_size=5),
#                 nn.Dropout2d(p=0.5),
#                 nn.MaxPool2d(2),
#                 nn.ReLU(),
#                 )
        
#         self.fc_params = nn.Linear(50*4*4, 500)
#         self.classifier = nn.Sequential(
#                 nn.ReLU(),
#                 nn.Dropout(p=0.5),
#                 nn.Linear(500, self.num_cls)
#                 )
    
#     def forward(self, x):
#         x = self.conv_params(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc_params(x)
#         y = self.classifier(x)
#         return x, y


# a = LeNet((3, 32, 32))
# b = torch.randn((32, 3, 32, 32))
# x, y = a(b)
# print(x.shape)
# print(y.shape)