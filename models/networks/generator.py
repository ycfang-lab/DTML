import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Residual module
    """
    def __init__(self, input_features:int):
        """
        args:
            input_features: input features
        """
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_features, input_features, 3),
            nn.InstanceNorm2d(input_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_features, input_features, 3), 
            nn.InstanceNorm2d(input_features)
        )

    def forward(self, x):
        x = x + self.block(x)
        return x


class GeneratorResNet(nn.Module):
    """
    Generator based on ResNet
    """
    def __init__(self, input_shape:tuple, num_residual_blocks:int, kernel_size=7):
        """
        args:
            input_shape: (channels, height, weight)
            num_residual_blocks: number of residual blocks
        """
        super(GeneratorResNet, self).__init__()
        channels = input_shape[0]

        # Initial convolution
        out_features = 64
        model = [
            nn.ReflectionPad2d((kernel_size-1)//2),
            nn.Conv2d(channels, out_features, kernel_size),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        input_features = out_features

        # down-sampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(input_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            input_features = out_features

        # residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # up-sampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(input_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            input_features = out_features
        
        # output layer
        model += [nn.ReflectionPad2d((kernel_size-1)//2), nn.Conv2d(out_features, channels, kernel_size), nn.Tanh()]
        
        # complete model
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


# a = GeneratorResNet((3,32,32), 3, 5)
# b = torch.randn((5,3,32,32))
# print(a(b).shape)