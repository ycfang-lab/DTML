import torch
from torch import nn
import torch.nn.functional as F


class DiscriminatorBlock(nn.Module):
    """
    Discriminator block
    """
    def __init__(self, input_features:int, output_features:int, normalize:bool = True):
        """
        args:
            input_features: input features
            output_features: output features
            normalize: normalize or not
        """
        super(DiscriminatorBlock, self).__init__()
        if normalize:
            self.block = nn.Sequential(
                nn.Conv2d(input_features, output_features, 4, stride=2, padding=1),
                nn.InstanceNorm2d(output_features),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(input_features, output_features,4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )

    def forward(self, x):
        x = self.block(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator
    """
    def __init__(self, input_shape:tuple, deep:int=4):
        """
        args:
            input_shape: (channels, height, weight)
        """
        super(Discriminator, self).__init__()
        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** deep, width // 2 ** deep)

        discrimiator_blocks = [DiscriminatorBlock(64 * 2 ** i, 128 * 2 ** i) for i in range(deep - 1)]
        self.model = nn.Sequential(
            DiscriminatorBlock(channels, 64, normalize=False),
            *discrimiator_blocks,
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64 * 2 ** (deep - 1), 1, 4, padding=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x

