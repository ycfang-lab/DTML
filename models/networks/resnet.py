from torch import nn
from torchvision import models

from utils import weights_init_normal

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}


class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.bottleneck.apply(weights_init_normal)
                self.fc.apply(weights_init_normal)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.apply(weights_init_normal)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
                x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    # def get_parameters(self):
    #     if self.new_cls:
    #             if self.use_bottleneck:
    #                     parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
    #                                                     {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
    #                                                     {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    #             else:
    #                     parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
    #                                                     {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    #     else:
    #             parameter_list = [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]
    #     return parameter_list