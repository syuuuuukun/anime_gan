from collections import OrderedDict

import torch.nn as nn
import torch
import torch.distributions as tdist

class Discriminator(nn.Module):
    def __init__(self, sa_block):
        super().__init__()


        self.layer1 = nn.Sequential(OrderedDict([
            ("conv1" , nn.Conv2d(3, 32, 4, 2, 1, bias=False,)),
            ("bn_1"  , nn.BatchNorm2d(32)),
            ("relu_1", nn.LeakyReLU(0.2, inplace=True))]))

        self.layer2 = nn.Sequential(OrderedDict([
            ("conv2", nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            ("bn_2" , nn.BatchNorm2d(64)),
            ("relu_2",nn.LeakyReLU(0.2, inplace=True))
                                    ]))

        self.layer3 = nn.Sequential(OrderedDict([
            ("conv3", nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            ("bn_3", nn.BatchNorm2d(128)),
            ("relu_3", nn.LeakyReLU(0.2, inplace=True))
                                    ]))

        self.layer4 = nn.Sequential(OrderedDict([
            ("conv3", nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            ("bn_3", nn.BatchNorm2d(256)),
            ("relu_3",nn.LeakyReLU(0.2, inplace=True))
                                    ]))

        self.final_conv = nn.Conv2d(256, 1, 4, 1, 0, bias=False)

        # self.layer1.conv1.weight = nn.Parameter(torch.rand(32, 3, 4, 4))
        # self.layer2.conv2.weight = nn.Parameter(torch.rand(64, 32, 4, 4))
        # self.layer3.conv3.weight = nn.Parameter(torch.rand(128, 64, 4, 4))
        # self.layer4.conv4.weight = nn.Parameter(torch.rand(256, 128, 4, 4))
        # self.final_conv.weight = nn.Parameter(torch.rand(256,1,4,4))

    def forward(self, x):
        x = self.main(x)
        x = self.final_conv(x).squeeze()
        return x


