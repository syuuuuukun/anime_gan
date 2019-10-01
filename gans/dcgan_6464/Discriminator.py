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
            ("conv4", nn.Conv2d(128, 256, 4, 2, 1, bias=False)),
            ("bn_4", nn.BatchNorm2d(256)),
            ("relu_4",nn.LeakyReLU(0.2, inplace=True))
                                    ]))

        self.final_conv = nn.Conv2d(256, 1, 4, 1, 0, bias=False)

        nn.init.normal_(self.layer1.conv1.weight, 0.0, 0.02)
        nn.init.normal_(self.layer2.conv2.weight, 0.0, 0.02)
        nn.init.normal_(self.layer3.conv3.weight, 0.0, 0.02)
        nn.init.normal_(self.layer4.conv4.weight, 0.0, 0.02)
        nn.init.normal_(self.final_conv.weight, 0.0, 0.02)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x).squeeze()
        return x


