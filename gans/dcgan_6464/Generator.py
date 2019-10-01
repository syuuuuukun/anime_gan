import torch.nn as nn
import torch
from collections import OrderedDict

class dc_Generator(nn.Module):

    def __init__(self, sa_block):
        super().__init__()

        self.layer1 = nn.Sequential(OrderedDict([
            ("fc", nn.Linear(100,4*4*512)),
            ("reshape" , Reshape(512,4,4)),
            ("conv1", nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)),
            ("bn_1", nn.BatchNorm2d(256)),
            ("relu1", nn.LeakyReLU(0.2, inplace=True))]))

        self.layer2 = nn.Sequential(OrderedDict([
            ("conv2", nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False)),
            ("bn_2" , nn.BatchNorm2d(128)),
            ("relu_2",nn.LeakyReLU(0.2, inplace=True))
                                    ]))

        self.layer3 = nn.Sequential(OrderedDict([
            ("conv3", nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False)),
            ("bn_3" , nn.BatchNorm2d(128)),
            ("relu_3",nn.LeakyReLU(0.2, inplace=True))
                                    ]))

        self.layer4 = nn.Sequential(OrderedDict([
            ("conv4", nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False)),
            ("Tanh" , nn.Tanh())
                                    ]))

        nn.init.normal_(self.layer1.conv1.weight, 0.0, 0.02)
        nn.init.normal_(self.layer2.conv2.weight, 0.0, 0.02)
        nn.init.normal_(self.layer3.conv3.weight, 0.0, 0.02)
        nn.init.normal_(self.layer4.conv4.weight, 0.0, 0.02)

    def forward(self, x):
        return self.main(x)

class Reshape(nn.Module):
    def __init__(self,c_size,h_size,w_size):
        super().__init__()
        self.c_size = c_size
        self.h_size = h_size
        self.w_size = w_size

    def forward(self,x):
        return x.view(-1,self.c_size,self.h_size,self.w_size)