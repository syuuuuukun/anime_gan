import torch.nn as nn
import torch

from spectral_norm import SpectralNorm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find("ConvTranspose2d") != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.xavier_normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)
        # torch.nn.init.constant_(m.bias, 0.0)


class Reshape(nn.Module):
    def __init__(self, c_size, h_size, w_size):
        super().__init__()
        self.c_size = c_size
        self.h_size = h_size
        self.w_size = w_size

    def forward(self, x):
        return x.view(-1, self.c_size, self.h_size, self.w_size)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])


class Reshape(nn.Module):
    def __init__(self, c_size, h_size, w_size):
        super().__init__()
        self.c_size = c_size
        self.h_size = h_size
        self.w_size = w_size

    def forward(self, x):
        return x.view(x.shape[0], self.c_size, self.h_size, self.w_size)


def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1, sn_layer=False):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=False)
    weights_init_normal(conv)
    if sn_layer:
        return SpectralNorm(conv)
    else:
        return conv


def linear_unit(indim, outdim, sn_layer=False):
    lin = nn.Linear(indim, outdim, bias=False)
    weights_init_normal(lin)
    if sn_layer:
        return SpectralNorm(lin)
    else:
        return lin


class ResidualBlock_gen(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sampling=None):
        super(ResidualBlock_gen, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, 3, stride)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 3, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.upsample = sampling
        self.conv3 = conv3x3(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        residual = self.conv3(self.upsample(residual))
        out = self.conv1(self.upsample(out))
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.add(out, residual)
        return out


class ResidualBlock_dis(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sampling=None):
        super(ResidualBlock_dis, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, 3, stride)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, 3, stride)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.sampling = sampling
        self.conv3 = conv3x3(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        residual = x
        out = self.relu1(x)
        residual = self.sampling(self.conv3(residual))
        out = self.conv1(out)
        out = self.relu2(out)
        out = self.sampling(self.conv2(out))
        out = torch.add(out, residual)
        return out


class res_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_layer = linear_unit(100, 1024 * 4 * 4)
        self.reshape = Reshape(1024, 4, 4)
        self.res1_1 = nn.Sequential(ResidualBlock_gen(1024, 1024, sampling=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res1 = nn.Sequential(ResidualBlock_gen(1024, 512, sampling=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res2 = nn.Sequential(ResidualBlock_gen(512, 256, sampling=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res3 = nn.Sequential(ResidualBlock_gen(256, 128, sampling=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res4 = nn.Sequential(ResidualBlock_gen(128, 64, sampling=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res5 = nn.Sequential(ResidualBlock_gen(64, 32, sampling=nn.UpsamplingNearest2d(scale_factor=2)))
        self.out = nn.Sequential(nn.BatchNorm2d(32), nn.ReLU(inplace=True), conv3x3(32, 3, 1, 1, 0))
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc_layer(x)
        x = self.reshape(x)
        x = self.res1_1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        # x = self.res6(x)
        x = self.out(x)
        x = self.tanh(x)
        return x


class res_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = ResidualBlock_dis(3, 64, sampling=nn.AvgPool2d(kernel_size=2))
        self.res2 = ResidualBlock_dis(64, 128, sampling=nn.AvgPool2d(kernel_size=2))
        self.res3 = ResidualBlock_dis(128, 256, sampling=nn.AvgPool2d(kernel_size=2))
        self.res4 = ResidualBlock_dis(256, 512, sampling=nn.AvgPool2d(kernel_size=2))
        self.res5 = ResidualBlock_dis(512, 1024, sampling=nn.AvgPool2d(kernel_size=2))
        self.res6 = nn.Sequential(ResidualBlock_dis(1024, 1024, sampling=nn.AvgPool2d(kernel_size=2)))

        self.conv = nn.Conv2d(1024, 1, 4, 1, bias=False)
        # self.conv = SpectralNorm(nn.Conv2d(1024,1,4,1,bias=False))

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        x4 = self.res4(x3)
        x5 = self.res5(x4)
        x6 = self.res6(x5)
        x = self.conv(x6)
        x = x.view(-1)
        return x

