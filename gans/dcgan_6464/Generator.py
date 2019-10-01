import torch.nn as nn

class dc_Generator(nn.Module):

    def __init__(self, sa_block):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(100,4*4*512),
            Reshape(512,4,4),
            
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

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