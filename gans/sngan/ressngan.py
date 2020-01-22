import torch
from torch.functional import F
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
import tqdm
from statistics import mean

import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class MyDataset(torch.utils.data.Dataset):

    def __init__(self,data,transform=None):
        self.transform = transform
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_l = self.data[idx]
        img_l = img_l/255
        
        label = 1
        return img_l.transpose(2,0,1),label
    
def conv3x3(in_channels, out_channels,kernel_size=3,stride=1,padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,upsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels,3,stride)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels,3,stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.upsample = upsample
        self.conv3 = conv3x3(in_channels, out_channels,1,1,0)
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        
    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
#         out = self.conv1(out)       
        if self.upsample:
            residual = self.conv3(self.upsample(residual))            
            out = self.conv1(self.upsample(out))
        else:
            out = self.conv1(out)           
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        if self.downsample:
            residual = self.conv3(self.downsample(residual))
            out = self.downsample(out)
        out = torch.add(out,residual)
#         out = self.relu(out)
        return out
    
    
class Reshape(nn.Module):
    def __init__(self,c_size,h_size,w_size):
        super().__init__()
        self.c_size = c_size
        self.h_size = h_size
        self.w_size = w_size
    def forward(self, x):
        return x.view(x.shape[0], self.c_size,self.h_size,self.w_size)
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

class Generator(nn.Module):
    def __init__(self, z_dims=512, d=64):
        super().__init__()
        self.deconv1 = nn.utils.spectral_norm(nn.ConvTranspose2d(z_dims, d * 8, 4, 1, 0))
        self.deconv2 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1))
        self.deconv3 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1))
        self.deconv4 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1))
        self.deconv5 = nn.utils.spectral_norm(nn.ConvTranspose2d(d * 2, d, 4, 2, 1))
        self.deconv6 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        
        self.fc1 = nn.Linear(512,512*4*4)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)  # 1 x 1
        x = F.relu(self.deconv1(input))  # 4 x 4
        x = F.relu(self.deconv2(x))  # 8 x 8
        x = F.relu(self.deconv3(x))  # 16 x 16
        x = F.relu(self.deconv4(x))  # 32 x 32
        x = F.relu(self.deconv5(x))  # 64 x 64
        x = F.tanh(self.deconv6(x))  # 128 x 128
        return x

class res_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc_layer = nn.Linear(100,512*4*4)
        self.reshape  = Reshape(512,4,4)
        self.res1 = nn.Sequential(ResidualBlock(512,256,upsample=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res2 = nn.Sequential(ResidualBlock(256,128,upsample=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res3 = nn.Sequential(ResidualBlock(128,64,upsample=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res4 = nn.Sequential(ResidualBlock(64,32,upsample=nn.UpsamplingNearest2d(scale_factor=2)))
        self.res5 = nn.Sequential(ResidualBlock(32,32,upsample=nn.UpsamplingNearest2d(scale_factor=2)))
        self.out  = nn.Conv2d(32, 3, 1, 1, bias=False)
        self.tanh = nn.Tanh()
        
        nn.init.xavier_normal_(self.fc_layer.weight)
        nn.init.xavier_normal_(self.out.weight)
    def forward(self, x):
        x = self.fc_layer(x)
        x = self.reshape(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.out(x)   
        x = self.tanh(x)
        return x

class res_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = ResidualBlock(3,64,downsample=nn.AvgPool2d(3,2,1))
        self.res2 = ResidualBlock(64,128,downsample=nn.AvgPool2d(3,2,1))
        self.res3 = ResidualBlock(128,256,downsample=nn.AvgPool2d(3,2,1))
        self.res4 = ResidualBlock(256,256,downsample=nn.AvgPool2d(3,2,1))
        self.res5 = ResidualBlock(256,512,downsample=nn.AvgPool2d(3,2,1))
        self.res6 = nn.Sequential(ResidualBlock(512,512),Flatten()) 
        self.fc = nn.Linear(512*4*4,1)
        
        nn.init.xavier_normal_(self.fc.weight.data)
    
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.fc(x)
        return x
    

    
if __name__ == "__main__":
    device = "cuda"
    batch_size = 1024
    model_G = res_Generator().to(device)
    model_D = res_Discriminator().to(device)

    model_G = torch.nn.DataParallel(model_G)
    model_D = torch.nn.DataParallel(model_D)

    loss_f = nn.CrossEntropyLoss().to(device)

    params_G = optim.Adam(model_G.parameters(),lr=0.0002,betas=(0.5, 0.99))
    params_D = optim.Adam(model_D.parameters(),lr=0.0002,betas=(0.5, 0.99))

    ones = torch.ones(batch_size*2).to(device).long() # 正例 1
    zeros = torch.zeros(batch_size*2).to(device).long() # 負例 0
    
    data = np.load("train_anime_128128.npy")
    data = torch.FloatTensor(data/255)
    test_z = torch.randn(50, 100).to(device)
    
    dis_num = 2

    for epoch in range(1000):
        losses_G = []
        losses_D = []
        rnd = np.random.permutation(data.shape[0])
        for i,batch in enumerate(range(0,data.shape[0],batch_size*dis_num)): 
            start_idx = ((batch_size*dis_num)*i)
            end_idx   = ((batch_size*dis_num)*(i+1))
            img = data[rnd[start_idx:end_idx]]
            for dis_n in range(dis_num):

                real_img = img[batch_size*dis_n:batch_size*(dis_n+1)]
                b_size = real_img.shape[0]
                z = torch.randn(b_size, 100).to(device)
                fake_img = model_G(z)
                fake_img_tensor = fake_img.detach()

                ##discriminator loss

                real_img = real_img.float()
                real_img = (real_img*2)-1
                real_img = real_img.to(device)

                real_out = model_D(real_img)
                
                "hinge"
                loss_D_real = torch.nn.ReLU()(1.0 - real_out).mean()
                "CrossEntropy"
    #             loss_D_real = loss_f(real_out, ones[: b_size])

                fake_out = model_D(fake_img_tensor)
                ##discriminator_fake loss
                "hinge"
                loss_D_fake = torch.nn.ReLU()(1.0 + fake_out).mean()
                "crossentropy"
    #             loss_D_fake = loss_f(fake_out, zeros[: b_size])
                loss_D = loss_D_real + loss_D_fake

                losses_D.append(loss_D.item())

                model_D.zero_grad()
                model_G.zero_grad()
                loss_D.backward()
                params_D.step()


            #Genrator
            z = torch.randn(b_size, 100).to(device)
            for gen_n in range(1):
                fake_img = model_G(z)
                fake_img_tensor = fake_img.detach()

                out = model_D(fake_img)
                loss_G = - out.mean()
    #             loss_G = loss_f(out, ones[: b_size])
                losses_G.append(loss_G.item())

                model_D.zero_grad()
                model_G.zero_grad()
                loss_G.backward()
                params_G.step()



        if (epoch%10) == 0:
            print("Genrator",sum(losses_G)/len(losses_G))
            print("Discriminator",sum(losses_D)/len(losses_D))
            with open("log.txt","a",) as w:
                a = "-"*50
                w.write(f"{a}\n{epoch}epoch\nGenerator: {sum(losses_G)/len(losses_G)}\nDiscriminator: {sum(losses_D)/len(losses_D)} \n")

            ##img_plot
            test_fake_img = model_G(test_z)
            test_fake_img = denorm(test_fake_img)
            fig = plt.figure(figsize=(20,20))
            for i,img_array in enumerate(test_fake_img.detach().cpu().numpy()):
                if i == 36:
                    break
                ax1 = fig.add_subplot(6,6,(i+1))
                img_array = img_array.transpose(1,2,0)
                ax1.imshow(img_array)
            plt.savefig(f"./result/{epoch}epoch_sa.jpg")
            plt.close()