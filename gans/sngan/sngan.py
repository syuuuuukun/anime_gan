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
    
class MyDataset(Dataset):

    def __init__(self,data,transform=None):
        self.transform = transform
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img_l = self.data[idx]/255       
        label = 1
        return img_l,label    
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("ConvTranspose2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    
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
        x = self.fc1(input)
        x = x.view(-1,512,4,4)
        
#         input = input.view(input.size(0), input.size(1), 1, 1)  # 1 x 1
#         x = F.relu(self.deconv1(input))  # 4 x 4
        x = F.relu(self.deconv2(x))  # 8 x 8
        x = F.relu(self.deconv3(x))  # 16 x 16
        x = F.relu(self.deconv4(x))  # 32 x 32
        x = F.relu(self.deconv5(x))  # 64 x 64
        x = F.tanh(self.deconv6(x))  # 128 x 128
        return x
    
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.layer1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.layer2 = nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False))
        self.layer3 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False))
        self.layer4 = nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False))
        self.layer5 = nn.utils.spectral_norm(nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False))
        self.layer6 = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        out = F.leaky_relu(self.layer1(input), 0.2, inplace=True)  # 64 x 64
        out = F.leaky_relu(self.layer2(out), 0.2, inplace=True)  # 32 x 32
        out = F.leaky_relu(self.layer3(out), 0.2, inplace=True)  # 16 x 16
        out = F.leaky_relu(self.layer4(out), 0.2, inplace=True)  # 8 x 8
        out = F.leaky_relu(self.layer5(out), 0.2, inplace=True)  # 4 x 4
        out = self.layer6(out)  # 1 x 1
        
        return out.view(-1, 1)
    
if __name__ == "__main__":
    device = "cuda"
    batch_size = 512
    model_G = Generator().to(device)
    model_D = Discriminator().to(device)

    model_G.apply(weights_init_normal)
    model_D.apply(weights_init_normal)
    
    model_G = torch.nn.DataParallel(model_G)
    model_D = torch.nn.DataParallel(model_D)
    
    loss_f = nn.CrossEntropyLoss().to(device)

    params_G = optim.Adam(model_G.parameters(),lr=0.0001,betas=(0.5, 0.99))
    params_D = optim.Adam(model_D.parameters(),lr=0.0004,betas=(0.5, 0.99))

    ones = torch.ones(batch_size*2).to(device).long() # 正例 1
    zeros = torch.zeros(batch_size*2).to(device).long() # 負例 0
    
    data = np.load("../train_anime_high2_128128.npy")
    
    dataset = MyDataset(data)
    loader = DataLoader(dataset, batch_size=256, shuffle=True,pin_memory=True)
    
    test_z = torch.randn(50, 512).to(device)
    
    dis_num = 5

    for epoch in range(1000):
        losses_G = []
        losses_D = []
        rnd = np.random.permutation(data.shape[0])
        for i,batch in enumerate(range(0,data.shape[0],batch_size*dis_num)): 
            for dis_n in range(dis_num):
                real_img = loader.__iter__().next()[0]
                b_size = real_img.shape[0]
                z = torch.randn(b_size, 512).to(device)
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
            z = torch.randn(b_size, 512).to(device)
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