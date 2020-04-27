#標準
import glob

#追加
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

#学習系
import torch
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets

##model
from Discriminator import Discriminator
from Generator import dc_Generator
from my_utils import gan_img_renorm,gp_loss



def to_numpy(x):
    return x.detach().cpu().numpy()

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform=None):
        self.transform = transform
        self.paths = path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.paths[idx]
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
            img = (img*2)-1
        return img
        


if __name__ == "__main__":
    gp = True
    multi_gpu = True

    train_data = np.load("../../train_data_64_50000.npy")
    print(train_data.shape)

    # dataloaderの準備
    # === 1. データの読み込み ===
    batch_size = 2000

    # dataloaderの準備
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MyDataset(train_data, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda"

    model_G = dc_Generator()
    model_D = Discriminator()
    loss_f = nn.BCEWithLogitsLoss()

    # if multi_gpu:
    #     _ = convert_model(model_D)
    #     _ = convert_model(model_G)
    #     model_D = DataParallelWithCallback(model_D)
    #     model_G = DataParallelWithCallback(model_G)

    model_G = model_G.to(device)
    model_D = model_D.to(device)


    params_G = optim.Adam(model_G.parameters(),lr=0.0002,betas=(0.5, 0.99))
    params_D = optim.Adam(model_D.parameters(),lr=0.0002,betas=(0.5, 0.99))

    ones = torch.ones(batch_size*2).to(device).float() # 正例 1
    zeros = torch.zeros(batch_size*2).to(device).float() # 負例 0

    test_z = torch.randn(batch_size, 100).to(device)
    test_z = test_z.normal_(0.0,1.0)

    print("dataset_num: ",len(dataset))
    print("1epoch_iteration: ",len(data_loader))
    for epoch in range(10000):
        losses_G = []
        losses_D = []
        for data in data_loader:
            b_size = data.shape[0]
            # print(b_size)

            # Genrator
            z = torch.randn(b_size, 100)
            z = z.normal_(0.0,1.0).to(device)
            # print(z)
            fake_img = model_G(z)
            fake_img_tensor = fake_img.detach()

            out = model_D(fake_img)
            loss_G = loss_f(out, ones[: b_size])
            losses_G.append(loss_G.item())

            model_D.zero_grad()
            model_G.zero_grad()
            loss_G.backward()
            params_G.step()

            ##discriminator loss
            real_img = data.float().to(device)
            real_out = model_D(real_img)

            loss_D_real = loss_f(real_out, ones[: b_size])

            fake_out = model_D(fake_img_tensor)
            ##discriminator_fake loss
            loss_D_fake = loss_f(fake_out, zeros[: b_size])
            loss_D = loss_D_real + loss_D_fake

            if gp:
                loss_D += gp_loss(fake_img_tensor,real_img,model_D)

            losses_D.append(loss_D.item())

            model_D.zero_grad()
            model_G.zero_grad()
            loss_D.backward()
            params_D.step()
        print(f"Generator: {np.mean(losses_G):.4} , Discriminator: {np.mean(losses_D):.4}")

        ##img_plot
        if (epoch%10) == 0:
            with torch.no_grad():   
                model_G.eval()
                test_fake_img = gan_img_renorm(model_G(test_z))
                fig = plt.figure(figsize=(12, 12))
                size = int(np.sqrt(len(test_fake_img))+1)
                for i, img_array in enumerate(to_numpy(test_fake_img)):
                    if i == 25:
                        break
                    ax1 = fig.add_subplot(5, 5, (i + 1))
                    ax1.imshow(img_array.transpose(1,2,0))
                plt.savefig(f"./result/{epoch}epoch.jpg")
                plt.close()
                model_G.train()
