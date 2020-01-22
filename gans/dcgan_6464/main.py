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


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform=None):
        self.transform = transform
        self.path = path

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.path[idx])).astype(np.float32)
        img_l = cv2.resize(img, (64, 64))
        img_h = cv2.resize(img, (256, 256))
        label = 1

        #         img_l -= 127.5
        if self.transform:
            img_l = self.transform(img_l)
            img_h = self.transform(img_h)
        return img_l / 255, img_h, label
        


if __name__ == "__main__":
    img_path = glob.glob("./gan_train/*")

    # dataloaderの準備
    # === 1. データの読み込み ===
    batch_size = 128

    # dataloaderの準備
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = MyDataset(img_path, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda:0"

    model_G = dc_Generator().to(device)
    model_D = Discriminator().to(device)
    loss_f = nn.BCEWithLogitsLoss().to(device)

    params_G = optim.Adam(model_G.parameters())
    params_D = optim.Adam(model_D.parameters())

    ones = torch.ones(batch_size*2).to(device).float() # 正例 1
    zeros = torch.zeros(batch_size*2).to(device).float() # 負例 0

    test_z = torch.rand(batch_size, 100).to(device)
    for epoch in range(1000):
        losses_G = []
        losses_D = []
        for data in data_loader:
            b_size = data.shape[0]

            # Genrator
            z = torch.rand(b_size, 100).to(device)
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
            real_img = data[0].float().to(device)
            real_out = model_D(real_img)

            loss_D_real = loss_f(real_out, ones[: b_size])

            fake_out = model_D(fake_img_tensor)
            ##discriminator_fake loss
            loss_D_fake = loss_f(fake_out, zeros[: b_size])
            loss_D = loss_D_real + loss_D_fake
            losses_D.append(loss_D.item())

            model_D.zero_grad()
            model_G.zero_grad()
            loss_D.backward()
            params_D.step()
        print("Generator:     ",loss_G.item())
        print("Discriminator: ",loss_D.item())

        if (epoch % 10) == 0:
            ##log
            with open("log.txt", "a", ) as w:
                a = "-" * 50
                w.write(
                    f"{a}\n{epoch}epoch\nGenerator: {sum(losses_G)/len(losses_G)}\nDiscriminator: {sum(losses_D)/len(losses_D)} \n")

            ##img_plot
            test_fake_img = model_G(test_z)
            fig = plt.figure(figsize=(12, 12))
            for i, img_array in enumerate(test_fake_img.detach().cpu().numpy()):
                if i == 16:
                    break
                ax1 = fig.add_subplot(4, 4, (i + 1))
                img_array = img_array.transpose(1, 2, 0)
                ax1.imshow(img_array)
            plt.savefig(f"./result/{epoch}epoch.jpg")
            plt.close()
