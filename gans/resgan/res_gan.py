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
from sync_batchnorm import convert_model, DataParallelWithCallback

from my_utils import gan_img_renorm, gp_loss, log_ploter
from model import *


class log_ploter(object):
    def __init__(self):
        self.names = None
        self.losses = []

    def ploter(self):
        print()
        for name, loss in zip(self.names, self.losses):
            print(f"{name}: {np.mean(loss):.4}")
        self.losses = [[] for i in range(len(self.names))]

    def get_var_names(self, vars):
        names = []
        for var in vars:
            for k, v in globals().items():
                if id(v) == id(var):
                    names.append(k)
        return names

    def updater(self, losses):
        if self.names is None:
            self.names = self.get_var_names(losses)
            self.losses = [[] for i in range(len(self.names))]

        for i, loss in enumerate(losses):
            self.losses[i].append(loss.item())


def to_numpy(x):
    return x.detach().cpu().numpy()


def gp_loss(fake, real, model_D, label=None, embed=None, epsilon=1e-3):
    b, c, h, w = fake.shape
    epsilon = torch.rand(b, 1, 1, 1, dtype=fake.dtype, device=fake.device)
    intpl = epsilon * fake + (1 - epsilon) * real
    intpl.requires_grad_()
    if label is None:
        f = model_D.forward(intpl)
        grad = torch.autograd.grad(f.sum(), intpl, create_graph=True)[0]
    else:
        f = model_D.forward(intpl, embed(label))
        grad = torch.autograd.grad(f[1].sum(), intpl, create_graph=True)[0]
    grad_norm = grad.view(b, -1).norm(dim=1)
    loss_gp = 5 * ((grad_norm ** 2).mean())
    return loss_gp


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img)
        img = self.transform(img)
        img = (img * 2) - 1
        return img

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

if __name__ == "__main__":
    device = "cuda:0"
    batch_size = 64
    multi_gpu = True
    model_G = res_Generator()
    model_D = res_Discriminator()
    print(model_G)
    print(model_D)

    model_G.apply(weights_init_normal)
    model_D.apply(weights_init_normal)

    if multi_gpu:
        _ = convert_model(model_D)
        _ = convert_model(model_G)
        model_D = DataParallelWithCallback(model_D)
        model_G = DataParallelWithCallback(model_G)

    model_D = model_D.to(device)
    model_G = model_G.to(device)

    params_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.99))
    params_D = optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.99))

    datas = np.load("../../data/yosida_full1data_256.npy")
    test_z = torch.randn(batch_size, 100).to(device)
    test_z = test_z.normal_(0.0, 1.0)

    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = MyDataset(datas, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    dis_num = 2
    global_steps = 0

    repoter = log_ploter()

    while True:
        for dis_n in range(dis_num):
            real_img = iter(data_loader).next()
            real_img = real_img.to(device)
            z = torch.randn(real_img.shape[0], 100).to(device)
            z = z.normal_(0.0, 1.0)
            fake_img = model_G(z)
            fake_img_tensor = fake_img.detach()

            real_out = model_D(real_img)

            loss_D_real = torch.nn.ReLU()(1.0 - real_out).mean()

            fake_out = model_D(fake_img_tensor)
            loss_D_fake = torch.nn.ReLU()(1.0 + fake_out).mean()
            loss_D = loss_D_real + loss_D_fake

            gp = gp_loss(fake_img_tensor, real_img, model_D)
            loss_D += gp

            loss_D /= dis_num

            model_D.zero_grad()
            model_G.zero_grad()
            loss_D.backward()
            params_D.step()
        # Genrator
        z = torch.randn(real_img.shape[0], 100).to(device)
        z = z.normal_(0.0, 1.0)
        for gen_n in range(1):
            fake_img = model_G(z)
            fake_img_tensor = fake_img.detach()

            out = model_D(fake_img)
            loss_G = -out.mean()

            model_D.zero_grad()
            model_G.zero_grad()
            loss_G.backward()
            params_G.step()

        repoter.updater([loss_D, gp, loss_G])

        if (global_steps % 100) == 0:
            repoter.ploter()
            ##img_plot
            test_fake_img = model_G(test_z)
            test_fake_img = denorm(test_fake_img)
            fig = plt.figure(figsize=(20, 20))
            size = int(np.sqrt(test_fake_img.shape[0])) + 1
            print(size, to_numpy(test_fake_img).shape)
            for i, img_array in enumerate(to_numpy(test_fake_img)[:4]):
                ax1 = fig.add_subplot(2, 2, (i + 1))
                img_array = img_array.transpose(1, 2, 0)
                ax1.imshow(img_array)
            plt.savefig(f"./result/{global_steps}epoch.jpg")
            plt.close()
        if (global_steps % 8800) == 0:
            _ = model_D.to("cpu")
            _ = model_G.to("cpu")
            torch.save(model_D.module.state_dict(), f"{global_steps}steps_modelD")
            torch.save(model_G.module.state_dict(), f"{global_steps}steps_modelG")
            _ = model_D.to(device)
            _ = model_G.to(device)
        global_steps += 1