import torch
from torch.functional import F
from torch import nn, optim
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, datasets
from torch.autograd import grad

import tqdm
from statistics import mean

import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from Generator import G
from Discriminator import D
from preset import resl_to_lr,resl_to_ch,resl_to_batch


class Train_LSGAN:
    def __init__(self, G, D, optim_G, optim_D, label_smoothing, batch, device):
        self.G = G
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D

        self.loss = nn.MSELoss()
        self.device = device

        self.batch = batch
        self.label_smoothing = label_smoothing
        self.ones  = torch.ones(batch).to(device)
        self.zeros = torch.zeros(batch).to(device)

        self.d_hat = 0
        self.last_d_hat = 0
        self.noise = 0

        # d_pres_hat = 0.1 * d_out + 0.9 * d_last_hat
        # noise = 0.2 * (max(0, d_pres_hat - 0.5) ** 2)

    def train_D(self, x, mode, d_iter=1):
        for _ in range(d_iter):
            latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)

            fake_x = self.G.forward(latent_z, mode)
            fake_y = self.D.forward(fake_x, mode)
            fake_loss = self.loss(fake_y, self.zeros)

            real_y = self.D.forward(x, mode)
            real_loss = self.loss(real_y, self.ones - self.label_smoothing)

            self.optim_D.zero_grad()
            loss_D = fake_loss + real_loss
            loss_D.backward()
            self.optim_D.step()
            
            self.last_d_hat = self.d_hat
            self.d_hat = 0.1 * real_y.mean().item() + 0.9 * self.last_d_hat
            noise = 0.2 * (max(0, self.d_hat - 0.5) ** 2)
            self.D.update_noise(noise)

        return {"loss_D"    : loss_D.item(),
                "fake_loss" : fake_loss.item(),
                "real_loss" : real_loss.item(),
                "noise_mag" : noise,
                "mean_y"    : real_y.mean(),
#                 "alpha_D"   : self.D.module.alpha}
                "alpha_D"   : self.D.alpha}

    def train_G(self, mode):
        latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)

        fake_x = self.G.forward(latent_z, mode)
        fake_y = self.D.forward(fake_x, mode)

        self.optim_G.zero_grad()
        loss_G = self.loss(fake_y, self.ones)
        loss_G.backward()
        self.optim_G.step()

        return {"loss_G"    : loss_G.item(),
#                 "alpha_G"   : self.G.module.alpha}
                "alpha_G"   : self.G.alpha}

    def grow(self, batch, optim_G, optim_D):
        self.batch   = batch
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.ones  = torch.ones(batch).to(self.device)
        self.zeros = torch.zeros(batch).to(self.device)

        
class Train_WGAN_GP:
    def __init__(self, G, D, optim_G, optim_D, gp_lambda, eps_drift, batch, device):
        self.G = G
        self.D = D
        self.optim_G = optim_G
        self.optim_D = optim_D

        self.device = device

        self.batch = batch
        self.gp_lambda = gp_lambda
        self.eps_drift = eps_drift

    def get_gp(self, x, fake_x, mode):
        alpha = torch.rand(self.batch, 1, 1, 1).to(self.device)

        x_hat = alpha * x.detach() + (1 - alpha) * fake_x.detach()
        x_hat.requires_grad_(True)

        pred_hat = self.D(x_hat, mode)
        gradients = grad(outputs=pred_hat, inputs=x_hat,
                         grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        grad_norm = gradients.view(self.batch, -1).norm(2, dim=1)
        return self.gp_lambda * grad_norm.sub(1).pow(2).mean()

    def train_D(self, x, mode, d_iter):
        for _ in range(d_iter):
            latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)

            fake_x = self.G(latent_z, mode)
            fake_y = self.D(fake_x, mode)
            fake_loss = fake_y.mean()

            real_y = self.D(x, mode)
            real_loss = real_y.mean()

            drift = real_y.pow(2).mean() * self.eps_drift

            gp = self.get_gp(x, fake_x, mode)

            self.optim_D.zero_grad()
            loss_D = fake_loss - real_loss + drift + gp
            loss_D.backward()
            self.optim_D.step()

        return {"loss_D"    : loss_D,
                "fake_loss" : fake_loss,
                "real_loss" : real_loss,
                "drift"     : drift,
                "gp"        : gp,
#                 "alpha_D"   : self.D.module.alpha}
                "alpha_D"   : self.D.alpha}

    def train_G(self, mode):
        latent_z = torch.randn(self.batch, 512, 1, 1).normal_().to(self.device)
        fake_x = self.G(latent_z, mode)
        fake_y = self.D(fake_x, mode)

        self.optim_G.zero_grad()
        loss_G = -1 * fake_y.mean()
        loss_G.backward()
        self.optim_G.step()
        return {"loss_G"    : loss_G,
#                 "alpha_G"   : self.G.module.alpha}
                "alpha_G"   : self.G.alpha}

    def grow(self, batch, optim_G, optim_D):
        self.batch   = batch
        self.optim_D = optim_D
        self.optim_G = optim_G
        
def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
def plot_results(step,latent_z,device,times):
    with torch.no_grad():
        fake_img = step.G(latent_z,"stabilization").to(device)
        fake_img = denorm(fake_img)
        fig = plt.figure(figsize=(20,20))
        for i,img_array in enumerate(fake_img.detach().cpu().numpy()):
            if i == 36:
                break
            ax1 = fig.add_subplot(6,6,(i+1))
            img_array = img_array.transpose(1,2,0)
            ax1.imshow(img_array)
        plt.savefig(f"./result/{times}epoch_sa.jpg")
        plt.close()    
if __name__ == "__main__":
    data_path = glob.glob("../train_anime_high2*")
    data_path = sorted(data_path)

    device = "cuda:1"
    model_G = G()
    model_D = D("wgangp")

    model_G = model_G.to(device)
    model_D = model_D.to(device)

    G_optimizer = optim.Adam(model_G.parameters(),lr=resl_to_lr[4],betas=(0.0, 0.99),weight_decay=0)
    D_optimizer = optim.Adam(model_D.parameters(),lr=resl_to_lr[4],betas=(0.0, 0.99),weight_decay=0)

#     step = Train_WGAN_GP(model_G, model_D, G_optimizer, D_optimizer,10.0, 0.001 ,512, device)
    step = Train_LSGAN(model_G, model_D, G_optimizer, D_optimizer, 0 ,512, device)

    temp = 0
    times = 0

    dataset = np.load(data_path[temp])
    dataset = torch.FloatTensor(dataset)/255
    dataset = (dataset*2) - 1
    latent_z = torch.randn(resl_to_batch[step.G.resl], 512, 1, 1).normal_().to(device)
    
    check_point = False
    
    while True:  
        if check_point:
            latent_z = torch.randn(resl_to_batch[step.G.resl], 512, 1, 1).normal_().to(device)
            ##data_chenge
            temp += 1
            dataset = np.load(data_path[temp])
            dataset = torch.FloatTensor(dataset)/255
            dataset = (dataset*2) - 1
            
            model_G.grow_network(device)
            model_D.grow_network(device)
            
            model_G = model_G.to(device)
            model_D = model_D.to(device)
            
            batch_size = resl_to_batch[model_G.resl]
            G_optimizer.param_groups = []
            G_optimizer.add_param_group({"params": list(model_G.parameters())})
            D_optimizer.param_groups = []
            D_optimizer.add_param_group({"params": list(model_D.parameters())})
            lr = resl_to_lr[model_G.resl]
            for x in G_optimizer.param_groups + D_optimizer.param_groups:
                x["lr"] = lr
            check_point = False
            
            step.grow(resl_to_batch[step.G.resl],D_optimizer,G_optimizer)
            print("update_model",model_G.resl,"size")
            torch.cuda.empty_cache()
        
        for epoch in range(5):
            batch_size = resl_to_batch[step.G.resl]
            if model_G.resl == 4:
                rnd = np.random.permutation(dataset.shape[0])
                for i in range(dataset.shape[0]//batch_size):
                    input_data = dataset[rnd[batch_size*i:batch_size*(i+1)]].to(device)
                    log_d = step.train_D(input_data,"stabilization",3)
                    log_g = step.train_G("stabilization")
                    print(log_d,log_g)
#                     print("loss_D",round(log_d["loss_D"].item(),4),"loss_G",round(log_g["loss_G"].item(),4))
                check_point = True
                plot_results(step,latent_z,device,times)
                times += 1

            elif 4 < model_G.resl < 128 :
                # stabilization_update
                rnd = np.random.permutation(dataset.shape[0])
                for i in range(dataset.shape[0]//batch_size):
                    input_data = dataset[rnd[batch_size*i:batch_size*(i+1)]].to(device)
#                     print(torch.max(input_data),torch.std(input_data))
                    log_d = step.train_D(input_data,"transition",3)
                    log_d = step.train_D(input_data,"stabilization",3)
                    log_g = step.train_G("transition")
                    log_g = step.train_G("stabilization")
                    print(log_d,log_g)
#                     print(log_d)
#                     print("loss_D",round(log_d["loss_D"].item(),4),"loss_G",round(log_g["loss_G"].item(),4))
                    step.G.update_alpha(1 / (dataset.shape[0]//batch_size))
                    step.D.update_alpha(1 / (dataset.shape[0]//batch_size))

                ## transition_update
#                 rnd = np.random.permutation(dataset.shape[0])
#                 for i in range(dataset.shape[0]//batch_size):
#                     input_data = dataset[rnd[batch_size*i:batch_size*(i+1)]].to(device)
#                     log_d = step.train_D(input_data,"stabilization",1)
#                     log_g = step.train_G("stabilization")
#                     print(log_d,log_g)
#                     print("loss_D",round(log_d["loss_D"].item(),4),"loss_G",round(log_g["loss_G"].item(),4))
                check_point = True
                plot_results(step,latent_z,device,times)
                times += 1
            else:
                rnd = np.random.permutation(dataset.shape[0])
                for i in range(dataset.shape[0]//batch_size):
                    input_data = dataset[rnd[batch_size*i:batch_size*(i+1)]].to(device)
                    log_d = step.train_D(input_data,"stabilization",3)
                    log_g = step.train_G("stabilization")
                    print(log_d,log_g)
#                     print("loss_D",round(log_d["loss_D"].item(),4),"loss_G",round(log_g["loss_G"].item(),4))
                plot_results(step,latent_z,device,times)
                times += 1
        step.G.alpha = 0
        step.D.alpha = 0
        times += epoch
            