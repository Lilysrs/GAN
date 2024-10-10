import torch
import torch.nn as nn
import utils, torch, time, os, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from WGAN_GP import Discriminator, Generator
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Learining_rate = 1e-4
Batch_size = 64
Image_size = 64
Channels_Img = 1
Z_dim = 100
Num_epochs = 5
Features_critic = 64
Features_Gen = 64
Critic_Iterations = 5
Lamda = 10

transforms = transforms.Compose(
    [
        transforms.Resize(Image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(Channels_Img)],[0.5 for _ in range(Channels_Img)]
        )
    ]
)

dataset = datasets.MNIST(root = "dataset/", transform = transforms, download = True)
datalaoder = DataLoader(dataset, batch_size = Batch_size, shuffle = True)
gen = Generator(Z_dim, Channels_Img, Features_Gen).to(device)
critic = Discriminator(Channels_Img, Features_critic).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(),lr = Learining_rate,betas = (0.0,0.9))
opt_disc = optim.Adam(critic.parameters(),lr=Learining_rate, betas = (0.0,0.9))

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_dim, 1,1).to(device)
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

gen.train()
critic.train()

for epoch in range(Num_epochs):
    for batch_idx, (real,_) in enumerate(datalaoder):
        real = real.to(device)

        for _ in range(Critic_Iterations):
                 
        ### Train Discriminator: max log(D(real)) + log(1-D(G(x)))
            noise = torch.randn(Batch_size,Z_dim,1,1).to(device)
            fake = gen(noise)
            disc_real = critic(real).reshape(-1)
            lossD_real = criterion(disc_real,torch.ones_like(disc_real))
            disc_fake = critic(fake).reshape(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            gp = gradient_penalty(critic,real,fake,device = device)
            lossD = (
                -(torch.mean(lossD_real) - torch.mean( lossD_fake)+ Lamda * gp)
                  )
            critic.zero_grad()
            lossD.backward(retain_graph = True)
            opt_disc.step()
        
        ### Train Generator min log(1-D(G(z))) <-----> max log(D(G(z)))
            output = critic(fake).reshape(-1)
            lossG = -torch.mean(output)
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()

        ### Tensorboard
            if batch_idx %100== 0:
                print(
                    f"Epoch [{epoch}/{Num_epochs}] Batch {batch_idx}/{len(DataLoader)} \
                    Loss D: {lossD: .4f}, Loss G: {lossG: .4f}"

                )

                with torch.no_grad():
                    fake = gen(fixed_noise)
                
                    img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)
                    img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                    writer_fake.add_image(
                        "MNIST Fake Images", img_grid_fake, global_step = step
                 )
                    writer_real.add_image(
                        "MNIST Real Images", img_grid_real, global_step = step
                 )

            step+=1
