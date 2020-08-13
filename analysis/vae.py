import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

LATENT_DIM = 16

class VAE(nn.Module):
    def __init__(self, z_dim=LATENT_DIM, en_activation=F.leaky_relu, dec_activation=F.relu):
        super(VAE, self).__init__()
        
        self.en_activ = en_activation
        self.dec_activ = dec_activation

        self.conv1 = nn.Conv2d( 1,   32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d( 32,  64, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d( 64, 128, 3, stride=2, padding=0)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=0)
        self.conv5 = nn.Conv2d(256, 256, 3, stride=2, padding=0)

        self.fc1 = nn.Linear(256 + 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3a = nn.Linear(64, z_dim)
        self.fc3b = nn.Linear(64, z_dim)

        self.fc1t = nn.Linear(z_dim + 5, z_dim * 2 * 2)
        self.fc2t = nn.Linear(z_dim * 2 * 2, z_dim * 2 * 2 * 4)
        self.fc3t = nn.Linear(z_dim * 2 * 2 * 4, z_dim * 2 * 2 * 8)


        # (Z x 8) x 2 x 2
        self.conv1t = nn.ConvTranspose2d(z_dim * 8, 128, 3, stride=2, padding=1, output_padding=1) 
        # 128 x 8 x 8
        self.conv2t = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        # 64 x 16 x 16
        self.conv3t = nn.ConvTranspose2d(64, 32,  3, stride=2, padding=1, output_padding=1)
        # 32 x 32 x 32
        self.conv4t = nn.ConvTranspose2d(32, 1,  3, stride=2, padding=1, output_padding=1)
        # 1 x 32 x 32
        # crop
        # 1 x 30 x 30

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def encode(self, X, params):
        X = self.en_activ(self.conv1(X))
        X = self.en_activ(self.conv2(X))
        X = self.en_activ(self.conv3(X))
        X = self.en_activ(self.conv4(X))

        X = X.reshape(-1,256)
        X = torch.cat([X, params], dim=1)
        X = self.en_activ(self.fc1(X))
        X = self.en_activ(self.fc2(X))
        return self.en_activ(self.fc3a(X)), self.en_activ(self.fc3b(X))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, params):
        z = self.en_activ(self.fc1t(torch.cat([z, params], dim=1)))
        z = self.en_activ(self.fc2t(z))
        z = self.en_activ(self.fc3t(z))
        z = z.view(-1, LATENT_DIM * 8, 2, 2)

        z = self.dec_activ(self.bn1(self.conv1t(z)))
        #print(EnergyDeposit.shape)
        z = self.dec_activ(self.bn2(self.conv2t(z)))
        #print(EnergyDeposit.shape)
        z = self.dec_activ(self.bn3(self.conv3t(z)))
        #print(EnergyDeposit.shape)
        z = self.dec_activ(self.conv4t(z))
        
        z = z[:,:,1:31,1:31]
        return z

    def forward(self, x, params):
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, params), mu, logvar