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
    def __init__(self, z_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.conv1a = nn.Conv2d(1,  32, 3, stride=1, padding=1)
        self.conv1 = nn.Conv2d( 32,   64, 3, stride=2, padding=1)
        # self.conv2a = nn.Conv2d(32,  64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d( 64,  128, 3, stride=2, padding=0)
        # self.conv3a = nn.Conv2d(64,  128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d( 128, 256, 3, stride=2, padding=0)
        # self.conv4a = nn.Conv2d(128,  256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, stride=2, padding=0)

        self.en_bn1a = nn.BatchNorm2d(32)
        self.en_bn1 = nn.BatchNorm2d(64)
        self.en_bn2a = nn.BatchNorm2d(64)
        self.en_bn2 = nn.BatchNorm2d(128)
        self.en_bn3a = nn.BatchNorm2d(128)
        self.en_bn3 = nn.BatchNorm2d(256)
        self.en_bn4a = nn.BatchNorm2d(256)
        self.en_bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 + 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3a = nn.Linear(64, z_dim)
        self.fc3b = nn.Linear(64, z_dim)

        self.en_bnfc1 = nn.BatchNorm1d(128)
        self.en_bnfc2 = nn.BatchNorm1d(64)

        self.fc1t = nn.Linear(z_dim, z_dim * 2 * 2)
        self.fc2t = nn.Linear(z_dim * 2 * 2 + 5, z_dim * 2 * 2 * 4)
        self.fc3t = nn.Linear(z_dim * 2 * 2 * 4, z_dim * 2 * 2 * 8)

        self.bnfc1t = nn.BatchNorm1d(z_dim * 2 * 2)
        self.bnfc2t = nn.BatchNorm1d(z_dim * 2 * 2 * 4)
        self.bnfc3t = nn.BatchNorm1d(z_dim * 2 * 2 * 8)

        self.conv1t = nn.ConvTranspose2d(z_dim * 8, 128, 3, stride=2, padding=1, output_padding=1) 
        # self.conv1ta = nn.ConvTranspose2d(256, 128,  3, stride=1, padding=0, output_padding=0)
        self.conv2t = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        # self.conv2ta = nn.ConvTranspose2d(128, 128,  3, stride=1, padding=0, output_padding=0)
        self.conv3t = nn.ConvTranspose2d(128, 64,  3, stride=2, padding=1, output_padding=1)
        # self.conv3ta = nn.ConvTranspose2d(64, 32,  3, stride=1, padding=0, output_padding=0)
        self.conv4t = nn.ConvTranspose2d(64, 32,  3, stride=2, padding=1, output_padding=1)
        self.conv4ta = nn.ConvTranspose2d(32, 1,  3, stride=1, padding=0, output_padding=0)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn1a = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn2a = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn3a = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)


    def encode(self, X, params):
        X = F.leaky_relu(self.en_bn1a(self.conv1a(X)))
        X = F.leaky_relu(self.en_bn1(self.conv1(X)))
        # X = F.leaky_relu(self.en_bn2a(self.conv2a(X)))
        X = F.leaky_relu(self.en_bn2(self.conv2(X)))
        # X = F.leaky_relu(self.en_bn3a(self.conv3a(X)))
        X = F.leaky_relu(self.en_bn3(self.conv3(X)))
        # X = F.leaky_relu(self.en_bn4a(self.conv4a(X)))
        X = F.leaky_relu(self.en_bn4(self.conv4(X)))

        X = X.reshape(-1,256)
        X = torch.cat([X, params], dim=1)
        X = F.leaky_relu(self.en_bnfc1(self.fc1(X)))
        X = F.leaky_relu(self.en_bnfc2(self.fc2(X)))
        return self.fc3a(X), self.fc3b(X)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, params):
        z = F.leaky_relu(self.bnfc1t(self.fc1t(z)))
        z = F.leaky_relu(self.bnfc2t(self.fc2t(torch.cat([z, params], dim=1))))
        z = F.leaky_relu(self.bnfc3t(self.fc3t(z)))
        z = z.view(-1, LATENT_DIM * 8, 2, 2)

        z = F.leaky_relu(self.bn1(self.conv1t(z)))
        # z = F.leaky_relu(self.bn1a(self.conv1ta(z)))
        # print(z.shape)
        z = F.leaky_relu(self.bn2(self.conv2t(z)))
        # z = F.leaky_relu(self.bn2a(self.conv2ta(z)))
        # print(z.shape)
        z = F.leaky_relu(self.bn3(self.conv3t(z)))
        # z = F.leaky_relu(self.bn3a(self.conv3ta(z)))
        # print(z.shape)
        z = F.leaky_relu(self.bn4(self.conv4t(z)))
        # z = F.relu(self.conv4t(z))
        z = F.relu(self.conv4ta(z))
        
        z = z[:,:,1:31,1:31]
        return z

    def forward(self, x, params):
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, params), mu, logvar
