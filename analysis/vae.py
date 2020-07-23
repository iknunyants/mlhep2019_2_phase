import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

LATENT_DIM = 32

class VAE(nn.Module):
    def __init__(self, h_dim=128, z_dim=LATENT_DIM):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(2)  # 15x15
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))  # 7x7
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(h_dim + 5, z_dim)
        self.fc2 = nn.Linear(h_dim + 5, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.fc4 = nn.Linear(h_dim + 5, 512)
        self.fc5 = nn.Linear(512, 3200)

        self.conv1t = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3))
        self.conv2t = nn.ConvTranspose2d(128, 128, kernel_size=(3, 3))
        self.conv3t = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3))
        self.conv4t = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2), output_padding=(1, 1))
        self.conv5t = nn.ConvTranspose2d(64, 32, kernel_size=(3, 3))
        self.conv6t = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3))
        self.conv7t = nn.ConvTranspose2d(16, 1, kernel_size=(3, 3))

        self.bn1t = nn.BatchNorm2d(128)
        self.bn2t = nn.BatchNorm2d(128)
        self.bn3t = nn.BatchNorm2d(64)
        self.bn4t = nn.BatchNorm2d(64)
        self.bn5t = nn.BatchNorm2d(32)
        self.bn6t = nn.BatchNorm2d(16)

    def encode(self, x, params):
        x = F.leaky_relu(self.bn1(self.conv1(x))) 	# 30x30
        x = self.bn2(self.conv2(x))
        x = F.leaky_relu(self.pool1(x)) 			# 15x15
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))
        x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = self.bn6(self.conv6(x)) 				# 7x7

        x = F.leaky_relu(nn.MaxPool2d(kernel_size=x.size()[2:])(x))
        x = F.leaky_relu(x.view(x.size(0), -1))
        x = torch.cat([x, params], dim=1)
        return F.leaky_relu(self.fc1(x)), F.leaky_relu(self.fc2(x))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, params):
        z = F.leaky_relu(self.fc3(z))
        z = F.leaky_relu(self.fc4(torch.cat([z, params], dim=1)))
        z = F.leaky_relu(self.fc5(z))
        z = z.view(-1, 128, 5, 5)						# 5x5x128
        z = F.leaky_relu(self.bn1t(self.conv1t(z)))		# 7x7x128
        z = F.leaky_relu(self.bn2t(self.conv2t(z)))		# 9x9x128
        z = F.leaky_relu(self.bn3t(self.conv3t(z)))		# 11x11x64
        z = F.leaky_relu(self.bn4t(self.conv4t(z)))		# 24x24x64
        z = F.leaky_relu(self.bn5t(self.conv5t(z)))		# 26x26x32
        z = F.leaky_relu(self.bn6t(self.conv6t(z)))		# 28x28x16
        z = F.leaky_relu(self.conv7t(z))				# 30x30x133

        return z

    def forward(self, x, params):
        mu, logvar = self.encode(x, params)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, params), mu, logvar
