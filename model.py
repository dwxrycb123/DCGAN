import torch.nn as nn 
from functools import reduce
import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

class Generator(nn.Module):
    '''
    Generator model 

    output images with given vectors.
    '''
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 1024, 4, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 3, 4, 2, 1), # kernel_size=4, stride=2, padding=1 => conv: size /= 2, deconv: size *= 2
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(-1, 128, 1, 1)
        return self.layers(x)
    
    def _train(self, optimizer, criterion, discriminator, fake_data, device):
        B = fake_data.size(0)

        optimizer.zero_grad()
        loss = criterion(discriminator(fake_data), torch.ones((B, 1)).to(device))

        loss.backward()
        optimizer.step()

        return loss.cpu().detach()

class Discriminator(nn.Module):
    '''
    Discriminator model 

    output images with given vectors.
    '''
    def __init__(self, leaky_slope=0.2):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1),
            nn.LeakyReLU(leaky_slope),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_slope),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(leaky_slope),

            nn.Conv2d(512, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(leaky_slope),

            nn.Conv2d(1024, 1, 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x).view(-1, 1) 

    def _train(self, optimizer, criterion, real_data, fake_data, device):
        B = real_data.size(0)

        optimizer.zero_grad()

        loss_real = criterion(self.forward(real_data), torch.ones((B, 1)).to(device))
        loss_fake = criterion(self.forward(fake_data), torch.zeros((B, 1)).to(device))

        loss = loss_real + loss_fake
        loss.backward()
        optimizer.step()

        return loss.cpu().detach()

