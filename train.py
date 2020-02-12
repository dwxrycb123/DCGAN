import torch
import torch.nn as nn
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
import os

from torchvision import transforms

def noise(B, feature_dim, device):
    return torch.randn((B, feature_dim)).to(device)

def train(G, D, args, train_dataloader, device):
    '''
    train the model with given hyper-paramaters   

    G : Generator \n
    D : Discriminator
    '''
    criterion = nn.BCELoss()

    G_optim = optim.Adam(G.parameters(), lr=args.lr)
    D_optim = optim.Adam(D.parameters(), lr=args.lr)

    G_losses = []
    D_losses = []
    images = []

    fix_noise = noise(60, 128, device)

    G.train()
    D.train()

    for epoch in range(args.pretrain+1, args.epochs+1):
        print('epoch:{}/{}'.format(epoch, args.epochs))
        D_loss = 0
        G_loss = 0

        # batch_num = len(train_dataloader)
        for i, data in tqdm(enumerate(train_dataloader, 0)):
            inputs, _ = data
            B = len(inputs)

            for j in range(args.D_epochs):
                fake_data = G(noise(B, 128, device)).detach() # fix Generator
                real_data = inputs.to(device)
                D_loss += D._train(D_optim, criterion, real_data, fake_data, device)

            fake_data = G(noise(B, 128, device))
            G_loss += G._train(G_optim, criterion, D, fake_data, device)
        
        image = G(fix_noise).cpu().detach()
        image = make_grid(image)
        transforms.ToPILImage()(image).save(os.path.join('.', 'process','epoch{}.jpg'.format(epoch)))        
        images.append(image)
        G_loss /= i
        D_loss /= i * args.D_epochs
        G_losses.append(G_loss)
        D_losses.append(D_loss)

        print('Epoch: {}, G_loss: {:4f}, D_loss: {:4f}'.format(epoch, G_loss, D_loss))
        if epoch % 3 == 0:
            torch.save(G.state_dict(), 'Generator_state_dict_{}epochs.pth'.format(epoch))
            torch.save(D.state_dict(), 'Discriminator_state_dict_{}epochs.pth'.format(epoch))


    print('Training finished. ')

    plt.plot(G_losses, label='Generator Losses')
    plt.plot(D_losses, label='Discriminator Losses')
    plt.legend()
    plt.savefig('loss.png')
    torch.save(G.state_dict(), 'Generator_state_dict.pth')

    # 
    imgs = [np.array(transforms.ToPILImage()(i)) for i in images]
    imageio.mimsave('progress.gif', imgs)